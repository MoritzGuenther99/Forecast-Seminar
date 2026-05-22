#!/usr/bin/env python3
"""
experiment_clusters.py  –  Compare forecasting performance across different
numbers of geographic cluster centroids.

Tests k ∈ {3, 5, 8, 10, 15} capacity-weighted K-Means configurations with
identical LightGBM hyperparameters and the same SMARD PV target, so the only
variable is how many representative weather locations feed the feature matrix.

Usage:  python experiment_clusters.py

Outputs saved to ./model_artifacts/:
    weather_cache_k{n}.parquet      – per-k Open-Meteo weather cache
    weather_cache_k{n}_meta.json    – centroid coordinates for cache validation
    cluster_count_results.json      – CV RMSE for every k, plus best-k summary
    cluster_count_comparison.png    – RMSE vs k line plot with error bars
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_model import (
    ARTIFACT_DIR,
    TRAIN_END,
    TRAIN_START,
    _lgbm_params,
    build_feature_df,
    cluster_plants,
    fetch_open_meteo_archive,
    fetch_smard_pv_series,
    load_mastr_plants,
)

K_VALUES: list[int] = [3, 5, 8, 10, 15]
_WEATHER_VARS = ("ghi", "temp", "cloud", "precip", "wind", "dhi")


# ---------------------------------------------------------------------------
# Per-k weather cache  (wide parquet + centroid meta JSON)
# ---------------------------------------------------------------------------

def _cache_path(k: int) -> Path:
    return ARTIFACT_DIR / f"weather_cache_k{k}.parquet"


def _meta_path(k: int) -> Path:
    return ARTIFACT_DIR / f"weather_cache_k{k}_meta.json"


def _centroids_match(clusters: list[dict], k: int) -> bool:
    """True if the stored meta centroids match the current cluster set."""
    mp = _meta_path(k)
    if not mp.exists():
        return False
    try:
        stored = json.loads(mp.read_text(encoding="utf-8"))
        return len(stored) == len(clusters) and all(
            s["lat"] == c["lat"] and s["lon"] == c["lon"]
            for s, c in zip(stored, clusters)
        )
    except Exception:
        return False


def _save_weather_cache(
    weather_dfs: dict[str, pd.DataFrame],
    clusters: list[dict],
    k: int,
) -> None:
    cols: dict[str, pd.Series] = {}
    for name, df in weather_dfs.items():
        for var in _WEATHER_VARS:
            cols[f"{name}|{var}"] = df[var]
    pd.DataFrame(cols).to_parquet(_cache_path(k))
    _meta_path(k).write_text(
        json.dumps([{"lat": c["lat"], "lon": c["lon"]} for c in clusters], indent=2),
        encoding="utf-8",
    )
    print(f"  [cache] Saved weather_cache_k{k}.parquet")


def _load_weather_cache(
    k: int,
    clusters: list[dict],
) -> dict[str, pd.DataFrame] | None:
    """Return cached weather data, or None if cache is missing or stale."""
    if not _cache_path(k).exists() or not _centroids_match(clusters, k):
        return None
    wide = pd.read_parquet(_cache_path(k))
    result: dict[str, pd.DataFrame] = {}
    for c in clusters:
        name = c["name"]
        try:
            result[name] = pd.DataFrame(
                {var: wide[f"{name}|{var}"] for var in _WEATHER_VARS},
                index=wide.index,
            )
        except KeyError:
            return None  # incomplete cache – re-fetch
    print(f"  [cache] Loaded weather_cache_k{k}.parquet")
    return result


# ---------------------------------------------------------------------------
# Single-k experiment
# ---------------------------------------------------------------------------

def run_k(
    k: int,
    df_plants: pd.DataFrame,
    smard: pd.Series | None,
) -> dict:
    """
    Full pipeline for one cluster count:
      cluster_plants → weather (cached) → build_feature_df →
      TimeSeriesSplit(n_splits=5) CV RMSE with identical hyperparameters.
    """
    header = f"  k = {k}  ({k} geographic clusters)"
    print(f"\n{'='*60}\n{header}\n{'='*60}")

    # Step 2: capacity-weighted K-Means
    clusters = cluster_plants(df_plants, k)

    # Step 3: weather data (load cache or fetch)
    weather_dfs = _load_weather_cache(k, clusters)
    if weather_dfs is None:
        print(f"  Fetching Open-Meteo archive for {k} centroids …")
        weather_dfs = {}
        for c in clusters:
            print(f"    {c['name']}  lat={c['lat']}, lon={c['lon']} …")
            weather_dfs[c["name"]] = fetch_open_meteo_archive(
                c["lat"], c["lon"], TRAIN_START, TRAIN_END
            )
            time.sleep(1.5)
        _save_weather_cache(weather_dfs, clusters, k)

    # Step 4: feature matrix
    common_index = weather_dfs[clusters[0]["name"]].index
    X_df         = build_feature_df(weather_dfs, common_index, clusters)
    n_features   = X_df.shape[1]
    X            = X_df.values
    print(f"  Feature matrix: {len(common_index):,} × {n_features}  "
          f"(= {k}×6 cluster + 20 shared)")

    # Align SMARD target; synthetic fallback if unavailable
    smard_ok = smard is not None and not smard.empty
    if smard_ok:
        y = smard.reindex(common_index).ffill().fillna(0.0).values
    else:
        rng = np.random.default_rng(42)
        y   = np.maximum(
            X_df["w_ghi"].values * 0.15
            + rng.normal(0, X_df["w_ghi"].values * 0.03 + 0.5),
            0.0,
        )

    mask  = np.isfinite(y) & (y >= 0)
    X, y  = X[mask], y[mask]
    idx_m = common_index[mask]

    # Last-6-months recency weighting (matches train_model.py §6)
    cutoff = pd.Timestamp(TRAIN_END - timedelta(days=180), tz="Europe/Berlin")
    sw     = np.where(idx_m >= cutoff, 2.0, 1.0)

    # Step 5: TimeSeriesSplit CV
    tscv       = TimeSeriesSplit(n_splits=5)
    rmse_folds: list[float] = []
    print("  TimeSeriesSplit CV (5 folds) …")
    for fold, (tr, va) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**_lgbm_params("regression"))
        m.fit(
            X[tr], y[tr],
            sample_weight=sw[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        rmse = float(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
        rmse_folds.append(rmse)
        print(f"    Fold {fold+1}: RMSE = {rmse:.2f} MW")

    mean_rmse = float(np.mean(rmse_folds))
    std_rmse  = float(np.std(rmse_folds))
    print(f"  → CV RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} MW")

    return {
        "k":             k,
        "n_features":    n_features,
        "cv_rmse_folds": [round(r, 4) for r in rmse_folds],
        "cv_rmse_mean":  round(mean_rmse, 4),
        "cv_rmse_std":   round(std_rmse,  4),
        "clusters":      clusters,
    }


# ---------------------------------------------------------------------------
# Result plot
# ---------------------------------------------------------------------------

def save_comparison_plot(results: list[dict]) -> None:
    k_vals   = [r["k"] for r in results]
    means    = np.array([r["cv_rmse_mean"] for r in results])
    stds     = np.array([r["cv_rmse_std"]  for r in results])
    best_idx = int(np.argmin(means))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        k_vals, means, yerr=stds,
        fmt="o-", capsize=5, lw=1.8,
        color="steelblue", ecolor="gray", elinewidth=1.2,
        label="CV RMSE ± std",
    )
    ax.scatter(
        [k_vals[best_idx]], [means[best_idx]],
        s=220, marker="*", color="gold", edgecolors="k",
        linewidths=0.8, zorder=5,
        label=f"Best: k={k_vals[best_idx]}  ({means[best_idx]:.1f} MW)",
    )

    # Arrow annotation on the best point
    ax.annotate(
        f"k={k_vals[best_idx]}\n{means[best_idx]:.1f} MW",
        xy=(k_vals[best_idx], means[best_idx]),
        xytext=(14, 12), textcoords="offset points", fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    # Value labels on all other points
    for i, (k, m) in enumerate(zip(k_vals, means)):
        if i == best_idx:
            continue
        ax.annotate(
            f"{m:.1f}",
            xy=(k, m),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8, color="dimgray",
        )

    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("CV RMSE (MW)", fontsize=11)
    ax.set_title(
        "Forecast RMSE vs. Number of Geographic Cluster Centroids",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ARTIFACT_DIR / "cluster_count_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[✓] cluster_count_comparison.png saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    print(f"[experiment_clusters] Training window: {TRAIN_START} → {TRAIN_END}")
    print(f"K values: {K_VALUES}\n")

    # Step 1: load MaStR plants from parquet cache (created by train_model.py)
    df_plants = load_mastr_plants()
    df_plants = df_plants[
        df_plants["lat"].between(47.0, 55.5) & df_plants["lon"].between(5.5, 15.5)
    ].copy()
    print(f"[MaStR] {len(df_plants):,} plants after Germany geo-filter")

    # Fetch SMARD target once and reuse across all k values
    print("\nFetching SMARD PV target …")
    smard    = fetch_smard_pv_series(TRAIN_START, TRAIN_END)
    smard_ok = smard is not None and not smard.empty
    if smard_ok:
        print(f"[SMARD] OK  ({len(smard):,} hourly values)")
    else:
        print("[SMARD] unavailable – will use synthetic fallback for all runs")

    # Run experiment for each k
    all_results: list[dict] = []
    for k in K_VALUES:
        all_results.append(run_k(k, df_plants, smard))

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'k':>4}  {'RMSE mean (MW)':>15}  {'std (MW)':>9}  {'features':>8}")
    print(f"  {'─'*4}  {'─'*15}  {'─'*9}  {'─'*8}")
    best_k = min(all_results, key=lambda r: r["cv_rmse_mean"])["k"]
    for r in all_results:
        tag = "  ← best" if r["k"] == best_k else ""
        print(
            f"  {r['k']:>4}  {r['cv_rmse_mean']:>15.2f}  "
            f"{r['cv_rmse_std']:>9.2f}  {r['n_features']:>8}{tag}"
        )

    # Save JSON (strip per-cluster raw data to keep file readable)
    output: dict = {
        "training_date":   date.today().isoformat(),
        "train_start":     TRAIN_START.isoformat(),
        "train_end":       TRAIN_END.isoformat(),
        "smard_available": smard_ok,
        "k_values":        K_VALUES,
        "best_k":          best_k,
        "best_rmse_mean":  round(
            min(r["cv_rmse_mean"] for r in all_results), 4
        ),
        "results": [
            {
                "k":             r["k"],
                "n_features":    r["n_features"],
                "cv_rmse_folds": r["cv_rmse_folds"],
                "cv_rmse_mean":  r["cv_rmse_mean"],
                "cv_rmse_std":   r["cv_rmse_std"],
                "centroids": [
                    {"name": c["name"], "lat": c["lat"], "lon": c["lon"],
                     "capacity_gw": c["capacity_gw"]}
                    for c in r["clusters"]
                ],
            }
            for r in all_results
        ],
    }
    out_json = ARTIFACT_DIR / "cluster_count_results.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] cluster_count_results.json saved")

    save_comparison_plot(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
