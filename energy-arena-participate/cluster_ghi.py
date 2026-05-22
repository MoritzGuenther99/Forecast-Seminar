#!/usr/bin/env python3
"""
cluster_ghi.py  –  GHI-pattern clustering as an alternative to geographic clustering.

Terren-Serrano et al. 2026: spatial weather pattern clustering selects representative
forecast locations by grouping 1°×1° grid points with similar GHI seasonality/daily
profiles, then weighting by MaStR installed capacity density.

Usage:  python cluster_ghi.py

Outputs saved to ./model_artifacts/:
    ghi_grid_cache.parquet          – cached hourly GHI for all grid points (one-time fetch)
    clusters_ghi.json               – 10 GHI-pattern cluster centroids (same schema as clusters.json)
    cluster_map_ghi.png             – grid points colored by cluster + capacity-weighted centroids
    cluster_comparison.png          – side-by-side geographic vs GHI-pattern cluster maps
    cluster_comparison_results.json – CV RMSE for both clustering approaches

Runtime estimate: ~15 min (80 API calls + 2× 5-fold CV training).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import shared utilities from train_model (same directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_model import (
    ARTIFACT_DIR,
    N_CLUSTERS,
    SUNNY_THRESHOLD,
    TRAIN_END,
    TRAIN_START,
    _lgbm_params,
    build_feature_df,
    fetch_open_meteo_archive,
    fetch_smard_pv_series,
    load_mastr_plants,
)

GHI_CACHE = ARTIFACT_DIR / "ghi_grid_cache.parquet"

# ---------------------------------------------------------------------------
# Step 1: Germany 1°×1° grid
# ---------------------------------------------------------------------------

def make_germany_grid() -> list[tuple[float, float]]:
    """
    1°×1° grid covering Germany: lat 47.5–55.5°N, lon 6.0–15.0°E.
    9 × 10 = 90 grid points; simple bounding box, no further land-mask needed.
    """
    lats = np.arange(47.5, 56.0, 1.0)  # 47.5 … 55.5  (9 values)
    lons = np.arange(6.0,  16.0, 1.0)  # 6.0  … 15.0  (10 values)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


# ---------------------------------------------------------------------------
# Step 2: Fetch and cache GHI grid data
# ---------------------------------------------------------------------------

def _col(lat: float, lon: float) -> str:
    return f"{lat}_{lon}"


def load_ghi_grid(grid_points: list[tuple[float, float]]) -> pd.DataFrame:
    """
    Return wide-format hourly GHI (index=timestamp Europe/Berlin, columns="{lat}_{lon}").
    Incrementally fetches and caches missing grid points so a failed run can resume.
    """
    ARTIFACT_DIR.mkdir(exist_ok=True)
    expected = {_col(*p) for p in grid_points}

    cached: pd.DataFrame
    if GHI_CACHE.exists():
        cached  = pd.read_parquet(GHI_CACHE)
        missing = [p for p in grid_points if _col(*p) not in cached.columns]
        if not missing:
            print(f"[GHI cache] All {len(cached.columns)} points loaded from {GHI_CACHE.name}")
            return cached[sorted(expected & set(cached.columns))]
        print(f"[GHI cache] {len(cached.columns)} / {len(grid_points)} points cached; "
              f"fetching {len(missing)} missing …")
    else:
        cached  = pd.DataFrame()
        missing = grid_points

    new_cols: dict[str, pd.Series] = {}
    for i, (lat, lon) in enumerate(missing):
        cname = _col(lat, lon)
        print(f"  [{i+1}/{len(missing)}] Open-Meteo ({lat:.1f}, {lon:.1f}) …", flush=True)
        try:
            df            = fetch_open_meteo_archive(lat, lon, TRAIN_START, TRAIN_END)
            new_cols[cname] = df["ghi"]
        except Exception as exc:
            print(f"    [!] {exc} – filling with NaN")
            new_cols[cname] = pd.Series(dtype=float)
        if i < len(missing) - 1:
            time.sleep(1.5)

    new_df = pd.DataFrame(new_cols)
    result = new_df if cached.empty else cached.join(new_df, how="outer")
    result = result.sort_index()
    result.to_parquet(GHI_CACHE)
    print(f"[GHI cache] Saved → {GHI_CACHE.name}  "
          f"({len(result.columns)} points, {len(result):,} hours)")
    return result


# ---------------------------------------------------------------------------
# Step 3: GHI feature matrix (41 features per grid point)
# ---------------------------------------------------------------------------

# Month → season index: 0=DJF, 1=MAM, 2=JJA, 3=SON
_SEASON_IDX: dict[int, int] = {
    12: 0, 1: 0, 2: 0,
    3: 1,  4: 1, 5: 1,
    6: 2,  7: 2, 8: 2,
    9: 3,  10: 3, 11: 3,
}


def build_ghi_features(
    ghi_wide: pd.DataFrame,
    grid_points: list[tuple[float, float]],
) -> np.ndarray:
    """
    41 aggregated GHI features per grid point (Terren-Serrano et al. 2026):
      12  monthly mean GHI        – captures seasonality
      24  hourly mean GHI         – captures daily irradiance profile
       4  seasonal sunny-day fraction (daily mean >= SUNNY_THRESHOLD, Nespoli 2019)
       1  annual mean GHI
    """
    rows: list[list[float]] = []
    for lat, lon in grid_points:
        col = _col(lat, lon)
        ghi = ghi_wide[col].dropna() if col in ghi_wide.columns else pd.Series(dtype=float)

        if ghi.empty:
            rows.append([0.0] * 41)
            continue

        # 12 monthly means
        by_month = ghi.groupby(ghi.index.month).mean()
        m_feats  = [float(by_month.get(m, 0.0)) for m in range(1, 13)]

        # 24 hourly means
        by_hour = ghi.groupby(ghi.index.hour).mean()
        h_feats = [float(by_hour.get(h, 0.0)) for h in range(24)]

        # 4 seasonal sunny-day fractions
        daily    = ghi.resample("D").mean()
        is_sunny = daily >= SUNNY_THRESHOLD
        cnt      = [0, 0, 0, 0]
        sunny_n  = [0, 0, 0, 0]
        for ts, flag in zip(daily.index, is_sunny.values):
            s         = _SEASON_IDX[ts.month]
            cnt[s]   += 1
            sunny_n[s] += int(flag)
        s_feats = [sunny_n[s] / cnt[s] if cnt[s] > 0 else 0.0 for s in range(4)]

        # 1 annual mean
        a_feat = [float(ghi.mean())]

        rows.append(m_feats + h_feats + s_feats + a_feat)

    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Step 4: GHI-pattern K-Means + capacity-weighted centroids
# ---------------------------------------------------------------------------

def _nearest_grid_idx(
    df_plants: pd.DataFrame,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> np.ndarray:
    """
    Index of the nearest grid point for each plant, computed in 100k-row chunks
    to keep peak RAM below ~130 MB regardless of dataset size.
    """
    plant_lats = df_plants["lat"].values
    plant_lons = df_plants["lon"].values
    n          = len(df_plants)
    out        = np.empty(n, dtype=np.int32)
    chunk      = 100_000
    for start in range(0, n, chunk):
        end            = min(start + chunk, n)
        dlat           = plant_lats[start:end, None] - grid_lats[None, :]
        dlon           = plant_lons[start:end, None] - grid_lons[None, :]
        out[start:end] = (dlat**2 + dlon**2).argmin(axis=1)
    return out


def cluster_ghi_patterns(
    ghi_wide: pd.DataFrame,
    grid_points: list[tuple[float, float]],
    df_plants: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
) -> tuple[list[dict], np.ndarray]:
    """
    Terren-Serrano et al. 2026: GHI-pattern clustering.

    1. Build 41-feature GHI matrix, normalize with StandardScaler.
    2. K-Means on normalized features.
    3. Assign MaStR plants to nearest grid point → capacity per grid point.
    4. Capacity-weighted centroid per cluster.

    Returns (clusters, grid_labels) where grid_labels[i] is the final sorted
    cluster index for grid_points[i].
    """
    print("[GHI-Cluster] Building GHI feature matrix …")
    F        = build_ghi_features(ghi_wide, grid_points)
    F_scaled = StandardScaler().fit_transform(F)

    print(f"[GHI-Cluster] K-Means ({n_clusters} clusters, n_init=10) …")
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(F_scaled)  # raw labels 0 … n_clusters-1

    grid_lats = np.array([p[0] for p in grid_points])
    grid_lons = np.array([p[1] for p in grid_points])

    # Per-grid-point installed capacity from MaStR plants
    print("[GHI-Cluster] Assigning plants to nearest grid point …")
    gp_idx      = _nearest_grid_idx(df_plants, grid_lats, grid_lons)
    plant_kw    = df_plants["kw"].values
    total_kw    = float(plant_kw.sum())
    gp_capacity = np.zeros(len(grid_points))
    for idx, kw in zip(gp_idx, plant_kw):
        gp_capacity[idx] += kw

    # Build unsorted cluster entries
    raw: list[dict] = []
    for c in range(n_clusters):
        mask   = labels == c
        lats_c = grid_lats[mask]
        lons_c = grid_lons[mask]
        kws_c  = gp_capacity[mask].copy()
        ck     = float(kws_c.sum())
        if ck == 0:
            # Equal-weight fallback for clusters that received no plant assignments
            kws_c = np.ones(mask.sum())
            ck    = float(mask.sum())
        raw.append({
            "raw_label":   c,
            "lat":         round(float((lats_c * kws_c).sum() / kws_c.sum()), 4),
            "lon":         round(float((lons_c * kws_c).sum() / kws_c.sum()), 4),
            "capacity_kw": float(gp_capacity[mask].sum()),
        })

    # Sort by installed capacity descending so cluster_0 = most important
    raw.sort(key=lambda c: c["capacity_kw"], reverse=True)
    raw_to_new = {c["raw_label"]: i for i, c in enumerate(raw)}
    new_labels = np.array([raw_to_new[l] for l in labels], dtype=np.int32)

    clusters: list[dict] = [
        {
            "name":        f"cluster_{i}",
            "lat":         rc["lat"],
            "lon":         rc["lon"],
            "weight":      round(rc["capacity_kw"] / total_kw, 6),
            "capacity_gw": round(rc["capacity_kw"] / 1e6, 3),
        }
        for i, rc in enumerate(raw)
    ]

    print("[GHI-Cluster] Summary (sorted by capacity):")
    for c in clusters:
        print(f"  {c['name']:10s}  lat={c['lat']:6.2f}  lon={c['lon']:5.2f}"
              f"  {c['capacity_gw']:5.1f} GW  w={c['weight']:.4f}")
    print(f"  Weight sum: {sum(c['weight'] for c in clusters):.6f}  (≈ 1.0)")

    return clusters, new_labels


# ---------------------------------------------------------------------------
# Step 5: Visualisations
# ---------------------------------------------------------------------------

_GER_XLIM = (6.0, 15.0)
_GER_YLIM = (47.0, 55.0)


def _style_germany_ax(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(*_GER_XLIM)
    ax.set_ylim(*_GER_YLIM)
    ax.set_xlabel("Lon (°E)")
    ax.set_ylabel("Lat (°N)")
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("#dce9f5")
    ax.grid(alpha=0.4, linestyle="--")


def _centroid_sizes(clusters: list[dict]) -> np.ndarray:
    cap = np.array([c["capacity_gw"] for c in clusters])
    return 80.0 + 600.0 * (cap / cap.max())


def save_cluster_map_ghi(
    grid_points: list[tuple[float, float]],
    grid_labels: np.ndarray,
    clusters: list[dict],
) -> None:
    """cluster_map_ghi.png: grid points colored by cluster + large centroid dots."""
    gp_lats = np.array([p[0] for p in grid_points])
    gp_lons = np.array([p[1] for p in grid_points])
    n       = len(clusters)

    fig, ax = plt.subplots(figsize=(7, 9))
    _style_germany_ax(ax, "GHI-Pattern Clusters – Grid Points & Capacity-Weighted Centroids")

    # Small dots: grid points colored by cluster
    ax.scatter(
        gp_lons, gp_lats,
        c=grid_labels, cmap="tab10", vmin=0, vmax=9,
        s=35, alpha=0.65, edgecolors="none", zorder=2,
    )
    # Large dots: cluster centroids sized by installed capacity
    ax.scatter(
        [c["lon"] for c in clusters],
        [c["lat"] for c in clusters],
        c=np.arange(n), cmap="tab10", vmin=0, vmax=9,
        s=_centroid_sizes(clusters),
        alpha=0.9, edgecolors="k", linewidths=0.8, zorder=4,
    )
    for i, c in enumerate(clusters):
        ax.annotate(
            f"C{i}\n{c['capacity_gw']:.1f} GW",
            xy=(c["lon"], c["lat"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, zorder=5,
        )

    fig.tight_layout()
    out = ARTIFACT_DIR / "cluster_map_ghi.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[✓] {out.name} saved")


def save_cluster_comparison(
    geo_clusters: list[dict],
    ghi_clusters: list[dict],
    grid_points: list[tuple[float, float]],
    grid_labels: np.ndarray,
) -> None:
    """cluster_comparison.png: side-by-side geographic vs GHI-pattern cluster maps."""
    gp_lats = np.array([p[0] for p in grid_points])
    gp_lons = np.array([p[1] for p in grid_points])
    n_geo   = len(geo_clusters)
    n_ghi   = len(ghi_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # ---- Left: geographic (capacity-weighted K-Means on plant locations) ----
    ax = axes[0]
    _style_germany_ax(
        ax,
        "Geographic Clustering\n(capacity-weighted K-Means on plant locations)",
    )
    ax.scatter(
        [c["lon"] for c in geo_clusters],
        [c["lat"] for c in geo_clusters],
        c=np.arange(n_geo), cmap="tab10", vmin=0, vmax=9,
        s=_centroid_sizes(geo_clusters),
        alpha=0.9, edgecolors="k", linewidths=0.8, zorder=3,
    )
    for i, c in enumerate(geo_clusters):
        ax.annotate(
            f"C{i}\n{c['capacity_gw']:.1f} GW",
            xy=(c["lon"], c["lat"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )

    # ---- Right: GHI-pattern (K-Means on GHI profiles) ----
    ax = axes[1]
    _style_germany_ax(
        ax,
        "GHI-Pattern Clustering\n(K-Means on monthly/hourly/seasonal GHI profiles)",
    )
    ax.scatter(
        gp_lons, gp_lats,
        c=grid_labels, cmap="tab10", vmin=0, vmax=9,
        s=22, alpha=0.5, edgecolors="none", zorder=2,
    )
    ax.scatter(
        [c["lon"] for c in ghi_clusters],
        [c["lat"] for c in ghi_clusters],
        c=np.arange(n_ghi), cmap="tab10", vmin=0, vmax=9,
        s=_centroid_sizes(ghi_clusters),
        alpha=0.9, edgecolors="k", linewidths=0.8, zorder=4,
    )
    for i, c in enumerate(ghi_clusters):
        ax.annotate(
            f"C{i}\n{c['capacity_gw']:.1f} GW",
            xy=(c["lon"], c["lat"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )

    fig.tight_layout()
    out = ARTIFACT_DIR / "cluster_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[✓] {out.name} saved")


# ---------------------------------------------------------------------------
# Step 6: Training comparison (identical hyperparameters, same SMARD target)
# ---------------------------------------------------------------------------

def _fetch_weather_for_clusters(clusters: list[dict], label: str) -> dict[str, pd.DataFrame]:
    print(f"[Weather/{label}] Fetching for {len(clusters)} centroids …")
    weather: dict[str, pd.DataFrame] = {}
    for c in clusters:
        print(f"  {c['name']}  lat={c['lat']}, lon={c['lon']} …")
        weather[c["name"]] = fetch_open_meteo_archive(
            c["lat"], c["lon"], TRAIN_START, TRAIN_END
        )
        time.sleep(1.5)
    return weather


def evaluate_clusters(
    clusters: list[dict],
    smard_series: pd.Series | None,
    label: str,
) -> dict:
    """
    Full pipeline for one cluster set: weather fetch → feature matrix →
    TimeSeriesSplit(n_splits=5) CV RMSE with identical LightGBM hyperparameters.
    """
    weather_dfs  = _fetch_weather_for_clusters(clusters, label)
    common_index = weather_dfs[clusters[0]["name"]].index

    print(f"[{label}] Building feature matrix ({len(common_index):,} hourly steps) …")
    X_df = build_feature_df(weather_dfs, common_index, clusters)
    X    = X_df.values

    smard_ok = smard_series is not None and not smard_series.empty
    if smard_ok:
        y = smard_series.reindex(common_index).ffill().fillna(0.0).values
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

    cutoff = pd.Timestamp(TRAIN_END - timedelta(days=180), tz="Europe/Berlin")
    sw     = np.where(idx_m >= cutoff, 2.0, 1.0)

    tscv       = TimeSeriesSplit(n_splits=5)
    rmse_folds: list[float] = []
    print(f"[{label}] TimeSeriesSplit CV (5 folds) …")
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
        print(f"  Fold {fold+1}: RMSE = {rmse:.2f} MW")

    result = {
        "label":         label,
        "cv_rmse_folds": [round(r, 4) for r in rmse_folds],
        "cv_rmse_mean":  round(float(np.mean(rmse_folds)), 4),
        "cv_rmse_std":   round(float(np.std(rmse_folds)),  4),
    }
    print(f"[{label}] CV RMSE: {result['cv_rmse_mean']:.2f} ± {result['cv_rmse_std']:.2f} MW")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    print(f"[cluster_ghi] Training window: {TRAIN_START} → {TRAIN_END}\n")

    # ------------------------------------------------------------------
    # 1. Grid
    # ------------------------------------------------------------------
    grid_points = make_germany_grid()
    print(f"[Grid] {len(grid_points)} points  "
          f"lat {min(p[0] for p in grid_points)}–{max(p[0] for p in grid_points)}°N  "
          f"lon {min(p[1] for p in grid_points)}–{max(p[1] for p in grid_points)}°E")

    # ------------------------------------------------------------------
    # 2. MaStR plants (shared with train_model; parquet cache reused)
    # ------------------------------------------------------------------
    df_plants = load_mastr_plants()
    df_plants = df_plants[
        df_plants["lat"].between(47.0, 55.5) & df_plants["lon"].between(5.5, 15.5)
    ].copy()
    print(f"[MaStR] {len(df_plants):,} plants after Germany geo-filter")

    # ------------------------------------------------------------------
    # 3. GHI grid time series (incrementally cached)
    # ------------------------------------------------------------------
    ghi_wide = load_ghi_grid(grid_points)

    # ------------------------------------------------------------------
    # 4. GHI-pattern clustering
    # ------------------------------------------------------------------
    ghi_clusters, grid_labels = cluster_ghi_patterns(
        ghi_wide, grid_points, df_plants
    )
    (ARTIFACT_DIR / "clusters_ghi.json").write_text(
        json.dumps(ghi_clusters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[✓] clusters_ghi.json saved\n")

    # ------------------------------------------------------------------
    # 5. Visualisations
    # ------------------------------------------------------------------
    save_cluster_map_ghi(grid_points, grid_labels, ghi_clusters)

    geo_clusters: list[dict] = []
    geo_path = ARTIFACT_DIR / "clusters.json"
    if geo_path.exists():
        geo_clusters = json.loads(geo_path.read_text(encoding="utf-8"))
        save_cluster_comparison(geo_clusters, ghi_clusters, grid_points, grid_labels)
    else:
        print("[!] clusters.json not found – run train_model.py first for comparison plot")

    # ------------------------------------------------------------------
    # 6. Model comparison
    # ------------------------------------------------------------------
    print("\n[Comparison] Fetching SMARD PV target …")
    smard    = fetch_smard_pv_series(TRAIN_START, TRAIN_END)
    smard_ok = smard is not None and not smard.empty
    print(f"[Comparison] SMARD: {'OK' if smard_ok else 'unavailable (synthetic fallback)'}\n")

    results: list[dict] = []

    if geo_clusters:
        results.append(evaluate_clusters(geo_clusters, smard, "Geographic"))
        print()

    results.append(evaluate_clusters(ghi_clusters, smard, "GHI-pattern"))

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:20s}  CV RMSE = {r['cv_rmse_mean']:.2f} ± {r['cv_rmse_std']:.2f} MW")

    comparison: dict = {"results": results}
    if len(results) == 2:
        geo_m  = results[0]["cv_rmse_mean"]
        ghi_m  = results[1]["cv_rmse_mean"]
        delta  = geo_m - ghi_m          # positive = GHI wins
        pct    = 100.0 * delta / geo_m
        winner = "GHI-pattern" if ghi_m < geo_m else "Geographic"
        margin = abs(pct)
        summary = (
            f"{winner} clustering wins by {margin:.1f}%  "
            f"(geo={geo_m:.2f} MW, ghi-pattern={ghi_m:.2f} MW)"
        )
        print(f"  → {summary}")
        comparison.update({
            "winner":    winner,
            "delta_mw":  round(delta, 4),
            "delta_pct": round(pct, 2),
            "summary":   summary,
        })

    (ARTIFACT_DIR / "cluster_comparison_results.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[✓] cluster_comparison_results.json saved")
    print("Done.")


if __name__ == "__main__":
    main()
