#!/usr/bin/env python3
"""
evaluate_models.py  –  offline verification of custom_model_2 vs custom_model_3.

Each model's transform_payload is called once per evaluation day using the
same inputs the live submission runner would use (SMARD data source, quantile
objective). SMARD PV actuals (module 125) serve as ground truth.

Metrics
-------
MAE           mean |actual − Q50| over the 24 hourly timesteps
Pinball       mean quantile loss over all five quantiles (Q10…Q90) and timesteps
WIS           mean Winkler Interval Score, averaged over the 80 % (Q10–Q90)
              and 50 % (Q25–Q75) central intervals
Coverage      % of actuals inside Q10–Q90  (ideal ≈ 80 %)

Note on custom_model_3
----------------------
Open-Meteo forecast API covers today − 9 days … today + 2 days.
For evaluation days older than ~9 days, NWP fetch returns nothing and the
model returns the unchanged all-zero baseline.  Those days inflate MAE/WIS
and drag down coverage — this is expected and documented in the output.

Outputs
-------
model_artifacts/model_comparison.json
model_artifacts/model_comparison.png
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

HERE         = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "model_artifacts"

# SMARD module 125 = Photovoltaik Einspeisung DE-LU, quarter-hour resolution
SMARD_MODULE = 125
SMARD_REGION = "DE-LU"

TZ_BERLIN = ZoneInfo("Europe/Berlin")

_TODAY     = date.today()
EVAL_END   = _TODAY - timedelta(days=2)    # buffer for SMARD publication lag
EVAL_START = EVAL_END - timedelta(days=29) # 30 calendar days inclusive

N_STEPS   = 24        # hourly evaluation window (one calendar day)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

MODEL_NAMES = ["custom_model_2", "custom_model_3"]
COLORS      = {"custom_model_2": "tab:blue", "custom_model_3": "tab:orange"}

# ---------------------------------------------------------------------------
# Import ChallengeContext / SmardCounterpartSpec from the repo
# ---------------------------------------------------------------------------

sys.path.insert(0, str(HERE))
from _starter_core import ChallengeContext, SmardCounterpartSpec  # noqa: E402

# ChallengeContext fed to both models on every transform_payload call.
# smard_counterpart tells custom_model_2's load_source_series where to pull
# historical PV actuals (module 125, ×4 to convert MWh/15-min → MW).
_PV_CONTEXT = ChallengeContext(
    challenge_id             = "eval_pv",
    challenge_name           = "Evaluation – DE-LU PV solar",
    target_code              = "day_ahead_solar",
    target_name              = "DE-LU PV actual generation",
    area                     = "DE-LU",
    areas                    = ["DE-LU"],
    forecast_objective       = "quantile",
    accepted_forecast_format = "[Q10, Q25, Q50, Q75, Q90]",
    reference_timezone       = "Europe/Berlin",
    target_period_timezone   = "Europe/Berlin",
    target_period_type       = "calendar_day",
    probabilistic_quantiles  = [0.10, 0.25, 0.50, 0.75, 0.90],
    max_ensemble_size        = 100,
    smard_counterpart        = SmardCounterpartSpec(
        module_id        = SMARD_MODULE,
        region           = SMARD_REGION,
        resolution       = "quarterhour",
        source_unit      = "MWh",
        target_unit      = "MW",
        value_multiplier = 4.0,   # MWh / 15-min slot → average MW
    ),
    baseline_supported = False,
    challenge_detail   = {},
)


# ---------------------------------------------------------------------------
# Ground truth: SMARD PV actuals
# (logic mirrors train_model.py fetch_smard_pv_series exactly)
# ---------------------------------------------------------------------------

def fetch_smard_pv_hourly(start: date, end: date) -> pd.Series | None:
    """
    Download DE-LU PV generation from the SMARD chart-data API.

    Returns an hourly MW Series (Europe/Berlin tz) covering [start, end],
    or None on failure.
    """
    index_url = (
        f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}"
        f"/index_quarterhour.json"
    )
    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        all_ts: list[int] = resp.json().get("timestamps", [])
    except Exception as exc:
        print(f"[SMARD] index fetch failed: {exc}")
        return None

    # Weekly chunk timestamps that overlap the requested window
    # (8-day buffer at start so we don't miss the first partial week)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) - 8 * 86_400_000
    end_ms   = int(pd.Timestamp(end + timedelta(days=2), tz="UTC").timestamp() * 1000)
    relevant = sorted(ts for ts in all_ts if start_ms <= ts <= end_ms)

    print(f"[SMARD] fetching {len(relevant)} weekly chunks …")
    records: list[tuple] = []
    for i, ts_ms in enumerate(relevant):
        chunk_url = (
            f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}/"
            f"{SMARD_MODULE}_{SMARD_REGION}_quarterhour_{ts_ms}.json"
        )
        try:
            r = requests.get(chunk_url, timeout=30)
            r.raise_for_status()
            for row in r.json().get("series", []):
                if row and row[1] is not None:
                    records.append((
                        pd.Timestamp(row[0], unit="ms", tz="UTC"),
                        float(row[1]) * 4.0,   # MWh / 15-min → MW
                    ))
        except Exception as exc:
            print(f"[SMARD]   chunk {ts_ms} failed: {exc}")
        if (i + 1) % 10 == 0:
            print(f"[SMARD]   {i + 1}/{len(relevant)} done")
        time.sleep(0.12)

    if not records:
        return None

    idx, vals = zip(*records)
    s = (
        pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)))
        .sort_index()
        .groupby(level=0).mean()        # deduplicate overlapping chunk edges
        .tz_convert("Europe/Berlin")
    )
    s_hourly = s.resample("h").mean()

    t0 = pd.Timestamp(start, tz="Europe/Berlin")
    t1 = pd.Timestamp(end + timedelta(days=1), tz="Europe/Berlin")
    return s_hourly[(s_hourly.index >= t0) & (s_hourly.index < t1)]


# ---------------------------------------------------------------------------
# Dynamic model loading
# ---------------------------------------------------------------------------

def load_model(name: str):
    """Import a custom_model_*.py module from the same directory."""
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_payload() -> dict:
    """Fresh 5-quantile baseline payload for one 24-hour evaluation day."""
    return {
        "challenge_id": "eval_pv",
        "area":         "DE-LU",
        "values":       [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(N_STEPS)],
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def pinball(q: float, y: float, f: float) -> float:
    """Pinball (quantile) loss for a single observation."""
    diff = y - f
    return q * diff if diff >= 0.0 else (q - 1.0) * diff


def winkler_interval_score(alpha: float, lo: float, hi: float, y: float) -> float:
    """
    Winkler Interval Score for a (1-alpha) central prediction interval.
    IS = width  +  (2/alpha) * max(lo-y, 0)  +  (2/alpha) * max(y-hi, 0)
    """
    return (
        (hi - lo)
        + (2.0 / alpha) * max(lo - y, 0.0)
        + (2.0 / alpha) * max(y - hi, 0.0)
    )


def day_metrics(
    actuals: np.ndarray,        # shape (N_STEPS,)
    q_mat:   list[list[float]], # shape (N_STEPS, 5): [Q10, Q25, Q50, Q75, Q90]
) -> dict[str, float]:
    """Compute aggregated metrics for one forecast day."""
    mae_vals, pb_vals, wis_vals, cov_vals = [], [], [], []

    for y, row in zip(actuals, q_mat):
        q10, q25, q50, q75, q90 = row

        # MAE – point forecast is Q50
        mae_vals.append(abs(y - q50))

        # Pinball – average over all five quantiles
        pb_vals.append(
            float(np.mean([pinball(q, y, f) for q, f in zip(QUANTILES, row)]))
        )

        # WIS – mean of 80 % interval score (IS_0.20) and 50 % interval score (IS_0.50)
        is80 = winkler_interval_score(0.20, q10, q90, y)
        is50 = winkler_interval_score(0.50, q25, q75, y)
        wis_vals.append(0.5 * (is80 + is50))

        # Coverage – 1 if actual inside Q10–Q90
        cov_vals.append(1.0 if q10 <= y <= q90 else 0.0)

    return {
        "mae":      float(np.mean(mae_vals)),
        "pinball":  float(np.mean(pb_vals)),
        "wis":      float(np.mean(wis_vals)),
        "coverage": float(np.mean(cov_vals) * 100.0),
    }


# ---------------------------------------------------------------------------
# Normalise model output to a (N_STEPS × 5) quantile matrix
# ---------------------------------------------------------------------------

def to_quantile_matrix(
    values: list, n_steps: int
) -> list[list[float]] | None:
    """
    Accept any of the three payload formats used by our models:
      - 5-element list  → [Q10, Q25, Q50, Q75, Q90]   (quantile objective)
      - scalar          → point forecast, zero spread
      - ≥ 10-element list → ensemble members, derive empirical quantiles
    Returns None if the format is unrecognised.
    """
    if len(values) < n_steps:
        return None

    row0 = values[0]

    # --- 5-quantile ---
    if isinstance(row0, list) and len(row0) == 5:
        return [list(map(float, v[:5])) for v in values[:n_steps]]

    # --- Point forecast (scalar) → replicate as zero-spread quantiles ---
    if isinstance(row0, (int, float)):
        return [[float(v)] * 5 for v in values[:n_steps]]

    # --- Ensemble (≥ 10 members) → empirical percentiles ---
    if isinstance(row0, list) and len(row0) >= 10:
        return [
            [
                float(np.percentile(v, 10)),
                float(np.percentile(v, 25)),
                float(np.percentile(v, 50)),
                float(np.percentile(v, 75)),
                float(np.percentile(v, 90)),
            ]
            for v in values[:n_steps]
        ]

    return None   # unrecognised


# ---------------------------------------------------------------------------
# Per-model evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    name:         str,
    smard_hourly: pd.Series,
    eval_dates:   list[date],
) -> dict:
    """
    Run model ``name`` over every day in eval_dates.

    Returns
    -------
    {"daily": {iso_date: metrics_dict}, "aggregated": metrics_dict}
    or {} if the model cannot be loaded / has no transform_payload.
    """
    sep = "=" * 60
    print(f"\n{sep}\nEvaluating  {name}\n{sep}")

    # --- Load module ---
    try:
        mod = load_model(name)
    except Exception as exc:
        print(f"  [!] Cannot load {name}: {exc}")
        return {}

    if not hasattr(mod, "transform_payload"):
        print(f"  [!] {name} has no transform_payload – skipping")
        return {}

    daily: dict[str, dict] = {}
    skipped = 0

    for day in eval_dates:
        # -- Ground truth slice --
        t0      = pd.Timestamp(day, tz="Europe/Berlin")
        t1      = t0 + pd.Timedelta(hours=24)
        gt_hour = smard_hourly[(smard_hourly.index >= t0) & (smard_hourly.index < t1)]

        if len(gt_hour) < N_STEPS:
            print(f"  {day}: only {len(gt_hour)} SMARD hours available – skipping")
            skipped += 1
            continue

        actuals = np.maximum(gt_hour.values[:N_STEPS].astype(float), 0.0)

        # -- Build baseline payload and call model --
        payload         = make_payload()
        target_start_dt = datetime(day.year, day.month, day.day, tzinfo=TZ_BERLIN)

        try:
            out = mod.transform_payload(
                payload,
                target_date      = day,
                target_start     = target_start_dt,
                challenge_id     = "eval_pv",
                area             = "DE-LU",
                entsoe_api_key   = "",
                api_base         = "",
                data_source      = "smard",
                challenge_context= _PV_CONTEXT,
                challenge_detail = {},
                forecast_objective = "quantile",
                tz_name          = "Europe/Berlin",
            )
        except Exception as exc:
            print(f"  {day}: transform_payload raised {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        # -- Normalise to quantile matrix --
        vals = out.get("values") or []
        q_mat = to_quantile_matrix(vals, N_STEPS)
        if q_mat is None:
            row0 = vals[0] if vals else None
            print(f"  {day}: unrecognised payload format "
                  f"(first row={str(row0)[:60]!r}) – skipping")
            skipped += 1
            continue

        # -- Compute and store metrics --
        m = day_metrics(actuals, q_mat)
        daily[day.isoformat()] = m
        print(
            f"  {day}  "
            f"MAE={m['mae']:6.1f} MW  "
            f"Pinball={m['pinball']:6.2f}  "
            f"WIS={m['wis']:7.1f}  "
            f"Cov={m['coverage']:.0f}%"
        )

    n_evaluated = len(daily)
    print(f"  → {n_evaluated} days evaluated, {skipped} skipped")

    if not daily:
        return {}

    aggregated = {
        k: float(np.mean([v[k] for v in daily.values()]))
        for k in ("mae", "pinball", "wis", "coverage")
    }
    return {"daily": daily, "aggregated": aggregated}


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def save_comparison_plot(eval_dates: list[date], results: dict[str, dict]) -> Path:
    """
    4-panel figure:  MAE | Pinball | WIS | Coverage  (daily, over 30 days).
    One line per model; horizontal 80 % target line on the coverage panel.
    """
    metric_cfg = [
        ("mae",      "MAE (MW)",             "lower is better"),
        ("pinball",  "Mean Pinball Loss",     "lower is better"),
        ("wis",      "Mean WIS",              "lower is better"),
        ("coverage", "Q10–Q90 Coverage (%)", "target ≈ 80 %"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()

    x    = np.arange(len(eval_dates))
    xlbl = [d.strftime("%m-%d") for d in eval_dates]

    for ax_i, (key, ylabel, note) in enumerate(metric_cfg):
        ax = axes_flat[ax_i]
        for name in MODEL_NAMES:
            res = results.get(name) or {}
            if not res.get("daily"):
                continue
            y_series = np.array(
                [res["daily"].get(d.isoformat(), {}).get(key, float("nan"))
                 for d in eval_dates],
                dtype=float,
            )
            ax.plot(
                x, y_series,
                label  = name,
                color  = COLORS.get(name, "gray"),
                alpha  = 0.85,
                lw     = 1.5,
                marker = ".",
                ms     = 4,
            )

        if key == "coverage":
            ax.axhline(80.0, ls="--", lw=0.9, color="black",
                       alpha=0.55, label="80 % target")

        ax.set_title(f"{ylabel}  ({note})", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        step = max(1, len(eval_dates) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(xlbl[::step], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Model Comparison  ·  {EVAL_START} – {EVAL_END}  "
        f"({(EVAL_END - EVAL_START).days + 1} days)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = ARTIFACT_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    n_days = (EVAL_END - EVAL_START).days + 1
    print(f"{'─'*60}")
    print(f"Evaluation period : {EVAL_START}  →  {EVAL_END}  ({n_days} days)")
    print(f"Models            : {', '.join(MODEL_NAMES)}")
    print(f"Ground truth      : SMARD module {SMARD_MODULE} ({SMARD_REGION}), ×4 MW")
    print(f"{'─'*60}\n")

    # ------------------------------------------------------------------
    # 1. Ground truth
    # ------------------------------------------------------------------
    print("Fetching SMARD PV actuals …")
    smard_hourly = fetch_smard_pv_hourly(EVAL_START, EVAL_END)
    if smard_hourly is None or smard_hourly.empty:
        print("[!] Failed to retrieve SMARD data. Cannot evaluate.")
        sys.exit(1)
    print(
        f"[✓] {len(smard_hourly)} hourly observations  "
        f"({smard_hourly.index[0].date()} – {smard_hourly.index[-1].date()})\n"
    )

    eval_dates = [EVAL_START + timedelta(days=i) for i in range(n_days)]

    # ------------------------------------------------------------------
    # 2. Evaluate each model
    # ------------------------------------------------------------------
    all_results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        all_results[name] = evaluate_model(name, smard_hourly, eval_dates)

    # ------------------------------------------------------------------
    # 3. Print comparison table
    # ------------------------------------------------------------------
    col = [24, 10, 13, 10, 14]
    sep = "─" * (sum(col) + len(col) - 1)
    header = (
        f"{'Model':<{col[0]}}"
        f"{'MAE (MW)':>{col[1]}}"
        f"{'Pinball':>{col[2]}}"
        f"{'WIS':>{col[3]}}"
        f"{'Coverage':>{col[4]}}"
    )
    print(f"\n{sep}\n{header}\n{sep}")
    for name in MODEL_NAMES:
        res = all_results.get(name) or {}
        if not res.get("aggregated"):
            print(
                f"{name:<{col[0]}}"
                f"{'N/A':>{col[1]}}"
                f"{'N/A':>{col[2]}}"
                f"{'N/A':>{col[3]}}"
                f"{'N/A':>{col[4]}}"
            )
            continue
        a = res["aggregated"]
        n = len(res.get("daily") or {})
        print(
            f"{name:<{col[0]}}"
            f"{a['mae']:>{col[1]}.1f}"
            f"{a['pinball']:>{col[2]}.3f}"
            f"{a['wis']:>{col[3]}.1f}"
            f"{a['coverage']:>{col[4]-1}.1f}%"
            f"  ({n} days)"
        )
    print(sep)

    # ------------------------------------------------------------------
    # 4. Save JSON
    # ------------------------------------------------------------------
    out_dict = {
        "eval_start":    EVAL_START.isoformat(),
        "eval_end":      EVAL_END.isoformat(),
        "n_days":        n_days,
        "metrics":       list(("mae", "pinball", "wis", "coverage")),
        "models":        all_results,
    }
    json_path = ARTIFACT_DIR / "model_comparison.json"
    json_path.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(f"\n[✓] {json_path}")

    # ------------------------------------------------------------------
    # 5. Save plot
    # ------------------------------------------------------------------
    png_path = save_comparison_plot(eval_dates, all_results)
    print(f"[✓] {png_path}")


if __name__ == "__main__":
    main()
