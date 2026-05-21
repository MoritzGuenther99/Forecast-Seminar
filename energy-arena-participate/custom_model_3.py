"""
custom_model_3.py  –  LightGBM PV solar forecast (inference).

Same transform_payload interface as custom_model_2.py.
Forecast locations are loaded at startup from clusters.json, which was
created by train_model.py using capacity-weighted K-Means on MaStR data.
Models are loaded once on the first call and cached at module level.

References:
    Nespoli et al. 2019:       150 W/m² daily mean GHI → sunny/cloudy threshold
    Terren-Serrano et al. 2026: NWP (ICON/GFS) forecast at inference time;
                                capacity-density spatial masking for site selection
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Module-level model + cluster cache  (populated on first transform_payload call)
# ---------------------------------------------------------------------------

_MODELS_LOADED  = False
_MODEL_POINT    = None
_MODEL_Q10      = None
_MODEL_Q90      = None
_FEATURE_NAMES: list[str]  = []
_CLUSTERS:      list[dict] = []   # loaded from clusters.json

_ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"

# Nespoli et al. 2019, §2.3
_SUNNY_THRESHOLD = 150.0  # W/m²


# ---------------------------------------------------------------------------
# Model + cluster loading
# ---------------------------------------------------------------------------

def _load_models() -> bool:
    """
    Lazy-load model files and clusters.json.
    Returns True on success; on any error prints a warning and returns False
    so transform_payload can fall back to the baseline payload.
    """
    global _MODELS_LOADED, _MODEL_POINT, _MODEL_Q10, _MODEL_Q90
    global _FEATURE_NAMES, _CLUSTERS

    if _MODELS_LOADED:
        return True

    try:
        import joblib  # optional dep; caught here so the module still imports without it

        _MODEL_POINT   = joblib.load(_ARTIFACT_DIR / "lgbm_point.pkl")
        _MODEL_Q10     = joblib.load(_ARTIFACT_DIR / "lgbm_quantile_low.pkl")
        _MODEL_Q90     = joblib.load(_ARTIFACT_DIR / "lgbm_quantile_high.pkl")
        _FEATURE_NAMES = json.loads(
            (_ARTIFACT_DIR / "feature_names.json").read_text(encoding="utf-8")
        )
        _CLUSTERS = json.loads(
            (_ARTIFACT_DIR / "clusters.json").read_text(encoding="utf-8")
        )
        _MODELS_LOADED = True
        print(
            f"[custom_model_3] Loaded {len(_CLUSTERS)} clusters, "
            f"{len(_FEATURE_NAMES)} features  ←  {_ARTIFACT_DIR}"
        )
        return True

    except Exception as exc:
        print(
            f"[custom_model_3] Cannot load models/clusters: {exc}. "
            "Run train_model.py first. Falling back to baseline payload."
        )
        return False


# ---------------------------------------------------------------------------
# Open-Meteo NWP forecast fetch
# Terren-Serrano et al. 2026: ICON/GFS NWP forecast data used at inference time
# ---------------------------------------------------------------------------

def _fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    *,
    past_days: int = 9,
) -> pd.DataFrame:
    """
    Hourly NWP forecast + recent past from the Open-Meteo forecast API.
    past_days=9 ensures all lag features (1d, 2d, 7d) are populated for tomorrow.
    """
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":     lat,
        "longitude":    lon,
        "hourly":       "shortwave_radiation,temperature_2m,cloudcover,precipitation,windspeed_10m,diffuse_radiation",
        "forecast_days": 2,
        "past_days":    past_days,
        "timezone":     "Europe/Berlin",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    h = resp.json()["hourly"]

    def _c(vals: list, fb: float = 0.0) -> list:
        return [v if v is not None else fb for v in vals]

    df = pd.DataFrame(
        {
            "ghi":    _c(h["shortwave_radiation"]),
            "temp":   _c(h["temperature_2m"],    fb=float("nan")),
            "cloud":  _c(h["cloudcover"],         fb=float("nan")),
            "precip": _c(h["precipitation"]),
            "wind":   _c(h["windspeed_10m"],      fb=float("nan")),
            "dhi":    _c(h["diffuse_radiation"]),
        },
        index=pd.to_datetime(h["time"]),
    )
    try:
        df.index = df.index.tz_localize(
            "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        )
    except Exception:
        df.index = df.index.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        df = df[df.index.notna()]

    df["temp"]  = df["temp"].ffill().fillna(10.0)
    df["cloud"] = df["cloud"].ffill().fillna(50.0)
    df["wind"]  = df["wind"].ffill().fillna(3.0)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Solar geometry  (must match train_model.py exactly)
# ---------------------------------------------------------------------------

def _cos_zenith_vec(timestamps: pd.DatetimeIndex, lat: float) -> np.ndarray:
    lat_r = np.radians(lat)
    doy   = np.array([t.timetuple().tm_yday for t in timestamps], dtype=float)
    decl  = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (284.0 + doy))))
    hour  = np.array([t.hour + t.minute / 60.0 for t in timestamps], dtype=float)
    ha    = np.radians((hour - 12.0) * 15.0)
    return np.maximum(
        np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha),
        0.0,
    )


# ---------------------------------------------------------------------------
# Feature engineering  (identical pipeline to train_model.py)
# ---------------------------------------------------------------------------

def _build_feature_df(
    weather_dfs: dict[str, pd.DataFrame],
    extended_index: pd.DatetimeIndex,
    target_date: date,
    locations: list[dict],
) -> pd.DataFrame:
    """
    Build feature matrix on extended_index (past 9 days + tomorrow),
    then filter to the 24 rows that correspond to target_date.

    The extended window is required so that lag-1d, lag-2d, lag-7d features
    are available for target_date without additional archive API calls.
    """
    feat  = pd.DataFrame(index=extended_index)
    w_sum = sum(loc["weight"] for loc in locations)

    # --- Per-cluster raw features ---
    for loc in locations:
        df = (
            weather_dfs[loc["name"]]
            .reindex(extended_index, method="nearest", tolerance=pd.Timedelta("90min"))
            .fillna(0.0)
        )
        for col in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
            feat[f"{loc['name']}_{col}"] = df[col].values

    # --- Capacity-weighted means ---
    for var in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
        feat[f"w_{var}"] = (
            sum(feat[f"{loc['name']}_{var}"] * loc["weight"] for loc in locations) / w_sum
        )

    # --- Time features ---
    feat["hour"]  = extended_index.hour.astype(float)
    feat["month"] = extended_index.month.astype(float)
    feat["dow"]   = extended_index.dayofweek.astype(float)
    feat["hour_sin"]  = np.sin(2 * np.pi * feat["hour"]  / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * feat["hour"]  / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # --- Solar geometry: capacity-weighted mean latitude ---
    rep_lat = sum(loc["lat"] * loc["weight"] for loc in locations) / w_sum
    feat["cos_zenith"] = _cos_zenith_vec(extended_index, rep_lat)

    # --- Clearness index Kt (Nespoli et al. 2019) ---
    ghi_cs = np.maximum(1000.0 * feat["cos_zenith"].values, 10.0)
    feat["kt"] = np.clip(feat["w_ghi"].values / ghi_cs, 0.0, 1.5)

    # --- Lag features on capacity-weighted GHI ---
    w_ghi = feat["w_ghi"].copy()
    feat["lag_1d_ghi"] = w_ghi.shift(24).values
    feat["lag_2d_ghi"] = w_ghi.shift(48).values
    feat["lag_7d_ghi"] = w_ghi.shift(168).values
    feat["roll3d_ghi"] = w_ghi.rolling(72, min_periods=1).mean().values

    # --- Sunny/cloudy flag (Nespoli et al. 2019, §2.3) ---
    daily_mean = {d: float(g.mean()) for d, g in w_ghi.groupby(w_ghi.index.date)}
    feat["is_sunny"] = np.array(
        [float(daily_mean.get(ts.date(), 0.0) >= _SUNNY_THRESHOLD) for ts in extended_index]
    )

    feat = feat.fillna(0.0)

    # Filter to the 24 hours of target_date in Europe/Berlin calendar time
    return feat.loc[extended_index.date == target_date]


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample_to_steps(hourly_24: np.ndarray, n_steps: int) -> np.ndarray:
    """Nearest-neighbour resample from 24 hourly values to n_steps."""
    if n_steps == 24:
        return hourly_24
    indices = np.clip((np.arange(n_steps) * 24 / n_steps).astype(int), 0, 23)
    return hourly_24[indices]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transform_payload(
    payload: dict,
    *,
    target_date: date,
    target_start: datetime,
    challenge_id: str,
    area: str,
    entsoe_api_key: str,
    api_base: str,
    data_source: str,
    challenge_context,
    challenge_detail: dict,
    forecast_objective: str,
    tz_name: str,
) -> dict:
    """
    LightGBM PV forecast using Open-Meteo NWP features at MaStR-derived cluster locations.
    Falls back to the unmodified baseline payload if models are unavailable or any
    API call fails.
    """
    n_steps = len(payload["values"])

    # ------------------------------------------------------------------
    # Step 1: Load models and cluster definitions (cached after first call)
    # ------------------------------------------------------------------
    if not _load_models():
        return payload

    # ------------------------------------------------------------------
    # Step 2: Fetch NWP forecast for all cluster centroids
    # past_days=9 ensures lag-7d features are populated for target_date.
    # Terren-Serrano et al. 2026: NWP (ICON/GFS) data used at inference time.
    # ------------------------------------------------------------------
    weather_dfs: dict[str, pd.DataFrame] = {}
    try:
        for loc in _CLUSTERS:
            weather_dfs[loc["name"]] = _fetch_open_meteo_forecast(
                loc["lat"], loc["lon"], past_days=9
            )
            time.sleep(0.2)
    except Exception as exc:
        print(f"[custom_model_3] Open-Meteo fetch failed: {exc}. Using baseline.")
        return payload

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix on the extended window, filter to target_date
    # ------------------------------------------------------------------
    try:
        extended_index = weather_dfs[_CLUSTERS[0]["name"]].index
        X_day_df       = _build_feature_df(weather_dfs, extended_index, target_date, _CLUSTERS)
    except Exception as exc:
        print(f"[custom_model_3] Feature engineering failed: {exc}. Using baseline.")
        return payload

    if len(X_day_df) == 0:
        print(
            f"[custom_model_3] No rows for {target_date} in forecast window "
            f"(index span: {extended_index[0].date()} – {extended_index[-1].date()}). "
            "Using baseline."
        )
        return payload

    # ------------------------------------------------------------------
    # Step 4: Reorder columns to training order, then predict
    # ------------------------------------------------------------------
    try:
        X_day = X_day_df[_FEATURE_NAMES].values
    except KeyError as exc:
        print(f"[custom_model_3] Feature mismatch (re-train may be needed): {exc}. Using baseline.")
        return payload

    try:
        pred_point = np.maximum(_MODEL_POINT.predict(X_day), 0.0)
        pred_q10   = np.maximum(_MODEL_Q10.predict(X_day),   0.0)
        pred_q90   = np.maximum(_MODEL_Q90.predict(X_day),   0.0)
    except Exception as exc:
        print(f"[custom_model_3] Model prediction failed: {exc}. Using baseline.")
        return payload

    # Derive per-hour std from Q10/Q90 spread (Gaussian: Q90-Q10 ≈ 3.29σ)
    std_per_hour = np.maximum((pred_q90 - pred_q10) / 3.29, 0.0)

    # ------------------------------------------------------------------
    # Step 5: Resample 24 hourly predictions → n_steps
    # ------------------------------------------------------------------
    mean_forecast = _resample_to_steps(pred_point, n_steps)
    std_per_step  = _resample_to_steps(std_per_hour, n_steps)

    # ------------------------------------------------------------------
    # Step 6: Night-hour mask
    # Timesteps below 1% of daily peak → set mean and std to 0.
    # ------------------------------------------------------------------
    peak = float(mean_forecast.max()) if mean_forecast.max() > 0 else 1.0
    night_mask            = mean_forecast < 0.01 * peak
    mean_forecast[night_mask] = 0.0
    std_per_step[night_mask]  = 0.0

    print(
        f"[custom_model_3] {target_date}  "
        f"peak={peak:.1f} MW  n_steps={n_steps}  "
        f"night_steps={int(night_mask.sum())}  "
        f"n_clusters={len(_CLUSTERS)}"
    )

    # ------------------------------------------------------------------
    # Step 7: Fill payload  (point / 5-quantile / 100-ensemble)
    # ------------------------------------------------------------------
    for index, original_value in enumerate(payload["values"]):
        fval = float(mean_forecast[index]) if index < len(mean_forecast) else 0.0
        std  = float(std_per_step[index])  if index < len(std_per_step)  else 1.0

        if isinstance(original_value, (int, float)):
            payload["values"][index] = fval

        elif isinstance(original_value, list) and len(original_value) == 5:
            # [Q10, Q25, Q50, Q75, Q90]
            payload["values"][index] = [
                float(max(0.0, fval - 2.0 * std)),
                float(max(0.0, fval - 0.7 * std)),
                float(fval),
                float(fval + 0.7 * std),
                float(fval + 2.0 * std),
            ]

        elif isinstance(original_value, list) and len(original_value) == 100:
            np.random.seed(index)
            ensemble = np.random.normal(loc=fval, scale=max(std, 0.01), size=100)
            payload["values"][index] = [float(v) for v in np.maximum(ensemble, 0.0)]

        else:
            if isinstance(original_value, list):
                payload["values"][index] = [float(v) for v in original_value]
            else:
                payload["values"][index] = float(original_value)

    return payload
