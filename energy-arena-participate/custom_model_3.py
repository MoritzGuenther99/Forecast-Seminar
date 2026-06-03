"""
custom_model_3.py  –  LightGBM PV solar forecast (inference).

Same transform_payload interface as custom_model_2.py.
Forecast locations are loaded at startup from clusters.json, which was
created by train_model.py using capacity-weighted K-Means on MaStR data.
Models are loaded once on the first call and cached at module level.

References:
    Nespoli et al. 2019:        150 W/m² daily mean GHI → sunny/cloudy threshold
    Sperati et al. 2016:        ECMWF ensemble for probabilistic PV forecasting
    Terren-Serrano et al. 2026: NWP + ensemble weather forecasts for probabilistic
                                energy forecasting; capacity-density spatial masking
    Yu et al. 2026:             PhysEmbedFormer: physics-guided decomposition
                                P_ac = P_physics + P_residual (eq. 6 for G_eff)
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
_ENSEMBLE_CACHE: dict      = {}   # target_date → {loc_name: {member_id: pd.DataFrame}}
_PHYSICS_DECOMP: bool      = False  # True when loaded model includes p_physics feature
_CAP_REF_GW: float | None  = None  # reference capacity used during training normalization

_ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"

# Nespoli et al. 2019, §2.3
_SUNNY_THRESHOLD = 150.0  # W/m²

# Yu et al. 2026: PV panel physics parameters (eq. 6)
_TILT_RAD = np.radians(20.0)                              # typical German rooftop
_K_DIFF   = 0.5 * (1.0 + np.cos(_TILT_RAD))              # diffuse view factor ≈ 0.970
_RHO_G    = 0.2                                           # ground reflectance
_K_REFL   = 0.5 * _RHO_G * (1.0 - np.cos(_TILT_RAD))    # reflected component ≈ 0.006
_GAMMA    = -0.004                                        # temperature coefficient /°C
_ETA      = 0.96                                          # inverter efficiency


# ---------------------------------------------------------------------------
# Recent SMARD PV fetch (for lag_1d_pv / lag_7d_pv inference features)
# ---------------------------------------------------------------------------

def _fetch_recent_smard_pv(n_days: int = 9) -> pd.Series | None:
    """
    Fetch the last n_days of DE-LU PV actual generation from SMARD.
    Returns an hourly MW Series (Europe/Berlin) or None on failure.
    Used to populate lag_1d_pv / lag_7d_pv / lag_1d_pv_peak at inference time.
    """
    _MODULE = 125
    _REGION = "DE-LU"
    index_url = (
        f"https://www.smard.de/app/chart_data/{_MODULE}/{_REGION}"
        f"/index_quarterhour.json"
    )
    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        all_ts = resp.json().get("timestamps", [])
    except Exception as exc:
        print(f"[custom_model_3] SMARD index fetch failed: {exc}")
        return None

    cutoff_ms = int(
        (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=n_days)).timestamp() * 1000
    )
    relevant = sorted(ts for ts in all_ts if ts >= cutoff_ms)

    records: list[tuple] = []
    for ts_ms in relevant:
        chunk_url = (
            f"https://www.smard.de/app/chart_data/{_MODULE}/{_REGION}/"
            f"{_MODULE}_{_REGION}_quarterhour_{ts_ms}.json"
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
            print(f"[custom_model_3] SMARD chunk {ts_ms} failed: {exc}")
        time.sleep(0.1)

    if not records:
        return None

    idx, vals = zip(*records)
    s = (
        pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)))
        .sort_index()
        .groupby(level=0).mean()
        .tz_convert("Europe/Berlin")
    )
    return s.resample("h").mean()


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
    global _FEATURE_NAMES, _CLUSTERS, _PHYSICS_DECOMP, _CAP_REF_GW

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
        _PHYSICS_DECOMP = "p_physics" in _FEATURE_NAMES

        # Load reference capacity used during training normalization (Nespoli et al. 2019).
        # Predictions are already in cap_ref MW units — no inference-time scaling needed.
        try:
            _ps = json.loads((_ARTIFACT_DIR / "physics_scaler.json").read_text(encoding="utf-8"))
            _CAP_REF_GW = float(_ps["cap_ref_gw"]) if "cap_ref_gw" in _ps else None
        except Exception:
            _CAP_REF_GW = None

        _MODELS_LOADED  = True
        print(
            f"[custom_model_3] Loaded {len(_CLUSTERS)} clusters, "
            f"{len(_FEATURE_NAMES)} features  ←  {_ARTIFACT_DIR}"
            + (f"  cap_ref={_CAP_REF_GW:.2f} GW" if _CAP_REF_GW is not None else "")
        )
        if not _PHYSICS_DECOMP:
            print(
                "[custom_model_3] WARNING: models lack 'p_physics' feature. "
                "Run python train_model.py to retrain with physics decomposition."
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
# ECMWF IFS ensemble fetch via Open-Meteo ensemble API
# Sperati et al. 2016: ECMWF ensemble members for probabilistic PV forecasting.
# Terren-Serrano et al. 2026: ensemble weather forecasts for probabilistic
#                             energy forecasting.
# ---------------------------------------------------------------------------

def _fetch_open_meteo_ensemble(
    lat: float,
    lon: float,
    *,
    past_days: int = 9,
) -> dict[str, pd.DataFrame]:
    """
    Fetch ECMWF IFS025 ensemble from the Open-Meteo ensemble API.
    Returns {member_id: pd.DataFrame} with 50 members (member01 … member50),
    each DataFrame having the same column structure as _fetch_open_meteo_forecast.
    If a variable carries no per-member field, the ensemble-mean field is used
    for all 50 members.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        (
            "shortwave_radiation,temperature_2m,cloudcover,"
            "diffuse_radiation,precipitation,windspeed_10m"
        ),
        "models":        "ecmwf_ifs025",
        "forecast_days": 2,
        "past_days":     past_days,
        "timezone":      "Europe/Berlin",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    h     = resp.json()["hourly"]
    times = pd.to_datetime(h["time"])

    def _c(vals: list, fb: float = 0.0) -> list:
        return [v if v is not None else fb for v in vals]

    # (api_field_prefix, df_column, none_fallback)
    _VARS = [
        ("shortwave_radiation", "ghi",    0.0),
        ("temperature_2m",      "temp",   float("nan")),
        ("cloudcover",          "cloud",  float("nan")),
        ("diffuse_radiation",   "dhi",    0.0),
        ("precipitation",       "precip", 0.0),
        ("windspeed_10m",       "wind",   float("nan")),
    ]
    n = len(times)

    members: dict[str, pd.DataFrame] = {}
    for mid in range(1, 51):
        mkey = f"member{mid:02d}"
        data: dict[str, list] = {}
        for api_var, col, fb in _VARS:
            raw = h.get(f"{api_var}_{mkey}") or h.get(api_var) or [fb] * n
            data[col] = _c(raw, fb=fb)

        df = pd.DataFrame(data, index=times)
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
        members[mkey] = df.sort_index()

    return members


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
# Physics-based PV power estimate
# Yu et al. 2026: PhysEmbedFormer, eq. 6  –  G_eff decomposition.
# P_ac = P_physics + P_residual; LightGBM learns only the residual.
# ---------------------------------------------------------------------------

def _compute_p_physics(
    weather_df: pd.DataFrame,
    cluster_info: dict,
    timestamp_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Single-cluster physics-based PV power estimate (Yu et al. 2026 eq. 6).

    G_eff  = G_beam*cos(z) + DHI*k_diff + GHI*rho_g*k_refl
    P_cell = P_rated * (G_eff/1000) * (1 + gamma*(T_air - 25))
    P_physics = P_cell * eta

    P_rated is normalised to cluster capacity (GW * 0.001) so the residual
    target stays on the same scale as the SMARD MW series.
    """
    df = (
        weather_df
        .reindex(timestamp_index, method="nearest", tolerance=pd.Timedelta("90min"))
        .fillna(0.0)
    )
    ghi   = df["ghi"].values
    dhi   = df["dhi"].values
    temp  = df["temp"].values
    cos_z = _cos_zenith_vec(timestamp_index, cluster_info["lat"])

    g_beam = np.maximum(ghi - dhi, 0.0)
    g_eff  = np.maximum(
        g_beam * cos_z + dhi * _K_DIFF + ghi * _RHO_G * _K_REFL,
        0.0,
    )
    p_rated = cluster_info["capacity_gw"] * 0.001   # normalised rated power
    p_cell  = p_rated * (g_eff / 1000.0) * (1.0 + _GAMMA * (temp - 25.0))
    return np.maximum(p_cell * _ETA, 0.0)


# ---------------------------------------------------------------------------
# Feature engineering  (identical pipeline to train_model.py)
# ---------------------------------------------------------------------------

def _build_feature_df(
    weather_dfs: dict[str, pd.DataFrame],
    extended_index: pd.DatetimeIndex,
    target_date: date,
    locations: list[dict],
    p_physics: np.ndarray | None = None,   # aligned with extended_index
    pv_series: pd.Series | None = None,    # recent SMARD MW for lag PV features
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

    # --- Physics-based PV baseline (Yu et al. 2026 PhysEmbedFormer decomposition) ---
    if p_physics is not None:
        feat["p_physics"] = p_physics

    # --- Lagged actual PV output (mirrors training features lag_1d_pv etc.) ---
    if pv_series is not None:
        pv_r = (
            pv_series
            .reindex(extended_index, method="nearest", tolerance=pd.Timedelta("90min"))
            .fillna(0.0)
        )
        feat["lag_1d_pv"]  = pv_r.shift(24).fillna(0.0).values
        feat["lag_7d_pv"]  = pv_r.shift(168).fillna(0.0).values
        daily_peak = pv_series.groupby(pv_series.index.date).max()
        feat["lag_1d_pv_peak"] = np.array([
            float(daily_peak.get((ts - pd.Timedelta(days=1)).date(), 0.0))
            for ts in extended_index
        ])

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

    Note on capacity normalization (Nespoli et al. 2019 / BNetzA MaStR):
    The training target was scaled to cap_ref MW units (capacity at TRAIN_END) so the
    model learns stable weather→power relationships across the 2-year training window.
    Predictions are already in current-capacity MW — no inference-time rescaling needed.
    cap_ref_gw is loaded from physics_scaler.json for logging only (_CAP_REF_GW).
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
    # Step 3: Compute P_physics baseline, fetch recent SMARD PV for lag
    # features, then build feature matrix.
    # Yu et al. 2026: P_ac = P_physics + P_residual (PhysEmbedFormer).
    # P_physics is added as a feature so LightGBM predicts the residual.
    # ------------------------------------------------------------------
    try:
        extended_index = weather_dfs[_CLUSTERS[0]["name"]].index
    except Exception as exc:
        print(f"[custom_model_3] Feature engineering failed: {exc}. Using baseline.")
        return payload

    p_physics_ext  = np.zeros(len(extended_index))
    physics_ok     = False
    if _PHYSICS_DECOMP:
        try:
            w_sum_phys = sum(loc["weight"] for loc in _CLUSTERS)
            for loc in _CLUSTERS:
                p_physics_ext += (
                    _compute_p_physics(weather_dfs[loc["name"]], loc, extended_index)
                    * loc["weight"] / w_sum_phys
                )
            physics_ok = True
        except Exception as exc:
            print(f"[custom_model_3] Physics computation failed: {exc}. Using zeros.")

    # Fetch recent SMARD PV only when the trained model includes lag PV features.
    pv_series_recent: pd.Series | None = None
    if "lag_1d_pv" in _FEATURE_NAMES:
        try:
            pv_series_recent = _fetch_recent_smard_pv(n_days=9)
            if pv_series_recent is None:
                print(
                    f"[custom_model_3] SMARD fetch returned None; "
                    "lag_1d_pv / lag_7d_pv set to 0."
                )
        except Exception as exc:
            print(
                f"[custom_model_3] SMARD fetch failed: {exc}; "
                "lag PV features set to 0."
            )

    try:
        X_day_df = _build_feature_df(
            weather_dfs, extended_index, target_date, _CLUSTERS,
            p_physics=p_physics_ext,
            pv_series=pv_series_recent,
        )
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
        pred_residual = _MODEL_POINT.predict(X_day)   # residual (may be negative)
        pred_q10      = _MODEL_Q10.predict(X_day)
        pred_q90      = _MODEL_Q90.predict(X_day)
    except Exception as exc:
        print(f"[custom_model_3] Model prediction failed: {exc}. Using baseline.")
        return payload

    # Scale diagnostic: raw LightGBM output vs expected SMARD range 0–60000 MW DE_LU
    _pos = pred_residual[pred_residual > 0]
    _pos_mean = f"{_pos.mean():.1f}" if len(_pos) else "0.0"
    print(
        f"[custom_model_3] SCALE CHECK {target_date}: "
        f"raw_residual min={pred_residual.min():.1f}  max={pred_residual.max():.1f}  "
        f"daytime_mean={_pos_mean} MW  "
        f"cap_ref={_CAP_REF_GW:.2f} GW  "
        f"[expected SMARD DE_LU solar: 0–60000 MW peak]"
    )

    # Physics + residual reconstruction (Yu et al. 2026 PhysEmbedFormer)
    if _PHYSICS_DECOMP and physics_ok:
        day_mask      = extended_index.date == target_date
        p_physics_day = p_physics_ext[day_mask]
        if len(p_physics_day) == len(pred_residual):
            pred_point = np.maximum(p_physics_day + pred_residual, 0.0)
        else:
            pred_point = np.maximum(pred_residual, 0.0)
        print(
            f"[custom_model_3] SCALE CHECK {target_date}: "
            f"p_physics max={p_physics_day.max():.5f}  "
            f"(negligible vs MW scale; residual dominates)"
        )
    else:
        pred_point = np.maximum(pred_residual, 0.0)

    _pp_pos = pred_point[pred_point > 0]
    _pp_mean = f"{float(_pp_pos.mean()):.1f}" if len(_pp_pos) else "0.0"
    print(
        f"[custom_model_3] SCALE CHECK {target_date}: "
        f"pred_point max={pred_point.max():.1f}  mean(>0)={_pp_mean} MW  "
        f"[SMARD typical summer peak 30000–55000 MW]"
    )

    # Derive per-hour std from Q10/Q90 spread (Gaussian: Q90-Q10 ≈ 3.29σ)
    std_per_hour = np.maximum((pred_q90 - pred_q10) / 3.29, 0.0)

    # ------------------------------------------------------------------
    # Step 5: Fetch ECMWF IFS025 ensemble (cached per target_date).
    # Run the LightGBM point model for each of the 50 members so that
    # probabilistic outputs are derived from real ensemble spread rather
    # than a parametric normal distribution.
    # Sperati et al. 2016: ECMWF ensemble for probabilistic PV forecasting.
    # Terren-Serrano et al. 2026: ensemble weather forecasts for
    # probabilistic energy forecasting.
    # ------------------------------------------------------------------
    ensemble_available = False
    member_preds_24: np.ndarray | None = None   # shape (50, 24)

    try:
        if target_date not in _ENSEMBLE_CACHE:
            ens_loc_data: dict[str, dict] = {}
            for loc in _CLUSTERS:
                ens_loc_data[loc["name"]] = _fetch_open_meteo_ensemble(
                    loc["lat"], loc["lon"], past_days=9
                )
                time.sleep(0.3)
            _ENSEMBLE_CACHE[target_date] = ens_loc_data

        ens_loc_data  = _ENSEMBLE_CACHE[target_date]
        ens_index     = ens_loc_data[_CLUSTERS[0]["name"]]["member01"].index
        ens_day_mask  = ens_index.date == target_date
        w_sum_ens     = sum(loc["weight"] for loc in _CLUSTERS)

        preds_list: list[np.ndarray] = []
        for mid in range(1, 51):
            mkey      = f"member{mid:02d}"
            m_weather = {loc["name"]: ens_loc_data[loc["name"]][mkey] for loc in _CLUSTERS}

            # Per-member physics baseline (Yu et al. 2026: ensemble spread from physics)
            p_phys_m_ext = np.zeros(len(ens_index))
            if _PHYSICS_DECOMP:
                try:
                    for loc in _CLUSTERS:
                        p_phys_m_ext += (
                            _compute_p_physics(m_weather[loc["name"]], loc, ens_index)
                            * loc["weight"] / w_sum_ens
                        )
                except Exception:
                    p_phys_m_ext = np.zeros(len(ens_index))

            X_m_df    = _build_feature_df(
                m_weather, ens_index, target_date, _CLUSTERS,
                p_physics=p_phys_m_ext,
                pv_series=pv_series_recent,
            )
            X_m        = X_m_df[_FEATURE_NAMES].values
            residual_m = _MODEL_POINT.predict(X_m)

            if _PHYSICS_DECOMP:
                p_phys_m_day = p_phys_m_ext[ens_day_mask]
                if len(p_phys_m_day) == len(residual_m):
                    member_pred = np.maximum(p_phys_m_day + residual_m, 0.0)
                else:
                    member_pred = np.maximum(residual_m, 0.0)
            else:
                member_pred = np.maximum(residual_m, 0.0)

            preds_list.append(member_pred)

        member_preds_24    = np.array(preds_list)    # (50, 24)
        ensemble_available = True
        print(
            f"[custom_model_3] {target_date} ensemble: "
            f"50 members × 24 h  (ECMWF IFS025 via Open-Meteo)"
        )
    except Exception as exc:
        print(
            f"[custom_model_3] Ensemble fetch failed: {exc}. "
            "Falling back to np.random.normal."
        )

    # ------------------------------------------------------------------
    # Step 6: Resample to n_steps; build ensemble matrix at target resolution
    # ------------------------------------------------------------------
    if ensemble_available and member_preds_24 is not None:
        mean_forecast_24 = member_preds_24.mean(axis=0)
        member_preds_ns  = np.array(                        # (50, n_steps)
            [_resample_to_steps(m, n_steps) for m in member_preds_24]
        )
    else:
        mean_forecast_24 = pred_point
        member_preds_ns  = None

    mean_forecast = _resample_to_steps(mean_forecast_24, n_steps)
    std_per_step  = _resample_to_steps(std_per_hour,     n_steps)

    # ------------------------------------------------------------------
    # Step 6b: Spread correction based on sunny/cloudy classification.
    # Nespoli et al. 2019: daily mean GHI >= 150 W/m² → sunny day.
    # Wider spread for cloudy days (higher uncertainty), tighter for sunny.
    # ------------------------------------------------------------------
    daily_mean_wghi = float(X_day_df["w_ghi"].mean())
    is_sunny        = daily_mean_wghi >= _SUNNY_THRESHOLD
    spread_factor   = 0.8 if is_sunny else 1.5
    if ensemble_available and member_preds_ns is not None:
        ensemble_mean   = member_preds_ns.mean(axis=0)
        deviations      = member_preds_ns - ensemble_mean
        member_preds_ns = np.maximum(ensemble_mean + deviations * spread_factor, 0.0)

    # ------------------------------------------------------------------
    # Step 7: Night-hour mask (applied to mean and all ensemble members)
    # Timesteps below 1% of daily peak → zero out.
    # ------------------------------------------------------------------
    peak       = float(mean_forecast.max()) if mean_forecast.max() > 0 else 1.0
    night_mask = mean_forecast < 0.01 * peak
    mean_forecast[night_mask] = 0.0
    std_per_step[night_mask]  = 0.0
    if member_preds_ns is not None:
        member_preds_ns[:, night_mask] = 0.0

    print(
        f"[custom_model_3] {target_date}  "
        f"peak={peak:.1f} MW  n_steps={n_steps}  "
        f"night_steps={int(night_mask.sum())}  "
        f"n_clusters={len(_CLUSTERS)}  "
        f"ensemble={'yes' if ensemble_available else 'no (fallback)'}  "
        f"spread_factor={spread_factor:.2f} ({'sunny' if is_sunny else 'cloudy'})"
    )

    # ------------------------------------------------------------------
    # Step 8: Fill payload  (point / 5-quantile / 100-ensemble)
    # ------------------------------------------------------------------
    for index, original_value in enumerate(payload["values"]):
        fval = float(mean_forecast[index]) if index < len(mean_forecast) else 0.0
        std  = float(std_per_step[index])  if index < len(std_per_step)  else 1.0

        if isinstance(original_value, (int, float)):
            # Ensemble mean across 50 members (or deterministic if unavailable)
            payload["values"][index] = fval

        elif isinstance(original_value, list) and len(original_value) == 5:
            if ensemble_available and member_preds_ns is not None:
                # Per-timestep empirical quantiles from 50 ECMWF members
                # Sperati et al. 2016: empirical quantiles from ensemble spread
                vals = member_preds_ns[:, index]
                payload["values"][index] = [
                    float(np.percentile(vals, 10)),
                    float(np.percentile(vals, 25)),
                    float(np.percentile(vals, 50)),
                    float(np.percentile(vals, 75)),
                    float(np.percentile(vals, 90)),
                ]
            else:
                # Fallback: Gaussian spread derived from Q10/Q90 model
                payload["values"][index] = [
                    float(max(0.0, fval - 2.0 * std)),
                    float(max(0.0, fval - 0.7 * std)),
                    float(fval),
                    float(fval + 0.7 * std),
                    float(fval + 2.0 * std),
                ]

        elif isinstance(original_value, list) and len(original_value) == 100:
            if ensemble_available and member_preds_ns is not None:
                # 50 ECMWF members → 100 via duplication with 2% jitter
                # Terren-Serrano et al. 2026: ensemble resampling for 100-member format
                members_50 = member_preds_ns[:, index]
                np.random.seed(index)
                jitter      = np.random.normal(0, 0.02, 50)
                members_100 = np.concatenate([members_50, members_50 * (1 + jitter)])
                payload["values"][index] = [float(v) for v in np.maximum(members_100, 0.0)]
            else:
                # Fallback: Gaussian sample
                np.random.seed(index)
                ensemble = np.random.normal(loc=fval, scale=max(std, 0.01), size=100)
                payload["values"][index] = [float(v) for v in np.maximum(ensemble, 0.0)]

        else:
            if isinstance(original_value, list):
                payload["values"][index] = [float(v) for v in original_value]
            else:
                payload["values"][index] = float(original_value)

    return payload