#!/usr/bin/env python3
"""
train_model.py  –  Train LightGBM PV forecast using MaStR-derived cluster locations.

Usage:  python train_model.py

The representative forecast locations are derived from the public
Marktstammdatenregister (MaStR) solar plant register by capacity-weighted
K-Means clustering, so they reflect the true geographic distribution of
installed PV capacity rather than hand-picked city coordinates.

Outputs saved to ./model_artifacts/:
    clusters.json               – 10 cluster centroids + capacity weights
    lgbm_point.pkl              – point forecast model  (trained on residual)
    lgbm_quantile_low.pkl       – Q10 model             (trained on residual)
    lgbm_quantile_high.pkl      – Q90 model             (trained on residual)
    feature_names.json          – ordered feature list  (includes p_physics)
    physics_scaler.json         – mean/std of P_physics training baseline
    training_stats.json         – RMSE per CV fold, run metadata
    validation_plot.png         – last-30-days actual vs P_physics+LightGBM
    feature_importance_plot.png – top-30 LightGBM feature importances
    cluster_map.png             – cluster centroids on Germany lat/lon map
    shap_summary.png            – SHAP beeswarm (last 30 days)
    shap_bar.png                – mean |SHAP| bar chart (last 30 days)

References:
    Nespoli et al. 2019:        150 W/m² daily mean GHI → sunny/cloudy threshold
    Terren-Serrano et al. 2026: ERA5 reanalysis for training, NWP for inference;
                                capacity-density spatial masking for site selection
    Yu et al. 2026:             PhysEmbedFormer: physics-guided decomposition
                                P_ac = P_physics + P_residual (eq. 6 for G_eff)
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"

N_CLUSTERS = 10

# Public MaStR bulk export – electricity generating units
MASTR_URL      = "https://www.marktstammdatenregister.de/MaStRDownload/Gesamtdatenexport_Strom_Einheiten.zip"
MASTR_ZIP_PATH = ARTIFACT_DIR / "mastr_solar.zip"        # secondary cache (large)
MASTR_PARQUET  = ARTIFACT_DIR / "mastr_clusters_raw.parquet"  # primary cache (small)

# Nespoli et al. 2019, §2.3
SUNNY_THRESHOLD = 150.0  # W/m²

# Yu et al. 2026: PV panel physics parameters (eq. 6)
_TILT_RAD = np.radians(20.0)                              # typical German rooftop
_K_DIFF   = 0.5 * (1.0 + np.cos(_TILT_RAD))              # diffuse view factor ≈ 0.970
_RHO_G    = 0.2                                           # ground reflectance
_K_REFL   = 0.5 * _RHO_G * (1.0 - np.cos(_TILT_RAD))    # reflected component ≈ 0.006
_GAMMA    = -0.004                                        # temperature coefficient /°C
_ETA      = 0.96                                          # inverter efficiency

# SMARD module 125 = PV actual generation, DE-LU, quarter-hour
SMARD_MODULE = 125
SMARD_REGION = "DE-LU"

_TODAY      = date.today()
TRAIN_END   = _TODAY - timedelta(days=5)   # buffer for SMARD publication lag
TRAIN_START = date(_TODAY.year - 2, _TODAY.month, _TODAY.day) - timedelta(days=5)


# ---------------------------------------------------------------------------
# MaStR download and parsing
# ---------------------------------------------------------------------------

def _download_zip(url: str, save_path: Path) -> None:
    """Stream-download a large ZIP with progress reporting."""
    print(f"[MaStR] Downloading {url}")
    headers = {
        "User-Agent": "PV-Forecast-Research/1.0 (academic; open-data)",
        "Accept":     "application/zip, application/octet-stream, */*",
    }
    with requests.get(url, headers=headers, stream=True, timeout=(30, 900)) as resp:
        resp.raise_for_status()
        total      = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct   = -1
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded / total * 100)
                    if pct >= last_pct + 5:
                        print(
                            f"  {downloaded / 1e6:5.0f} / {total / 1e6:.0f} MB  ({pct}%)",
                            flush=True,
                        )
                        last_pct = pct
    print(f"[MaStR] Download complete: {save_path.stat().st_size / 1e6:.0f} MB")


def _load_solar_xml(path: Path) -> pd.DataFrame | None:
    """
    Parse one MaStR EinheitenSolar_*.xml file (UTF-16, iterative streaming).

    Records are <EinheitSolar> elements. Operational status is the numeric
    field EinheitBetriebsstatus == "35" (= InBetrieb). Not every record has
    coordinates; those without are skipped via dropna.
    Returns a tidy DataFrame with columns [lat, lon, kw, land] or None on failure.
    """
    rows: list[dict] = []

    try:
        # ET.iterparse detects the UTF-16 BOM automatically from the binary stream
        for _event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag != "EinheitSolar":
                continue

            # 35 = InBetrieb (numeric catalogue value)
            if (elem.findtext("EinheitBetriebsstatus") or "").strip() != "35":
                elem.clear()
                continue

            def _f(tag: str) -> str:
                return (elem.findtext(tag) or "").strip().replace(",", ".")

            rows.append({
                "lat":  _f("Breitengrad"),
                "lon":  _f("Laengengrad"),
                "kw":   _f("Nettonennleistung"),
                "land": (elem.findtext("Bundesland") or "").strip(),
            })
            elem.clear()
    except ET.ParseError as exc:
        print(f"  [!] XML parse error in {path.name}: {exc}")
        return None
    except Exception as exc:
        print(f"  [!] Failed reading {path.name}: {exc}")
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ("lat", "lon", "kw"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "kw"])
    df = df[df["kw"] > 0].copy()
    return df[["lat", "lon", "kw", "land"]].reset_index(drop=True)


def _parse_mastr_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extract EinheitenSolar_*.xml file(s) from ZIP and parse them."""
    with zipfile.ZipFile(zip_path) as zf:
        solar_files = sorted(
            n for n in zf.namelist()
            if "EinheitenSolar" in n and n.endswith(".xml")
        )

    if not solar_files:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        raise RuntimeError(
            f"No EinheitenSolar_*.xml found in ZIP. First 20 entries: {names[:20]}"
        )

    print(f"[MaStR] EinheitenSolar XML parts: {len(solar_files)} files")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mastr_"))
    try:
        all_dfs: list[pd.DataFrame] = []
        with zipfile.ZipFile(zip_path) as zf:
            for fname in solar_files:
                extracted = Path(zf.extract(fname, path=tmp_dir))
                size_mb   = extracted.stat().st_size / 1e6
                print(f"  Parsing {fname}  ({size_mb:.0f} MB extracted) …")
                df_part = _load_solar_xml(extracted)
                if df_part is not None and len(df_part) > 0:
                    print(f"    → {len(df_part):,} valid plants")
                    all_dfs.append(df_part)
                else:
                    print(f"    [!] No valid rows from {fname}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_dfs:
        raise RuntimeError("[MaStR] Could not extract any solar plant records from ZIP.")

    return pd.concat(all_dfs, ignore_index=True)


def load_mastr_plants() -> pd.DataFrame:
    """
    Return a cleaned DataFrame of operational DE solar plants: [lat, lon, kw, land?].

    Cache hierarchy (fastest first):
      1. MASTR_PARQUET  – compact, loads in seconds
      2. MASTR_ZIP_PATH – re-parse if parquet missing but ZIP exists
      3. Download ZIP from MaStR and create both caches
    """
    ARTIFACT_DIR.mkdir(exist_ok=True)

    if MASTR_PARQUET.exists():
        print(f"[MaStR] Loading from parquet cache: {MASTR_PARQUET}")
        return pd.read_parquet(MASTR_PARQUET)

    if not MASTR_ZIP_PATH.exists():
        _download_zip(MASTR_URL, MASTR_ZIP_PATH)
    else:
        print(f"[MaStR] Using cached ZIP: {MASTR_ZIP_PATH}  ({MASTR_ZIP_PATH.stat().st_size/1e6:.0f} MB)")

    df = _parse_mastr_from_zip(MASTR_ZIP_PATH)
    df.to_parquet(MASTR_PARQUET)
    print(f"[MaStR] Parquet cache saved: {MASTR_PARQUET}  ({len(df):,} plants)")
    return df


# ---------------------------------------------------------------------------
# Capacity-weighted K-Means clustering
# ---------------------------------------------------------------------------

def cluster_plants(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> list[dict]:
    """
    Capacity-weighted K-Means → N representative forecast locations.

    Terren-Serrano et al. 2026: capacity-density spatial masking –
    cluster centroids weight forecast locations by installed capacity,
    not by plant count.

    Returns a list of dicts sorted by capacity (descending):
        name, lat, lon, weight (capacity fraction), capacity_gw
    """
    total_gw = df["kw"].sum() / 1e6
    print(f"[MaStR] {len(df):,} operational plants  |  {total_gw:.1f} GW total installed")

    if "land" in df.columns:
        print("[MaStR] Capacity by Bundesland (top 16):")
        by_land = df.groupby("land")["kw"].sum().sort_values(ascending=False) / 1e6
        for land, gw in by_land.head(16).items():
            print(f"  {str(land):30s}  {gw:6.1f} GW")

    # Subsample to ≤ 500 k rows for KMeans memory efficiency
    if len(df) > 500_000:
        print(f"[MaStR] Subsampling to 500,000 plants for KMeans …")
        df_sample = df.sample(500_000, random_state=42)
    else:
        df_sample = df.copy()

    X_sample = df_sample[["lat", "lon"]].values
    w_sample = df_sample["kw"].values

    print(f"[KMeans] Fitting {n_clusters} clusters (n_init=5, capacity-weighted) …")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    km.fit(X_sample, sample_weight=w_sample)

    # Assign ALL plants to clusters (not just the subsample)
    labels    = km.predict(df[["lat", "lon"]].values)
    total_kw  = df["kw"].values.sum()

    clusters: list[dict] = []
    for i in range(n_clusters):
        mask  = labels == i
        c_kw  = df["kw"].values[mask]
        c_lat = df["lat"].values[mask]
        c_lon = df["lon"].values[mask]
        ck    = float(c_kw.sum())
        if ck == 0:
            continue

        clusters.append({
            "name":        f"cluster_{i}",          # renamed after sort below
            "lat":         round(float((c_lat * c_kw).sum() / ck), 4),
            "lon":         round(float((c_lon * c_kw).sum() / ck), 4),
            "weight":      round(ck / total_kw, 6),
            "capacity_gw": round(ck / 1e6, 3),
        })

    # Sort by capacity descending so cluster_0 is always the most important
    clusters.sort(key=lambda c: c["capacity_gw"], reverse=True)
    for i, c in enumerate(clusters):
        c["name"] = f"cluster_{i}"

    print(f"[KMeans] Cluster summary (sorted by capacity):")
    for c in clusters:
        print(
            f"  {c['name']:10s}  lat={c['lat']:6.2f}  lon={c['lon']:5.2f}"
            f"  {c['capacity_gw']:5.1f} GW  w={c['weight']:.4f}"
        )
    weight_sum = sum(c["weight"] for c in clusters)
    print(f"  Weight sum: {weight_sum:.6f}  (should be ≈ 1.0)")

    return clusters


# ---------------------------------------------------------------------------
# Open-Meteo archive (ERA5-Land reanalysis)
# ---------------------------------------------------------------------------

def fetch_open_meteo_archive(
    lat: float,
    lon: float,
    start: date,
    end: date,
    *,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Hourly ERA5-Land reanalysis from the Open-Meteo archive API (no key required).
    Terren-Serrano et al. 2026: reanalysis data used for model training.
    """
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "hourly":     "shortwave_radiation,temperature_2m,cloudcover,precipitation,windspeed_10m,diffuse_radiation",
        "timezone":   "Europe/Berlin",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            break
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Open-Meteo archive failed ({lat},{lon}): {exc}") from exc
            wait = 10 * (attempt + 1)
            print(f"  [Open-Meteo] Retry {attempt+1}/{retries} in {wait}s: {exc}")
            time.sleep(wait)

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
    df.index = df.index.tz_localize(
        "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = df[df.index.notna()]
    df["temp"]  = df["temp"].ffill().fillna(10.0)
    df["cloud"] = df["cloud"].ffill().fillna(50.0)
    df["wind"]  = df["wind"].ffill().fillna(3.0)
    return df.sort_index()


# ---------------------------------------------------------------------------
# SMARD PV actual generation
# ---------------------------------------------------------------------------

def fetch_smard_pv_series(start: date, end: date) -> pd.Series | None:
    """
    Fetch DE-LU PV actual generation from the SMARD JSON chart-data API.
    Module 125 = Photovoltaik Einspeisung.
    Values are in MWh per 15-min slot; × 4 → average MW.
    Returns an hourly MW Series (Europe/Berlin) or None on failure.
    """
    index_url = (
        f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}"
        f"/index_quarterhour.json"
    )
    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        all_ts = resp.json().get("timestamps", [])
    except Exception as exc:
        print(f"[SMARD] index fetch failed: {exc}")
        return None

    # Weekly timestamp boundaries (8-day buffer at start, 2-day at end)
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
        if (i + 1) % 20 == 0:
            print(f"[SMARD]   {i+1}/{len(relevant)} done")
        time.sleep(0.15)

    if not records:
        return None

    idx, vals = zip(*records)
    s = (
        pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)))
        .sort_index()
        .groupby(level=0).mean()      # deduplicate
        .tz_convert("Europe/Berlin")
    )
    s_hourly = s.resample("h").mean()
    t0 = pd.Timestamp(start, tz="Europe/Berlin")
    t1 = pd.Timestamp(end + timedelta(days=1), tz="Europe/Berlin")
    return s_hourly[(s_hourly.index >= t0) & (s_hourly.index < t1)]


# ---------------------------------------------------------------------------
# Solar geometry (physics-informed feature)
# ---------------------------------------------------------------------------

def cos_zenith_vec(timestamps: pd.DatetimeIndex, lat: float) -> np.ndarray:
    """
    Approximate cos(solar zenith angle) at a given latitude.
    Used as a physics-informed feature (Nespoli et al. 2019).
    """
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
# LightGBM is trained on P_residual = P_actual - P_physics.
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

    P_rated is normalised: capacity_gw * 0.001 (shape matters, not absolute scale).
    """
    df = (
        weather_df
        .reindex(timestamp_index, method="nearest", tolerance=pd.Timedelta("90min"))
        .fillna(0.0)
    )
    ghi   = df["ghi"].values
    dhi   = df["dhi"].values
    temp  = df["temp"].values
    cos_z = cos_zenith_vec(timestamp_index, cluster_info["lat"])

    g_beam = np.maximum(ghi - dhi, 0.0)
    g_eff  = np.maximum(
        g_beam * cos_z + dhi * _K_DIFF + ghi * _RHO_G * _K_REFL,
        0.0,
    )
    p_rated = cluster_info["capacity_gw"] * 0.001
    p_cell  = p_rated * (g_eff / 1000.0) * (1.0 + _GAMMA * (temp - 25.0))
    return np.maximum(p_cell * _ETA, 0.0)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_df(
    weather_dfs: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    locations: list[dict],
    p_physics: np.ndarray | None = None,   # aligned with common_index
) -> pd.DataFrame:
    """
    Build the complete hourly feature matrix.

    Features:
      60  per-cluster raw weather  (N clusters × 6 variables)
       6  capacity-weighted means  (w_ghi, w_temp, w_cloud, w_precip, w_wind, w_dhi)
       3  time: hour, month, dow
       4  cyclic sin/cos encodings
       1  cos(solar zenith angle)       – physics-informed (Nespoli et al. 2019)
       1  clearness index Kt            – Nespoli et al. 2019
       3  lag-1d, lag-2d, lag-7d GHI
       1  3-day rolling mean GHI
       1  binary sunny/cloudy flag      – daily mean GHI >= 150 W/m²
       1  p_physics                     – Yu et al. 2026 physics baseline (optional)

    Total: 80 features for N_CLUSTERS=10 (81 with p_physics).
    """
    feat  = pd.DataFrame(index=common_index)
    w_sum = sum(loc["weight"] for loc in locations)  # ≈ 1.0

    # --- Per-cluster raw features ---
    for loc in locations:
        df = (
            weather_dfs[loc["name"]]
            .reindex(common_index, method="nearest", tolerance=pd.Timedelta("90min"))
            .fillna(0.0)
        )
        for col in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
            feat[f"{loc['name']}_{col}"] = df[col].values

    # --- Capacity-weighted means (Terren-Serrano et al. 2026: density-weighted aggregate) ---
    for var in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
        feat[f"w_{var}"] = (
            sum(feat[f"{loc['name']}_{var}"] * loc["weight"] for loc in locations) / w_sum
        )

    # --- Time features ---
    feat["hour"]  = common_index.hour.astype(float)
    feat["month"] = common_index.month.astype(float)
    feat["dow"]   = common_index.dayofweek.astype(float)
    feat["hour_sin"]  = np.sin(2 * np.pi * feat["hour"]  / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * feat["hour"]  / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # --- Solar geometry: capacity-weighted mean latitude ---
    rep_lat = sum(loc["lat"] * loc["weight"] for loc in locations) / w_sum
    feat["cos_zenith"] = cos_zenith_vec(common_index, rep_lat)

    # --- Clearness index Kt = GHI / GHI_clear_sky (Nespoli et al. 2019) ---
    ghi_cs = np.maximum(1000.0 * feat["cos_zenith"].values, 10.0)
    feat["kt"] = np.clip(feat["w_ghi"].values / ghi_cs, 0.0, 1.5)

    # --- Lag features on capacity-weighted GHI (shift is position-based) ---
    w_ghi = feat["w_ghi"].copy()
    feat["lag_1d_ghi"] = w_ghi.shift(24).values
    feat["lag_2d_ghi"] = w_ghi.shift(48).values
    feat["lag_7d_ghi"] = w_ghi.shift(168).values
    feat["roll3d_ghi"] = w_ghi.rolling(72, min_periods=1).mean().values

    # --- Sunny/cloudy flag: daily mean GHI >= 150 W/m² (Nespoli et al. 2019, §2.3) ---
    daily_mean = {d: float(g.mean()) for d, g in w_ghi.groupby(w_ghi.index.date)}
    feat["is_sunny"] = np.array(
        [float(daily_mean.get(ts.date(), 0.0) >= SUNNY_THRESHOLD) for ts in common_index]
    )

    # --- Physics-based PV baseline (Yu et al. 2026 PhysEmbedFormer decomposition) ---
    if p_physics is not None:
        feat["p_physics"] = p_physics

    return feat.fillna(0.0)


# ---------------------------------------------------------------------------
# LightGBM helpers
# ---------------------------------------------------------------------------

def _lgbm_params(objective: str, alpha: float | None = None) -> dict:
    p: dict = dict(
        objective=objective,
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if objective == "quantile":
        p["metric"] = "quantile"
        p["alpha"]  = alpha
    else:
        p["metric"] = "rmse"
    return p


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    print(f"Training window: {TRAIN_START} → {TRAIN_END}")

    # ------------------------------------------------------------------
    # 1. Load MaStR PV plant data
    # ------------------------------------------------------------------
    df_plants = load_mastr_plants()

    # Restrict to Germany (remove outliers / foreign entries)
    df_plants = df_plants[
        df_plants["lat"].between(47.0, 55.5) & df_plants["lon"].between(5.5, 15.5)
    ].copy()
    print(f"[MaStR] After Germany geo-filter: {len(df_plants):,} plants")

    # ------------------------------------------------------------------
    # 2. Capacity-weighted clustering → 10 representative locations
    # ------------------------------------------------------------------
    clusters = cluster_plants(df_plants, N_CLUSTERS)
    (ARTIFACT_DIR / "clusters.json").write_text(
        json.dumps(clusters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[✓] Saved clusters.json  ({len(clusters)} clusters)")

    # ------------------------------------------------------------------
    # 3. Fetch Open-Meteo archive for each cluster centroid
    # ------------------------------------------------------------------
    weather_dfs: dict[str, pd.DataFrame] = {}
    for c in clusters:
        print(f"[Open-Meteo] {c['name']}  lat={c['lat']}, lon={c['lon']} …")
        weather_dfs[c["name"]] = fetch_open_meteo_archive(
            c["lat"], c["lon"], TRAIN_START, TRAIN_END
        )
        time.sleep(1.5)

    # ------------------------------------------------------------------
    # 4. Compute P_physics baseline, then build feature matrix.
    # Yu et al. 2026: P_ac = P_physics + P_residual (PhysEmbedFormer, eq. 6).
    # ------------------------------------------------------------------
    common_index = weather_dfs[clusters[0]["name"]].index
    print(f"Computing physics-based PV baseline (Yu et al. 2026 eq. 6) …")
    w_sum_phys     = sum(c["weight"] for c in clusters)
    p_physics_full = np.zeros(len(common_index))
    for c in clusters:
        p_physics_full += (
            _compute_p_physics(weather_dfs[c["name"]], c, common_index)
            * c["weight"] / w_sum_phys
        )
    print(
        f"  P_physics: mean={p_physics_full.mean():.4f}, "
        f"max={p_physics_full.max():.4f}, "
        f"nonzero={int((p_physics_full > 0).sum()):,} h"
    )

    print(f"Building feature matrix ({len(common_index):,} hourly steps) …")
    X_df          = build_feature_df(weather_dfs, common_index, clusters, p_physics=p_physics_full)
    feature_names = list(X_df.columns)
    X             = X_df.values
    print(f"  Shape: {X.shape[0]:,} × {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 5. Fetch SMARD PV target
    # ------------------------------------------------------------------
    print("Fetching SMARD PV actual generation …")
    target_series = fetch_smard_pv_series(TRAIN_START, TRAIN_END)
    smard_ok      = target_series is not None and not target_series.empty

    if smard_ok:
        print(f"[✓] SMARD: {len(target_series):,} hourly values")
        y = target_series.reindex(common_index).ffill().fillna(0.0).values
    else:
        print("[!] SMARD unavailable – using synthetic target (GHI × 0.15 + noise)")
        rng = np.random.default_rng(42)
        y   = np.maximum(
            X_df["w_ghi"].values * 0.15
            + rng.normal(0, X_df["w_ghi"].values * 0.03 + 0.5),
            0.0,
        )

    # Physics decomposition: LightGBM learns P_residual = P_actual - P_physics
    # Yu et al. 2026: PhysEmbedFormer – physics-guided target decomposition.
    y_residual = y - p_physics_full    # may be negative at night/overcast hours

    # Drop rows where SMARD target is NaN or negative
    mask      = np.isfinite(y) & (y >= 0)
    X         = X[mask]
    y_smard_m = y[mask]                # original SMARD values (for validation)
    y_train   = y_residual[mask]       # residual target for LightGBM
    p_phys_m  = p_physics_full[mask]   # physics baseline (for validation)
    idx_m     = common_index[mask]
    print(f"  Samples after NaN removal: {len(y_train):,}")

    # ------------------------------------------------------------------
    # 6. Sample weights: last 6 months → 2×, older → 1×
    # ------------------------------------------------------------------
    cutoff        = pd.Timestamp(TRAIN_END - timedelta(days=180), tz="Europe/Berlin")
    sample_weight = np.where(idx_m >= cutoff, 2.0, 1.0)
    print(
        f"  Recent (2×): {(sample_weight == 2.0).sum():,}  "
        f"  Older (1×): {(sample_weight == 1.0).sum():,}"
    )

    # ------------------------------------------------------------------
    # 7. TimeSeriesSplit cross-validation (RMSE reporting)
    # ------------------------------------------------------------------
    tscv       = TimeSeriesSplit(n_splits=5)
    rmse_folds: list[float] = []
    print("TimeSeriesSplit CV …")
    for fold, (tr, va) in enumerate(tscv.split(X)):
        cv_m = lgb.LGBMRegressor(**_lgbm_params("regression"))
        cv_m.fit(
            X[tr], y_train[tr],
            sample_weight=sample_weight[tr],
            eval_set=[(X[va], y_train[va])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        # RMSE on reconstructed full forecast (physics + residual) vs SMARD
        pred_va = np.maximum(p_phys_m[va] + cv_m.predict(X[va]), 0.0)
        rmse    = float(np.sqrt(mean_squared_error(y_smard_m[va], pred_va)))
        rmse_folds.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.2f} MW")
    print(f"CV RMSE: {np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f} MW")

    # ------------------------------------------------------------------
    # 8. Final models on all data
    # ------------------------------------------------------------------
    print("Fitting final models on all data …")
    model_point = lgb.LGBMRegressor(**_lgbm_params("regression"))
    model_point.fit(X, y_train, sample_weight=sample_weight)

    model_q10 = lgb.LGBMRegressor(**_lgbm_params("quantile", alpha=0.10))
    model_q10.fit(X, y_train, sample_weight=sample_weight)

    model_q90 = lgb.LGBMRegressor(**_lgbm_params("quantile", alpha=0.90))
    model_q90.fit(X, y_train, sample_weight=sample_weight)
    print("[✓] All three models trained.")

    # Feature importances (top 15)
    imps    = model_point.feature_importances_
    top_idx = np.argsort(imps)[::-1][:15]
    print("\nTop-15 feature importances (point model):")
    for i in top_idx:
        print(f"  {feature_names[i]:40s}  {imps[i]:.0f}")

    # ------------------------------------------------------------------
    # 9. Save all artifacts
    # ------------------------------------------------------------------
    joblib.dump(model_point, ARTIFACT_DIR / "lgbm_point.pkl")
    joblib.dump(model_q10,   ARTIFACT_DIR / "lgbm_quantile_low.pkl")
    joblib.dump(model_q90,   ARTIFACT_DIR / "lgbm_quantile_high.pkl")

    (ARTIFACT_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )

    stats = {
        "training_date":        date.today().isoformat(),
        "train_start":          TRAIN_START.isoformat(),
        "train_end":            TRAIN_END.isoformat(),
        "n_samples":            int(len(y_train)),
        "n_features":           len(feature_names),
        "n_clusters":           len(clusters),
        "smard_available":      smard_ok,
        "physics_decomposition": True,
        "cv_rmse_folds":        [round(r, 4) for r in rmse_folds],
        "cv_rmse_mean":         round(float(np.mean(rmse_folds)), 4),
        "cv_rmse_std":          round(float(np.std(rmse_folds)),  4),
        "clusters":             clusters,
    }
    (ARTIFACT_DIR / "training_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    physics_scaler = {
        "mean": float(p_phys_m.mean()),
        "std":  float(p_phys_m.std()),
    }
    (ARTIFACT_DIR / "physics_scaler.json").write_text(
        json.dumps(physics_scaler, indent=2), encoding="utf-8"
    )
    print(
        f"[✓] physics_scaler.json saved  "
        f"(mean={physics_scaler['mean']:.4f}, std={physics_scaler['std']:.4f})"
    )

    print(f"\n[✓] All artifacts saved to {ARTIFACT_DIR}")

    # ------------------------------------------------------------------
    # 10. Validation plot – last 30 days of the dataset
    # ------------------------------------------------------------------
    cut      = idx_m[-1] - pd.Timedelta(days=30)
    pm       = idx_m >= cut
    y_actual = y_smard_m[pm]
    y_pred   = np.maximum(p_phys_m[pm] + model_point.predict(X[pm]), 0.0)
    rmse_p   = float(np.sqrt(mean_squared_error(y_actual, y_pred)))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(idx_m[pm], y_actual, label="Actual (SMARD)", alpha=0.85, lw=0.8)
    ax.plot(idx_m[pm], y_pred,   label="Predicted (P_physics + LightGBM residual)", alpha=0.85, lw=0.8)
    ax.set_title(f"PV Forecast – Last 30 Days Validation  (RMSE = {rmse_p:.1f} MW)")
    ax.set_ylabel("MW")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "validation_plot.png", dpi=130)
    plt.close(fig)
    print(f"[✓] Validation plot saved  (RMSE {rmse_p:.1f} MW)")

    # ------------------------------------------------------------------
    # 11. Feature importance plot – top 30 features
    # ------------------------------------------------------------------
    imps_all  = model_point.feature_importances_
    top30_idx = np.argsort(imps_all)[::-1][:30]
    top30_names = [feature_names[i] for i in top30_idx]
    top30_vals  = imps_all[top30_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top30_names))
    ax.barh(y_pos, top30_vals[::-1], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30_names[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (LightGBM splits)")
    ax.set_title("Top 30 Feature Importances – Point Model")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "feature_importance_plot.png", dpi=130)
    plt.close(fig)
    print("[✓] feature_importance_plot.png saved")

    # ------------------------------------------------------------------
    # 12. Cluster map – centroids on Germany bounding box
    # ------------------------------------------------------------------
    cap_gws   = np.array([c["capacity_gw"] for c in clusters])
    dot_sizes = 50 + 750 * (cap_gws / cap_gws.max())

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(6, 15)
    ax.set_ylim(47, 55)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("PV Cluster Centroids – Capacity-Weighted K-Means")
    ax.set_facecolor("#dce9f5")
    ax.grid(alpha=0.4, linestyle="--")

    ax.scatter(
        [c["lon"] for c in clusters],
        [c["lat"] for c in clusters],
        s=dot_sizes,
        c=np.arange(len(clusters)),
        cmap="tab10",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    for i, c in enumerate(clusters):
        ax.annotate(
            f"C{i}\n{c['capacity_gw']:.1f} GW",
            xy=(c["lon"], c["lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            zorder=4,
        )
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "cluster_map.png", dpi=130)
    plt.close(fig)
    print("[✓] cluster_map.png saved")

    # ------------------------------------------------------------------
    # 13. SHAP analysis – point model on last 30 days of test data
    # ------------------------------------------------------------------
    X_shap      = X[pm]
    explainer   = shap.TreeExplainer(model_point)
    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[✓] shap_summary.png saved")

    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[✓] shap_bar.png saved")

    print("Done.")


if __name__ == "__main__":
    main()
