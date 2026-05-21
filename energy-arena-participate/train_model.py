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
    lgbm_point.pkl              – point forecast model
    lgbm_quantile_low.pkl       – Q10 model
    lgbm_quantile_high.pkl      – Q90 model
    feature_names.json          – ordered feature list
    training_stats.json         – RMSE per CV fold, run metadata
    validation_plot.png         – last-30-days actual vs predicted

References:
    Nespoli et al. 2019:       150 W/m² daily mean GHI → sunny/cloudy threshold
    Terren-Serrano et al. 2026: ERA5 reanalysis for training, NWP for inference;
                                capacity-density spatial masking for site selection
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
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


def _load_solar_csv(path: Path) -> pd.DataFrame | None:
    """
    Parse one MaStR solar CSV file.

    Handles the common MaStR export format (semicolon-separated, decimal comma)
    with multiple encoding fallbacks.  Returns a tidy DataFrame with columns
    [lat, lon, kw, land?] or None on failure.
    """
    # Columns we need (patterns matched case-insensitively after stripping soft hyphens)
    COL_PATTERNS: dict[str, list[str]] = {
        "lat":    ["breitengrad"],
        "lon":    ["längengrad", "laengengrad"],
        "kw":     ["nettonennleistung"],
        "status": ["betriebsstatus"],
        "land":   ["bundesland"],
    }

    detected_enc = detected_sep = detected_dec = None
    col_map: dict[str, str] = {}

    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        for sep, dec in ((";", ","), (";", "."), (",", ".")):
            try:
                hdr_df = pd.read_csv(
                    path, sep=sep, decimal=dec, encoding=enc, nrows=0, low_memory=False
                )
                if len(hdr_df.columns) < 5:
                    continue  # wrong separator

                # Normalise header: strip soft hyphens (U+00AD), quotes, whitespace
                norm_to_orig = {
                    c.replace("­", "").strip().strip('"').lower(): c
                    for c in hdr_df.columns
                }

                tmp_map: dict[str, str] = {}
                for key, patterns in COL_PATTERNS.items():
                    for pat in patterns:
                        if pat in norm_to_orig:
                            tmp_map[key] = norm_to_orig[pat]
                            break

                if "lat" in tmp_map and "lon" in tmp_map and "kw" in tmp_map:
                    detected_enc, detected_sep, detected_dec = enc, sep, dec
                    col_map = tmp_map
                    break
            except Exception:
                continue
        if detected_enc:
            break

    if not detected_enc:
        print(f"  [!] Could not detect CSV format for {path.name}")
        return None

    # Read only the needed columns
    usecols = list(set(col_map.values()))
    try:
        df = pd.read_csv(
            path,
            sep=detected_sep,
            decimal=detected_dec,
            encoding=detected_enc,
            usecols=usecols,
            low_memory=False,
        )
    except Exception as exc:
        print(f"  [!] Read failed for {path.name}: {exc}")
        return None

    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Filter operational plants
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip() == "In Betrieb"].copy()

    # Parse lat/lon/kw to float (handles any remaining decimal-comma strings)
    for col in ("lat", "lon", "kw"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df = df.dropna(subset=["lat", "lon", "kw"])
    df = df[df["kw"] > 0].copy()

    out_cols = ["lat", "lon", "kw"] + (["land"] if "land" in df.columns else [])
    return df[out_cols].reset_index(drop=True)


def _parse_mastr_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extract *EinheitenSolar*.csv file(s) from ZIP and parse them."""
    with zipfile.ZipFile(zip_path) as zf:
        solar_files = sorted(
            n for n in zf.namelist()
            if "EinheitenSolar" in n and n.endswith(".csv")
        )

    if not solar_files:
        # Show first few entries to help debug
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        raise RuntimeError(
            f"No *EinheitenSolar*.csv found in ZIP. First 20 entries: {names[:20]}"
        )

    print(f"[MaStR] Solar CSV(s) in ZIP: {solar_files}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mastr_"))
    try:
        all_dfs: list[pd.DataFrame] = []
        with zipfile.ZipFile(zip_path) as zf:
            for fname in solar_files:
                extracted = Path(zf.extract(fname, path=tmp_dir))
                size_mb   = extracted.stat().st_size / 1e6
                print(f"  Parsing {fname}  ({size_mb:.0f} MB extracted) …")
                df_part = _load_solar_csv(extracted)
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
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )
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
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_df(
    weather_dfs: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    locations: list[dict],
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

    Total: 80 features for N_CLUSTERS=10.
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
    # 4. Build feature matrix
    # ------------------------------------------------------------------
    common_index = weather_dfs[clusters[0]["name"]].index
    print(f"Building feature matrix ({len(common_index):,} hourly steps) …")
    X_df         = build_feature_df(weather_dfs, common_index, clusters)
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

    # Drop rows where target is NaN or negative
    mask          = np.isfinite(y) & (y >= 0)
    X, y          = X[mask], y[mask]
    idx_m         = common_index[mask]
    print(f"  Samples after NaN removal: {len(y):,}")

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
            X[tr], y[tr],
            sample_weight=sample_weight[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        rmse = float(np.sqrt(mean_squared_error(y[va], cv_m.predict(X[va]))))
        rmse_folds.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.2f} MW")
    print(f"CV RMSE: {np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f} MW")

    # ------------------------------------------------------------------
    # 8. Final models on all data
    # ------------------------------------------------------------------
    print("Fitting final models on all data …")
    model_point = lgb.LGBMRegressor(**_lgbm_params("regression"))
    model_point.fit(X, y, sample_weight=sample_weight)

    model_q10 = lgb.LGBMRegressor(**_lgbm_params("quantile", alpha=0.10))
    model_q10.fit(X, y, sample_weight=sample_weight)

    model_q90 = lgb.LGBMRegressor(**_lgbm_params("quantile", alpha=0.90))
    model_q90.fit(X, y, sample_weight=sample_weight)
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
        "training_date":   date.today().isoformat(),
        "train_start":     TRAIN_START.isoformat(),
        "train_end":       TRAIN_END.isoformat(),
        "n_samples":       int(len(y)),
        "n_features":      len(feature_names),
        "n_clusters":      len(clusters),
        "smard_available": smard_ok,
        "cv_rmse_folds":   [round(r, 4) for r in rmse_folds],
        "cv_rmse_mean":    round(float(np.mean(rmse_folds)), 4),
        "cv_rmse_std":     round(float(np.std(rmse_folds)),  4),
        "clusters":        clusters,
    }
    (ARTIFACT_DIR / "training_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[✓] All artifacts saved to {ARTIFACT_DIR}")

    # ------------------------------------------------------------------
    # 10. Validation plot – last 30 days of the dataset
    # ------------------------------------------------------------------
    cut      = idx_m[-1] - pd.Timedelta(days=30)
    pm       = idx_m >= cut
    y_actual = y[pm]
    y_pred   = model_point.predict(X[pm])
    rmse_p   = float(np.sqrt(mean_squared_error(y_actual, y_pred)))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(idx_m[pm], y_actual, label="Actual (SMARD)", alpha=0.85, lw=0.8)
    ax.plot(idx_m[pm], y_pred,   label="Predicted (LightGBM)", alpha=0.85, lw=0.8)
    ax.set_title(f"PV Forecast – Last 30 Days Validation  (RMSE = {rmse_p:.1f} MW)")
    ax.set_ylabel("MW")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "validation_plot.png", dpi=130)
    plt.close(fig)
    print(f"[✓] Validation plot saved  (RMSE {rmse_p:.1f} MW)")
    print("Done.")


if __name__ == "__main__":
    main()
