#!/usr/bin/env python3
"""
generate_paper_figures.py – Wissenschaftliche Abbildungen für die Seminararbeit.

Erzeugt sechs Abbildungen im Stil von figures/SMARD_Fundamentalplot_v2.png:
  fig_1_ensemble_21jun.png        – Ensemble-Forecast 21.06.2026
  fig_2_leaderboard_c10.png       – Challenge 10 Leaderboard (19.–25.06.2026)
  fig_3_mae_c10.png               – MAE-Verlauf Challenge 10 (30 Tage)
  fig_4_crps_c16.png              – CRPS-Verlauf Challenge 16 (30 Tage)
  fig_5_cap_problem.png           – Ensemble 22.–29.06.2026 (Cap-Plateau)
  fig_6_regime_change.png         – Ensemble 29.06.2026 (Regime-Wechsel)

Benötigt (model_artifacts/):
  lgbm_point.pkl  feature_names.json  clusters.json  physics_scaler.json

Manuelle Daten: Abschnitt "MANUELLE DATEN" unten ausfüllen.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

# Windows-Terminal: UTF-8 erzwingen, damit Umlaute in print() funktionieren
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "model_artifacts"
FIGURES_DIR  = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TZ_BERLIN  = ZoneInfo("Europe/Berlin")
DPI        = 300
CACHE_DIR  = ARTIFACT_DIR / "fig_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return CACHE_DIR / f"{safe}.pkl"


def _cache_load(key: str):
    p = _cache_path(key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _cache_save(key: str, obj) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(obj, f)

# ---------------------------------------------------------------------------
# Matplotlib-Stil (konsistent mit SMARD_Fundamentalplot_v2)
# ---------------------------------------------------------------------------
STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "legend.framealpha": 0.88,
    "legend.edgecolor":  "#999999",
    "lines.linewidth":   1.4,
    "axes.linewidth":    0.9,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
    "grid.color":        "#aaaaaa",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#fafafa",
}

# Farbpalette (aus SMARD-Plot übernommen)
C_TRUTH    = "#333333"   # Ground Truth (dunkelgrau/schwarz)
C_ENS      = "#4393c3"   # Ensemble-Mitglieder (blau)
C_ENS_MEAN = "#2166ac"   # Ensemble-Mittelwert
C_SPINNER  = "#e66101"   # Vergleichsteilnehmer spinner (orange)
C_BAR_NORM = "#4393c3"   # Balken normal
C_BAR_OUT  = "#d7191c"   # Balken Ausreißer
C_MY_ROW   = "#f4a736"   # eigene Zeile in Leaderboard

# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════
#  MANUELLE DATEN – aus Energy-Arena-Screenshots eintragen
# ═══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

MY_USERNAME = "guenther99"   # Nutzername auf der Plattform

# ------ Plot 2: Challenge-10-Leaderboard 19.–25.06.2026 -------------------
# Quelle: Screenshot Energy Arena, Evaluation Period 19.6.–25.6.2026
# (Rang, Teilnehmer, WIS, LQS, MAE_Median, Coverage_50%, Coverage_95%)
# WIS/MAE in MW; Coverage als Dezimalzahl (30.65 % → 0.3065)
LEADERBOARD_C10: list[tuple] = [
    (1, "TelescopeEnergy", 1034.17, 517.09, 1434.98, 0.3065, 0.5610),
    (2, "guenther99",      1226.11, 613.06, 1911.06, 0.3244, 0.4955),
    (3, "spinner",         1800.18, 900.09, 2683.74, 0.1533, 0.3616),
    (4, "NaiveBenchmark",  2193.10, 1096.55, 3695.08, 0.2113, 0.5848),
    (5, "Mkay",            2610.69, 1305.35, 4676.28, 0.1994, 0.6339),
    (6, "OpenForecast",    2610.69, 1305.35, 4676.28, 0.1994, 0.6339),
    (7, "Wetterfrosch",    6709.02, 3354.51, 9039.99, 0.1830, 0.5149),
]

# ------ Plot 3: Tägliche MAE Challenge 10 (27.05.–25.06.2026) ------------
# Quelle: Energy Arena API /leaderboard/timeseries (Challenge 10, Metrik MAE)
# Berlin-Datum = UTC-target_start + 1 Tag (22:00 UTC = 00:00 CEST)
# Fehlende Tage = keine Einreichung (nicht im Wettbewerb aktiv)
DAILY_MAE_C10: dict[str, float] = {
    "2026-05-27": 3272.61,
    "2026-05-28": 3601.08,
    "2026-05-29": 4777.37,
    "2026-05-30": 1698.16,
    "2026-05-31": 2760.84,
    "2026-06-01": 2059.49,
    "2026-06-02": 1253.62,
    "2026-06-03": 3629.39,
    "2026-06-04": 5035.63,
    "2026-06-05": 2532.34,
    "2026-06-06": 1774.08,
    "2026-06-07": 3550.82,
    "2026-06-08": 1880.30,
    # 09.–11.06.: keine Einreichung
    "2026-06-12": 9365.31,   # ← AUSREISSER
    # 13.–14.06.: keine Einreichung
    "2026-06-15": 2098.64,
    "2026-06-16": 1539.04,
    "2026-06-17": 1848.26,
    "2026-06-18": 2963.63,
    "2026-06-19": 1704.32,
    # 20.06.: keine Einreichung
    "2026-06-21": 1205.84,
    "2026-06-22": 1357.87,
    "2026-06-23": 1517.59,
    "2026-06-24": 1302.51,
    "2026-06-25": 2174.29,
}
OUTLIER_MAE_C10 = ["2026-06-12"]

# ------ Plot 4: Tägliche CRPS Challenge 16 (27.05.–25.06.2026) -----------
# Quelle: Energy Arena API /leaderboard/timeseries (Challenge 16, Metrik CRPS)
# Berlin-Datum = UTC-target_start + 1 Tag
# Fehlende Tage = keine Einreichung
DAILY_CRPS_C16: dict[str, float] = {
    "2026-05-27": 2295.39,
    "2026-05-28": 2436.34,
    "2026-05-29": 4238.00,
    "2026-05-30":  901.17,
    "2026-05-31": 2611.43,
    "2026-06-01": 6319.93,   # ← AUSREISSER
    "2026-06-02": 2409.52,
    "2026-06-03": 1532.12,
    "2026-06-04": 2449.84,
    "2026-06-05": 1577.60,
    "2026-06-06": 1451.95,
    "2026-06-07": 2784.26,
    "2026-06-08": 1387.69,
    "2026-06-09": 2811.91,
    "2026-06-10": 1082.75,
    "2026-06-11": 1414.24,
    "2026-06-12": 10786.67,  # ← AUSREISSER
    # 13.06.: keine Einreichung
    "2026-06-14": 1828.76,
    "2026-06-15": 1687.20,
    "2026-06-16": 1245.77,
    "2026-06-17": 2245.53,
    "2026-06-18": 3631.90,
    "2026-06-19": 2918.72,
    # 20.06.: keine Einreichung
    "2026-06-21": 1723.19,
    "2026-06-22": 2997.96,
    "2026-06-23": 2674.77,
    "2026-06-24": 3721.36,
    "2026-06-25": 3487.58,
}
OUTLIER_CRPS_C16 = ["2026-06-01", "2026-06-12"]

# ------ Plots 1 & 5 & 6: spinner-Forecast (24 stündliche MW-Werte) --------
# Für Plot 1 (21.06.2026): Liste mit 24 Werten – aus Energy Arena ableiten
# None → wird im Plot weggelassen
SPINNER_FORECAST_21JUN: list[float] | None = None
# Beispiel: SPINNER_FORECAST_21JUN = [0,0,0,0,0,500,3200,8100,...,0,0]

# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════
#  PHYSIK-HILFSFUNKTIONEN (aus custom_model_3.py übernommen)
# ═══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

_TILT_RAD = np.radians(20.0)
_K_DIFF   = 0.5 * (1.0 + np.cos(_TILT_RAD))
_RHO_G    = 0.2
_K_REFL   = 0.5 * _RHO_G * (1.0 - np.cos(_TILT_RAD))
_GAMMA    = -0.004
_ETA      = 0.96


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


def _compute_p_physics(
    weather_df: pd.DataFrame,
    cluster_info: dict,
    timestamp_index: pd.DatetimeIndex,
) -> np.ndarray:
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
        g_beam * cos_z + dhi * _K_DIFF + ghi * _RHO_G * _K_REFL, 0.0
    )
    p_rated = cluster_info["capacity_gw"] * 0.001
    p_cell  = p_rated * (g_eff / 1000.0) * (1.0 + _GAMMA * (temp - 25.0))
    return np.maximum(p_cell * _ETA, 0.0)

_SUNNY_THRESHOLD = 150.0


def _build_feature_df(
    weather_dfs: dict[str, pd.DataFrame],
    extended_index: pd.DatetimeIndex,
    target_date: date,
    locations: list[dict],
    p_physics: np.ndarray | None = None,
    pv_series: pd.Series | None = None,
) -> pd.DataFrame:
    feat  = pd.DataFrame(index=extended_index)
    w_sum = sum(loc["weight"] for loc in locations)

    for loc in locations:
        df = (
            weather_dfs[loc["name"]]
            .reindex(extended_index, method="nearest", tolerance=pd.Timedelta("90min"))
            .fillna(0.0)
        )
        for col in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
            feat[f"{loc['name']}_{col}"] = df[col].values

    for var in ("ghi", "temp", "cloud", "precip", "wind", "dhi"):
        feat[f"w_{var}"] = (
            sum(feat[f"{loc['name']}_{var}"] * loc["weight"] for loc in locations) / w_sum
        )

    feat["hour"]      = extended_index.hour.astype(float)
    feat["month"]     = extended_index.month.astype(float)
    feat["dow"]       = extended_index.dayofweek.astype(float)
    feat["hour_sin"]  = np.sin(2 * np.pi * feat["hour"]  / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * feat["hour"]  / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    rep_lat = sum(loc["lat"] * loc["weight"] for loc in locations) / w_sum
    feat["cos_zenith"] = _cos_zenith_vec(extended_index, rep_lat)

    ghi_cs = np.maximum(1000.0 * feat["cos_zenith"].values, 10.0)
    feat["kt"] = np.clip(feat["w_ghi"].values / ghi_cs, 0.0, 1.5)

    w_ghi = feat["w_ghi"].copy()
    feat["lag_1d_ghi"] = w_ghi.shift(24).values
    feat["lag_2d_ghi"] = w_ghi.shift(48).values
    feat["lag_7d_ghi"] = w_ghi.shift(168).values
    feat["roll3d_ghi"] = w_ghi.rolling(72, min_periods=1).mean().values

    daily_mean = {d: float(g.mean()) for d, g in w_ghi.groupby(w_ghi.index.date)}
    feat["is_sunny"] = np.array(
        [float(daily_mean.get(ts.date(), 0.0) >= _SUNNY_THRESHOLD)
         for ts in extended_index]
    )

    if p_physics is not None:
        feat["p_physics"] = p_physics

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
    return feat.loc[extended_index.date == target_date]


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------

def load_model_artifacts() -> tuple[object, list[str], list[dict], bool]:
    """Lädt LightGBM-Modell, Feature-Namen und Cluster-Konfiguration."""
    model     = joblib.load(ARTIFACT_DIR / "lgbm_point.pkl")
    feat_names = json.loads((ARTIFACT_DIR / "feature_names.json").read_text())
    clusters  = json.loads((ARTIFACT_DIR / "clusters.json").read_text())
    physics   = "p_physics" in feat_names
    print(f"[Modell] {len(clusters)} Cluster, {len(feat_names)} Features, "
          f"physics_decomp={physics}")
    return model, feat_names, clusters, physics


# ---------------------------------------------------------------------------
# SMARD – PV-Einspeisung DE-LU (15-Min, Modul 125)
# ---------------------------------------------------------------------------

SMARD_MODULE = 125
SMARD_REGION = "DE-LU"


def fetch_smard_qh(start: date, end: date) -> pd.Series:
    """
    Lädt viertelstündliche PV-Einspeisedaten von SMARD (Modul 125, DE-LU).
    Rückgabe: pd.Series in MW, Index Europe/Berlin.
    """
    idx_url = (
        f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}"
        f"/index_quarterhour.json"
    )
    all_ts = requests.get(idx_url, timeout=30).json().get("timestamps", [])

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) - 8 * 86_400_000
    end_ms   = int(pd.Timestamp(end + timedelta(days=2), tz="UTC").timestamp() * 1000)
    relevant = sorted(ts for ts in all_ts if start_ms <= ts <= end_ms)

    records: list[tuple] = []
    for ts_ms in relevant:
        chunk_url = (
            f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}/"
            f"{SMARD_MODULE}_{SMARD_REGION}_quarterhour_{ts_ms}.json"
        )
        try:
            for row in requests.get(chunk_url, timeout=30).json().get("series", []):
                if row and row[1] is not None:
                    records.append((
                        pd.Timestamp(row[0], unit="ms", tz="UTC"),
                        float(row[1]) * 4.0,   # MWh/15-Min → MW
                    ))
        except Exception:
            pass
        time.sleep(0.08)

    if not records:
        raise RuntimeError(f"Keine SMARD-Daten für {start}–{end}")

    idx, vals = zip(*records)
    s = (
        pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), name="pv_mw")
        .sort_index()
        .pipe(lambda x: x[~x.index.duplicated(keep="first")])
    )
    s.index = s.index.tz_convert("Europe/Berlin")
    t0 = pd.Timestamp(start, tz="Europe/Berlin")
    t1 = pd.Timestamp(end + timedelta(days=1), tz="Europe/Berlin")
    return s.loc[(s.index >= t0) & (s.index < t1)]


def smard_hourly(start: date, end: date) -> pd.Series:
    """SMARD-Daten auf Stundenbasis (für LightGBM-Feature-Matrix)."""
    qh = fetch_smard_qh(start, end)
    return qh.resample("h").mean()


# ---------------------------------------------------------------------------
# Open-Meteo: deterministischer Forecast + Ensemble
# ---------------------------------------------------------------------------

def _parse_forecast_df(h: dict) -> pd.DataFrame:
    def _c(vals, fb=0.0):
        return [v if v is not None else fb for v in vals]

    df = pd.DataFrame(
        {
            "ghi":    _c(h["shortwave_radiation"]),
            "temp":   _c(h["temperature_2m"],   fb=float("nan")),
            "cloud":  _c(h["cloudcover"],        fb=float("nan")),
            "precip": _c(h["precipitation"]),
            "wind":   _c(h["windspeed_10m"],     fb=float("nan")),
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


def fetch_nwp_forecast(lat: float, lon: float, past_days: int = 9) -> pd.DataFrame:
    key = f"nwp_{lat:.4f}_{lon:.4f}_pd{past_days}"
    cached = _cache_load(key)
    if cached is not None:
        return cached
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "shortwave_radiation,temperature_2m,cloudcover,"
                  "precipitation,windspeed_10m,diffuse_radiation",
        "forecast_days": 2, "past_days": past_days,
        "timezone": "Europe/Berlin",
    }
    h = requests.get(url, params=params, timeout=30).json()["hourly"]
    result = _parse_forecast_df(h)
    _cache_save(key, result)
    return result


def fetch_ensemble(
    lat: float, lon: float, past_days: int = 9
) -> dict[str, pd.DataFrame]:
    """Fetch ECMWF IFS025 ensemble (50 members) from Open-Meteo."""
    key = f"ens_{lat:.4f}_{lon:.4f}_pd{past_days}"
    cached = _cache_load(key)
    if cached is not None:
        return cached
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ("shortwave_radiation,temperature_2m,cloudcover,"
                   "diffuse_radiation,precipitation,windspeed_10m"),
        "models": "ecmwf_ifs025",
        "forecast_days": 2, "past_days": past_days,
        "timezone": "Europe/Berlin",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    times = pd.to_datetime(h["time"])

    def _c(vals, fb=0.0):
        return [v if v is not None else fb for v in vals]

    _VARS = [
        ("shortwave_radiation", "ghi",    0.0),
        ("temperature_2m",      "temp",   10.0),
        ("cloudcover",          "cloud",  50.0),
        ("diffuse_radiation",   "dhi",    0.0),
        ("precipitation",       "precip", 0.0),
        ("windspeed_10m",       "wind",   3.0),
    ]

    members: dict[str, pd.DataFrame] = {}
    for mid in range(1, 51):
        mkey = f"member{mid:02d}"
        cols: dict[str, list] = {}
        for api_var, col, fb in _VARS:
            key_m = f"{api_var}_{mkey}"
            key_e = f"{api_var}_member01"  # fallback to member01 as mean
            if key_m in h:
                cols[col] = _c(h[key_m], fb)
            elif f"{api_var}" in h:
                cols[col] = _c(h[api_var], fb)
            else:
                cols[col] = [fb] * len(times)

        df = pd.DataFrame(cols, index=times)
        try:
            df.index = df.index.tz_localize(
                "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
            )
        except Exception:
            df.index = df.index.tz_localize(
                "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
            )
            df = df[df.index.notna()]
        members[mkey] = df.sort_index()

    _cache_save(key, members)
    return members


# ---------------------------------------------------------------------------
# Ensemble-Inferenz für einen Zieltag
# ---------------------------------------------------------------------------

def run_ensemble_for_date(
    target_date: date,
    model,
    feat_names: list[str],
    clusters: list[dict],
    physics_decomp: bool,
) -> np.ndarray:
    """
    Führt LightGBM-Ensemble-Inferenz für target_date durch.
    Gibt ndarray (50, 24) in MW zurück.

    Nutzt dieselbe Pipeline wie custom_model_3.py:
      • Open-Meteo NWP (deterministisch) für Lag-Features
      • Open-Meteo ECMWF IFS025 Ensemble (50 Mitglieder) für Wetter-Features
      • LightGBM point model für jeden Member

    past_days wird automatisch aus dem Abstand zu heute berechnet.
    """
    today     = date.today()
    past_days = (today - target_date).days + 2   # +2 Puffer für forecast_days

    if past_days > 92:
        raise ValueError(
            f"target_date {target_date} liegt >92 Tage zurück – "
            "Open-Meteo unterstützt maximal 92 past_days."
        )

    print(f"[Ensemble] Lade Wetterdaten für {target_date} (past_days={past_days}) …")

    # Deterministischer NWP-Forecast für alle Cluster
    nwp_dfs: dict[str, pd.DataFrame] = {}
    for loc in clusters:
        nwp_dfs[loc["name"]] = fetch_nwp_forecast(loc["lat"], loc["lon"], past_days)
        time.sleep(0.3)

    # Extended time index über NWP-Fenster
    ens_index = nwp_dfs[clusters[0]["name"]].index

    # SMARD-Lag-Features (PV-Einspeisung von gestern / letzter Woche)
    lag_start = target_date - timedelta(days=10)
    try:
        pv_series = smard_hourly(lag_start, target_date)
        print(f"[SMARD] {len(pv_series)} Stundenwerte für Lag-Features geladen")
    except Exception as exc:
        print(f"[SMARD] Lag-Features nicht verfügbar: {exc}. Setze auf 0.")
        pv_series = None

    # Ensemble-Mitglieder pro Cluster abrufen
    print(f"[Ensemble] Lade ECMWF IFS025 für {len(clusters)} Cluster …")
    ens_loc: dict[str, dict[str, pd.DataFrame]] = {}
    for i, loc in enumerate(clusters):
        print(f"  [{i+1}/{len(clusters)}] {loc['name']}", flush=True)
        ens_loc[loc["name"]] = fetch_ensemble(loc["lat"], loc["lon"], past_days)
        time.sleep(0.5)

    # Inferenz für jeden der 50 Members
    w_sum     = sum(loc["weight"] for loc in clusters)
    preds_list: list[np.ndarray] = []

    for mid in range(1, 51):
        mkey      = f"member{mid:02d}"
        m_weather = {loc["name"]: ens_loc[loc["name"]][mkey] for loc in clusters}

        p_phys_m = np.zeros(len(ens_index))
        if physics_decomp:
            for loc in clusters:
                p_phys_m += (
                    _compute_p_physics(m_weather[loc["name"]], loc, ens_index)
                    * loc["weight"] / w_sum
                )

        X_df = _build_feature_df(
            m_weather, ens_index, target_date, clusters,
            p_physics=p_phys_m if physics_decomp else None,
            pv_series=pv_series,
        )

        if X_df.empty:
            preds_list.append(np.zeros(24))
            continue

        # Feature-Spalten in Trainingsreihenfolge bringen; fehlende → 0
        X_aligned = pd.DataFrame(0.0, index=X_df.index, columns=feat_names)
        common    = [f for f in feat_names if f in X_df.columns]
        X_aligned[common] = X_df[common]

        residual_m = model.predict(X_aligned.values)

        if physics_decomp:
            day_mask = ens_index.date == target_date
            p_day    = p_phys_m[day_mask]
            if len(p_day) == len(residual_m):
                pred = np.maximum(p_day + residual_m, 0.0)
            else:
                pred = np.maximum(residual_m, 0.0)
        else:
            pred = np.maximum(residual_m, 0.0)

        # Nacht-Maske (< 3 % des Tagespeaks → 0)
        peak = pred.max() if pred.max() > 0 else 1.0
        pred[pred < 0.03 * peak] = 0.0

        preds_list.append(pred)

    members_24 = np.array(preds_list)  # (50, 24)
    print(f"[Ensemble] {target_date}: Mittelwert={members_24.mean():.1f} MW, "
          f"Max={members_24.max():.1f} MW")
    return members_24


# ---------------------------------------------------------------------------
# PLOT 1 + 5 + 6 – Ensemble-Forecast für einen oder mehrere Tage
# ---------------------------------------------------------------------------

def _tz_day_index_qh(d: date) -> pd.DatetimeIndex:
    start = pd.Timestamp(d, tz="Europe/Berlin")
    return pd.date_range(start, periods=96, freq="15min")


def plot_ensemble_day(
    out_name: str,
    target_date: date,
    truth_qh: pd.Series,       # 96 Werte, 15-Min
    members_h: np.ndarray,     # (50, 24) stündlich
    spinner_h: list | None = None,
    title: str | None = None,
    annotation: str | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=DPI)

    hours   = np.arange(24) + 0.5
    hours_x = [pd.Timestamp(target_date, tz="Europe/Berlin") + pd.Timedelta(hours=h)
                for h in hours]
    qh_idx  = _tz_day_index_qh(target_date)

    # Ground Truth (15-Min)
    truth_day = truth_qh.reindex(qh_idx, method="nearest", tolerance=pd.Timedelta("10min"))
    ax.plot(
        truth_day.index, truth_day.values,
        color=C_TRUTH, linewidth=1.8, linestyle="--",
        label="Ground Truth (SMARD)", zorder=5,
    )

    # Ensemble-Mitglieder
    for i, m in enumerate(members_h):
        ax.step(
            hours_x, m, where="post",
            color=C_ENS, linewidth=0.5, alpha=0.25, zorder=2,
            label="ECMWF IFS025 Ensemble (50 Mitgl.)" if i == 0 else "",
        )

    # Ensemble-Mittelwert
    ens_mean = members_h.mean(axis=0)
    ax.step(
        hours_x, ens_mean, where="post",
        color=C_ENS_MEAN, linewidth=1.8, zorder=4,
        label="Ensemble-Mittelwert",
    )

    # Vergleichsteilnehmer spinner (optional)
    if spinner_h is not None:
        ax.step(
            hours_x, spinner_h, where="post",
            color=C_SPINNER, linewidth=2.0, zorder=4,
            label="spinner (Referenz)",
        )

    # Horizontale Linie (z. B. Kapazitäts-Cap)
    if hline is not None:
        ax.axhline(
            hline, color="#d7191c", linewidth=1.4, linestyle=":",
            label=hline_label or f"{hline:,.0f} MW",
        )

    if annotation:
        ax.text(
            0.02, 0.97, annotation,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="#555555",
            bbox=dict(fc="white", ec="none", alpha=0.7),
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_BERLIN))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel(f"Uhrzeit ({target_date.strftime('%d.%m.%Y')}, MEZ/MESZ)")
    ax.set_ylabel("PV-Einspeisung (MW)")
    ax.set_title(title or f"Ensemble-Prognose – {target_date.strftime('%d.%m.%Y')}")
    ax.set_xlim(
        pd.Timestamp(target_date, tz="Europe/Berlin"),
        pd.Timestamp(target_date + timedelta(days=1), tz="Europe/Berlin"),
    )
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    leg_loc = "upper right" if annotation else "upper left"
    ax.legend(seen.values(), seen.keys(), loc=leg_loc, fontsize=10)

    fig.text(
        0.99, 0.01,
        "Quelle: SMARD (Ground Truth) | Open-Meteo ECMWF IFS025 (Ensemble) | LightGBM",
        ha="right", va="bottom", fontsize=8, color="#888888",
    )
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_multi_day_ensemble(
    out_name: str,
    date_range: list[date],
    truth_qh: pd.Series,           # langer 15-Min-Zeitreihe
    members_per_day: dict[date, np.ndarray],  # date → (50, 24)
    title: str | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
    annotation: str | None = None,
) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)

    tz = "Europe/Berlin"
    t0 = pd.Timestamp(date_range[0],  tz=tz)
    t1 = pd.Timestamp(date_range[-1] + timedelta(days=1), tz=tz)

    # Ground Truth
    truth_win = truth_qh.loc[t0:t1]
    ax.plot(
        truth_win.index, truth_win.values,
        color=C_TRUTH, linewidth=1.6, linestyle="--",
        label="Ground Truth (SMARD)", zorder=5,
    )

    # Ensemble je Tag
    first = True
    for d in date_range:
        if d not in members_per_day:
            continue
        members = members_per_day[d]
        hours_x = [
            pd.Timestamp(d, tz=tz) + pd.Timedelta(hours=h + 0.5)
            for h in range(24)
        ]
        for i, m in enumerate(members):
            ax.step(
                hours_x, m, where="post",
                color=C_ENS, linewidth=0.4, alpha=0.18, zorder=2,
                label="ECMWF IFS025 Ensemble (50 Mitgl.)" if (first and i == 0) else "",
            )
        ens_mean = members.mean(axis=0)
        ax.step(
            hours_x, ens_mean, where="post",
            color=C_ENS_MEAN, linewidth=1.4, zorder=4,
            label="Ensemble-Mittelwert" if first else "",
        )
        first = False

    if hline is not None:
        ax.axhline(
            hline, color="#d7191c", linewidth=1.4, linestyle=":",
            label=hline_label or f"{hline:,.0f} MW",
        )

    if annotation:
        ax.text(
            0.01, 0.97, annotation,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="#555555",
            bbox=dict(fc="white", ec="none", alpha=0.7),
        )

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=TZ_BERLIN))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Datum (MEZ/MESZ)")
    ax.set_ylabel("PV-Einspeisung (MW)")
    ax.set_title(title or "Ensemble-Prognose – Mehrtagesansicht")
    ax.set_xlim(t0, t1)
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    leg_loc = "upper right" if annotation else "upper left"
    ax.legend(seen.values(), seen.keys(), loc=leg_loc, fontsize=10)

    fig.text(
        0.99, 0.01,
        "Quelle: SMARD (Ground Truth) | Open-Meteo ECMWF IFS025 (Ensemble) | LightGBM",
        ha="right", va="bottom", fontsize=8, color="#888888",
    )
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# PLOT 7 – 7-Tage-Gesamtübersicht (Ground Truth + Ensemble-Band)
# ---------------------------------------------------------------------------

def plot_7day_overview(
    out_name: str,
    date_range: list[date],
    truth_qh: pd.Series,
    members_per_day: dict[date, np.ndarray],
    title: str | None = None,
) -> None:
    """Ground Truth + Ensemble-Mittelwert + IQR-Band, kein Cap-Fokus."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)

    tz = "Europe/Berlin"
    t0 = pd.Timestamp(date_range[0],  tz=tz)
    t1 = pd.Timestamp(date_range[-1] + timedelta(days=1), tz=tz)

    # Ground Truth
    truth_win = truth_qh.loc[t0:t1]
    if not truth_win.empty:
        ax.plot(
            truth_win.index, truth_win.values,
            color=C_TRUTH, linewidth=1.6, linestyle="--",
            label="Ground Truth (SMARD)", zorder=5,
        )

    # Ensemble-Band (25–75 % Quantil) + Mittelwert
    first = True
    for d in date_range:
        if d not in members_per_day:
            continue
        members = members_per_day[d]           # (50, 24)
        hours_x = [
            pd.Timestamp(d, tz=tz) + pd.Timedelta(hours=h + 0.5)
            for h in range(24)
        ]
        q25 = np.percentile(members, 25, axis=0)
        q75 = np.percentile(members, 75, axis=0)
        mean_m = members.mean(axis=0)

        # IQR-Fläche
        ax.fill_between(
            hours_x, q25, q75,
            step="post", color=C_ENS, alpha=0.30, zorder=2,
            label="Ensemble IQR (25–75 %)" if first else "",
        )
        # Mittelwert
        ax.step(
            hours_x, mean_m, where="post",
            color=C_ENS_MEAN, linewidth=1.4, zorder=4,
            label="Ensemble-Mittelwert" if first else "",
        )
        first = False

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=TZ_BERLIN))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Datum (MEZ/MESZ)")
    ax.set_ylabel("PV-Einspeisung (MW)")
    ax.set_title(title or "PV-Forecast-Gesamtübersicht 22.–29.06.2026")
    ax.set_xlim(t0, t1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10)

    fig.text(
        0.99, 0.01,
        "Quelle: SMARD (Ground Truth) | Open-Meteo ECMWF IFS025 (Ensemble) | LightGBM",
        ha="right", va="bottom", fontsize=8, color="#888888",
    )
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# PLOT 2 – Leaderboard-Tabelle
# ---------------------------------------------------------------------------

def plot_leaderboard(
    out_name: str,
    data: list[tuple],
    my_name: str,
    title: str = "Challenge 10 – Leaderboard (19.–25.06.2026)",
) -> None:
    plt.rcParams.update(STYLE)

    col_labels = ["Rang", "Teilnehmer", "WIS (MW)", "LQS", "MAE\nMedian (MW)",
                  "Coverage\n50 %", "Coverage\n95 %"]

    rows = []
    for rank, name, wis, lqs, mae, cov50, cov95 in data:
        rows.append([
            str(rank), name,
            f"{wis:,.0f}", f"{lqs:.3f}", f"{mae:,.0f}",
            f"{cov50:.1%}", f"{cov95:.1%}",
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)

    fig_h = max(2.4, 0.5 + n_rows * 0.52)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=DPI)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Header-Style
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#2166ac")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        tbl[(0, j)].set_height(0.18)

    # Zeilen-Style
    for i, (_, name, *_rest) in enumerate(data, start=1):
        for j in range(n_cols):
            cell = tbl[(i, j)]
            if name == my_name:
                cell.set_facecolor("#fff3cd")
                cell.set_text_props(fontweight="bold", color="#7b4f00")
            elif i % 2 == 0:
                cell.set_facecolor("#f0f4f8")
            else:
                cell.set_facecolor("white")
            cell.set_height(0.14)

    # Spaltenbreiten
    col_widths = [0.06, 0.18, 0.12, 0.10, 0.14, 0.12, 0.12]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            tbl[(i, j)].set_width(w)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.text(
        0.99, 0.01,
        f"Quelle: Energy Arena (energy-arena.org) | Stand: 25.06.2026",
        ha="right", va="bottom", fontsize=8, color="#888888",
    )
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# PLOT 3 & 4 – Tägliche Metriken (Balkendiagramm)
# ---------------------------------------------------------------------------

def plot_daily_metric(
    out_name: str,
    daily_values: dict[str, float],
    metric_label: str,
    outlier_dates: list[str],
    title: str,
    ylabel: str,
) -> None:
    plt.rcParams.update(STYLE)

    dates = sorted(daily_values.keys())
    vals  = [daily_values[d] for d in dates]
    ts    = pd.to_datetime(dates)

    colors = [C_BAR_OUT if d in outlier_dates else C_BAR_NORM for d in dates]

    fig, ax = plt.subplots(figsize=(13, 4.5), dpi=DPI)
    bars = ax.bar(ts, vals, color=colors, width=0.7, edgecolor="none", zorder=3)

    # Annotations für Ausreißer
    max_val = max(vals) if vals else 1.0
    for d, v, c in zip(ts, vals, colors):
        if c == C_BAR_OUT:
            ax.annotate(
                f"{v:,.0f}",
                xy=(d, v), xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, color=C_BAR_OUT, fontweight="bold",
            )

    # Referenzlinie: Mittelwert (ohne Ausreißer)
    clean_vals = [v for d, v in zip(dates, vals) if d not in outlier_dates]
    if clean_vals:
        mean_clean = np.mean(clean_vals)
        ax.axhline(
            mean_clean, color="#555555", linewidth=1.2, linestyle="--",
            label=f"Mittelwert (ohne Ausreißer): {mean_clean:,.0f} {metric_label}",
            zorder=4,
        )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(ts[0] - pd.Timedelta(days=1), ts[-1] + pd.Timedelta(days=1))
    ax.set_ylim(bottom=0, top=max_val * 1.18)

    # Legende: Farberklärung
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=C_BAR_NORM, label="Regulärer Tag"),
        Patch(facecolor=C_BAR_OUT,  label="Ausreißertag"),
    ]
    if clean_vals:
        from matplotlib.lines import Line2D
        legend_elems.append(
            Line2D([0], [0], color="#555555", linestyle="--",
                   label=f"Mittelwert: {mean_clean:,.0f} {metric_label}")
        )
    ax.legend(handles=legend_elems, loc="upper right", fontsize=10)

    fig.text(
        0.99, 0.01,
        "Quelle: Energy Arena (energy-arena.org) | Challenge-Ergebnisse",
        ha="right", va="bottom", fontsize=8, color="#888888",
    )
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("generate_paper_figures.py – Seminararbeit Abbildungen")
    print("=" * 65)

    # Modell laden
    model, feat_names, clusters, physics_decomp = load_model_artifacts()

    # -----------------------------------------------------------------------
    # PLOT 1 – Ensemble-Forecast 21.06.2026 (Challenge 16)
    # -----------------------------------------------------------------------
    target_21 = date(2026, 6, 21)
    print(f"\n[Plot 1] Ensemble {target_21} …")
    try:
        # Ground Truth
        truth_qh_21 = fetch_smard_qh(target_21, target_21)
        # Ensemble-Inferenz
        members_21 = run_ensemble_for_date(
            target_21, model, feat_names, clusters, physics_decomp
        )
        plot_ensemble_day(
            "fig_1_ensemble_21jun.png",
            target_21,
            truth_qh_21,
            members_21,
            spinner_h=SPINNER_FORECAST_21JUN,
            title="Ensemble-Prognose PV-Einspeisung – 21.06.2026 (Challenge 16)",
            annotation=(
                "ECMWF IFS025: 50 Ensemble-Mitglieder\n"
                "LightGBM-Inferenz pro Mitglied\n"
                "Hinweis: Rückwirkend rekonstruiert\n"
                "(Reanalyse, kein originaler Forecast)"
            ),
        )
    except Exception as exc:
        print(f"[!] Plot 1 fehlgeschlagen: {exc}")

    # -----------------------------------------------------------------------
    # PLOT 2 – Leaderboard Challenge 10 (19.–25.06.2026)
    # -----------------------------------------------------------------------
    print("\n[Plot 2] Leaderboard Challenge 10 …")
    if LEADERBOARD_C10:
        plot_leaderboard(
            "fig_2_leaderboard_c10.png",
            LEADERBOARD_C10,
            MY_USERNAME,
            title="Challenge 10 – Quantile-Forecast Leaderboard (19.–25.06.2026)",
        )
    else:
        print("[!] LEADERBOARD_C10 leer – bitte Daten eintragen.")

    # -----------------------------------------------------------------------
    # PLOT 3 – Tägliche MAE Challenge 10 (27.05.–25.06.2026)
    # -----------------------------------------------------------------------
    print("\n[Plot 3] MAE-Verlauf Challenge 10 …")
    if DAILY_MAE_C10:
        plot_daily_metric(
            "fig_3_mae_c10.png",
            DAILY_MAE_C10,
            "MW",
            OUTLIER_MAE_C10,
            title="Challenge 10 – Täglicher MAE (27.05.–25.06.2026)",
            ylabel="MAE (MW)",
        )
    else:
        print("[!] DAILY_MAE_C10 leer – bitte Daten eintragen.")

    # -----------------------------------------------------------------------
    # PLOT 4 – Tägliche CRPS Challenge 16 (27.05.–25.06.2026)
    # -----------------------------------------------------------------------
    print("\n[Plot 4] CRPS-Verlauf Challenge 16 …")
    if DAILY_CRPS_C16:
        plot_daily_metric(
            "fig_4_crps_c16.png",
            DAILY_CRPS_C16,
            "MW",
            OUTLIER_CRPS_C16,
            title="Challenge 16 – Täglicher CRPS (27.05.–25.06.2026)",
            ylabel="CRPS (MW)",
        )
    else:
        print("[!] DAILY_CRPS_C16 leer – bitte Daten eintragen.")

    # -----------------------------------------------------------------------
    # PLOTS 5 & 6 – Ensemble 22.–29.06.2026 (Cap-Problem + Regime-Wechsel)
    # -----------------------------------------------------------------------
    cap_dates = [date(2026, 6, d) for d in range(22, 30)]
    today     = date.today()

    # Ground Truth für gesamten Zeitraum
    print("\n[Plot 5+6] Lade SMARD für 22.–29.06.2026 …")
    truth_avail_end = today   # SMARD i.d.R. am nächsten Tag verfügbar
    smard_end = min(date(2026, 6, 29), truth_avail_end)
    try:
        truth_qh_range = fetch_smard_qh(date(2026, 6, 22), smard_end)
        print(f"[SMARD] {len(truth_qh_range)} Viertelstundenwerte geladen "
              f"(bis {smard_end})")
    except Exception as exc:
        print(f"[!] SMARD-Abruf fehlgeschlagen: {exc}")
        truth_qh_range = pd.Series(dtype=float)

    # Ensemble-Inferenz je Tag (nur für verfügbare Tage)
    members_range: dict[date, np.ndarray] = {}
    for d in cap_dates:
        days_ago = (today - d).days
        if days_ago > 92:
            print(f"  {d}: >92 Tage zurück – übersprungen.")
            continue
        if days_ago < 0:
            print(f"  {d}: liegt in der Zukunft – übersprungen.")
            continue
        print(f"\n[Plot 5+6] Ensemble {d} …")
        try:
            members_range[d] = run_ensemble_for_date(
                d, model, feat_names, clusters, physics_decomp
            )
        except Exception as exc:
            print(f"  [!] {d}: {exc}")

    # Plot 5: Gesamter Mehrtages-Zeitraum
    if members_range:
        print("\n[Plot 5] Mehrtages-Ensemble …")
        plot_multi_day_ensemble(
            "fig_5_cap_problem.png",
            sorted(members_range.keys()),
            truth_qh_range,
            members_range,
            title="PV-Ensemble-Prognose 22.–29.06.2026 – Kapazitäts-Cap",
            hline=41_000.0,
            hline_label="Plateau-Referenz ≈ 41 000 MW (lag_1d_pv-abhängig)",
            annotation="Plateau-Niveau variiert mit lag_1d_pv\n(Vortageswert als Anker)",
        )

    # Plot 6: Nur 29.06.2026
    target_29 = date(2026, 6, 29)
    if target_29 in members_range:
        print("\n[Plot 6] Einzeltag 29.06.2026 (Regime-Wechsel) …")
        truth_29 = (
            truth_qh_range.loc[
                truth_qh_range.index.date == target_29
            ]
            if not truth_qh_range.empty else pd.Series(dtype=float)
        )
        plot_ensemble_day(
            "fig_6_regime_change.png",
            target_29,
            truth_29,
            members_range[target_29],
            title="Ensemble-Prognose PV-Einspeisung – 29.06.2026 (Wetterumschwung)",
            annotation=(
                "Regime-Wechsel: Sonnig → Bewölkt\n"
                "Hohe Ensemble-Streuung bei Wetterumbruch"
            ),
        )
    else:
        print(f"[!] Plot 6: Keine Ensemble-Daten für {target_29}")

    # Plot 7: 7-Tage-Gesamtübersicht (Ground Truth + Ensemble-Band)
    if members_range:
        print("\n[Plot 7] 7-Tage-Gesamtübersicht …")
        plot_7day_overview(
            "fig_7_overview_22_29jun.png",
            sorted(members_range.keys()),
            truth_qh_range,
            members_range,
            title="PV-Einspeisung 22.–29.06.2026 – Forecast-Gesamtübersicht",
        )

    print("\n" + "=" * 65)
    print(f"Fertig! Alle Abbildungen in: {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
