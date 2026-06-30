#!/usr/bin/env python3
"""
SMARD Fundamentalplot – Solareinspeisung DE_LU 2024
Lädt 15-Min-Daten von der SMARD-API (kein API-Key nötig) und erstellt
drei Subplots zu den fundamentalen Mustern der PV-Einspeisung.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
SMARD_MODULE = 4068
SMARD_REGION = "DE"
INSTALLED_CAPACITY_MW = 66_600

START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)

OUTPUT_PATH = Path(__file__).parent / "SMARD_Fundamentalplot_v2.png"

SEASON_COLORS = {
    "Winter":   "#2166ac",
    "Frühling": "#4dac26",
    "Sommer":   "#d7191c",
    "Herbst":   "#f4a736",
}

SEASON_MONTHS = {
    "Winter":   (12, 1, 2),
    "Frühling": (3, 4, 5),
    "Sommer":   (6, 7, 8),
    "Herbst":   (9, 10, 11),
}


def get_season(month: int) -> str:
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    return "Herbst"


# ---------------------------------------------------------------------------
# SMARD-Datenabruf
# ---------------------------------------------------------------------------
def fetch_smard_series(start: date, end: date) -> pd.Series:
    """Lädt 15-Min-PV-Einspeisedaten (MW) von der SMARD-API und
    gibt sie als pd.Series (Europe/Berlin, normiert auf Kapazitätsfaktor) zurück."""
    index_url = (
        f"https://www.smard.de/app/chart_data/{SMARD_MODULE}/{SMARD_REGION}"
        f"/index_quarterhour.json"
    )
    print(f"[SMARD] Index abrufen: {index_url}")
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()
    all_ts = resp.json().get("timestamps", [])

    # 8 Tage Puffer am Anfang, 2 Tage am Ende (wöchentliche Chunks)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) - 8 * 86_400_000
    end_ms   = int(pd.Timestamp(end + timedelta(days=2), tz="UTC").timestamp() * 1000)
    relevant = sorted(ts for ts in all_ts if start_ms <= ts <= end_ms)

    print(f"[SMARD] {len(relevant)} wöchentliche Chunks werden geladen …")
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
                        float(row[1]) * 4.0,   # MWh/15-Min → MW
                    ))
        except Exception as exc:
            print(f"[SMARD]   Chunk {ts_ms} fehlgeschlagen: {exc}")
        if (i + 1) % 10 == 0:
            print(f"[SMARD]   {i+1}/{len(relevant)} erledigt")
        time.sleep(0.1)

    if not records:
        raise RuntimeError("Keine Daten von SMARD erhalten!")

    idx, vals = zip(*records)
    s = pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), name="pv_mw")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s.index = s.index.tz_convert("Europe/Berlin")

    # Auf gewünschten Zeitraum filtern
    s = s.loc[str(START_DATE) : str(END_DATE)]

    # Auf Kapazitätsfaktor normieren
    s = s / INSTALLED_CAPACITY_MW

    print(f"[SMARD] {len(s)} Datenpunkte geladen.")
    print(f"[SMARD] Zeitraum: {s.index[0]} bis {s.index[-1]}")
    print(f"[SMARD] Max. Kapazitätsfaktor: {s.max():.4f}")
    return s


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(s: pd.Series) -> None:
    df = s.to_frame("cf")
    df["hour_frac"] = df.index.hour + df.index.minute / 60.0
    df["hour_int"]  = df.index.hour
    df["month"]     = df.index.month
    df["date"]      = df.index.date
    df["season"]    = df["month"].apply(get_season)

    # --- Aggregationen ---
    # (1) Stundenmittel + Std je Jahreszeit
    grp = df.groupby(["season", "hour_int"])["cf"]
    s_mean = grp.mean()
    s_std  = grp.std()

    # (2) Tagesmaximum
    daily_max = df.groupby("date")["cf"].max()
    daily_max.index = pd.to_datetime(daily_max.index).tz_localize("Europe/Berlin")

    # (3) Jahreszeiten je Tag (für Plot-3-Einfärbung)
    date_season = df.groupby("date")["season"].first()

    # ---------------------------------------------------------------------------
    # Matplotlib-Stilvorgaben (IEEE-ähnlich)
    # ---------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          16,
        "axes.titlesize":     17,
        "axes.labelsize":     15,
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    13,
        "legend.framealpha":  0.88,
        "legend.edgecolor":   "#999999",
        "lines.linewidth":    1.6,
        "axes.linewidth":     0.9,
        "axes.grid":          True,
        "grid.alpha":         0.28,
        "grid.linewidth":     0.5,
        "grid.color":         "#aaaaaa",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "#fafafa",
    })

    DPI    = 300
    FIG_W  = 5490 / DPI   # ≈ 18.3 Zoll
    FIG_H  = 3500 / DPI   # ≈ 11.67 Zoll

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.26,
        left=0.07, right=0.97,
        top=0.91, bottom=0.09,
        height_ratios=[1.0, 1.05],
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    hours = np.arange(24)

    # -----------------------------------------------------------------------
    # (a) Mittlerer Tagesgang nach Jahreszeit
    # -----------------------------------------------------------------------
    for season in ["Winter", "Frühling", "Sommer", "Herbst"]:
        color = SEASON_COLORS[season]
        if season not in s_mean.index.get_level_values(0):
            continue
        mean_h = s_mean[season].reindex(hours, fill_value=0.0).values
        std_h  = s_std[season].reindex(hours, fill_value=0.0).values
        ax1.plot(hours, mean_h, color=color, label=season, linewidth=2.2, zorder=3)
        ax1.fill_between(
            hours,
            np.clip(mean_h - std_h, 0, None),
            mean_h + std_h,
            color=color, alpha=0.14, zorder=2,
        )

    ax1.set_xlim(0, 23)
    ax1.set_ylim(bottom=0)
    ax1.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax1.set_xticklabels(["0", "3", "6", "9", "12", "15", "18", "21", "23"])
    ax1.set_xlabel("Uhrzeit (h)")
    ax1.set_ylabel("Kapazitätsfaktor")
    ax1.set_title("(a) Mittlerer Tagesgang nach Jahreszeit\n± 1 Standardabweichung")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # -----------------------------------------------------------------------
    # (b) Tagesmaximum über das Jahr
    # -----------------------------------------------------------------------
    ax2.plot(
        daily_max.index, daily_max.values,
        color="#555555", linewidth=0.85, alpha=0.75,
        label="Tagesmaximum",
    )
    mean_max = daily_max.mean()
    ax2.axhline(
        mean_max,
        color="#d7191c", linewidth=1.8, linestyle="--",
        label=f"Jahresmittel: {mean_max:.3f}",
        zorder=4,
    )

    # Jahreszeiten als farbige Unterlegung
    season_month_map = {m: s for s, months in SEASON_MONTHS.items() for m in months}
    for month_start, season in season_month_map.items():
        if month_start == 12:
            xs = pd.Timestamp("2024-12-01", tz="Europe/Berlin")
            xe = pd.Timestamp("2024-12-31 23:59", tz="Europe/Berlin")
        else:
            xs = pd.Timestamp(f"2024-{month_start:02d}-01", tz="Europe/Berlin")
            xe = xs + pd.offsets.MonthEnd(1)
            xe = xe.replace(tzinfo=xs.tzinfo)
        ax2.axvspan(xs, xe, color=SEASON_COLORS[season], alpha=0.07, linewidth=0)

    ax2.set_xlim(daily_max.index[0], daily_max.index[-1])
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_xlabel("Datum")
    ax2.set_ylabel("Kapazitätsfaktor")
    ax2.set_title("(b) Tagesmaximum der PV-Einspeisung 2024")
    ax2.legend(loc="upper right")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # -----------------------------------------------------------------------
    # (c) Alle Tagesprofile übereinandergelegt
    # -----------------------------------------------------------------------
    for d, season in date_season.items():
        day_df = df[df["date"] == d].sort_values("hour_frac")
        if len(day_df) < 4:
            continue
        ax3.plot(
            day_df["hour_frac"].values,
            day_df["cf"].values,
            color=SEASON_COLORS[season],
            alpha=0.10,
            linewidth=0.55,
            rasterized=True,
        )

    # Mittelwert-Overlay je Jahreszeit (für Übersichtlichkeit)
    for season in ["Winter", "Frühling", "Sommer", "Herbst"]:
        if season not in s_mean.index.get_level_values(0):
            continue
        mean_h = s_mean[season].reindex(hours, fill_value=0.0).values
        ax3.plot(
            hours, mean_h,
            color=SEASON_COLORS[season], linewidth=2.2,
            alpha=0.85, zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], color=c, linewidth=2.2, label=s)
        for s, c in SEASON_COLORS.items()
    ]
    ax3.legend(handles=legend_handles, loc="upper right")

    ax3.set_xlim(0, 24)
    ax3.set_ylim(bottom=0)
    ax3.set_xticks(range(0, 25, 2))
    ax3.set_xlabel("Uhrzeit (h)")
    ax3.set_ylabel("Kapazitätsfaktor")
    ax3.set_title(
        "(c) Alle Tagesprofile 2024 – nach Jahreszeit eingefärbt\n"
        "(transparente Linien = Einzeltage, opake Linien = Saisonmittel)"
    )
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # -----------------------------------------------------------------------
    # Haupttitel
    # -----------------------------------------------------------------------
    fig.suptitle(
        "Solareinspeisung DE_LU – Fundamentale Muster (SMARD, 2024)",
        fontsize=19, fontweight="bold", y=0.975,
    )

    # Fußnote
    fig.text(
        0.5, 0.005,
        f"Quelle: SMARD.de | Modul {SMARD_MODULE} | Normierung: {INSTALLED_CAPACITY_MW:,} MW installierte Kapazität",
        ha="center", va="bottom", fontsize=10, color="#666666",
    )

    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"\n[PLOT] Gespeichert: {OUTPUT_PATH}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SMARD Fundamentalplot – Solareinspeisung DE_LU")
    print(f"Zeitraum : {START_DATE} bis {END_DATE}")
    print(f"Kapazität: {INSTALLED_CAPACITY_MW:,} MW")
    print("=" * 60)

    series = fetch_smard_series(START_DATE, END_DATE)
    make_plot(series)
    print("Fertig!")
