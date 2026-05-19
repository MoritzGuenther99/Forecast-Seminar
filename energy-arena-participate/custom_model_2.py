from __future__ import annotations

from datetime import date, datetime, timedelta
import numpy as np
import requests

from data_loaders import load_source_series


# Schwellenwert aus Nespoli et al. 2019 (Abschnitt 2.3):
# Tagesmittel der globalen Einstrahlung >= 150 W/m² → sunny
SUNNY_THRESHOLD = 150.0  # W/m²

# Repräsentativer Standort für DE_LU (Mitteldeutschland)
_DWD_LAT = 51.0
_DWD_LON = 10.0

# Skalierung: unterhalb _GHI_MIN wird kein Skalierungsfaktor angewendet
# (Nachtstunden / Schwachlicht, um Division durch ~0 zu vermeiden)
_GHI_MIN   = 10.0  # W/m²
_SCALE_MAX  = 3.0  # maximaler Skalierungsfaktor (dämpft Ausreißer)


def _fetch_dwd_hourly_irradiance(query_date: date) -> np.ndarray | None:
    """
    Ruft stündliche GHI-Werte (W/m²) für query_date via brightsky.dev ab.

    brightsky liefert DWD-Daten:
    - Vergangene Tage:  DWD SYNOP-Beobachtungen
    - Zukünftige Tage: DWD MOSMIX-Prognose

    Das Feld 'solar' ist der Stundensummenwert in Wh/m².
    Da 1 Wh/m² pro Stunde = 1 W/m² mittlere Leistung, sind die Werte
    direkt als W/m² je Zeitschritt interpretierbar.

    Rückgabe: 24-Element-Array (W/m²) oder None bei Fehler.
    Bei DST-Tagen (23/25 h) wird auf 24 Werte normiert.
    """
    url = (
        f"https://api.brightsky.dev/weather"
        f"?lat={_DWD_LAT}&lon={_DWD_LON}"
        f"&date={query_date.isoformat()}"
        f"&tz=Europe/Berlin"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        solar = np.array(
            [
                float(e["solar"]) if e.get("solar") is not None else 0.0
                for e in resp.json().get("weather", [])
            ],
            dtype=float,
        )
        if solar.size == 0:
            return None
        if solar.size >= 24:
            return solar[:24]
        return np.pad(solar, (0, 24 - solar.size), mode="edge")
    except Exception as e:
        print(f"[DWD] Fehler fuer {query_date}: {e}")
        return None


def _resample_to_steps(hourly_24: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Resampled ein 24-Element-Stunden-Array auf n_steps per Nearest-Neighbor.
    Funktioniert für beliebige Auflösungen (15 min → 96, 30 min → 48, stündlich → 24).
    """
    if n_steps == 24:
        return hourly_24
    indices = np.clip((np.arange(n_steps) * 24 / n_steps).astype(int), 0, 23)
    return hourly_24[indices]


def _classify_target_day_fallback(historical_data: list[np.ndarray]) -> str:
    """Fallback ohne DWD: Mehrheitsvoting der letzten 3 Tage mit relativem PV-Median-Schwellenwert."""
    pv_threshold = float(np.median([float(np.mean(d)) for d in historical_data]))
    recent = historical_data[:3]
    sunny_votes = sum(1 for d in recent if float(np.mean(d)) >= pv_threshold)
    result = "sunny" if sunny_votes >= len(recent) - sunny_votes else "cloudy"
    print(f"[Fallback] Mehrheitsvoting letzte {len(recent)} Tage → '{result}'")
    return result


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
    Cluster-basierter Forecast nach Nespoli et al. 2019, erweitert um DWD-Einstrahlungsdaten.

    Ablauf
    ------
    1.  Lade historische PV-Daten der letzten 7 Tage (SMARD / ENTSO-E).
    2.  Rufe stündliche DWD-GHI für alle 7 historischen Tage + Ziel-Tag ab (je 1 API-Call).
    3.  Klassifiziere historische Tage per DWD-Beobachtung (150 W/m²-Schwelle).
        Fallback: relativer PV-Median-Schwellenwert.
    4.  Klassifiziere Ziel-Tag per DWD MOSMIX-Prognose (150 W/m²-Schwelle).
        Fallback: Mehrheitsvoting der letzten 3 Tage.
    5.  Bilde Cluster-Mittelwert (PV) nur aus Tagen des gleichen Cluster-Labels.
    6.  Skaliere den Mittelwert-Forecast per Zeitschritt:
            scale[t] = GHI_Ziel[t] / GHI_Cluster_Mittel[t]
        Nur tagsüber (GHI_Cluster_Mittel >= 10 W/m²), gecappt bei 3×.
    7.  Nachtstunden: Zeitschritte unter 1 % des Tagespeaks → Mean und Std auf 0.
    8.  Befülle Payload (Point / Quantil / Ensemble).
    """
    n_steps = len(payload["values"])

    # ------------------------------------------------------------------
    # Schritt 1: Historische PV-Daten laden (letzte 7 Tage)
    # ------------------------------------------------------------------
    historical_data: list[np.ndarray] = []
    historical_dates: list[date] = []

    for days_back in range(1, 8):
        past_date = target_date - timedelta(days=days_back)
        try:
            series = load_source_series(
                data_source=data_source,
                challenge_context=challenge_context,
                delivery_date=past_date,
                entsoe_api_key=entsoe_api_key,
            )
            values = series.values.astype(float)
            if len(values) >= n_steps:
                values = values[:n_steps]
            else:
                values = np.pad(values, (0, n_steps - len(values)), mode="edge")
            historical_data.append(values)
            historical_dates.append(past_date)
        except Exception as e:
            print(f"[!] Konnte Daten fuer {past_date} nicht laden: {e}")

    if not historical_data:
        print("[!] Keine historischen Daten verfuegbar, nutze Baseline.")
        return payload

    # ------------------------------------------------------------------
    # Schritt 2: Stündliche DWD-GHI für alle Tage abrufen (1 Call pro Tag)
    # ------------------------------------------------------------------
    all_dates = historical_dates + [target_date]
    ghi_by_date: dict[date, np.ndarray | None] = {
        d: _fetch_dwd_hourly_irradiance(d) for d in all_dates
    }

    # ------------------------------------------------------------------
    # Schritt 3: Historische Tage klassifizieren
    # ------------------------------------------------------------------
    pv_threshold = float(np.median([float(np.mean(d)) for d in historical_data]))
    labels: list[str] = []

    for d, pv_values in zip(historical_dates, historical_data):
        ghi = ghi_by_date[d]
        if ghi is not None:
            daily_mean = float(np.mean(ghi))
            label = "sunny" if daily_mean >= SUNNY_THRESHOLD else "cloudy"
            print(f"[DWD]  {d}  →  {daily_mean:.1f} W/m²  →  '{label}'")
        else:
            label = "sunny" if float(np.mean(pv_values)) >= pv_threshold else "cloudy"
            print(f"[Fallback]  {d}  →  PV-Mittel={np.mean(pv_values):.1f}  →  '{label}'")
        labels.append(label)

    # ------------------------------------------------------------------
    # Schritt 4: Ziel-Tag per DWD MOSMIX-Prognose klassifizieren
    # ------------------------------------------------------------------
    ghi_target = ghi_by_date[target_date]
    if ghi_target is not None:
        daily_mean = float(np.mean(ghi_target))
        target_cluster = "sunny" if daily_mean >= SUNNY_THRESHOLD else "cloudy"
        print(f"[DWD]  Ziel-Tag {target_date}  →  {daily_mean:.1f} W/m²  →  '{target_cluster}'")
    else:
        target_cluster = _classify_target_day_fallback(historical_data)
    print(f"[cluster] Ziel-Tag → '{target_cluster}'")

    # ------------------------------------------------------------------
    # Schritt 5: Cluster-Matrix und PV-Mittelwert berechnen
    # ------------------------------------------------------------------
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == target_cluster]

    if cluster_indices:
        cluster_pv    = [historical_data[i] for i in cluster_indices]
        cluster_dates = [historical_dates[i] for i in cluster_indices]
        print(f"[✓] Cluster '{target_cluster}': {len(cluster_pv)} Tag(e) von {len(historical_data)}.")
    else:
        cluster_pv    = list(historical_data)
        cluster_dates = list(historical_dates)
        print(f"[!] Kein Tag im Cluster '{target_cluster}'. Nutze alle {len(historical_data)} Tage.")

    matrix        = np.array(cluster_pv)
    mean_forecast = np.maximum(np.mean(matrix, axis=0), 0.0)
    std_per_step  = np.std(matrix, axis=0)

    # ------------------------------------------------------------------
    # Schritt 6: GHI-basierter Skalierungsfaktor anwenden
    #
    # scale[t] = GHI_Ziel[t] / GHI_Cluster_Mittel[t]
    #
    # Interpretation: Wenn morgen laut MOSMIX 20 % mehr GHI erwartet wird
    # als im historischen Cluster-Durchschnitt, skalieren wir den
    # PV-Mittelwert ebenfalls um Faktor 1.2.
    # Nachtstunden (GHI_Cluster < _GHI_MIN) werden nicht skaliert (Faktor = 1).
    # Der Faktor wird auf [0, _SCALE_MAX] gecappt.
    # ------------------------------------------------------------------
    ghi_cluster_arrays = [ghi_by_date[d] for d in cluster_dates if ghi_by_date[d] is not None]

    if ghi_target is not None and ghi_cluster_arrays:
        ghi_target_steps = _resample_to_steps(ghi_target, n_steps)
        ghi_hist_mean    = np.mean(np.array(ghi_cluster_arrays), axis=0)
        ghi_hist_steps   = _resample_to_steps(ghi_hist_mean, n_steps)

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_scale = np.where(
                ghi_hist_steps >= _GHI_MIN,
                ghi_target_steps / ghi_hist_steps,
                1.0,
            )
        scale = np.clip(raw_scale, 0.0, _SCALE_MAX)

        daytime    = ghi_hist_steps >= _GHI_MIN
        mean_scale = float(np.mean(scale[daytime])) if daytime.any() else 1.0
        print(f"[✓] GHI-Skalierung: mittlerer Tagesfaktor = {mean_scale:.2f}")

        mean_forecast = np.maximum(mean_forecast * scale, 0.0)
        std_per_step  = std_per_step * scale
    else:
        print("[!] GHI-Skalierung nicht moeglich (fehlende DWD-Daten). Unskalierten Forecast verwenden.")

    # ------------------------------------------------------------------
    # Schritt 7: Nachtstunden auf 0 setzen
    # Zeitschritte unter 1 % des Tagespeaks → Mean und Std hart auf 0
    # ------------------------------------------------------------------
    peak = float(mean_forecast.max()) if mean_forecast.max() > 0 else 1.0
    night_mask = mean_forecast < 0.01 * peak
    mean_forecast[night_mask] = 0.0
    std_per_step[night_mask]  = 0.0

    print(f"[✓] Forecast (erste 5 Werte): {mean_forecast[:5]}")

    # ------------------------------------------------------------------
    # Schritt 8: Payload befüllen (Point / Quantil / Ensemble)
    # ------------------------------------------------------------------
    for index, original_value in enumerate(payload["values"]):
        fval = float(mean_forecast[index]) if index < len(mean_forecast) else 0.0
        std  = float(std_per_step[index])  if index < len(std_per_step)  else 1.0

        if isinstance(original_value, (int, float)):
            payload["values"][index] = fval

        elif isinstance(original_value, list) and len(original_value) == 5:
            payload["values"][index] = [
                float(max(0, fval - 2.0 * std)),   # Q10
                float(max(0, fval - 0.7 * std)),   # Q25
                float(fval),                        # Q50
                float(fval + 0.7 * std),            # Q75
                float(fval + 2.0 * std),            # Q90
            ]

        elif isinstance(original_value, list) and len(original_value) == 100:
            np.random.seed(index)
            ensemble = np.random.normal(loc=fval, scale=max(std, 0.01), size=100)
            ensemble = np.maximum(ensemble, 0)
            payload["values"][index] = [float(v) for v in ensemble]

        else:
            if isinstance(original_value, list):
                payload["values"][index] = [float(v) for v in original_value]
            else:
                payload["values"][index] = float(original_value)

    return payload
