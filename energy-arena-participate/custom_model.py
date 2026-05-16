"""
Eigenes Solar Forecast Modell fuer DE_LU.
Strategie: Durchschnitt der letzten 7 Tage pro Stunde als Forecast.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

from data_loaders import load_source_series


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
    n_steps = len(payload["values"])
    historical_data = []

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
                values = np.pad(values, (0, n_steps - len(values)), mode='edge')
            historical_data.append(values)
        except Exception as e:
            print(f"[!] Konnte Daten fuer {past_date} nicht laden: {e}")

    if not historical_data:
        print("[!] Keine historischen Daten verfuegbar, nutze Baseline.")
        return payload

    matrix = np.array(historical_data)
    mean_forecast = np.maximum(np.mean(matrix, axis=0), 0)

    print(f"[✓] Forecast aus {len(historical_data)} Tagen. Erste 5: {mean_forecast[:5]}")

    for index, original_value in enumerate(payload["values"]):
        forecast_value = float(mean_forecast[index]) if index < len(mean_forecast) else 0.0
        std = float(np.std(matrix[:, index])) if index < matrix.shape[1] else 1.0

        if isinstance(original_value, (int, float)):
            payload["values"][index] = forecast_value
        elif isinstance(original_value, list) and len(original_value) == 5:
            payload["values"][index] = [
                float(max(0, forecast_value - 2.0 * std)),
                float(max(0, forecast_value - 0.7 * std)),
                float(forecast_value),
                float(forecast_value + 0.7 * std),
                float(forecast_value + 2.0 * std),
            ]
        elif isinstance(original_value, list) and len(original_value) == 100:
            np.random.seed(index)
            ensemble = np.random.normal(loc=forecast_value, scale=max(std, 0.01), size=100)
            ensemble = np.maximum(ensemble, 0)
            payload["values"][index] = [float(v) for v in ensemble]
        else:
            if isinstance(original_value, list):
                payload["values"][index] = [float(v) for v in original_value]
            else:
                payload["values"][index] = float(original_value)

    return payload