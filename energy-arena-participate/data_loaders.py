"""
Helper functions for loading historical source data for starter models.

These functions are intentionally separated from the command-line entry points
so students can reuse them directly inside custom_model.py.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from _starter_core import (
    _fetch_source_series,
    fetch_entsoe_series,
    fetch_smard_series,
)


def load_smard_series(*, challenge_context: Any, delivery_date: date) -> pd.Series:
    """Load one SMARD source series for the selected challenge context."""
    return fetch_smard_series(
        context=challenge_context,
        delivery_date=delivery_date,
    )


def load_entsoe_series(
    *,
    entsoe_api_key: str,
    challenge_context: Any,
    delivery_date: date,
) -> pd.Series:
    """Load one ENTSO-E source series for the selected challenge context."""
    return fetch_entsoe_series(
        api_key=entsoe_api_key,
        context=challenge_context,
        delivery_date=delivery_date,
    )


def load_source_series(
    *,
    data_source: str,
    challenge_context: Any,
    delivery_date: date,
    entsoe_api_key: str = "",
) -> pd.Series:
    """Load one source series from either SMARD or ENTSO-E."""
    return _fetch_source_series(
        data_source=data_source,
        context=challenge_context,
        delivery_date=delivery_date,
        entsoe_api_key=entsoe_api_key,
    )
