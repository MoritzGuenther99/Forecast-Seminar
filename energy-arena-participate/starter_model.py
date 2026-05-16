"""
Starter-model helpers for building the baseline payload.

Students can use this as the simplest reference implementation before replacing
parts of the logic in custom_model.py.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from _starter_core import TARGET_BASELINES, build_payload_from_source


def build_starter_payload(
    *,
    target_date: date,
    challenge_context: Any,
    data_source: str,
    entsoe_api_key: str,
) -> dict:
    """
    Build the default starter payload for one challenge and target day.

    This uses the built-in historical baseline logic from the repository.
    """
    return build_payload_from_source(
        target_date=target_date,
        context=challenge_context,
        data_source=data_source,
        entsoe_api_key=entsoe_api_key,
    )
