"""CSV and DataFrame helpers for LLX2026 batch prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .model import predict, prediction_interval


class BatchFormatError(ValueError):
    """Raised when a batch table does not contain the required inputs."""


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "drying_time": ("dt", "t-t0", "drying_time", "drying time"),
    "curing_age": ("t0", "curing_age", "curing age"),
    "relative_humidity": ("rh", "h", "relative_humidity", "relative humidity"),
    "volume_to_surface": ("vtos", "v/s", "vs", "volume_to_surface"),
    "water_cement_ratio": ("wc", "w/c", "w_c", "water_cement_ratio"),
    "aggregate_content": (
        "agg_total",
        "agg",
        "aggregate",
        "aggregate_content",
        "total aggregate content",
    ),
}

OUTPUT_COLUMNS = (
    "Predicted_Shrinkage_ue",
    "PI90_Lower_ue",
    "PI90_Upper_ue",
    "PI95_Lower_ue",
    "PI95_Upper_ue",
)


def _column_lookup(columns: Iterable[object]) -> dict[str, object]:
    return {str(column).strip().lower(): column for column in columns}


def resolve_columns(frame: pd.DataFrame) -> dict[str, object]:
    """Resolve supported CSV headings to canonical model input names."""

    available = _column_lookup(frame.columns)
    resolved: dict[str, object] = {}
    missing: list[str] = []

    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((available[name] for name in aliases if name in available), None)
        if match is None:
            missing.append(f"{canonical} ({'/'.join(aliases[:2])})")
        else:
            resolved[canonical] = match

    if missing:
        raise BatchFormatError("Missing required columns: " + ", ".join(missing))
    return resolved


def predict_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with LLX2026 predictions appended."""

    if frame.empty:
        raise BatchFormatError("The input table is empty.")

    columns = resolve_columns(frame)
    numeric = {
        name: pd.to_numeric(frame[source], errors="coerce")
        for name, source in columns.items()
    }
    invalid = [name for name, values in numeric.items() if values.isna().any()]
    if invalid:
        raise BatchFormatError(
            "Missing or non-numeric values found in: " + ", ".join(invalid)
        )

    values = predict(
        drying_time=numeric["drying_time"].to_numpy(),
        curing_age=numeric["curing_age"].to_numpy(),
        relative_humidity=numeric["relative_humidity"].to_numpy(),
        volume_to_surface=numeric["volume_to_surface"].to_numpy(),
        water_cement_ratio=numeric["water_cement_ratio"].to_numpy(),
        aggregate_content=numeric["aggregate_content"].to_numpy(),
    )
    lo90, hi90 = prediction_interval(values, numeric["drying_time"].to_numpy(), 0.90)
    lo95, hi95 = prediction_interval(values, numeric["drying_time"].to_numpy(), 0.95)

    result = frame.copy()
    result[OUTPUT_COLUMNS[0]] = values
    result[OUTPUT_COLUMNS[1]] = lo90
    result[OUTPUT_COLUMNS[2]] = hi90
    result[OUTPUT_COLUMNS[3]] = lo95
    result[OUTPUT_COLUMNS[4]] = hi95
    return result


def predict_csv(source: str | Path, destination: str | Path | None = None) -> pd.DataFrame:
    """Read a CSV, calculate predictions, and optionally write the result."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    result = predict_dataframe(pd.read_csv(source_path))
    if destination is not None:
        result.to_csv(Path(destination), index=False)
    return result
