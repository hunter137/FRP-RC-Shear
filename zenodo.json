"""Numerical implementation of the LLX2026 shrinkage equation.

This module has no GUI or file-system dependencies.  It is the stable public
interface for scripts, notebooks, and third-party applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ModelParameters:
    """Coefficients and reference values used by the LLX2026 formulation."""

    theta: tuple[float, ...] = (
        1062.3529,
        0.319716,
        1.500000,
        61.51684,
        1.349781,
        0.845264,
        0.361052,
        0.115219,
        0.389485,
    )
    volume_surface_reference: float = 22.7273
    aggregate_volume_reference: float = 0.7015
    aggregate_density: float = 2650.0

    def __post_init__(self) -> None:
        if len(self.theta) != 9:
            raise ValueError("LLX2026 requires exactly nine coefficients.")


DEFAULT_PARAMETERS = ModelParameters()
THETA = np.asarray(DEFAULT_PARAMETERS.theta, dtype=float)
VS_REF = DEFAULT_PARAMETERS.volume_surface_reference
VA_REF = DEFAULT_PARAMETERS.aggregate_volume_reference
RHO_AGG = DEFAULT_PARAMETERS.aggregate_density

# Upper drying-age limits followed by 90% and 95% additive half-widths (με).
# These values come from experiment-grouped out-of-fold residuals.
PI_HALF_WIDTHS = np.array(
    [
        [7.0, 61.1152, 72.3735],
        [28.0, 88.2015, 112.9973],
        [180.0, 108.6639, 144.0709],
        [1000.0, 102.5550, 117.9668],
        [np.inf, 89.8207, 109.9725],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ShrinkageInputs:
    """One prediction request, expressed in the units used by the GUI."""

    drying_time: float
    curing_age: float
    relative_humidity: float
    volume_to_surface: float
    water_cement_ratio: float
    aggregate_content: float


@dataclass(frozen=True)
class Prediction:
    """Point prediction and empirical prediction limits, in microstrain."""

    value: float
    pi90_lower: float
    pi90_upper: float
    pi95_lower: float
    pi95_upper: float

    def as_dict(self) -> dict[str, float]:
        return {
            "prediction": self.value,
            "pi90_lower": self.pi90_lower,
            "pi90_upper": self.pi90_upper,
            "pi95_lower": self.pi95_lower,
            "pi95_upper": self.pi95_upper,
        }


def _positive(name: str, value: NDArray[np.float64]) -> None:
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must contain finite positive values.")


def aggregate_volume_fraction(
    aggregate_content: ArrayLike,
    density: ArrayLike = RHO_AGG,
) -> NDArray[np.float64]:
    """Convert aggregate content (kg/m³) to a nominal volume fraction.

    No clipping is performed: an impossible input is reported instead of being
    silently changed.
    """

    aggregate = np.asarray(aggregate_content, dtype=float)
    rho = np.asarray(density, dtype=float)
    _positive("aggregate_content", aggregate)
    _positive("density", rho)
    volume_fraction = aggregate / rho
    if np.any(volume_fraction >= 1):
        raise ValueError("aggregate_content / density must be smaller than 1.")
    return volume_fraction


def predict(
    drying_time: ArrayLike,
    curing_age: ArrayLike,
    relative_humidity: ArrayLike,
    volume_to_surface: ArrayLike,
    water_cement_ratio: ArrayLike,
    *,
    aggregate_content: ArrayLike | None = None,
    aggregate_volume_fraction_value: ArrayLike | None = None,
    parameters: ModelParameters = DEFAULT_PARAMETERS,
) -> NDArray[np.float64]:
    """Predict positive drying-shrinkage magnitude in microstrain.

    Supply either ``aggregate_content`` or a measured
    ``aggregate_volume_fraction_value``.  Arrays are broadcast in the usual
    NumPy manner, which makes this function suitable for batch calculations.
    """

    dt = np.asarray(drying_time, dtype=float)
    t0 = np.asarray(curing_age, dtype=float)
    rh = np.asarray(relative_humidity, dtype=float)
    vs = np.asarray(volume_to_surface, dtype=float)
    wc = np.asarray(water_cement_ratio, dtype=float)

    _positive("drying_time", dt)
    _positive("curing_age", t0)
    _positive("volume_to_surface", vs)
    _positive("water_cement_ratio", wc)
    if np.any(~np.isfinite(rh)) or np.any((rh < 0) | (rh > 100)):
        raise ValueError("relative_humidity must be between 0 and 100 percent.")

    if (aggregate_content is None) == (aggregate_volume_fraction_value is None):
        raise ValueError(
            "Supply exactly one of aggregate_content or "
            "aggregate_volume_fraction_value."
        )

    if aggregate_volume_fraction_value is None:
        va = aggregate_volume_fraction(aggregate_content, parameters.aggregate_density)
    else:
        va = np.asarray(aggregate_volume_fraction_value, dtype=float)
        if np.any(~np.isfinite(va)) or np.any((va <= 0) | (va >= 1)):
            raise ValueError("aggregate_volume_fraction_value must lie between 0 and 1.")

    th = np.asarray(parameters.theta, dtype=float)
    humidity = rh / 100.0
    amplitude = th[0] * (wc / 0.5) ** th[1]
    humidity_factor = (1 - humidity**th[2]) / (1 - 0.5**th[2])
    tau = th[3] * (vs / parameters.volume_surface_reference) ** th[4]
    time_factor = (dt / (dt + tau)) ** th[5]
    early_age_factor = 1 + th[6] * np.exp(-dt / 50.0)
    curing_factor = 1 - th[7] * np.log(t0 / 7.0)
    aggregate_factor = (
        (1 - va) / (1 - parameters.aggregate_volume_reference)
    ) ** th[8]
    return amplitude * humidity_factor * time_factor * early_age_factor * curing_factor * aggregate_factor


def prediction_interval(
    mean_prediction: ArrayLike,
    drying_time: ArrayLike,
    level: float = 0.90,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return an empirical, drying-age-stratified prediction interval."""

    if level not in (0.90, 0.95):
        raise ValueError("level must be either 0.90 or 0.95.")

    mean = np.asarray(mean_prediction, dtype=float)
    dt = np.asarray(drying_time, dtype=float)
    _positive("drying_time", dt)
    mean, dt = np.broadcast_arrays(mean, dt)
    if np.any(~np.isfinite(mean)):
        raise ValueError("mean_prediction must contain finite values.")

    age_index = np.searchsorted(PI_HALF_WIDTHS[:-1, 0], dt, side="left")
    width_column = 1 if level == 0.90 else 2
    half_width = PI_HALF_WIDTHS[age_index, width_column]
    return np.maximum(0.0, mean - half_width), mean + half_width


def evaluate(inputs: ShrinkageInputs, *, parameters: ModelParameters = DEFAULT_PARAMETERS) -> Prediction:
    """Evaluate one typed input record and return a typed result."""

    value = float(
        predict(
            drying_time=inputs.drying_time,
            curing_age=inputs.curing_age,
            relative_humidity=inputs.relative_humidity,
            volume_to_surface=inputs.volume_to_surface,
            water_cement_ratio=inputs.water_cement_ratio,
            aggregate_content=inputs.aggregate_content,
            parameters=parameters,
        )
    )
    lo90, hi90 = prediction_interval(value, inputs.drying_time, 0.90)
    lo95, hi95 = prediction_interval(value, inputs.drying_time, 0.95)
    return Prediction(value, float(lo90), float(hi90), float(lo95), float(hi95))


def predict_with_intervals(**kwargs: float) -> dict[str, float]:
    """Compatibility wrapper returning one result as a plain dictionary."""

    parameters = kwargs.pop("parameters", DEFAULT_PARAMETERS)
    value = float(predict(parameters=parameters, **kwargs))
    dt = float(kwargs["drying_time"])
    lo90, hi90 = prediction_interval(value, dt, 0.90)
    lo95, hi95 = prediction_interval(value, dt, 0.95)
    return Prediction(value, float(lo90), float(hi90), float(lo95), float(hi95)).as_dict()


def inputs_from_mapping(values: Mapping[str, float]) -> ShrinkageInputs:
    """Build :class:`ShrinkageInputs` from a mapping with public API names."""

    return ShrinkageInputs(
        drying_time=float(values["drying_time"]),
        curing_age=float(values["curing_age"]),
        relative_humidity=float(values["relative_humidity"]),
        volume_to_surface=float(values["volume_to_surface"]),
        water_cement_ratio=float(values["water_cement_ratio"]),
        aggregate_content=float(values["aggregate_content"]),
    )
