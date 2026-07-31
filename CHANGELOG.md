"""Reusable plotting functions for LLX2026 predictions."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .model import ShrinkageInputs, predict, prediction_interval


def development_figure(
    inputs: ShrinkageInputs,
    *,
    maximum_time: float = 365.0,
    points: int = 360,
) -> Figure:
    """Create a shrinkage-development plot without opening a GUI window."""

    if maximum_time <= 1:
        raise ValueError("maximum_time must be greater than one day.")
    if points < 20:
        raise ValueError("points must be at least 20.")

    drying_time = np.linspace(1.0, float(maximum_time), int(points))
    mean = predict(
        drying_time=drying_time,
        curing_age=inputs.curing_age,
        relative_humidity=inputs.relative_humidity,
        volume_to_surface=inputs.volume_to_surface,
        water_cement_ratio=inputs.water_cement_ratio,
        aggregate_content=inputs.aggregate_content,
    )
    lo90, hi90 = prediction_interval(mean, drying_time, 0.90)
    lo95, hi95 = prediction_interval(mean, drying_time, 0.95)

    figure = Figure(figsize=(7.2, 5.0), dpi=100)
    axis = figure.add_subplot(111)
    axis.fill_between(drying_time, lo95, hi95, color="#3498DB", alpha=0.15, label="95% PI")
    axis.fill_between(drying_time, lo90, hi90, color="#3498DB", alpha=0.32, label="90% PI")
    axis.plot(drying_time, mean, color="#1F4E79", linewidth=2.0, label="Predicted mean")

    if 1 <= inputs.drying_time <= maximum_time:
        marker = float(
            predict(
                drying_time=inputs.drying_time,
                curing_age=inputs.curing_age,
                relative_humidity=inputs.relative_humidity,
                volume_to_surface=inputs.volume_to_surface,
                water_cement_ratio=inputs.water_cement_ratio,
                aggregate_content=inputs.aggregate_content,
            )
        )
        marker_lo, marker_hi = prediction_interval(marker, inputs.drying_time, 0.90)
        axis.errorbar(
            [inputs.drying_time],
            [marker],
            yerr=[[marker - float(marker_lo)], [float(marker_hi) - marker]],
            fmt="o",
            color="#E74C3C",
            markersize=6,
            capsize=4,
            linewidth=1.5,
            zorder=5,
            label=f"t = {inputs.drying_time:g} d",
        )
        axis.annotate(
            f"{marker:.0f}\n[{float(marker_lo):.0f}, {float(marker_hi):.0f}]",
            xy=(inputs.drying_time, marker),
            xytext=(8, -30),
            textcoords="offset points",
            fontsize=8,
            color="#E74C3C",
        )

    aggregate_fraction = inputs.aggregate_content / 2650.0
    conditions = (
        f"t₀ = {inputs.curing_age:g} d,  RH = {inputs.relative_humidity:g}%,  "
        f"V/S = {inputs.volume_to_surface:g} mm,  w/c = {inputs.water_cement_ratio:g},  "
        f"aggregate = {inputs.aggregate_content:g} kg/m³  (Vₐ = {aggregate_fraction:.3f})"
    )
    axis.text(
        0.02,
        0.97,
        conditions,
        transform=axis.transAxes,
        fontsize=8,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "#F5F5F5", "edgecolor": "#BDC3C7", "alpha": 0.9},
    )
    axis.set_xlabel("Drying time, t − t₀ (days)", fontsize=11)
    axis.set_ylabel("Predicted drying-shrinkage magnitude (με)", fontsize=11)
    axis.set_title("LLX2026 drying-shrinkage prediction", fontsize=12, fontweight="bold")
    axis.set_xlim(0, maximum_time)
    axis.set_ylim(bottom=0)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="lower right", fontsize=9, framealpha=0.9)
    figure.tight_layout()
    return figure
