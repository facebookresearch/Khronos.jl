# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Adjoint optimization module, compatible with meep.adjoint (mpa).

Provides DesignRegion, OptimizationProblem, objective quantity classes,
and topology optimization filters/projections.
"""

from .optimization_problem import OptimizationProblem
from .design_region import DesignRegion
from .objective import EigenmodeCoefficient, FourierFields, Near2FarFields, LDOS
from .filters import (
    conic_filter,
    cylindrical_filter,
    gaussian_filter,
    tanh_projection,
    smoothed_projection,
    heaviside_projection,
    constraint_solid,
    constraint_void,
    indicator_solid,
    indicator_void,
    gray_indicator,
    get_conic_radius_from_eta_e,
    get_eta_from_conic,
)

__all__ = [
    "OptimizationProblem",
    "DesignRegion",
    "EigenmodeCoefficient",
    "FourierFields",
    "Near2FarFields",
    "LDOS",
    "conic_filter",
    "cylindrical_filter",
    "gaussian_filter",
    "tanh_projection",
    "smoothed_projection",
    "heaviside_projection",
    "constraint_solid",
    "constraint_void",
    "indicator_solid",
    "indicator_void",
    "gray_indicator",
    "get_conic_radius_from_eta_e",
    "get_eta_from_conic",
]
