# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
tidy3d-compatible API for Khronos.jl GPU FDTD solver.

Usage:
    import khronos.tidy3d as td
    import khronos.web as web

    sim = td.Simulation(
        size=(2e-6, 2e-6, 6e-6),
        structures=[td.Structure(geometry=td.Box(...), medium=td.Medium(...))],
        sources=[td.PlaneWave(source_time=td.GaussianPulse(...), ...)],
        monitors=[td.FluxMonitor(center=..., size=..., freqs=..., name="flux")],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
        run_time=1e-12,
    )
    data = web.run(sim)
    flux = data["flux"].flux
"""

# Constants
from .constants import C_0, EPSILON_0, MU_0, ETA_0, HBAR, K_B, Q_e, inf

# Geometry
from .geometry import Box, Sphere, Cylinder

# Medium / Materials
from .medium import Medium, Lorentz, Drude, Sellmeier, PEC, PECMedium

# Structure
from .structure import Structure

# Source time profiles
from .source import GaussianPulse, ContinuousWave

# Sources
from .source import (
    PlaneWave, PointDipole, UniformCurrentSource,
    ModeSource, GaussianBeam, TFSF,
)

# Monitors
from .monitor import (
    FieldMonitor, FluxMonitor, ModeMonitor,
    FieldTimeMonitor, DiffractionMonitor,
    FieldProjectionAngleMonitor, ModeSpec,
)

# Boundaries
from .boundary import (
    BoundarySpec, Boundary, PML, StablePML, Absorber,
    Periodic, BlochBoundary, PECBoundary, PMCBoundary,
)

# Grid
from .grid import GridSpec, AutoGrid, UniformGrid

# Simulation
from .simulation import Simulation

# Data
from .data import SimulationData, FieldData, FluxData, ModeData

__all__ = [
    # Constants
    "C_0", "EPSILON_0", "MU_0", "ETA_0", "HBAR", "K_B", "Q_e", "inf",
    # Geometry
    "Box", "Sphere", "Cylinder",
    # Materials
    "Medium", "Lorentz", "Drude", "Sellmeier", "PEC", "PECMedium",
    # Structure
    "Structure",
    # Source time
    "GaussianPulse", "ContinuousWave",
    # Sources
    "PlaneWave", "PointDipole", "UniformCurrentSource",
    "ModeSource", "GaussianBeam", "TFSF",
    # Monitors
    "FieldMonitor", "FluxMonitor", "ModeMonitor",
    "FieldTimeMonitor", "DiffractionMonitor",
    "FieldProjectionAngleMonitor", "ModeSpec",
    # Boundaries
    "BoundarySpec", "Boundary", "PML", "StablePML", "Absorber",
    "Periodic", "BlochBoundary", "PECBoundary", "PMCBoundary",
    # Grid
    "GridSpec", "AutoGrid", "UniformGrid",
    # Simulation
    "Simulation",
    # Data
    "SimulationData", "FieldData", "FluxData", "ModeData",
]
