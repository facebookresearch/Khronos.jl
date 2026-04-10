# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Meep-compatible Python API for the Khronos FDTD engine.

Usage::

    import khronos.meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8),
        resolution=10,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1), material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                           component=mp.Ez, center=mp.Vector3(-7))],
    )
    flux = sim.add_flux(0.15, 0.1, 50, mp.FluxRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2)))
    sim.run(until_after_sources=200)
    print(mp.get_fluxes(flux))

This module re-exports all public names so that ``import khronos.meep as mp``
provides the same namespace as ``import meep as mp``.
"""

# ------------------------------------------------------------------- #
# Constants and enums
# ------------------------------------------------------------------- #
from .constants import (
    Ex, Ey, Ez, Hx, Hy, Hz,
    Dx, Dy, Dz, Bx, By, Bz,
    Er, Ep, Hr, Hp, Dr, Dp, Br, Bp,
    Dielectric, Permeability,
    ALL_COMPONENTS,
    X, Y, Z, R, P, NO_DIRECTION,
    Low, High,
    NO_PARITY, EVEN_Z, ODD_Z, EVEN_Y, ODD_Y, TE, TM,
    AUTOMATIC, CYLINDRICAL, ALL,
    inf,
    is_electric, is_magnetic, component_direction,
)

# ------------------------------------------------------------------- #
# Unit control
# ------------------------------------------------------------------- #
from ._units import set_length_scale, get_length_scale

# ------------------------------------------------------------------- #
# Vector3
# ------------------------------------------------------------------- #
from .geom import Vector3

# ------------------------------------------------------------------- #
# Materials
# ------------------------------------------------------------------- #
from .geom import (
    Medium,
    Susceptibility,
    LorentzianSusceptibility,
    DrudeSusceptibility,
    MaterialGrid,
    # Predefined materials
    vacuum, air, metal,
    perfect_electric_conductor,
    perfect_magnetic_conductor,
)

# ------------------------------------------------------------------- #
# Geometry
# ------------------------------------------------------------------- #
from .geom import (
    GeometricObject,
    Block,
    Sphere,
    Cylinder,
    Prism,
    Cone,
    Ellipsoid,
    Wedge,
    Volume,
    FluxRegion,
    ModeRegion,
    Near2FarRegion,
    ForceRegion,
    EnergyRegion,
    Matrix,
    Lattice,
    interpolate,
    # Khronos extensions
    LayerSpec,
    LayerStack,
    import_gdsii,
)

# ------------------------------------------------------------------- #
# Symmetry
# ------------------------------------------------------------------- #
from .geom import (
    Symmetry,
    Mirror,
    Rotate2,
    Rotate4,
)

# ------------------------------------------------------------------- #
# Sources
# ------------------------------------------------------------------- #
from .source import (
    SourceTime,
    GaussianSource,
    ContinuousSource,
    CustomSource,
    Source,
    EigenModeSource,
    GaussianBeamSource,
    GaussianBeam3DSource,
    GaussianBeam2DSource,
    # Khronos extension
    TFSFSource,
)

# ------------------------------------------------------------------- #
# Boundaries
# ------------------------------------------------------------------- #
from .boundaries import (
    PML,
    Absorber,
)

# ------------------------------------------------------------------- #
# Simulation
# ------------------------------------------------------------------- #
from .simulation import (
    Simulation,
    make_output_directory,
    delete_directory,
)

# ------------------------------------------------------------------- #
# DFT objects and data extraction
# ------------------------------------------------------------------- #
from .dft import (
    DftFlux,
    DftFields,
    DftNear2Far,
    DftEnergy,
    DftForce,
    FluxData,
    ForceData,
    EigenmodeData,
    get_fluxes,
    get_flux_freqs,
    get_forces,
    get_force_freqs,
    get_near2far_freqs,
    get_ldos_freqs,
    complexarray,
    # Khronos extensions
    DftDiffraction,
    get_diffraction_efficiencies,
    get_diffraction_freqs,
)

# ------------------------------------------------------------------- #
# Run functions and step functions
# ------------------------------------------------------------------- #
from .run_functions import (
    # Supported stop conditions
    stop_when_fields_decayed,
    stop_when_energy_decayed,
    stop_when_dft_decayed,
    stop_after_walltime,
    stop_on_interrupt,
    # Unsupported step functions (warn but don't crash)
    at_beginning,
    at_end,
    at_every,
    at_time,
    after_time,
    before_time,
    after_sources,
    after_sources_and_time,
    during_sources,
    combine_step_funcs,
    synchronized_magnetic,
    in_volume,
    in_point,
    to_appended,
    with_prefix,
    when_true,
    when_false,
    # Output functions (unsupported)
    output_epsilon,
    output_mu,
    output_efield,
    output_efield_x,
    output_efield_y,
    output_efield_z,
    output_hfield,
    output_hfield_x,
    output_hfield_y,
    output_hfield_z,
    output_dfield,
    output_bfield,
    output_sfield,
    output_poynting,
    output_poynting_x,
    output_poynting_y,
    output_poynting_z,
    output_hpwr,
    output_dpwr,
    output_tot_pwr,
    output_png,
    # Harminv
    Harminv,
    # Other
    DiffractedPlanewave,
    verbosity,
    quiet,
    # Khronos extensions
    run_batch,
    run_batch_concurrent,
    run_batch_multi_gpu,
    compute_LEE,
    compute_incoherent_LEE,
)


# ------------------------------------------------------------------- #
# Visualization
# ------------------------------------------------------------------- #
from .visualization import plot2D, plot_eps, plot_boundaries, plot_sources, plot_monitors, plot_fields


# ------------------------------------------------------------------- #
# Adjoint module is available as khronos.meep.adjoint
# ------------------------------------------------------------------- #
from . import adjoint


# ------------------------------------------------------------------- #
# Module-level aliases for common patterns
# ------------------------------------------------------------------- #

# FreqRange for Medium valid_freq_range
class FreqRange:
    def __init__(self, min=-inf, max=inf):
        self.min = min
        self.max = max


__all__ = [
    # Constants
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "Dx", "Dy", "Dz", "Bx", "By", "Bz",
    "Dielectric", "Permeability", "ALL_COMPONENTS",
    "X", "Y", "Z", "R", "P", "NO_DIRECTION",
    "Low", "High",
    "NO_PARITY", "EVEN_Z", "ODD_Z", "EVEN_Y", "ODD_Y", "TE", "TM",
    "AUTOMATIC", "CYLINDRICAL", "ALL",
    "inf",
    # Units
    "set_length_scale", "get_length_scale",
    # Core types
    "Vector3", "Volume",
    "Medium", "Susceptibility", "LorentzianSusceptibility", "DrudeSusceptibility",
    "MaterialGrid",
    "vacuum", "air", "metal", "perfect_electric_conductor", "perfect_magnetic_conductor",
    # Geometry
    "GeometricObject", "Block", "Sphere", "Cylinder", "Prism",
    "Cone", "Ellipsoid", "Wedge",
    "FluxRegion", "ModeRegion", "Near2FarRegion", "ForceRegion", "EnergyRegion",
    "Matrix", "Lattice", "interpolate",
    # Symmetry
    "Symmetry", "Mirror", "Rotate2", "Rotate4",
    # Sources
    "SourceTime", "GaussianSource", "ContinuousSource", "CustomSource",
    "Source", "EigenModeSource",
    "GaussianBeamSource", "GaussianBeam3DSource", "GaussianBeam2DSource",
    # Boundaries
    "PML", "Absorber",
    # Simulation
    "Simulation", "make_output_directory", "delete_directory",
    # DFT
    "DftFlux", "DftFields", "DftNear2Far", "DftEnergy", "DftForce",
    "FluxData", "ForceData", "EigenmodeData",
    "get_fluxes", "get_flux_freqs", "get_forces", "get_force_freqs",
    "get_near2far_freqs", "get_ldos_freqs", "complexarray",
    # Run functions
    "stop_when_fields_decayed", "stop_when_energy_decayed",
    "stop_when_dft_decayed", "stop_after_walltime", "stop_on_interrupt",
    "at_beginning", "at_end", "at_every", "at_time",
    "after_time", "before_time", "after_sources", "after_sources_and_time",
    "during_sources", "combine_step_funcs", "synchronized_magnetic",
    "in_volume", "in_point", "to_appended", "with_prefix",
    "when_true", "when_false",
    "output_epsilon", "output_mu",
    "output_efield", "output_efield_x", "output_efield_y", "output_efield_z",
    "output_hfield", "output_hfield_x", "output_hfield_y", "output_hfield_z",
    "output_dfield", "output_bfield", "output_sfield",
    "output_poynting", "output_poynting_x", "output_poynting_y", "output_poynting_z",
    "output_hpwr", "output_dpwr", "output_tot_pwr", "output_png",
    "Harminv", "DiffractedPlanewave",
    "verbosity", "quiet",
    "FreqRange",
    # Khronos extensions
    "TFSFSource",
    "LayerSpec", "LayerStack", "import_gdsii",
    "DftDiffraction", "get_diffraction_efficiencies", "get_diffraction_freqs",
    "run_batch", "run_batch_concurrent", "run_batch_multi_gpu",
    "compute_LEE", "compute_incoherent_LEE",
    # Visualization
    "plot2D", "plot_eps", "plot_boundaries", "plot_sources", "plot_monitors", "plot_fields",
    # Adjoint
    "adjoint",
]
