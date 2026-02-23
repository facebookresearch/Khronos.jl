# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Grid specification: GridSpec, AutoGrid, UniformGrid.

Since Khronos currently supports only uniform grids, AutoGrid computes the
optimal uniform resolution based on the worst-case material refractive index.
"""

import math
from .constants import C_0


class AutoGrid:
    """Automatic grid sizing based on wavelength and material properties."""

    def __init__(self, min_steps_per_wvl=10, max_scale=1.4, dl_min=0):
        self.min_steps_per_wvl = min_steps_per_wvl
        self.max_scale = max_scale
        self.dl_min = dl_min


class UniformGrid:
    """Uniform grid with explicit cell size."""

    def __init__(self, dl):
        self.dl = dl  # cell size in meters


class GridSpec:
    """Grid specification for the simulation domain."""

    def __init__(self, grid_x=None, grid_y=None, grid_z=None,
                 wavelength=None, override_structures=None, uniform_dl=None):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.wavelength = wavelength
        self.override_structures = override_structures or []
        self.uniform_dl = uniform_dl

    @classmethod
    def auto(cls, min_steps_per_wvl=10, wavelength=None, max_scale=1.4,
             dl_min=0, **kwargs):
        """Create auto grid spec that determines resolution from materials."""
        ag = AutoGrid(min_steps_per_wvl=min_steps_per_wvl,
                      max_scale=max_scale, dl_min=dl_min)
        return cls(grid_x=ag, grid_y=ag, grid_z=ag, wavelength=wavelength)

    @classmethod
    def uniform(cls, dl):
        """Create uniform grid with explicit cell size (meters)."""
        return cls(uniform_dl=dl)

    def make_resolution(self, structures, sources):
        """Compute the uniform Khronos resolution (grid pts per μm).

        For AutoGrid: resolution = min_steps_per_wvl * max_n / wavelength_um
        For UniformGrid: resolution = 1 / dl_um
        """
        if self.uniform_dl is not None:
            dl_um = self.uniform_dl * 1e6  # meters → μm
            return 1.0 / dl_um

        # Determine wavelength
        wavelength = self.wavelength
        if wavelength is None and sources:
            # Get from source center frequency
            freq0 = None
            for src in sources:
                if hasattr(src, 'source_time') and hasattr(src.source_time, 'freq0'):
                    freq0 = src.source_time.freq0
                    break
            if freq0 is not None and freq0 > 0:
                wavelength = C_0 / freq0  # meters

        if wavelength is None:
            wavelength = 1.0e-6  # default 1 μm

        wavelength_um = wavelength * 1e6  # meters → μm

        # Find maximum refractive index across all structures
        max_n = 1.0
        for s in structures:
            if hasattr(s, 'medium') and hasattr(s.medium, 'refractive_index'):
                n = s.medium.refractive_index(wavelength)
                max_n = max(max_n, abs(n))

        # Get min_steps_per_wvl from the grid spec
        min_steps = 10  # default
        for grid_1d in [self.grid_x, self.grid_y, self.grid_z]:
            if isinstance(grid_1d, AutoGrid):
                min_steps = max(min_steps, grid_1d.min_steps_per_wvl)

        # Resolution: enough points per wavelength in the highest-index material
        # λ_material = λ_0 / n → need min_steps points per λ_material
        # dl = λ_material / min_steps = λ_0 / (n * min_steps)
        # resolution = 1/dl = n * min_steps / λ_0
        resolution_um = max_n * min_steps / wavelength_um

        return math.ceil(resolution_um)
