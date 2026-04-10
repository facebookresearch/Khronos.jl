# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Boundary conditions matching the meep Python API."""

from .constants import ALL, X, Y, Z, Low, High
from ._units import meep_to_khronos_length


class PML:
    """Perfectly matched layer.

    Parameters
    ----------
    thickness : float
        PML thickness in meep length units.
    direction : int
        Direction (mp.X, mp.Y, mp.Z, or mp.ALL). Default ALL.
    side : int
        Side (mp.Low, mp.High, or mp.ALL). Default ALL.
    R_asymptotic : float
        Asymptotic reflectivity.
    mean_stretch : float
        Coordinate stretching factor.
    pml_profile : callable
        PML profile function.
    """

    def __init__(self, thickness, direction=ALL, side=ALL,
                 R_asymptotic=1e-15, mean_stretch=1.0,
                 pml_profile=None):
        self.thickness = thickness
        self.direction = direction
        self.side = side
        self.R_asymptotic = R_asymptotic
        self.mean_stretch = mean_stretch
        self.pml_profile = pml_profile


class Absorber(PML):
    """Adiabatic absorber (more stable than PML for some cases).

    Same parameters as PML.
    """
    pass


class Mirror:
    """Mirror symmetry."""

    def __init__(self, direction=X, phase=1.0):
        self.direction = direction
        self.phase = phase


class Rotate2:
    """2-fold rotational symmetry."""

    def __init__(self, direction=Z, phase=1.0):
        self.direction = direction
        self.phase = phase


class Rotate4:
    """4-fold rotational symmetry."""

    def __init__(self, direction=Z, phase=1.0):
        self.direction = direction
        self.phase = phase


def _build_pml_params(boundary_layers, resolution, cell_size):
    """Convert a list of meep PML/Absorber objects to Khronos boundary params.

    Returns
    -------
    boundaries : list of [float, float] per axis
        PML thickness per side in Khronos units (μm).
    boundary_conditions : list of [bc_minus, bc_plus] per axis
        Julia boundary condition objects.
    absorbers_list : list of absorber info per axis (or None)
    pml_thicknesses : same as boundaries (for cell size expansion)
    """
    # Initialize: no PML on any axis
    pml_thickness = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # [axis][side]
    is_absorber = [[False, False], [False, False], [False, False]]

    for pml in boundary_layers:
        thickness_um = meep_to_khronos_length(pml.thickness)
        absorber = isinstance(pml, Absorber)

        if pml.direction == ALL:
            axes = [0, 1, 2]
        else:
            axes = [pml.direction]

        for ax in axes:
            if pml.side == ALL:
                sides = [0, 1]
            else:
                sides = [pml.side]

            for side in sides:
                pml_thickness[ax][side] = thickness_um
                is_absorber[ax][side] = absorber

    return pml_thickness, is_absorber
