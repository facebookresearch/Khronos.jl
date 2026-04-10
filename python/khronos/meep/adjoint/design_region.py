# Copyright (c) Meta Platforms, Inc. and affiliates.
"""DesignRegion for topology optimization."""

from ..geom import Vector3, Volume, MaterialGrid
from .._units import meep_to_khronos_length, meep_to_khronos_freq


class DesignRegion:
    """A region of continuously-varying material for inverse design.

    Parameters
    ----------
    design_parameters : MaterialGrid
        The material grid defining the design parameterization.
    volume : Volume, optional
        Volume of the design region (alternative to center/size).
    size : Vector3, optional
        Size of the design region.
    center : Vector3, optional
        Center of the design region.
    """

    def __init__(self, design_parameters, volume=None, size=None,
                 center=None):
        self.design_parameters = design_parameters

        if volume is not None:
            self.center = volume.center
            self.size = volume.size
        else:
            self.center = center if center is not None else Vector3()
            self.size = size if size is not None else Vector3()
        if not isinstance(self.center, Vector3):
            self.center = Vector3(*self.center)
        if not isinstance(self.size, Vector3):
            self.size = Vector3(*self.size)

        self._khronos_dr = None

    @property
    def num_design_params(self):
        mg = self.design_parameters
        return mg.Nx * mg.Ny * mg.Nz

    def update_design_parameters(self, weights):
        """Update the design weights."""
        self.design_parameters.update_weights(weights)

    def update_beta(self, beta):
        """Update the projection strength."""
        self.design_parameters.beta = beta

    def _to_khronos(self, K):
        """Convert to Khronos DesignRegion."""
        mg = self.design_parameters
        c = [meep_to_khronos_length(x) for x in self.center]
        s = [meep_to_khronos_length(x) for x in self.size]

        eps_min = mg.medium1.epsilon
        eps_max = mg.medium2.epsilon

        self._khronos_dr = K.DesignRegion(
            center=c,
            size=s,
            Nx=mg.Nx,
            Ny=mg.Ny,
            ε_min=float(eps_min),
            ε_max=float(eps_max),
        )
        return self._khronos_dr
