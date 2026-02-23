# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Boundary conditions: BoundarySpec, Boundary, PML, Absorber, Periodic, Bloch."""

from .constants import to_length


class PML:
    """Perfectly matched layer boundary."""

    def __init__(self, num_layers=12):
        self.num_layers = num_layers
        # Approximate thickness: ~1 μm for 12 layers at typical resolution
        self._thickness_per_layer = 0.04e-6  # will be computed from resolution


class StablePML(PML):
    """More stable PML variant."""
    pass


class Absorber:
    """Adiabatic absorber boundary (more stable than PML for curved structures)."""

    def __init__(self, num_layers=40):
        self.num_layers = num_layers


class Periodic:
    """Periodic boundary condition."""
    pass


class BlochBoundary:
    """Bloch-periodic boundary with phase shift."""

    def __init__(self, bloch_vec=0.0):
        self.bloch_vec = bloch_vec  # in rad/m


class PECBoundary:
    """Perfect electric conductor boundary."""
    pass


class PMCBoundary:
    """Perfect magnetic conductor boundary."""
    pass


class Boundary:
    """Pair of boundary conditions for one axis (minus and plus sides)."""

    def __init__(self, plus=None, minus=None):
        self.plus = plus or PML()
        self.minus = minus or PML()

    @classmethod
    def pml(cls, **kwargs):
        return cls(plus=PML(**kwargs), minus=PML(**kwargs))

    @classmethod
    def stable_pml(cls, **kwargs):
        return cls(plus=StablePML(**kwargs), minus=StablePML(**kwargs))

    @classmethod
    def absorber(cls, **kwargs):
        return cls(plus=Absorber(**kwargs), minus=Absorber(**kwargs))

    @classmethod
    def periodic(cls):
        return cls(plus=Periodic(), minus=Periodic())

    @classmethod
    def pec(cls):
        return cls(plus=PECBoundary(), minus=PECBoundary())

    @classmethod
    def pmc(cls):
        return cls(plus=PMCBoundary(), minus=PMCBoundary())

    @classmethod
    def bloch(cls, bloch_vec=0.0):
        return cls(plus=BlochBoundary(bloch_vec), minus=BlochBoundary(bloch_vec))


class BoundarySpec:
    """Boundary specification for all 3 axes."""

    def __init__(self, x=None, y=None, z=None):
        self.x = x or Boundary.pml()
        self.y = y or Boundary.pml()
        self.z = z or Boundary.pml()

    @classmethod
    def all_sides(cls, boundary=None):
        if boundary is None:
            boundary = PML()
        b = Boundary(plus=boundary, minus=boundary)
        return cls(x=b, y=b, z=b)

    @classmethod
    def pml(cls, **kwargs):
        return cls.all_sides(PML(**kwargs))

    def _to_khronos(self, K, resolution):
        """Convert to Khronos boundary parameters.

        Returns (boundaries, boundary_conditions, absorbers, pml_thicknesses).
        """
        boundaries = []
        boundary_conditions = []
        absorbers_list = [None, None, None]
        pml_thicknesses = []

        for axis_idx, axis_bc in enumerate([self.x, self.y, self.z]):
            axis_boundaries = [0.0, 0.0]
            axis_bcs = []
            axis_absorbers = [None, None]

            for side_idx, side_bc in enumerate([axis_bc.minus, axis_bc.plus]):
                if isinstance(side_bc, PML) or isinstance(side_bc, StablePML):
                    thickness = side_bc.num_layers / resolution  # physical thickness
                    axis_boundaries[side_idx] = thickness
                    axis_bcs.append(K.PML())
                elif isinstance(side_bc, Absorber):
                    axis_absorbers[side_idx] = K.Absorber(num_layers=side_bc.num_layers)
                    axis_bcs.append(K.PML())  # PML type but with absorber overlay
                elif isinstance(side_bc, Periodic):
                    axis_bcs.append(K.Periodic())
                elif isinstance(side_bc, BlochBoundary):
                    axis_bcs.append(K.Bloch(k=float(side_bc.bloch_vec)))
                elif isinstance(side_bc, PECBoundary):
                    axis_bcs.append(K.PECBoundary())
                elif isinstance(side_bc, PMCBoundary):
                    axis_bcs.append(K.PMCBoundary())
                else:
                    axis_bcs.append(K.PML())
                    axis_boundaries[side_idx] = 12 / resolution

            boundaries.append(axis_boundaries)
            boundary_conditions.append(axis_bcs)
            if any(a is not None for a in axis_absorbers):
                absorbers_list[axis_idx] = axis_absorbers
            pml_thicknesses.append(axis_boundaries)

        return boundaries, boundary_conditions, absorbers_list, pml_thicknesses
