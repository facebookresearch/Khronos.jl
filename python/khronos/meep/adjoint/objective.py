# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Objective quantity classes for adjoint optimization."""

import warnings
from ..geom import Vector3, Volume, FluxRegion
from ..constants import AUTOMATIC, NO_PARITY, _COMPONENT_TO_KHRONOS
from .._units import meep_to_khronos_length, meep_to_khronos_freq


class ObjectiveQuantity:
    """Base class for quantities that can be optimized."""

    def __init__(self, sim):
        self.sim = sim
        self._khronos_monitor = None

    def __call__(self):
        raise NotImplementedError

    def register_monitors(self, frequencies):
        raise NotImplementedError

    def place_adjoint_source(self, dJ):
        raise NotImplementedError


class EigenmodeCoefficient(ObjectiveQuantity):
    """Eigenmode coefficient objective.

    Measures the overlap of the DFT fields at a monitor plane with a
    specific waveguide eigenmode.

    Parameters
    ----------
    sim : Simulation
        Parent simulation.
    volume : Volume
        Monitor plane.
    mode : int
        Mode index (1-indexed).
    forward : bool
        Whether to measure the forward (+) or backward (-) coefficient.
    kpoint_func : callable, optional
        Function to compute k-point (NOT SUPPORTED).
    eig_parity : int
        Parity filtering.
    decimation_factor : int
        DFT decimation factor.
    subtracted_dft_fields : FluxData, optional
        Incident field data to subtract.
    """

    def __init__(self, sim, volume, mode, forward=True,
                 kpoint_func=None, kpoint_func_overlap_idx=0,
                 decimation_factor=0, subtracted_dft_fields=None,
                 eig_parity=NO_PARITY, **kwargs):
        super().__init__(sim)
        if isinstance(volume, Volume):
            self.center = volume.center
            self.size = volume.size
        else:
            self.center = volume.center if hasattr(volume, 'center') else Vector3()
            self.size = volume.size if hasattr(volume, 'size') else Vector3()
        self.mode = mode
        self.forward = forward
        self.eig_parity = eig_parity
        self.decimation_factor = decimation_factor
        self.subtracted_dft_fields = subtracted_dft_fields

    def _to_khronos(self, K, frequencies):
        """Create Khronos objective."""
        c = [meep_to_khronos_length(x) for x in self.center]
        s = [meep_to_khronos_length(x) for x in self.size]
        k_freqs = [meep_to_khronos_freq(f) for f in frequencies]
        direction = "+" if self.forward else "-"

        mon = K.ModeMonitor(
            center=c,
            size=s,
            frequencies=k_freqs,
        )
        self._khronos_monitor = mon

        return K.EigenmodeCoefficient(
            monitor=mon,
            mode_index=self.mode,
            direction=1 if self.forward else -1,
        )


class FourierFields(ObjectiveQuantity):
    """Fourier-domain field objective.

    Measures the DFT of a specific field component over a volume.

    Parameters
    ----------
    sim : Simulation
        Parent simulation.
    volume : Volume
        Monitor volume.
    component : int
        Field component (mp.Ez, etc.).
    yee_grid : bool
        Whether to use Yee grid positions.
    """

    def __init__(self, sim, volume, component, yee_grid=False,
                 decimation_factor=0, subtracted_dft_fields=None):
        super().__init__(sim)
        if isinstance(volume, Volume):
            self.center = volume.center
            self.size = volume.size
        else:
            self.center = volume.center if hasattr(volume, 'center') else Vector3()
            self.size = volume.size if hasattr(volume, 'size') else Vector3()
        self.component = component
        self.yee_grid = yee_grid

    def _to_khronos(self, K, frequencies):
        c = [meep_to_khronos_length(x) for x in self.center]
        s = [meep_to_khronos_length(x) for x in self.size]
        k_freqs = [meep_to_khronos_freq(f) for f in frequencies]

        comp_name = _COMPONENT_TO_KHRONOS.get(self.component)
        if comp_name is None:
            raise ValueError(f"Unsupported component: {self.component}")
        k_comp = getattr(K, comp_name)()

        mon = K.DFTMonitor(
            center=c, size=s, frequencies=k_freqs, component=k_comp
        )
        self._khronos_monitor = mon

        return K.FourierFieldsObjective(monitor=mon, component=k_comp)


class Near2FarFields(ObjectiveQuantity):
    """Near-to-far-field objective (limited support)."""

    def __init__(self, sim, Near2FarRegions, far_pts, nperiods=1,
                 decimation_factor=0, norm_near_fields=None,
                 greencyl_tol=1e-3):
        super().__init__(sim)
        self.regions = Near2FarRegions
        self.far_pts = far_pts
        self.nperiods = nperiods
        warnings.warn(
            "Near2FarFields objective has limited support in Khronos.",
            stacklevel=2,
        )


class LDOS(ObjectiveQuantity):
    """Local density of states (NOT SUPPORTED)."""

    def __init__(self, sim, **kwargs):
        super().__init__(sim)
        raise NotImplementedError(
            "LDOS objective is not supported by the Khronos backend."
        )
