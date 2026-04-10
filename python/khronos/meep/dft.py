# Copyright (c) Meta Platforms, Inc. and affiliates.
"""DFT monitor proxy objects and data extraction functions."""

from .geom import Vector3
from ._units import meep_to_khronos_freq


class DftFlux:
    """Proxy for a flux monitor, returned by Simulation.add_flux().

    Stores the monitor configuration and, after simulation, holds
    references to the Khronos Julia monitor for data extraction.
    """

    def __init__(self, sim, center, size, freqs, name=None):
        self._sim = sim
        self.center = center
        self.size = size
        self.freqs = list(freqs)
        self.nfreqs = len(self.freqs)
        self.name = name or f"flux_{id(self)}"
        self._khronos_monitor = None
        self._flux_data = None  # for get_flux_data / load_minus_flux_data
        self._scale = 1.0

    @property
    def freq(self):
        return list(self.freqs)


class DftFields:
    """Proxy for DFT field monitors, returned by Simulation.add_dft_fields()."""

    def __init__(self, sim, center, size, freqs, components, name=None):
        self._sim = sim
        self.center = center
        self.size = size
        self.freqs = list(freqs)
        self.nfreqs = len(self.freqs)
        self.components = components
        self.name = name or f"dft_fields_{id(self)}"
        self._khronos_monitors = {}  # component → Julia monitor

    @property
    def freq(self):
        return list(self.freqs)


class DftNear2Far:
    """Proxy for near-to-far-field monitor."""

    def __init__(self, sim, center, size, freqs, name=None,
                 layer_stack=None, theta=None, phi=None, proj_distance=1e6):
        self._sim = sim
        self.center = center
        self.size = size
        self.freqs = list(freqs)
        self.nfreqs = len(self.freqs)
        self.name = name or f"near2far_{id(self)}"
        self.layer_stack = layer_stack
        self.theta = theta
        self.phi = phi
        self.proj_distance = proj_distance
        self._khronos_monitor = None


class DftEnergy:
    """Proxy for energy density monitor (NOT SUPPORTED)."""

    def __init__(self, sim, center, size, freqs, name=None):
        self._sim = sim
        self.freqs = list(freqs)
        self.name = name or f"energy_{id(self)}"


class DftForce:
    """Proxy for force monitor (NOT SUPPORTED)."""

    def __init__(self, sim, center, size, freqs, name=None):
        self._sim = sim
        self.freqs = list(freqs)
        self.name = name or f"force_{id(self)}"


class DftDiffraction:
    """Proxy for a diffraction order monitor.

    NOT available in meep — this is a Khronos extension.
    Decomposes DFT fields into diffraction orders via spatial FFT.
    """

    def __init__(self, sim, center, size, freqs, name=None):
        self._sim = sim
        self.center = center
        self.size = size
        self.freqs = list(freqs)
        self.nfreqs = len(self.freqs)
        self.name = name or f"diffraction_{id(self)}"
        self._khronos_monitor = None

    @property
    def freq(self):
        return list(self.freqs)


class FluxData:
    """Opaque container for saved flux DFT data (for normalization pattern)."""

    def __init__(self, E, H, cE, cH):
        self.E = E
        self.H = H
        self.cE = cE
        self.cH = cH


class ForceData:
    """Opaque container for saved force DFT data."""

    def __init__(self, offdiag1, offdiag2, diag):
        self.offdiag1 = offdiag1
        self.offdiag2 = offdiag2
        self.diag = diag


class EigenmodeData:
    """Result of get_eigenmode_coefficients."""

    def __init__(self, alpha, vgrp, kpoints, kdom, cscale=None):
        self.alpha = alpha  # shape: (bands, nfreqs, 2) - 2 = fwd/bwd
        self.vgrp = vgrp
        self.kpoints = kpoints
        self.kdom = kdom
        self.cscale = cscale


# ------------------------------------------------------------------- #
# Module-level data extraction functions
# ------------------------------------------------------------------- #

def get_fluxes(dft_flux):
    """Get power flux spectrum from a DftFlux object.

    Returns a list of flux values, one per frequency.
    """
    if dft_flux._flux_data is not None:
        return list(dft_flux._flux_data)
    if dft_flux._khronos_monitor is not None:
        from .._bridge import get_khronos
        import numpy as np
        K = get_khronos()
        flux = np.array(K.get_flux(dft_flux._khronos_monitor))
        dft_flux._flux_data = flux * dft_flux._scale
        return list(dft_flux._flux_data)
    return [0.0] * dft_flux.nfreqs


def get_flux_freqs(dft_flux):
    """Get frequency list from a DftFlux object."""
    return list(dft_flux.freqs)


def get_forces(dft_force):
    """Get force spectrum (NOT SUPPORTED)."""
    raise NotImplementedError("Force monitors are not supported by the Khronos backend.")


def get_force_freqs(dft_force):
    """Get frequency list from a DftForce object."""
    return list(dft_force.freqs)


def get_near2far_freqs(dft_n2f):
    """Get frequency list from a DftNear2Far object."""
    return list(dft_n2f.freqs)


def get_ldos_freqs(ldos):
    """Get frequency list from LDOS (NOT SUPPORTED)."""
    raise NotImplementedError("LDOS is not supported by the Khronos backend.")


def get_diffraction_efficiencies(dft_diffraction, max_order=5):
    """Get power per diffraction order from a DftDiffraction monitor.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    dft_diffraction : DftDiffraction
        Diffraction monitor (from ``sim.add_diffraction_monitor``).
    max_order : int
        Maximum diffraction order to compute (default 5).

    Returns
    -------
    dict of (int, int) → list of float
        Maps ``(m, n)`` order tuples to lists of power values (one per frequency).
    """
    if dft_diffraction._khronos_monitor is None:
        raise RuntimeError("Simulation has not been run yet")
    from .._bridge import get_khronos
    K = get_khronos()
    return dict(K.get_diffraction_efficiencies(
        dft_diffraction._khronos_monitor, max_order=max_order))


def get_diffraction_freqs(dft_diffraction):
    """Get frequency list from a DftDiffraction object."""
    return list(dft_diffraction.freqs)


def complexarray(re, im):
    """Combine real and imaginary arrays into a complex array."""
    import numpy as np
    return np.array(re) + 1j * np.array(im)


# ------------------------------------------------------------------- #
# Helper to build meep-style frequency list from fcen/df/nfreq
# ------------------------------------------------------------------- #

def _make_freq_list(args, kwargs):
    """Parse meep-style flux/monitor frequency arguments.

    Supports two calling conventions:
      add_flux(fcen, df, nfreq, *regions)
      add_flux([freq_array], *regions)

    Returns (freqs, remaining_args).
    """
    if len(args) == 0:
        raise ValueError("No arguments provided")

    # Check if first arg is an array/list of frequencies
    first = args[0]
    if hasattr(first, '__len__') and not isinstance(first, str):
        return list(first), args[1:]

    # Otherwise: fcen, df, nfreq, ...
    if len(args) < 3:
        raise ValueError("Expected fcen, df, nfreq or a frequency array")

    fcen, df, nfreq = float(args[0]), float(args[1]), int(args[2])
    if nfreq == 1:
        freqs = [fcen]
    else:
        freqs = [fcen - df / 2 + i * df / (nfreq - 1)
                 for i in range(nfreq)]
    return freqs, args[3:]
