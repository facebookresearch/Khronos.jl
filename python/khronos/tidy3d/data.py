# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SimulationData and result data classes."""


def _np():
    import numpy as np
    return np


class SimulationData:
    """Container for simulation results, accessed by monitor name."""

    def __init__(self, simulation, data_map):
        self.simulation = simulation
        self._data = data_map  # name → data object

    def __getitem__(self, name):
        if name not in self._data:
            available = list(self._data.keys())
            raise KeyError(f"Monitor '{name}' not found. Available: {available}")
        return self._data[name]

    @property
    def log(self):
        return ""  # No log for local execution


class FieldData:
    """DFT field data for a FieldMonitor."""

    def __init__(self, components):
        self._components = components  # dict of component_name → numpy array

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._components:
            return self._components[name]
        raise AttributeError(f"No field component '{name}'")


class FluxData:
    """Flux data from a FluxMonitor."""

    def __init__(self, flux_values):
        self.flux = _np().array(flux_values)


class ModeData:
    """Mode decomposition data from a ModeMonitor."""

    def __init__(self, amps_plus, amps_minus, n_complex=None):
        np = _np()
        self._amps_plus = np.array(amps_plus)
        self._amps_minus = np.array(amps_minus)
        self.n_complex = n_complex

    @property
    def amps(self):
        """Mode amplitudes as a structured object with .sel() method."""
        return _ModeAmps(self._amps_plus, self._amps_minus)


class _ModeAmps:
    """Helper for mode amplitude selection (mimics xarray interface)."""

    def __init__(self, plus, minus):
        self._plus = plus
        self._minus = minus

    def sel(self, mode_index=0, direction="+"):
        if direction == "+":
            return self._plus
        return self._minus


class DiffractionData:
    """Diffraction order data from a DiffractionMonitor."""

    def __init__(self, orders):
        self.orders = orders  # dict of (m,n) → power array


class FieldProjectionAngleData:
    """Far-field projection data."""

    def __init__(self, fields, theta, phi, freqs):
        self._fields = fields
        self.theta = theta
        self.phi = phi
        self.freqs = freqs

    @property
    def radar_cross_section(self):
        """Compute RCS from far-field data."""
        # Placeholder — would compute |E_far|^2 * 4*pi*r^2 / |E_inc|^2
        return self._fields
