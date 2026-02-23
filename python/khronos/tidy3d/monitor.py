# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Monitor types: FieldMonitor, FluxMonitor, ModeMonitor, etc."""

from .constants import to_freq, to_length


class FieldMonitor:
    """Frequency-domain field monitor (records all 6 components)."""

    def __init__(self, center, size, freqs, name, fields=None):
        self.center = tuple(center)
        self.size = tuple(size)
        self.freqs = list(freqs)
        self.name = name
        self._components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    def _to_khronos_monitors(self, K):
        """Create 6 DFTMonitors (one per component)."""
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freqs_k = [to_freq(f) for f in self.freqs]
        monitors = []
        for comp_name in self._components:
            comp = getattr(K, comp_name)()
            monitors.append(K.DFTMonitor(
                component=comp,
                center=list(c),
                size=list(s),
                frequencies=freqs_k,
            ))
        return monitors


class FluxMonitor:
    """Power flux monitor."""

    def __init__(self, center, size, freqs, name):
        self.center = tuple(center)
        self.size = tuple(size)
        self.freqs = list(freqs)
        self.name = name

    def _to_khronos_monitors(self, K):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freqs_k = [to_freq(f) for f in self.freqs]
        return [K.FluxMonitor(
            center=list(c),
            size=list(s),
            frequencies=freqs_k,
        )]


class ModeMonitor:
    """Mode decomposition monitor."""

    def __init__(self, center, size, freqs, name, mode_spec=None):
        self.center = tuple(center)
        self.size = tuple(size)
        self.freqs = list(freqs)
        self.name = name
        self.mode_spec = mode_spec or {}

    def _to_khronos_monitors(self, K, geometry_objects=None):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freqs_k = [to_freq(f) for f in self.freqs]

        mode_spec_kwargs = {
            "num_modes": self.mode_spec.get("num_modes", 1),
            "mode_solver_resolution": self.mode_spec.get("mode_solver_resolution", 50),
        }
        if geometry_objects is not None:
            mode_spec_kwargs["geometry"] = geometry_objects

        return [K.ModeMonitor(
            center=list(c),
            size=list(s),
            frequencies=freqs_k,
            mode_spec=K.ModeSpec(**mode_spec_kwargs),
        )]


class FieldTimeMonitor:
    """Time-domain field monitor."""

    def __init__(self, center, size, name, start=0.0, interval=1):
        self.center = tuple(center)
        self.size = tuple(size)
        self.name = name
        self.start = start
        self.interval = interval

    def _to_khronos_monitors(self, K):
        # TimeMonitor takes integer grid coordinates and a recording length
        # For now, return an empty list (time monitor needs special handling)
        return []


class DiffractionMonitor:
    """Diffraction order monitor."""

    def __init__(self, center, size, freqs, name, normal_dir="+"):
        self.center = tuple(center)
        self.size = tuple(size)
        self.freqs = list(freqs)
        self.name = name

    def _to_khronos_monitors(self, K):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freqs_k = [to_freq(f) for f in self.freqs]
        return [K.DiffractionMonitor(
            center=list(c),
            size=list(s),
            frequencies=freqs_k,
        )]


class FieldProjectionAngleMonitor:
    """Far-field projection monitor at specified angles."""

    def __init__(self, center, size, freqs, name, theta=None, phi=None,
                 proj_distance=1e6, normal_dir="+"):
        self.center = tuple(center)
        self.size = tuple(size)
        self.freqs = list(freqs)
        self.name = name
        self.theta = theta or [0.0]
        self.phi = phi or [0.0]
        self.proj_distance = proj_distance
        self.normal_dir = normal_dir

    def _to_khronos_monitors(self, K):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freqs_k = [to_freq(f) for f in self.freqs]
        nd = "+" if self.normal_dir == "+" else "-"
        return [K.Near2FarMonitor(
            center=list(c),
            size=list(s),
            frequencies=freqs_k,
            theta=list(self.theta),
            phi=list(self.phi),
            r=to_length(self.proj_distance),
        )]


# ModeSpec for mode sources/monitors
class ModeSpec:
    """Mode solver specification."""

    def __init__(self, num_modes=1, target_neff=0.0, **kwargs):
        self.num_modes = num_modes
        self.target_neff = target_neff
        self._kwargs = kwargs

    def to_dict(self):
        d = {"num_modes": self.num_modes, "target_neff": self.target_neff}
        d.update(self._kwargs)
        return d
