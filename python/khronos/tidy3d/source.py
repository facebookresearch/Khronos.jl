# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Source types and time profiles."""

from .constants import to_freq, to_length, inf


# ------------------------------------------------------------------- #
# Time profiles
# ------------------------------------------------------------------- #

class GaussianPulse:
    """Gaussian pulse time profile."""

    def __init__(self, freq0, fwidth):
        self.freq0 = freq0  # Hz
        self.fwidth = fwidth  # Hz

    def _to_khronos(self, K):
        import math
        fcen = to_freq(self.freq0)
        fwidth_k = 2 * math.pi * to_freq(self.fwidth)
        return K.GaussianPulseSource(fcen=fcen, fwidth=fwidth_k)


class ContinuousWave:
    """Continuous wave time profile."""

    def __init__(self, freq0):
        self.freq0 = freq0  # Hz

    def _to_khronos(self, K):
        return K.ContinuousWaveSource(fcen=to_freq(self.freq0))


# ------------------------------------------------------------------- #
# Spatial sources
# ------------------------------------------------------------------- #

class PlaneWave:
    """Plane wave source."""

    def __init__(self, source_time, center, size, direction="+",
                 pol_angle=0.0, angle_theta=None, angle_phi=None):
        self.source_time = source_time
        self.center = tuple(center)
        self.size = tuple(size)
        self.direction = direction
        self.pol_angle = pol_angle
        self.angle_theta = angle_theta
        self.angle_phi = angle_phi

    def _to_khronos(self, K, gp):
        # Determine propagation direction from the zero-size axis
        normal_axis = next(i for i, s in enumerate(self.size) if s == 0)
        k_vector = [0.0, 0.0, 0.0]
        k_vector[normal_axis] = 1.0 if self.direction == "+" else -1.0

        c = [to_length(x) for x in self.center]
        s = [to_length(x) if x != inf else 1e6 for x in self.size]

        return K.PlaneWaveSource(
            time_profile=self.source_time._to_khronos(K),
            center=c,
            size=s,
            k_vector=k_vector,
            polarization_angle=float(self.pol_angle),
        )


class PointDipole:
    """Point dipole source."""

    _pol_map = {"Ex": "Ex", "Ey": "Ey", "Ez": "Ez",
                "Hx": "Hx", "Hy": "Hy", "Hz": "Hz"}

    def __init__(self, source_time, center, polarization="Ez"):
        self.source_time = source_time
        self.center = tuple(center)
        self.polarization = polarization

    def _to_khronos(self, K, gp):
        comp = getattr(K, self.polarization)()
        c = [to_length(x) for x in self.center]
        return K.UniformSource(
            time_profile=self.source_time._to_khronos(K),
            component=comp,
            center=c,
            size=[0.0, 0.0, 0.0],
        )


class UniformCurrentSource:
    """Uniform current source."""

    def __init__(self, source_time, center, size, polarization="Ez"):
        self.source_time = source_time
        self.center = tuple(center)
        self.size = tuple(size)
        self.polarization = polarization

    def _to_khronos(self, K, gp):
        comp = getattr(K, self.polarization)()
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        return K.UniformSource(
            time_profile=self.source_time._to_khronos(K),
            component=comp,
            center=c,
            size=s,
        )


class ModeSource:
    """Waveguide mode source."""

    def __init__(self, source_time, center, size, mode_spec=None,
                 mode_index=0, direction="+", num_freqs=1):
        self.source_time = source_time
        self.center = tuple(center)
        self.size = tuple(size)
        self.mode_spec = mode_spec or {}
        self.mode_index = mode_index
        self.direction = direction
        self.num_freqs = num_freqs

    def _to_khronos(self, K, gp, geometry_objects=None):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        freq = to_freq(self.source_time.freq0)

        kwargs = dict(
            time_profile=self.source_time._to_khronos(K),
            frequency=freq,
            center=c,
            size=s,
            mode_index=self.mode_index + 1,  # Julia 1-indexed
            mode_solver_resolution=self.mode_spec.get("mode_solver_resolution", 50),
            solver_tolerance=self.mode_spec.get("solver_tolerance", 1e-6),
        )
        if geometry_objects is not None:
            kwargs["geometry"] = geometry_objects
        return K.ModeSource(**kwargs)


class GaussianBeam:
    """Gaussian beam source."""

    def __init__(self, source_time, center, size, direction="+",
                 pol_angle=0.0, waist_radius=1.0, waist_distance=0.0):
        self.source_time = source_time
        self.center = tuple(center)
        self.size = tuple(size)
        self.direction = direction
        self.pol_angle = pol_angle
        self.waist_radius = waist_radius
        self.waist_distance = waist_distance

    def _to_khronos(self, K, gp):
        normal_axis = next(i for i, s in enumerate(self.size) if s == 0)
        k_vector = [0.0, 0.0, 0.0]
        k_vector[normal_axis] = 1.0 if self.direction == "+" else -1.0

        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]

        beam_center = list(c)
        beam_center[normal_axis] -= to_length(self.waist_distance) * (1.0 if self.direction == "+" else -1.0)

        # Polarization vector from pol_angle
        transverse = [i for i in range(3) if i != normal_axis]
        import math
        pol = [0.0, 0.0, 0.0]
        pol[transverse[0]] = math.cos(self.pol_angle)
        pol[transverse[1]] = math.sin(self.pol_angle)

        return K.GaussianBeamSource(
            time_profile=self.source_time._to_khronos(K),
            center=c,
            size=s,
            beam_waist=to_length(self.waist_radius),
            beam_center=beam_center,
            k_vector=k_vector,
            polarization=pol,
        )


class TFSF:
    """Total-field/scattered-field source."""

    def __init__(self, source_time, center=(0, 0, 0), size=(1, 1, 1),
                 direction="+", injection_axis=2, pol_angle=0.0):
        self.source_time = source_time
        self.center = tuple(center)
        self.size = tuple(size)
        self.direction = direction
        self.injection_axis = injection_axis
        self.pol_angle = pol_angle

    def _to_khronos(self, K, gp):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) for x in self.size]
        d = 1 if self.direction == "+" else -1
        return K.TFSFSource(
            time_profile=self.source_time._to_khronos(K),
            center=c,
            size=s,
            injection_axis=self.injection_axis + 1,  # Julia 1-indexed
            direction=d,
            polarization_angle=float(self.pol_angle),
        )
