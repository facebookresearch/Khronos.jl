# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Source types matching the meep Python API."""

import math
from .constants import (
    Ex, Ey, Ez, Hx, Hy, Hz, inf, AUTOMATIC, ALL_COMPONENTS, NO_PARITY,
    _COMPONENT_TO_KHRONOS, is_electric,
)
from ._units import meep_to_khronos_length, meep_to_khronos_freq, meep_to_khronos_time
from .geom import Vector3, _jl_vec


# ------------------------------------------------------------------- #
# Time profiles (SourceTime)
# ------------------------------------------------------------------- #

class SourceTime:
    """Base class for source time profiles."""

    def __init__(self):
        self.frequency = 0

    def fourier_transform(self, freq):
        """Fourier transform of the source at the given frequency."""
        raise NotImplementedError


class GaussianSource(SourceTime):
    """Gaussian-envelope pulse source.

    Parameters
    ----------
    frequency : float
        Center frequency in meep units (a/λ).
    fwidth : float
        Frequency width in meep units.
    width : float
        Temporal width (= 1/fwidth). Specify either width or fwidth.
    start_time : float
        Start time in meep time units.
    cutoff : float
        Number of widths before cutoff.
    is_integrated : bool
        Whether this is an "integrated" source (current vs dipole).
    wavelength : float
        Wavelength in meep units (= 1/frequency). Alternative to frequency.
    """

    def __init__(self, frequency=None, fwidth=None, width=0,
                 start_time=0, cutoff=5.0, is_integrated=False,
                 wavelength=None):
        super().__init__()
        if wavelength is not None and frequency is None:
            frequency = 1.0 / wavelength
        if frequency is None:
            raise ValueError("Must specify frequency or wavelength")
        self.frequency = frequency
        if fwidth is not None:
            self.fwidth = fwidth
        elif width > 0:
            self.fwidth = 1.0 / width
        else:
            self.fwidth = frequency * 0.2  # default: 20% bandwidth
        self.start_time = start_time
        self.cutoff = cutoff
        self.is_integrated = is_integrated

    def _to_khronos(self, K):
        fcen = meep_to_khronos_freq(self.frequency)
        fwidth_k = 2 * math.pi * meep_to_khronos_freq(self.fwidth)
        return K.GaussianPulseSource(fcen=fcen, fwidth=fwidth_k)

    def fourier_transform(self, freq):
        """Approximate Fourier transform of the Gaussian source at freq."""
        df = self.fwidth
        w = 1.0 / df if df > 0 else 1.0
        return math.exp(-0.5 * ((freq - self.frequency) * w * 2 * math.pi) ** 2)


class ContinuousSource(SourceTime):
    """Continuous-wave (monochromatic) source."""

    def __init__(self, frequency=None, start_time=0, end_time=1e20,
                 width=0, fwidth=None, slowness=3.0, wavelength=None,
                 is_integrated=False):
        super().__init__()
        if wavelength is not None and frequency is None:
            frequency = 1.0 / wavelength
        if frequency is None:
            raise ValueError("Must specify frequency or wavelength")
        self.frequency = frequency
        self.start_time = start_time
        self.end_time = end_time
        self.slowness = slowness
        self.is_integrated = is_integrated
        if fwidth is not None:
            self.fwidth = fwidth
        elif width > 0:
            self.fwidth = 1.0 / width
        else:
            self.fwidth = 0

    def _to_khronos(self, K):
        fcen = meep_to_khronos_freq(self.frequency)
        return K.ContinuousWaveSource(fcen=fcen)


class CustomSource(SourceTime):
    """Source with user-defined time dependence."""

    def __init__(self, src_func, start_time=-1e20, end_time=1e20,
                 is_integrated=False, center_frequency=0, fwidth=0):
        super().__init__()
        self.src_func = src_func
        self.start_time = start_time
        self.end_time = end_time
        self.is_integrated = is_integrated
        self.frequency = center_frequency
        self.fwidth = fwidth

    def _to_khronos(self, K, dt=None, num_steps=None):
        """Sample the source function and convert to CustomSourceData.

        Since Khronos needs discrete samples, we sample the function over
        the simulation time range. This requires dt and num_steps to be
        provided by the Simulation at build time.
        """
        if dt is None or num_steps is None:
            raise ValueError(
                "CustomSource requires dt and num_steps for sampling. "
                "These are provided automatically during simulation build."
            )
        import numpy as np
        times = np.arange(num_steps) * dt
        values = np.array([self.src_func(t) for t in times], dtype=complex)
        times_k = [meep_to_khronos_time(t) for t in times]
        values_k = [complex(v) for v in values]
        fcen = meep_to_khronos_freq(self.frequency) if self.frequency > 0 else 0.1
        return K.CustomSourceData(times=times_k, values=values_k, fcen=fcen)


# ------------------------------------------------------------------- #
# Spatial sources
# ------------------------------------------------------------------- #

class Source:
    """A current source in the simulation domain.

    Parameters
    ----------
    src : SourceTime
        Time profile (GaussianSource, ContinuousSource, etc.).
    component : int
        Field component (mp.Ex, mp.Ey, mp.Ez, mp.Hx, etc.).
    center : Vector3
        Center of the source region.
    size : Vector3
        Size of the source region (0 → point source).
    volume : Volume
        Alternative to center/size.
    amplitude : complex
        Overall amplitude multiplier.
    amp_func : callable, optional
        Spatial amplitude function f(Vector3) → complex. NOT SUPPORTED.
    amp_data : numpy array, optional
        NumPy amplitude array. NOT SUPPORTED.
    """

    def __init__(self, src, component, center=None, size=None,
                 volume=None, amplitude=1.0, amp_func=None,
                 amp_func_file="", amp_data=None):
        self.src = src
        self.component = component
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
        self.amplitude = amplitude
        self.amp_func = amp_func
        self.amp_data = amp_data

        if amp_func is not None:
            import warnings
            warnings.warn(
                "amp_func is not supported by the Khronos backend. "
                "The source will use uniform amplitude.",
                stacklevel=2,
            )

    def _to_khronos(self, K, gp):
        comp_name = _COMPONENT_TO_KHRONOS.get(self.component)
        if comp_name is None:
            raise ValueError(f"Unsupported source component: {self.component}")

        comp = getattr(K, comp_name)()
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        s = _jl_vec([meep_to_khronos_length(x) for x in self.size])

        return K.UniformSource(
            time_profile=self.src._to_khronos(K),
            component=comp,
            center=c,
            size=s,
            amplitude=complex(self.amplitude),
        )


class EigenModeSource:
    """Eigenmode source (launches a specific waveguide mode).

    Parameters
    ----------
    src : SourceTime
        Time profile.
    center, size, volume : spatial extent of the source plane.
    eig_band : int
        Mode index (1-indexed).
    direction : int
        Normal direction (AUTOMATIC to infer from zero-size dimension).
    eig_kpoint : Vector3
        Initial k-vector guess for mode solver.
    eig_parity : int
        Parity filtering (NO_PARITY, EVEN_Z, ODD_Z, etc.).
    eig_resolution : int
        Mode solver resolution (0 = automatic).
    eig_tolerance : float
        Mode solver tolerance.
    """

    def __init__(self, src, center=None, size=None, volume=None,
                 eig_band=1, direction=AUTOMATIC,
                 eig_kpoint=None, eig_match_freq=True,
                 eig_parity=NO_PARITY, eig_resolution=0,
                 eig_tolerance=1e-12, component=ALL_COMPONENTS,
                 amplitude=1.0, **kwargs):
        self.src = src
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
        self.eig_band = eig_band
        self.direction = direction
        self.eig_kpoint = eig_kpoint or Vector3()
        self.eig_match_freq = eig_match_freq
        self.eig_parity = eig_parity
        self.eig_resolution = eig_resolution
        self.eig_tolerance = eig_tolerance
        self.component = component
        self.amplitude = amplitude

    def _to_khronos(self, K, gp, geometry_objects=None):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        s = _jl_vec([meep_to_khronos_length(x) for x in self.size])
        freq = meep_to_khronos_freq(self.src.frequency)

        mode_solver_res = self.eig_resolution if self.eig_resolution > 0 else 50

        kwargs = dict(
            time_profile=self.src._to_khronos(K),
            frequency=freq,
            center=c,
            size=s,
            mode_index=self.eig_band,  # already 1-indexed in meep
            mode_solver_resolution=mode_solver_res,
            solver_tolerance=float(self.eig_tolerance),
            amplitude=complex(self.amplitude),
        )
        if geometry_objects is not None:
            kwargs["geometry"] = geometry_objects
        return K.ModeSource(**kwargs)


class GaussianBeamSource:
    """Gaussian beam source (3D)."""

    def __init__(self, src, center=None, size=None, volume=None,
                 beam_x0=None, beam_kdir=None, beam_w0=None, beam_E0=None,
                 amplitude=1.0, **kwargs):
        self.src = src
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
        self.beam_x0 = beam_x0 or Vector3()
        self.beam_kdir = beam_kdir or Vector3(0, 0, 1)
        self.beam_w0 = beam_w0 or 1.0
        self.beam_E0 = beam_E0 or Vector3(1, 0, 0)
        self.amplitude = amplitude

    def _to_khronos(self, K, gp):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        s = _jl_vec([meep_to_khronos_length(x) for x in self.size])

        beam_center = _jl_vec([meep_to_khronos_length(x) for x in
                       (self.center + self.beam_x0)])
        k_vector = _jl_vec([float(self.beam_kdir.x), float(self.beam_kdir.y),
                     float(self.beam_kdir.z)])
        pol = _jl_vec([float(self.beam_E0.x), float(self.beam_E0.y),
               float(self.beam_E0.z)])

        return K.GaussianBeamSource(
            time_profile=self.src._to_khronos(K),
            center=c,
            size=s,
            beam_waist=meep_to_khronos_length(self.beam_w0),
            beam_center=beam_center,
            k_vector=k_vector,
            polarization=pol,
        )


class TFSFSource:
    """Total-Field/Scattered-Field source box.

    Creates a TFSF box that separates the simulation domain into
    total-field (inside) and scattered-field (outside) regions.
    Implemented internally as 6 equivalent current surfaces.

    This source type is NOT available in meep — it is a Khronos extension.

    Parameters
    ----------
    src : SourceTime
        Time profile.
    center : Vector3
        Center of the TFSF box.
    size : Vector3
        Size of the TFSF box.
    direction : int
        Propagation direction along injection_axis (+1 or -1).
    injection_axis : int
        Axis of propagation (0=x, 1=y, 2=z). Default 2 (z).
    pol_angle : float
        Polarization angle in radians.
    amplitude : complex
        Amplitude multiplier.
    """

    def __init__(self, src, center=None, size=None, direction=1,
                 injection_axis=2, pol_angle=0.0, amplitude=1.0):
        self.src = src
        self.center = center if center is not None else Vector3()
        self.size = size if size is not None else Vector3()
        if not isinstance(self.center, Vector3):
            self.center = Vector3(*self.center)
        if not isinstance(self.size, Vector3):
            self.size = Vector3(*self.size)
        self.direction = direction
        self.injection_axis = injection_axis
        self.pol_angle = pol_angle
        self.amplitude = amplitude

    def _to_khronos(self, K, gp):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        s = _jl_vec([meep_to_khronos_length(x) for x in self.size])
        result = K.TFSFSource(
            time_profile=self.src._to_khronos(K),
            center=c,
            size=s,
            injection_axis=self.injection_axis + 1,  # 0-indexed → 1-indexed
            direction=self.direction,
            polarization_angle=float(self.pol_angle),
            amplitude=complex(self.amplitude),
        )
        # TFSFSource returns a list of 6 EquivalentSources
        if hasattr(result, '__iter__'):
            return list(result)
        return result


# Aliases
GaussianBeam3DSource = GaussianBeamSource
GaussianBeam2DSource = GaussianBeamSource
