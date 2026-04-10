# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Run control functions matching the meep Python API.

Step functions and most run callbacks are NOT SUPPORTED because Khronos
runs the FDTD loop entirely on GPU without Python callbacks. Only the
termination conditions that map to Khronos's built-in stop conditions
are functional.
"""

import warnings


# ------------------------------------------------------------------- #
# Termination conditions (supported)
# ------------------------------------------------------------------- #

class _StopCondition:
    """Base class for stop conditions."""
    pass


class stop_when_fields_decayed(_StopCondition):
    """Stop when field component at a point decays.

    In Khronos, this is approximated via DFT convergence
    (stop_when_dft_decayed) since per-point field monitoring
    is not available.
    """

    def __init__(self, dt, component, pt, decay_by):
        self.dt = dt
        self.component = component
        self.pt = pt
        self.decay_by = decay_by

    def _to_khronos(self, K, run_time_k):
        return K.stop_when_dft_decayed(
            tolerance=float(self.decay_by),
            minimum_runtime=run_time_k * 0.1,
            maximum_runtime=run_time_k,
        )


class stop_when_energy_decayed(_StopCondition):
    """Stop when energy decays below threshold."""

    def __init__(self, dt=50, decay_by=1e-11):
        self.dt = dt
        self.decay_by = decay_by

    def _to_khronos(self, K, run_time_k):
        return K.stop_when_dft_decayed(
            tolerance=float(self.decay_by),
            minimum_runtime=run_time_k * 0.1,
            maximum_runtime=run_time_k,
        )


class stop_when_dft_decayed(_StopCondition):
    """Stop when DFT monitors converge."""

    def __init__(self, tol=1e-11, minimum_run_time=0, maximum_run_time=None):
        self.tol = tol
        self.minimum_run_time = minimum_run_time
        self.maximum_run_time = maximum_run_time

    def _to_khronos(self, K, run_time_k):
        min_rt = self.minimum_run_time if self.minimum_run_time > 0 else run_time_k * 0.1
        max_rt = self.maximum_run_time if self.maximum_run_time is not None else run_time_k
        return K.stop_when_dft_decayed(
            tolerance=float(self.tol),
            minimum_runtime=float(min_rt),
            maximum_runtime=float(max_rt),
        )


class stop_after_walltime(_StopCondition):
    """Stop after wall-clock time limit (NOT SUPPORTED)."""

    def __init__(self, seconds):
        self.seconds = seconds

    def _to_khronos(self, K, run_time_k):
        warnings.warn(
            "stop_after_walltime is not supported by Khronos. "
            "Using maximum_runtime instead.",
            stacklevel=3,
        )
        return K.stop_when_dft_decayed(
            tolerance=1e-20,
            minimum_runtime=0,
            maximum_runtime=run_time_k,
        )


class stop_on_interrupt(_StopCondition):
    """Stop on keyboard interrupt (NOT SUPPORTED — runs complete)."""

    def _to_khronos(self, K, run_time_k):
        return None


# ------------------------------------------------------------------- #
# Step functions (NOT SUPPORTED — raise or warn)
# ------------------------------------------------------------------- #

def _unsupported_step_func(name):
    """Factory for unsupported step function stubs."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Step function '{name}' is not supported by the Khronos backend. "
            "Khronos runs the FDTD loop entirely on GPU without Python callbacks. "
            "This call will be ignored.",
            stacklevel=2,
        )
        return None
    wrapper.__name__ = name
    return wrapper


at_beginning = _unsupported_step_func("at_beginning")
at_end = _unsupported_step_func("at_end")
at_every = _unsupported_step_func("at_every")
at_time = _unsupported_step_func("at_time")
after_time = _unsupported_step_func("after_time")
before_time = _unsupported_step_func("before_time")
after_sources = _unsupported_step_func("after_sources")
after_sources_and_time = _unsupported_step_func("after_sources_and_time")
during_sources = _unsupported_step_func("during_sources")
combine_step_funcs = _unsupported_step_func("combine_step_funcs")
synchronized_magnetic = _unsupported_step_func("synchronized_magnetic")
in_volume = _unsupported_step_func("in_volume")
in_point = _unsupported_step_func("in_point")
to_appended = _unsupported_step_func("to_appended")
with_prefix = _unsupported_step_func("with_prefix")
when_true = _unsupported_step_func("when_true")
when_false = _unsupported_step_func("when_false")


# ------------------------------------------------------------------- #
# Output functions (NOT SUPPORTED)
# ------------------------------------------------------------------- #

output_epsilon = _unsupported_step_func("output_epsilon")
output_mu = _unsupported_step_func("output_mu")
output_efield = _unsupported_step_func("output_efield")
output_efield_x = _unsupported_step_func("output_efield_x")
output_efield_y = _unsupported_step_func("output_efield_y")
output_efield_z = _unsupported_step_func("output_efield_z")
output_hfield = _unsupported_step_func("output_hfield")
output_hfield_x = _unsupported_step_func("output_hfield_x")
output_hfield_y = _unsupported_step_func("output_hfield_y")
output_hfield_z = _unsupported_step_func("output_hfield_z")
output_dfield = _unsupported_step_func("output_dfield")
output_bfield = _unsupported_step_func("output_bfield")
output_sfield = _unsupported_step_func("output_sfield")
output_poynting = _unsupported_step_func("output_poynting")
output_poynting_x = _unsupported_step_func("output_poynting_x")
output_poynting_y = _unsupported_step_func("output_poynting_y")
output_poynting_z = _unsupported_step_func("output_poynting_z")
output_hpwr = _unsupported_step_func("output_hpwr")
output_dpwr = _unsupported_step_func("output_dpwr")
output_tot_pwr = _unsupported_step_func("output_tot_pwr")
output_png = _unsupported_step_func("output_png")


# ------------------------------------------------------------------- #
# Harminv (NOT SUPPORTED)
# ------------------------------------------------------------------- #

class Harminv:
    """Harmonic inversion (NOT SUPPORTED by Khronos backend).

    Harminv monitors a field at a point during time-stepping to extract
    resonant frequencies and Q-factors. Since Khronos doesn't support
    per-timestep callbacks, this is not available.
    """

    def __init__(self, c, pt, fcen, df, mxbands=300):
        warnings.warn(
            "Harminv is not supported by the Khronos backend. "
            "Use DFT monitors to extract frequency information instead.",
            stacklevel=2,
        )
        self.component = c
        self.pt = pt
        self.fcen = fcen
        self.df = df
        self.mxbands = mxbands
        self.freqs = []
        self.Q = []
        self.amp = []
        self.modes = []


# ------------------------------------------------------------------- #
# Miscellaneous
# ------------------------------------------------------------------- #

class DiffractedPlanewave:
    """Diffracted planewave for mode decomposition."""

    def __init__(self, g, axis, s, p):
        self.g = g
        self.axis = axis
        self.s = s
        self.p = p


class Verbosity:
    """Control output verbosity."""

    def __init__(self):
        self.meep = 1

    def __call__(self, val):
        self.meep = val


verbosity = Verbosity()


def quiet(quietval=True):
    """Suppress output."""
    verbosity.meep = 0 if quietval else 1


# ------------------------------------------------------------------- #
# Batch execution (Khronos extension — not in meep)
# ------------------------------------------------------------------- #

def run_batch(sim, source_configs, until=None, until_after_sources=None):
    """Run multiple simulations sequentially with different sources.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    sim : Simulation
        Template simulation (geometry, monitors, boundaries are shared).
    source_configs : list
        Each entry is a list of Source objects for one simulation.
    until : float, optional
        Run time in meep units.
    until_after_sources : float or StopCondition, optional
        Run duration or stop condition after sources turn off.

    Returns
    -------
    list of dict
        One result dict per simulation, containing monitor data.
    """
    from .._bridge import get_khronos, get_jl
    from ._units import meep_to_khronos_time
    K = get_khronos()
    jl = get_jl()

    # Build template simulation
    sim.init_sim()

    kwargs = {}
    if until is not None:
        kwargs["until"] = meep_to_khronos_time(float(until))
    if until_after_sources is not None:
        if isinstance(until_after_sources, _StopCondition):
            max_t = sim._estimate_run_time()
            kwargs["stop_condition"] = until_after_sources._to_khronos(K, max_t)
        else:
            kwargs["until_after_sources"] = meep_to_khronos_time(
                float(until_after_sources))

    # Build source configs in Julia format
    jl_configs = []
    for sources in source_configs:
        jl_sources = []
        for src in sources:
            from .._bridge import get_gp
            gp = get_gp()
            result = src._to_khronos(K, gp)
            if isinstance(result, list):
                jl_sources.extend(result)
            else:
                jl_sources.append(result)
        jl_configs.append(jl_sources)

    results = K.run_batch(jl_configs, **kwargs)
    return list(results)


def run_batch_concurrent(sim, source_configs, max_concurrent=0, **kwargs):
    """Run multiple simulations concurrently on a single GPU.

    NOT available in meep — this is a Khronos extension.
    Uses multi-stream GPU execution for overlapping kernel dispatch.

    Parameters
    ----------
    sim : Simulation
        Template simulation.
    source_configs : list
        Each entry is a list of Source objects.
    max_concurrent : int
        Maximum concurrent simulations (0 = auto-detect from GPU memory).

    Returns
    -------
    list of dict
        One result dict per simulation.
    """
    from .._bridge import get_khronos
    K = get_khronos()

    sim.init_sim()
    results = K.run_batch_concurrent(
        sim._jl_sim, max_concurrent=max_concurrent, **kwargs)
    return list(results)


def run_batch_multi_gpu(sim, source_configs, n_gpus=1, **kwargs):
    """Distribute batch simulations across multiple GPUs on a single node.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    sim : Simulation
        Template simulation.
    source_configs : list
        Each entry is a list of Source objects.
    n_gpus : int
        Number of GPUs to use.

    Returns
    -------
    list of dict
        One result dict per simulation.
    """
    from .._bridge import get_khronos
    K = get_khronos()

    sim.init_sim()
    results = K.run_batch_multi_gpu(
        sim._jl_sim, n_gpus=n_gpus, **kwargs)
    return list(results)


# ------------------------------------------------------------------- #
# LEE computation (Khronos extension — not in meep)
# ------------------------------------------------------------------- #

def compute_LEE(power, theta, phi, cone_half_angle=90.0):
    """Compute Light Extraction Efficiency from far-field power.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    power : array-like
        Far-field power matrix of shape (n_theta, n_phi).
    theta : array-like
        Polar angles in degrees.
    phi : array-like
        Azimuthal angles in degrees.
    cone_half_angle : float
        Extraction cone half-angle in degrees (default 90 = full hemisphere).

    Returns
    -------
    float
        Ratio of power within the cone to total hemispherical power.
    """
    import math
    from .._bridge import get_khronos
    K = get_khronos()
    return float(K.compute_LEE(
        power, list(theta), list(phi),
        cone_half_angle=math.radians(cone_half_angle)))


def compute_incoherent_LEE(batch_results, theta, phi, cone_half_angle=90.0):
    """Compute LEE from incoherent sum of batch simulation results.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    batch_results : list
        Results from run_batch or run_batch_multi_gpu.
    theta : array-like
        Polar angles in degrees.
    phi : array-like
        Azimuthal angles in degrees.
    cone_half_angle : float
        Extraction cone half-angle in degrees.

    Returns
    -------
    float
        Incoherently-averaged LEE.
    """
    import math
    from .._bridge import get_khronos
    K = get_khronos()
    return float(K.compute_incoherent_LEE(
        batch_results, list(theta), list(phi),
        cone_half_angle=math.radians(cone_half_angle)))
