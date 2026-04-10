# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Simulation class matching the meep Python API.

This is the central class of the meep-compatible wrapper. It stores all
simulation parameters and builds a Khronos Julia simulation on demand
when run() is called (deferred construction pattern).
"""

import warnings

from .constants import (
    Ex, Ey, Ez, Hx, Hy, Hz, Dielectric, Permeability,
    X, Y, Z, ALL, AUTOMATIC, inf,
    _COMPONENT_TO_KHRONOS,
)
from ._units import (
    meep_to_khronos_length, meep_to_khronos_freq, meep_to_khronos_time,
    khronos_to_meep_time,
)
from .geom import Vector3, Medium, Block, GeometricObject, FluxRegion, ModeRegion, Volume
from .boundaries import PML, Absorber, _build_pml_params
from .dft import (
    DftFlux, DftFields, DftNear2Far, DftEnergy, DftForce,
    EigenmodeData, FluxData, _make_freq_list,
)
from .run_functions import _StopCondition


class Simulation:
    """FDTD simulation, compatible with meep.Simulation.

    Parameters
    ----------
    cell_size : Vector3
        Size of the computational cell in meep units.
    resolution : float
        Pixels per meep distance unit.
    geometry : list of GeometricObject
        Dielectric structures. Later objects override earlier ones.
    sources : list of Source
        Current sources.
    boundary_layers : list of PML or Absorber
        Absorbing boundary layers.
    symmetries : list of Symmetry
        Mirror/rotational symmetries (stored but NOT enforced).
    default_material : Medium
        Background medium.
    eps_averaging : bool
        Subpixel smoothing (maps to Khronos AnisotropicSmoothing).
    dimensions : int
        Dimensionality (2 or 3). Inferred from cell_size if not given.
    Courant : float
        Courant stability factor.
    k_point : Vector3 or False
        Bloch-periodic k-vector. False = no Bloch boundaries.
    force_complex_fields : bool
        Force complex-valued fields (always True in Khronos DFT).
    """

    def __init__(
        self,
        cell_size=None,
        resolution=10,
        geometry=None,
        sources=None,
        boundary_layers=None,
        symmetries=None,
        default_material=None,
        eps_averaging=True,
        dimensions=3,
        force_complex_fields=False,
        m=0,
        k_point=False,
        kz_2d="complex",
        extra_materials=None,
        material_function=None,
        epsilon_func=None,
        epsilon_input_file="",
        Courant=0.5,
        ensure_periodicity=True,
        num_chunks=0,
        split_chunks_evenly=True,
        chunk_layout=None,
        filename_prefix=None,
        output_volume=None,
        output_single_precision=False,
        subpixel_tol=1e-4,
        subpixel_maxeval=100000,
        geometry_center=None,
        force_all_components=False,
        collect_stats=False,
        accurate_fields_near_cylorigin=False,
        progress_interval=4,
        # --- Khronos extensions (not in meep) ---
        backend="cuda",
        precision="float32",
        grid_dl_x=None,
        grid_dl_y=None,
        grid_dl_z=None,
    ):
        # Core parameters
        if cell_size is None:
            cell_size = Vector3()
        if not isinstance(cell_size, Vector3):
            cell_size = Vector3(*cell_size)
        self.cell_size = cell_size
        self.resolution = resolution
        self.geometry = list(geometry) if geometry else []
        self.sources = list(sources) if sources else []
        self.boundary_layers = list(boundary_layers) if boundary_layers else []
        self.symmetries = list(symmetries) if symmetries else []
        self.default_material = default_material or Medium()
        self.eps_averaging = eps_averaging
        self.dimensions = dimensions
        self.force_complex_fields = force_complex_fields
        self.Courant = Courant
        self.k_point = k_point
        self.geometry_center = geometry_center or Vector3()
        if not isinstance(self.geometry_center, Vector3):
            self.geometry_center = Vector3(*self.geometry_center)

        # Khronos extensions
        self.backend = backend
        self.precision = precision
        self.grid_dl_x = grid_dl_x
        self.grid_dl_y = grid_dl_y
        self.grid_dl_z = grid_dl_z

        # Internal state
        self._jl_sim = None
        self._is_initialized = False
        self._current_time = 0.0

        # Deferred monitors: added via add_flux/add_mode_monitor/etc
        self._flux_monitors = []         # list of DftFlux
        self._field_monitors = []        # list of DftFields
        self._n2f_monitors = []          # list of DftNear2Far
        self._diffraction_monitors = []  # list of DftDiffraction
        self._all_dft_objects = []       # ordered list of all DFT objects

        # Symmetry warning
        if self.symmetries:
            warnings.warn(
                "Symmetries are stored but NOT enforced by the Khronos backend. "
                "The simulation will run the full domain.",
                stacklevel=2,
            )

        if material_function is not None or epsilon_func is not None:
            warnings.warn(
                "material_function and epsilon_func are not supported by Khronos. "
                "Use explicit geometry objects instead.",
                stacklevel=2,
            )

    @property
    def fields(self):
        """Compatibility property. Returns self if initialized, None otherwise."""
        return self if self._is_initialized else None

    # ------------------------------------------------------------------- #
    # Monitor registration (deferred — actual Julia objects created at run)
    # ------------------------------------------------------------------- #

    def add_flux(self, *args, **kwargs):
        """Add a DFT flux monitor.

        Calling conventions:
          add_flux(fcen, df, nfreq, FluxRegion(...), ...)
          add_flux([freq_array], FluxRegion(...), ...)
        """
        freqs, regions = _make_freq_list(args, kwargs)
        decimation_factor = kwargs.get("decimation_factor", 0)

        # Use center/size from first FluxRegion
        if not regions:
            raise ValueError("Must provide at least one FluxRegion")
        fr = regions[0]
        if not isinstance(fr, FluxRegion):
            raise TypeError(f"Expected FluxRegion, got {type(fr)}")

        dft = DftFlux(self, fr.center, fr.size, freqs)
        self._flux_monitors.append(dft)
        self._all_dft_objects.append(dft)
        return dft

    def add_mode_monitor(self, *args, **kwargs):
        """Add a mode monitor (same as add_flux for Khronos)."""
        return self.add_flux(*args, **kwargs)

    def add_dft_fields(self, components, *args, **kwargs):
        """Add DFT field monitors for specific components.

        Calling conventions:
          add_dft_fields([mp.Ez], fcen, df, nfreq, where=Volume(...))
          add_dft_fields([mp.Ez, mp.Hx], [freq_array], where=Volume(...))
        """
        where = kwargs.pop("where", None)
        yee_grid = kwargs.pop("yee_grid", False)
        decimation_factor = kwargs.pop("decimation_factor", 0)

        freqs, _ = _make_freq_list(args, kwargs)

        if where is not None:
            center = where.center
            size = where.size
        else:
            center = Vector3()
            size = self.cell_size

        dft = DftFields(self, center, size, freqs, components)
        self._field_monitors.append(dft)
        self._all_dft_objects.append(dft)
        return dft

    def add_near2far(self, *args, layer_stack=None, theta=None, phi=None,
                     proj_distance=1e6, **kwargs):
        """Add near-to-far-field monitor.

        Parameters
        ----------
        layer_stack : LayerStack, optional
            Khronos extension: dielectric layer stack for transfer-matrix
            far-field projection through layered media.
        theta : list of float, optional
            Polar observation angles (radians).
        phi : list of float, optional
            Azimuthal observation angles (radians).
        proj_distance : float
            Far-field projection distance.
        """
        freqs, regions = _make_freq_list(args, kwargs)

        if not regions:
            raise ValueError("Must provide at least one Near2FarRegion")
        fr = regions[0]

        dft = DftNear2Far(self, fr.center, fr.size, freqs,
                          layer_stack=layer_stack, theta=theta, phi=phi,
                          proj_distance=proj_distance)
        self._n2f_monitors.append(dft)
        self._all_dft_objects.append(dft)
        return dft

    def add_diffraction_monitor(self, *args, center=None, size=None, **kwargs):
        """Add a diffraction order monitor.

        NOT available in meep — this is a Khronos extension.
        Decomposes DFT fields into diffraction orders via spatial FFT.

        Parameters
        ----------
        center : Vector3
            Center of the monitor plane.
        size : Vector3
            Size of the monitor plane (one dimension must be 0).
        """
        from .dft import DftDiffraction
        freqs, _ = _make_freq_list(args, kwargs)
        if center is None:
            center = Vector3()
        if size is None:
            size = Vector3()
        if not isinstance(center, Vector3):
            center = Vector3(*center)
        if not isinstance(size, Vector3):
            size = Vector3(*size)

        dft = DftDiffraction(self, center, size, freqs)
        self._diffraction_monitors.append(dft)
        self._all_dft_objects.append(dft)
        return dft

    def add_energy(self, *args, **kwargs):
        raise NotImplementedError("Energy monitors are not supported by Khronos.")

    def add_force(self, *args, **kwargs):
        raise NotImplementedError("Force monitors are not supported by Khronos.")

    # ------------------------------------------------------------------- #
    # Simulation lifecycle
    # ------------------------------------------------------------------- #

    def init_sim(self):
        """Initialize the simulation (build the Khronos Julia object)."""
        if self._is_initialized:
            return
        self._jl_sim = self._build_khronos_sim()
        self._is_initialized = True

    def reset_meep(self):
        """Reset the simulation state for a new run."""
        self._jl_sim = None
        self._is_initialized = False
        self._current_time = 0.0
        # Clear monitor references but keep registrations
        for dft in self._flux_monitors:
            dft._khronos_monitor = None
            dft._flux_data = None
        for dft in self._field_monitors:
            dft._khronos_monitors = {}
        for dft in self._n2f_monitors:
            dft._khronos_monitor = None

    def restart_fields(self):
        """Reset fields but keep the structure."""
        self.reset_meep()

    def run(self, *step_funcs, **kwargs):
        """Run the simulation.

        Parameters
        ----------
        until : float
            Run for this many meep time units.
        until_after_sources : float or _StopCondition
            Run until sources turn off, then continue for the given time
            or until the stop condition is satisfied.
        num_gpus : int
            Khronos extension: number of GPUs for domain decomposition.
            Default 1 (single GPU).

        Step functions are accepted for API compatibility but are NOT
        executed (warnings are emitted).
        """
        until = kwargs.get("until", None)
        until_after_sources = kwargs.get("until_after_sources", None)
        num_gpus = kwargs.get("num_gpus", 1)

        # Filter out None step functions (from unsupported wrappers)
        real_step_funcs = [f for f in step_funcs if f is not None]
        if real_step_funcs:
            warnings.warn(
                "Step functions are not supported by the Khronos backend. "
                "They will be ignored.",
                stacklevel=2,
            )

        # Build simulation if needed
        if not self._is_initialized:
            self.init_sim()

        from .._bridge import get_khronos, get_jl
        K = get_khronos()
        jl = get_jl()

        # Multi-GPU domain decomposition (Khronos extension)
        if num_gpus > 1:
            K.plan_chunks(self._jl_sim, num_gpus)

        if until is not None:
            # Run for a fixed time
            run_time_k = meep_to_khronos_time(float(until))
            K.run(self._jl_sim, until=run_time_k)
            self._current_time = float(until)

        elif until_after_sources is not None:
            # Determine run time
            if isinstance(until_after_sources, _StopCondition):
                # Use the stop condition
                # Estimate a reasonable max run time based on source bandwidth
                max_time_k = self._estimate_run_time()
                stop_cond = until_after_sources._to_khronos(K, max_time_k)
                K.run(self._jl_sim, until_after_sources=stop_cond)
            elif isinstance(until_after_sources, (int, float)):
                run_time_k = meep_to_khronos_time(float(until_after_sources))
                K.run(self._jl_sim, until_after_sources=run_time_k)
            else:
                raise ValueError(
                    f"Unsupported until_after_sources type: {type(until_after_sources)}"
                )

        else:
            # Default: run for a reasonable time based on sources
            run_time_k = self._estimate_run_time()
            K.run(self._jl_sim, until=run_time_k)

        # Extract monitor data after run
        self._extract_all_monitor_data(K, jl)

    def meep_time(self):
        """Return the current simulation time in meep units."""
        if self._jl_sim is not None:
            from .._bridge import get_khronos
            K = get_khronos()
            return khronos_to_meep_time(float(K.current_time(self._jl_sim)))
        return self._current_time

    def estimate_memory(self, verbose=False):
        """Estimate GPU memory requirements before running.

        NOT available in meep — this is a Khronos extension.

        Parameters
        ----------
        verbose : bool
            Print detailed per-chunk breakdown.

        Returns
        -------
        dict with keys: total_bytes, total_mb, recommended_gpus, and more.
        """
        if not self._is_initialized:
            self.init_sim()
        from .._bridge import get_khronos
        K = get_khronos()
        result = K.estimate_memory(self._jl_sim, verbose=verbose)
        return {
            "total_bytes": int(result.total_bytes),
            "total_mb": int(result.total_bytes) / 1e6,
            "recommended_gpus": int(result.recommended_gpus),
            "field_bytes": int(result.field_bytes),
            "geometry_bytes": int(result.geometry_bytes),
            "monitor_bytes": int(result.monitor_bytes),
        }

    # ------------------------------------------------------------------- #
    # Data extraction
    # ------------------------------------------------------------------- #

    def get_flux_data(self, dft_flux):
        """Save DFT flux data for later subtraction (normalization pattern)."""
        import numpy as np
        if dft_flux._khronos_monitor is None:
            raise RuntimeError("Simulation has not been run yet")
        from .._bridge import get_khronos
        K = get_khronos()
        # Store raw flux values
        flux = np.array(K.get_flux(dft_flux._khronos_monitor))
        return FluxData(flux, None, None, None)

    def load_flux_data(self, dft_flux, fdata):
        """Load previously saved flux data into monitor."""
        dft_flux._flux_data = fdata.E

    def load_minus_flux_data(self, dft_flux, fdata):
        """Load negated flux data (for reflection subtraction).

        After this call, subsequent get_fluxes will return the
        subtracted flux: F_new - F_saved.
        """
        dft_flux._scale = -1.0
        dft_flux._flux_data = None  # will recompute on next get_fluxes

    def get_eigenmode_coefficients(self, dft_flux, bands, eig_parity=0,
                                   eig_vol=None, eig_resolution=0,
                                   eig_tolerance=1e-12, **kwargs):
        """Decompose fields into eigenmodes.

        Parameters
        ----------
        dft_flux : DftFlux
            Flux monitor to decompose.
        bands : list of int
            Mode indices (1-indexed).
        eig_parity : int
            Parity filtering.

        Returns
        -------
        EigenmodeData with alpha[bands, freqs, 2] (fwd/bwd).
        """
        import numpy as np
        if dft_flux._khronos_monitor is None:
            raise RuntimeError("Simulation has not been run yet")

        from .._bridge import get_khronos
        K = get_khronos()

        a_plus, a_minus = K.compute_mode_amplitudes(
            dft_flux._khronos_monitor.monitor_data
        )
        a_plus = np.array(a_plus)
        a_minus = np.array(a_minus)

        # Reshape to (bands, freqs, 2)
        nfreqs = len(dft_flux.freqs)
        nbands = len(bands)
        alpha = np.zeros((nbands, nfreqs, 2), dtype=complex)
        # For now, only first mode is available from Khronos
        if nbands >= 1:
            alpha[0, :, 0] = a_plus.flatten()[:nfreqs]
            alpha[0, :, 1] = a_minus.flatten()[:nfreqs]

        return EigenmodeData(
            alpha=alpha,
            vgrp=np.ones(nbands),  # placeholder
            kpoints=[Vector3()] * nbands,
            kdom=[Vector3()] * nbands,
        )

    def get_dft_array(self, dft_obj, component, num_freq):
        """Get a DFT field array for a specific component and frequency.

        Parameters
        ----------
        dft_obj : DftFields or DftFlux
            DFT monitor object.
        component : int
            Field component (mp.Ex, mp.Ez, etc.).
        num_freq : int
            Frequency index (0-based).

        Returns
        -------
        numpy array of complex field values.
        """
        import numpy as np
        if isinstance(dft_obj, DftFields):
            comp_name = _COMPONENT_TO_KHRONOS.get(component)
            if comp_name and comp_name in dft_obj._khronos_monitors:
                from .._bridge import get_khronos
                K = get_khronos()
                fields = K.get_dft_fields(dft_obj._khronos_monitors[comp_name])
                arr = np.array(fields)
                # Return the slice for the requested frequency
                if arr.ndim > 2 and num_freq < arr.shape[-1]:
                    return arr[..., num_freq]
                return arr
        elif isinstance(dft_obj, DftFlux):
            # Flux monitors can also return DFT arrays
            comp_name = _COMPONENT_TO_KHRONOS.get(component)
            if dft_obj._khronos_monitor is not None:
                from .._bridge import get_khronos
                K = get_khronos()
                fields = K.get_dft_fields(dft_obj._khronos_monitor)
                return np.array(fields)

        return np.array([])

    def get_array(self, component=None, vol=None, center=None, size=None,
                  cmplx=False, arr=None, frequency=0):
        """Get a field or material array.

        Only supports Dielectric component (epsilon) in this backend.
        Live field access is not available.
        """
        import numpy as np
        if component == Dielectric:
            warnings.warn(
                "get_array(Dielectric) returns a uniform array based on "
                "the default material. Full epsilon map is not yet available.",
                stacklevel=2,
            )
            # Return a uniform array of the default epsilon
            if center is None:
                center = Vector3()
            if size is None:
                size = self.cell_size
            if not isinstance(size, Vector3):
                size = Vector3(*size)
            nx = max(1, int(size.x * self.resolution))
            ny = max(1, int(size.y * self.resolution))
            nz = max(1, int(size.z * self.resolution)) if size.z > 0 else 1
            eps = self.default_material.epsilon
            if nz == 1:
                return np.full((nx, ny), eps)
            return np.full((nx, ny, nz), eps)

        raise NotImplementedError(
            f"get_array for component {component} is not supported by Khronos. "
            "Live field access is not available; use DFT monitors instead."
        )

    def get_field_point(self, component, pt):
        """Get instantaneous field at a point (NOT SUPPORTED)."""
        raise NotImplementedError(
            "get_field_point is not supported. Khronos does not provide "
            "live field access. Use DFT monitors to get frequency-domain fields."
        )

    def get_epsilon_point(self, pt, frequency=0):
        """Get permittivity at a point (approximate)."""
        return self.default_material.epsilon

    def change_sources(self, new_sources):
        """Replace sources for the next run."""
        self.sources = list(new_sources)
        self._is_initialized = False

    def change_k_point(self, new_k):
        """Change the Bloch k-point."""
        self.k_point = new_k
        self._is_initialized = False

    # ------------------------------------------------------------------- #
    # Utilities
    # ------------------------------------------------------------------- #

    def output_dft(self, dft_obj, fname):
        """Output DFT data to HDF5 (NOT SUPPORTED)."""
        raise NotImplementedError("HDF5 output is not supported by Khronos.")

    def plot2D(self, **kwargs):
        """2D visualization (NOT SUPPORTED)."""
        raise NotImplementedError(
            "plot2D is not supported by Khronos. Use matplotlib directly "
            "on the data arrays returned by get_dft_array or get_fluxes."
        )

    def plot3D(self, **kwargs):
        """3D visualization (NOT SUPPORTED)."""
        raise NotImplementedError("plot3D is not supported by Khronos.")

    def dump_structure(self, fname):
        raise NotImplementedError("Structure serialization is not supported.")

    def load_structure(self, fname):
        raise NotImplementedError("Structure serialization is not supported.")

    def dump_fields(self, fname):
        raise NotImplementedError("Field serialization is not supported.")

    def load_fields(self, fname):
        raise NotImplementedError("Field serialization is not supported.")

    # ------------------------------------------------------------------- #
    # Internal: build Khronos simulation
    # ------------------------------------------------------------------- #

    def _estimate_run_time(self):
        """Estimate a reasonable run time from source bandwidths."""
        max_width = 0
        for src in self.sources:
            st = src.src if hasattr(src, 'src') else src
            if hasattr(st, 'fwidth') and st.fwidth > 0:
                max_width = max(max_width, 1.0 / st.fwidth)
        if max_width > 0:
            # Run for ~10 pulse widths after sources
            return meep_to_khronos_time(max_width * 10)
        # Fallback: 200 meep time units
        return meep_to_khronos_time(200)

    def _build_khronos_sim(self):
        """Convert to a Khronos.Simulation Julia object."""
        from .._bridge import get_khronos, get_gp, get_jl
        from .geom import _jl_vec
        K = get_khronos()
        gp = get_gp()
        jl = get_jl()

        # Choose backend (Khronos extension)
        _device_map = {"cuda": K.CUDADevice, "cpu": K.CPUDevice, "metal": K.MetalDevice}
        _prec_map = {"float32": "Float32", "float64": "Float64"}
        device_cls = _device_map.get(self.backend, K.CUDADevice)
        prec_str = _prec_map.get(self.precision, "Float32")
        K.choose_backend(device_cls(), jl.seval(prec_str))

        resolution = self.resolution

        # Compute PML thicknesses
        pml_thickness, is_absorber = _build_pml_params(
            self.boundary_layers, resolution, self.cell_size
        )

        # Cell size in Khronos units (including PML)
        cell_size = _jl_vec([
            meep_to_khronos_length(self.cell_size[i]) +
            pml_thickness[i][0] + pml_thickness[i][1]
            for i in range(3)
        ])

        # Cell center (shifted for asymmetric PML)
        cell_center = _jl_vec([
            meep_to_khronos_length(self.geometry_center[i]) +
            (pml_thickness[i][1] - pml_thickness[i][0]) / 2
            for i in range(3)
        ])

        # Build boundary conditions
        boundaries_py = [[pml_thickness[i][0], pml_thickness[i][1]] for i in range(3)]
        boundaries = jl.seval("Vector{Vector{Float32}}")([_jl_vec(b) for b in boundaries_py])
        boundary_conditions_py = []
        for i in range(3):
            axis_bcs = []
            for side in range(2):
                if is_absorber[i][side]:
                    axis_bcs.append(K.PML())  # absorber uses PML type
                elif pml_thickness[i][side] > 0:
                    axis_bcs.append(K.PML())
                elif self.k_point is not False and self.k_point is not None:
                    kv = self.k_point
                    if not isinstance(kv, Vector3):
                        kv = Vector3(*kv)
                    k_val = [kv.x, kv.y, kv.z][i]
                    if k_val == 0:
                        axis_bcs.append(K.Periodic())
                    else:
                        axis_bcs.append(K.Bloch(k=float(k_val)))
                else:
                    axis_bcs.append(K.Periodic())
            boundary_conditions_py.append(axis_bcs)

        # Convert to Julia Vector{Vector{BoundaryCondition}}
        jl_bc_outer = jl.seval("Vector{Vector{Khronos.BoundaryCondition}}")()
        for axis_bcs in boundary_conditions_py:
            jl_bc_inner = jl.seval("Khronos.BoundaryCondition[]")
            for bc in axis_bcs:
                jl.seval("push!")(jl_bc_inner, bc)
            jl.seval("push!")(jl_bc_outer, jl_bc_inner)
        boundary_conditions = jl_bc_outer

        # Build geometry objects (meep: last wins → Khronos: first wins)
        geometry_objects = []
        for obj in reversed(self.geometry):
            if isinstance(obj, GeometricObject):
                geometry_objects.append(obj._to_khronos(K, gp))
            else:
                # Already a Khronos object?
                geometry_objects.append(obj)

        # Background medium as lowest-priority object
        bg_size = Vector3(
            self.cell_size.x * 10 if self.cell_size.x > 0 else 1e6,
            self.cell_size.y * 10 if self.cell_size.y > 0 else 1e6,
            self.cell_size.z * 10 if self.cell_size.z > 0 else 1e6,
        )
        bg_block = Block(size=bg_size, center=self.geometry_center,
                         material=self.default_material)
        geometry_objects.append(bg_block._to_khronos(K, gp))

        # Build Julia geometry vector early (needed by EigenModeSource)
        jl_geometry = jl.seval("Khronos.Object[]")
        for g in geometry_objects:
            jl.seval("push!")(jl_geometry, g)

        # Convert sources
        khronos_sources = []
        for src in self.sources:
            if hasattr(src, '_to_khronos'):
                # EigenModeSource needs geometry for mode solving
                from .source import EigenModeSource
                if isinstance(src, EigenModeSource):
                    result = src._to_khronos(K, gp, geometry_objects=jl_geometry)
                else:
                    result = src._to_khronos(K, gp)
                if isinstance(result, list):
                    khronos_sources.extend(result)
                else:
                    khronos_sources.append(result)

        # Convert monitors
        khronos_monitors = []

        # Flux monitors → Khronos FluxMonitor
        for dft in self._flux_monitors:
            c = _jl_vec([meep_to_khronos_length(x) for x in dft.center])
            s = _jl_vec([meep_to_khronos_length(x) for x in dft.size])
            k_freqs = _jl_vec([meep_to_khronos_freq(f) for f in dft.freqs])
            mon = K.FluxMonitor(center=c, size=s, frequencies=k_freqs)
            dft._khronos_monitor = mon
            khronos_monitors.append(mon)

        # Field monitors → Khronos DFTMonitor per component
        for dft in self._field_monitors:
            c = _jl_vec([meep_to_khronos_length(x) for x in dft.center])
            s = _jl_vec([meep_to_khronos_length(x) for x in dft.size])
            k_freqs = _jl_vec([meep_to_khronos_freq(f) for f in dft.freqs])
            for comp in dft.components:
                comp_name = _COMPONENT_TO_KHRONOS.get(comp)
                if comp_name:
                    k_comp = getattr(K, comp_name)()
                    mon = K.DFTMonitor(
                        center=c, size=s, frequencies=k_freqs, component=k_comp
                    )
                    dft._khronos_monitors[comp_name] = mon
                    khronos_monitors.append(mon)

        # Build Khronos Simulation
        # Convert Python lists to Julia Vectors with correct element types
        jl_sources = jl.seval("Khronos.Source[]")
        for s in khronos_sources:
            jl.seval("push!")(jl_sources, s)

        jl_monitors = jl.seval("Khronos.Monitor[]")
        for m in khronos_monitors:
            jl.seval("push!")(jl_monitors, m)

        sim_kwargs = dict(
            cell_size=cell_size,
            cell_center=cell_center,
            resolution=resolution,
            geometry=jl_geometry if geometry_objects else None,
            sources=jl_sources if khronos_sources else None,
            boundaries=boundaries,
            boundary_conditions=boundary_conditions,
            Courant=float(self.Courant),
        )

        # Monitors
        if khronos_monitors:
            sim_kwargs["monitors"] = jl_monitors

        # Subpixel smoothing (extended to support "volume" mode)
        if self.eps_averaging is True or self.eps_averaging == "anisotropic":
            sim_kwargs["subpixel_smoothing"] = K.AnisotropicSmoothing()
        elif self.eps_averaging == "volume":
            sim_kwargs["subpixel_smoothing"] = K.VolumeAveraging()
        # False / "none" → NoSmoothing (default, don't pass)

        # Diffraction monitors → Khronos DiffractionMonitor
        for dft in self._diffraction_monitors:
            c = _jl_vec([meep_to_khronos_length(x) for x in dft.center])
            s = _jl_vec([meep_to_khronos_length(x) for x in dft.size])
            k_freqs = _jl_vec([meep_to_khronos_freq(f) for f in dft.freqs])
            mon = K.DiffractionMonitor(center=c, size=s, frequencies=k_freqs)
            dft._khronos_monitor = mon
            khronos_monitors.append(mon)

        # Near2Far monitors → Khronos Near2FarMonitor
        for dft in self._n2f_monitors:
            c = _jl_vec([meep_to_khronos_length(x) for x in dft.center])
            s = _jl_vec([meep_to_khronos_length(x) for x in dft.size])
            k_freqs = _jl_vec([meep_to_khronos_freq(f) for f in dft.freqs])
            n2f_kwargs = dict(center=c, size=s, frequencies=k_freqs)
            if dft.theta is not None:
                n2f_kwargs["theta"] = list(dft.theta)
            if dft.phi is not None:
                n2f_kwargs["phi"] = list(dft.phi)
            n2f_kwargs["r"] = meep_to_khronos_length(dft.proj_distance)
            if dft.layer_stack is not None:
                n2f_kwargs["layer_stack"] = dft.layer_stack._to_khronos(K)
            mon = K.Near2FarMonitor(**n2f_kwargs)
            dft._khronos_monitor = mon
            khronos_monitors.append(mon)

        self._jl_sim = K.Simulation(**sim_kwargs)

        # Non-uniform grid override (Khronos extension)
        if self.grid_dl_x is not None:
            self._jl_sim.Δx = jl.Vector(
                [meep_to_khronos_length(d) for d in self.grid_dl_x])
            self._jl_sim.Nx = len(self.grid_dl_x)
        if self.grid_dl_y is not None:
            self._jl_sim.Δy = jl.Vector(
                [meep_to_khronos_length(d) for d in self.grid_dl_y])
            self._jl_sim.Ny = len(self.grid_dl_y)
        if self.grid_dl_z is not None:
            self._jl_sim.Δz = jl.Vector(
                [meep_to_khronos_length(d) for d in self.grid_dl_z])
            self._jl_sim.Nz = len(self.grid_dl_z)

        return self._jl_sim

    def _extract_all_monitor_data(self, K, jl):
        """Extract data from all monitors after a run completes."""
        import numpy as np
        # Flux monitors
        for dft in self._flux_monitors:
            if dft._khronos_monitor is not None:
                flux = np.array(K.get_flux(dft._khronos_monitor))
                if dft._scale != 1.0:
                    # Subtract mode: combine old and new flux
                    if dft._flux_data is not None:
                        dft._flux_data = flux + dft._scale * dft._flux_data
                    else:
                        dft._flux_data = flux
                else:
                    dft._flux_data = flux

        # Field monitors - data extracted on demand via get_dft_array


# ------------------------------------------------------------------- #
# Utility functions (module-level, matching meep API)
# ------------------------------------------------------------------- #

def make_output_directory(dir_name=""):
    """Create a temporary output directory."""
    import tempfile
    return tempfile.mkdtemp(prefix="meep-")


def delete_directory(dir_name):
    """Delete a directory tree."""
    import shutil
    shutil.rmtree(dir_name, ignore_errors=True)
