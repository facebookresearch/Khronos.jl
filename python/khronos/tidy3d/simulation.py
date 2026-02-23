# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Simulation: the main entry point for defining an FDTD simulation."""

from .constants import to_length, to_time, to_freq, inf
from .grid import GridSpec
from .boundary import BoundarySpec, PML, Absorber, Periodic, BlochBoundary
from .geometry import Box
from .medium import Medium


class Simulation:
    """FDTD simulation definition (tidy3d-compatible API).

    Mirrors td.Simulation: define domain, structures, sources, monitors,
    boundaries, and grid specification. Call web.run(sim) to execute.
    """

    def __init__(
        self,
        size,
        center=(0, 0, 0),
        medium=None,
        structures=(),
        sources=(),
        monitors=(),
        boundary_spec=None,
        grid_spec=None,
        symmetry=(0, 0, 0),
        run_time=1e-12,
        shutoff=1e-5,
        courant=0.5,
        subpixel=False,
        normalize_index=0,
    ):
        self.size = tuple(size)  # meters
        self.center = tuple(center)  # meters
        self.medium = medium or Medium(permittivity=1.0)
        self.structures = list(structures)
        self.sources = list(sources)
        self.monitors = list(monitors)
        self.boundary_spec = boundary_spec or BoundarySpec.all_sides(PML())
        self.grid_spec = grid_spec or GridSpec.auto()
        self.symmetry = tuple(symmetry)
        self.run_time = run_time  # seconds
        self.shutoff = shutoff
        self.courant = courant
        self.subpixel = subpixel
        self.normalize_index = normalize_index

        # Internal state (populated at run time)
        self._jl_sim = None
        self._monitor_index_map = {}  # name → (khronos_monitor_idx, type)

    def _build_khronos_sim(self):
        """Convert this tidy3d Simulation to a Khronos.Simulation Julia object."""
        from ._bridge import get_khronos, get_gp, get_jl
        K = get_khronos()
        gp = get_gp()
        jl = get_jl()

        # Choose backend
        K.choose_backend(K.CUDADevice(), jl.seval("Float32"))

        # Compute resolution from grid spec
        resolution = self.grid_spec.make_resolution(self.structures, self.sources)

        # Convert boundaries and compute PML thicknesses
        boundaries, boundary_conditions, absorbers, pml_thicknesses = \
            self.boundary_spec._to_khronos(K, resolution)

        # Cell size: user's domain size + PML thicknesses (PML is inside cell_size in Khronos)
        cell_size = [
            to_length(self.size[i]) + pml_thicknesses[i][0] + pml_thicknesses[i][1]
            for i in range(3)
        ]

        # Cell center: shift to account for asymmetric PML
        cell_center = [
            to_length(self.center[i]) + (pml_thicknesses[i][1] - pml_thicknesses[i][0]) / 2
            for i in range(3)
        ]

        # Build geometry objects (reversed order: tidy3d last-wins → Khronos first-wins)
        geometry_objects = []
        for s in reversed(self.structures):
            geometry_objects.append(s._to_khronos(K, gp))

        # Append background medium as the last (lowest priority) object
        bg_box = Box(center=self.center,
                     size=tuple(s * 10 for s in self.size))
        bg_obj = K.Object(bg_box._to_khronos(gp), self.medium._to_khronos(K))
        geometry_objects.append(bg_obj)

        # Convert sources
        khronos_sources = []
        for src in self.sources:
            result = src._to_khronos(K, gp)
            if isinstance(result, list):
                khronos_sources.extend(result)  # TFSF returns multiple sources
            else:
                khronos_sources.append(result)

        # Convert monitors and build index map
        khronos_monitors = []
        self._monitor_index_map = {}
        for mon in self.monitors:
            start_idx = len(khronos_monitors)
            jl_monitors = mon._to_khronos_monitors(K)
            khronos_monitors.extend(jl_monitors)
            self._monitor_index_map[mon.name] = {
                "monitor": mon,
                "indices": list(range(start_idx, start_idx + len(jl_monitors))),
                "khronos_monitors": jl_monitors,
            }

        # Build Khronos Simulation
        sim_kwargs = dict(
            cell_size=cell_size,
            cell_center=cell_center,
            resolution=resolution,
            geometry=geometry_objects if geometry_objects else None,
            sources=khronos_sources,
            monitors=jl.seval("Khronos.Monitor")(khronos_monitors) if khronos_monitors else None,
            boundaries=boundaries,
            Courant=float(self.courant),
            symmetry=tuple(int(s) for s in self.symmetry),
        )

        # Add boundary conditions if any non-PML BCs exist
        has_special_bc = any(
            not isinstance(bc, type(K.PML()))
            for axis_bcs in boundary_conditions
            for bc in axis_bcs
        )
        sim_kwargs["boundary_conditions"] = boundary_conditions

        # Add absorbers if any exist
        if any(a is not None for a in absorbers):
            sim_kwargs["absorbers"] = absorbers

        # Subpixel smoothing
        if self.subpixel:
            sim_kwargs["subpixel_smoothing"] = K.AnisotropicSmoothing()

        self._jl_sim = K.Simulation(**sim_kwargs)
        return self._jl_sim
