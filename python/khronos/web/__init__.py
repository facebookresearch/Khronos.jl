# Copyright (c) Meta Platforms, Inc. and affiliates.
"""web.run() — local GPU execution mimicking tidy3d's cloud API."""

import numpy as np  # noqa: lazy via venv
from ..tidy3d.data import SimulationData, FieldData, FluxData, ModeData, DiffractionData
from ..tidy3d.monitor import FieldMonitor, FluxMonitor, ModeMonitor, DiffractionMonitor
from ..tidy3d.constants import to_time


def run(simulation, task_name=None, path=None, verbose=True, **kwargs):
    """Run a simulation locally on GPU.

    Mimics td.web.run() but executes on local hardware via Khronos.jl.
    """
    from ..tidy3d._bridge import get_khronos, get_jl

    K = get_khronos()
    jl = get_jl()

    # Build the Khronos simulation
    if verbose:
        print(f"Building Khronos simulation{f' ({task_name})' if task_name else ''}...")
    jl_sim = simulation._build_khronos_sim()

    # Run
    run_time_khronos = to_time(simulation.run_time)

    if verbose:
        print(f"Running FDTD (run_time={simulation.run_time*1e12:.1f} ps)...")

    K.run(
        jl_sim,
        until_after_sources=K.stop_when_dft_decayed(
            tolerance=float(simulation.shutoff),
            minimum_runtime=run_time_khronos * 0.1,
            maximum_runtime=run_time_khronos,
        ),
    )

    # Extract results
    if verbose:
        print("Extracting monitor data...")

    data_map = _extract_monitor_data(K, jl, simulation)

    if verbose:
        print("Done.")

    return SimulationData(simulation, data_map)


def _extract_monitor_data(K, jl, simulation):
    """Extract results from all monitors into Python data objects."""
    data_map = {}

    for name, info in simulation._monitor_index_map.items():
        monitor = info["monitor"]
        khronos_monitors = info["khronos_monitors"]

        if isinstance(monitor, FieldMonitor):
            components = {}
            comp_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
            for i, comp_name in enumerate(comp_names):
                if i < len(khronos_monitors):
                    fields = K.get_dft_fields(khronos_monitors[i])
                    components[comp_name] = np.array(fields)
            data_map[name] = FieldData(components)

        elif isinstance(monitor, FluxMonitor):
            if khronos_monitors:
                flux = np.array(K.get_flux(khronos_monitors[0]))
                data_map[name] = FluxData(flux)

        elif isinstance(monitor, ModeMonitor):
            if khronos_monitors:
                a_plus, a_minus = K.compute_mode_amplitudes(
                    khronos_monitors[0].monitor_data
                )
                data_map[name] = ModeData(
                    np.array(a_plus), np.array(a_minus)
                )

        elif isinstance(monitor, DiffractionMonitor):
            if khronos_monitors:
                orders = K.get_diffraction_efficiencies(khronos_monitors[0])
                data_map[name] = DiffractionData(dict(orders))

        else:
            data_map[name] = None

    return data_map
