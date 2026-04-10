# Precompile workload for PackageCompiler sysimage.
# Exercises the core Khronos API paths to eliminate JIT latency.
#
# Usage (standalone test):
#   julia --project=/path/to/Khronos.jl precompile/workload.jl
#
# This script is called by PackageCompiler during sysimage creation.

import Khronos
using GeometryPrimitives

# Use CPU backend for sysimage build (works everywhere)
Khronos.choose_backend(Khronos.CPUDevice(), Float32)

# --- 2D TM simulation: waveguide with flux monitor ---
sim_2d = Khronos.Simulation(
    cell_size = [10.0f0, 6.0f0, 0.0f0],
    cell_center = [0.0f0, 0.0f0, 0.0f0],
    resolution = 10,
    geometry = [
        Khronos.Object(
            shape = Cuboid([0.0, 0.0, 0.0], [10.0, 1.0, 1e-6]),
            material = Khronos.Material(ε = 12.0),
        ),
    ],
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 0.15, fwidth = 0.1),
            component = Khronos.Ez(),
            center = [-4.0, 0.0, 0.0],
            size = [0.0, 2.0, 0.0],
        ),
    ],
    monitors = [
        Khronos.FluxMonitor(
            center = [4.0, 0.0, 0.0],
            size = [0.0, 2.0, 0.0],
            frequencies = [0.1, 0.15, 0.2],
        ),
        Khronos.DFTMonitor(
            center = [0.0, 0.0, 0.0],
            size = [8.0, 4.0, 0.0],
            frequencies = [0.15],
            component = Khronos.Ez(),
        ),
    ],
    boundaries = [[1.0f0, 1.0f0], [1.0f0, 1.0f0], [0.0f0, 0.0f0]],
)

# Run a short simulation to exercise the stepping kernel
Khronos.run(sim_2d, until = 5.0)

# Exercise data extraction
flux_data = sim_2d.monitors[1].monitor_data
if flux_data !== nothing
    Khronos.get_flux(flux_data)
end

# --- 3D simulation: small box with PML ---
sim_3d = Khronos.Simulation(
    cell_size = [4.0f0, 4.0f0, 4.0f0],
    cell_center = [0.0f0, 0.0f0, 0.0f0],
    resolution = 5,
    geometry = [
        Khronos.Object(
            shape = Sphere([0.0, 0.0, 0.0], 0.5),
            material = Khronos.Material(ε = 4.0),
        ),
    ],
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 0.3, fwidth = 0.2),
            component = Khronos.Ez(),
            center = [-1.0, 0.0, 0.0],
            size = [0.0, 2.0, 2.0],
        ),
    ],
    monitors = [
        Khronos.FluxMonitor(
            center = [1.0, 0.0, 0.0],
            size = [0.0, 2.0, 2.0],
            frequencies = [0.2, 0.3, 0.4],
        ),
    ],
    boundaries = [[0.5f0, 0.5f0], [0.5f0, 0.5f0], [0.5f0, 0.5f0]],
)

Khronos.run(sim_3d, until = 2.0)

flux_data_3d = sim_3d.monitors[1].monitor_data
if flux_data_3d !== nothing
    Khronos.get_flux(flux_data_3d)
end

# Exercise stop_when_dft_decayed (function construction only)
stop_fn = Khronos.stop_when_dft_decayed(
    tolerance = 1e-4,
    minimum_runtime = 1.0,
    maximum_runtime = 10.0,
)

println("Precompile workload completed successfully.")
