# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Inject an oblique Gaussian beam and plot its response.

import Khronos
using CairoMakie
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

function build_simulation(λ, waist_radius, resolution)

    dpml = 1.0
    monitor_xy = 10.0
    monitor_height = 1.0

    sxy = 2 * dpml + monitor_xy
    sz = 2 * dpml + 4 * monitor_height
    cell_size = [sxy, sxy, sz]

    boundary_layers = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

    sources = [
        Khronos.GaussianBeamSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 1 / λ, fwidth = 0.2 * 1 / λ),
            center = [0, 0, -2 * monitor_height],
            size = [sxy, sxy, 0],
            beam_center = [0.0, 0.0, 0.0],
            beam_waist = waist_radius,
            k_vector = [0.5, 0.0, 1.0],
            polarization = [1.0, 0.0, 0.0],
        ),
    ]

    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0, 0, 0],
            size = [monitor_xy, 0, 4 * sz],
            frequencies = [1 ./ λ],
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = cell_size,
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = boundary_layers,
        monitors = monitors,
    )

    return sim
end

function run_gaussian_beam()
    λ = 1.0
    waist_radius = 0.75
    resolution = 20.0
    sim = build_simulation(λ, waist_radius, resolution)
    Khronos.run(
        sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance = 1e-5,
            minimum_runtime = 0.0,
            maximum_runtime = 300.0,
        ),
    )
    return sim
end

# --- Run 1: includes JIT compilation ---
println("=" ^ 60)
println("RUN 1 (cold — includes JIT compilation)")
println("=" ^ 60)
t1 = time()
sim = run_gaussian_beam()
println("Run 1 total wall time: $(round(time() - t1, digits=3))s\n")

# --- Run 2: fresh sim, JIT already done ---
println("=" ^ 60)
println("RUN 2 (warm — JIT already compiled)")
println("=" ^ 60)
t2 = time()
sim = run_gaussian_beam()
println("Run 2 total wall time: $(round(time() - t2, digits=3))s\n")

# Visualize the source profile
λ = 1.0
frequencies = 0.75:0.005:1.25
scene = Khronos.plot_timesource(sim, sim.sources[1], frequencies)
save("sources.png", scene)

# Visualize the response of the monitor
scene = Khronos.plot_monitor(sim.monitors[1], 1)
save("gaussian_beam.png", scene)
