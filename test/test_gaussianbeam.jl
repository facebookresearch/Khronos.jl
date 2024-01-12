# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)
"""
    build_simulation(d, n, λ)
TBW
"""
function build_simulation(λ, waist_radius, resolution)

    dpml = 1.0
    monitor_xy = 10.0
    monitor_height = 1.0

    sxy = 2 * dpml + monitor_xy
    sz = 2 * dpml + 4 * monitor_height
    cell_size = [sxy, sxy, sz]

    boundary_layers = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

    sources = [
        fdtd.GaussianBeamSource(
            time_profile = fdtd.GaussianPulseSource(fcen = 1 / λ, fwidth = 0.2 * 1 / λ),
            center = [0, 0, -2 * monitor_height],
            size = [sxy, sxy, 0],
            beam_center = [0.0, 0.0, 0.0],
            beam_waist = waist_radius,
            k_vector = [0.5, 0.0, 1.0],
            polarization = [1.0, 0.0, 0.0],
        ),
    ]

    monitors = [
        fdtd.DFTMonitor(
            component = fdtd.Ex(),
            center = [0, 0, 0],
            size = [monitor_xy, 0, 4 * sz],
            frequencies = [1 ./ λ],
        ),
    ]

    sim = fdtd.Simulation(
        cell_size = cell_size,
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = boundary_layers,
        monitors = monitors,
    )

    return sim
end

function run_simulation!(sim::fdtd.SimulationData)
    fdtd.run(
        sim,
        until_after_sources = fdtd.stop_when_dft_decayed(
            tolerance = 1e-11,
            minimum_runtime = 0.0,
            maximum_runtime = 300.0,
        ),
    )
end

function plot_source_profile!(sim::fdtd.SimulationData)
    fdtd.prepare_simulation!(sim)
    Ex = sum(sim.source_data[1].amplitude_data, dims = 3)[:, :, 1]
    heatmap(Ex)
end

λ = 1.0
waist_radius = 0.75
resolution = 20.0

sim = build_simulation(λ, waist_radius, resolution)
#plot_source_profile!(sim)
run_simulation!(sim)

fdtd.plot_monitor(sim.monitors[1], 1)
