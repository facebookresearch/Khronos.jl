# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)

"""
    build_simulation(λ, resolution)
TBW
"""
function build_simulation(λ, resolution)

    dpml = 1.0
    monitor_xy = 2.0
    monitor_height = 1.0

    sxy = 2 * dpml + monitor_xy
    sz = 2 * dpml + 2 * monitor_height
    cell_size = [sxy, sxy, sz]
    
    boundary_layers = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

    sources = [
        fdtd.PlaneWaveSource(
            time_profile = fdtd.ContinuousWaveSource(fcen = 1 / λ),
            center = [0,0,0],
            size = [Inf, Inf, 0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
        )
    ]
    
    sim = fdtd.Simulation(
        cell_size = cell_size,
        cell_center = [0.0,0.0,0.0],
        resolution = resolution,
        sources = sources,
        boundaries = boundary_layers,
    )

    return sim
end

λ = 1.0
resolution = 20.0

sim = build_simulation(λ, resolution)
fdtd.run(sim, until=60)
scene = fdtd.plot2D(sim, fdtd.Ex(), fdtd.Volume([0., 0., 0.],[2.0, 2.0, 4.0]), plot_geometry=false)
