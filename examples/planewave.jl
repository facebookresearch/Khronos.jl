# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Simulate the propagation of a planewave in a homogeonous medium.

import Khronos
using CairoMakie
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

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
        Khronos.PlaneWaveSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / λ),
            center = [0, 0, 0],
            size = [Inf, Inf, 0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
            amplitude = 1im,
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = cell_size,
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = boundary_layers,
    )

    return sim
end

λ = 1.0
resolution = 40.0

sim = build_simulation(λ, resolution)
Khronos.run(sim, until = 60)
scene = Khronos.plot2D(
    sim,
    Khronos.Ex(),
    Khronos.Volume([0.0, 0.0, 0.0], [2.0, 2.0, 4.0]),
    plot_geometry = false,
)
save("planewave.png", scene)
