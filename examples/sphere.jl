# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)

function build_sphere_sim(resolution, radius; include_loss = false)

    s_xyz = 2.0 + 1.0 + 2 * radius

    src_z = -s_xyz / 2.0 + 1.0
    #TODO swap out for actual planewave source once ready
    sources = [
        fdtd.UniformSource(
            time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
            component = fdtd.Ex(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
        fdtd.UniformSource(
            time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
            component = fdtd.Hy(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
    ]

    if include_loss
        mat = fdtd.Material(ε = 3, σD = 5)
    else
        mat = fdtd.Material(ε = 3)
    end

    geometry = [fdtd.Object(Ball([0.0, 0.0, 0.0], radius), mat)]

    sim = fdtd.Simulation(
        cell_size = [s_xyz, s_xyz, s_xyz],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        geometry = geometry,
    )

    return sim
end

sim = build_sphere_sim(20.0, 1.0; include_loss = true)
t_end = 10.0;
fdtd.run(sim, until = t_end)

scene = fdtd.plot2D(sim, fdtd.Ex(), fdtd.Volume([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]))
save("sphere.png", scene)
