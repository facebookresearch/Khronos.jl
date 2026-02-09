# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Simulate the scattering of a planewave off of a conductive sphere.

import Khronos
using CairoMakie
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

"""
    build_sphere_sim(resolution, radius; include_loss = false)

Construct a simulation object at a given `resolution` (pixels/micron) and sphere
`radius` (in microns). Optionally make the sphere lossy (`include=true`).
"""
function build_sphere_sim(resolution, radius; include_loss = false)

    s_xyz = 2.0 + 1.0 + 2 * radius

    src_z = -s_xyz / 2.0 + 1.0
    λ = 1.0

    sources = [
        Khronos.PlaneWaveSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / λ),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
            amplitude = 1im,
        ),
    ]

    mat = include_loss ? Khronos.Material(ε = 3, σD = 5) : Khronos.Material(ε = 3)

    geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], radius), mat)]

    sim = Khronos.Simulation(
        cell_size = [s_xyz, s_xyz, s_xyz],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        geometry = geometry,
    )

    return sim, s_xyz
end

function run_sphere()
    sim, s_xyz = build_sphere_sim(20.0, 1.0; include_loss = true)
    t_end = 10.0;
    Khronos.run(sim, until = t_end)
    return sim, s_xyz
end

# --- Run 1: includes JIT compilation ---
println("=" ^ 60)
println("RUN 1 (cold — includes JIT compilation)")
println("=" ^ 60)
t1 = time()
sim, s_xyz = run_sphere()
println("Run 1 total wall time: $(round(time() - t1, digits=3))s\n")

# --- Run 2: fresh sim, JIT already done ---
println("=" ^ 60)
println("RUN 2 (warm — JIT already compiled)")
println("=" ^ 60)
t2 = time()
sim, s_xyz = run_sphere()
println("Run 2 total wall time: $(round(time() - t2, digits=3))s\n")

# Visualize the fields
scene = Khronos.plot2D(
    sim,
    Khronos.Ex(),
    Khronos.Volume([0.0, 0.0, 0.0], [s_xyz, s_xyz, s_xyz]),
)
save("sphere.png", scene)
