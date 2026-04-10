# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Simulate a dipole in vacuum.

import Khronos
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

function build_dipole_sim()
    # Place a z-polarized dipole with a wavelength 1μm at the center
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
            # we add a pi phase shift to force a sine wave, rather than a cosine.
            amplitude = 1im,
        ),
    ]

    # Build the simulation object, such that it spans a cube 10μm×10μm×10μm. Place PML 1μm thick.
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,#40,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )
    return sim
end

function run_dipole()
    sim = build_dipole_sim()
    t_end = 1.0#40.0;
    Khronos.run(sim, until = t_end)
    return sim
end

# --- Run 1: includes JIT compilation ---
println("=" ^ 60)
println("RUN 1 (cold — includes JIT compilation)")
println("=" ^ 60)
t1 = time()
sim = run_dipole()
println("Run 1 total wall time: $(round(time() - t1, digits=3))s\n")

# --- Run 2: fresh sim, JIT already done ---
println("=" ^ 60)
println("RUN 2 (warm — JIT already compiled)")
println("=" ^ 60)
t2 = time()
sim = run_dipole()
println("Run 2 total wall time: $(round(time() - t2, digits=3))s\n")

# Plot cross sections of the result
scene = Khronos.plot2D(
    sim,
    Khronos.Ex(),
    Khronos.Volume([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    symmetric_field_scaling = false,
)
save("dipole.png", scene)
