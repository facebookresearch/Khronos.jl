# (c) Meta Platforms, Inc. and affiliates.
#
# Excitation of a dielectric waveguide using a mode source.

import Khronos
using CairoMakie
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

function build_waveguide_simulation(λ::Number)

    geometry = [
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 0.5, 0.22]),
            Khronos.Material(ε = 3.4^2),
        ), # Waveguide - Si
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            Khronos.Material(ε = 1.44^2),
        ), # Background material -- SiO2
    ]

    sources = [
        Khronos.ModeSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / λ),
            frequency = 1.0 / λ,
            mode_solver_resolution = 50,
            mode_index = 1,
            center = [0.0, 0.0, 0.0],
            size = [0.0, 2.0, 2.0],
            solver_tolerance = 1e-6,
            geometry = geometry,
        ),
    ]

    # Build the simulation object, such that it spans a cube 10μm×10μm×10μm. Place PML 1μm thick.
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 6.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 25,
        geometry = geometry,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )
    return sim
end

λ = 1.55
sim = build_waveguide_simulation(λ)

# scene = Khronos.plot2D(
#     sim,
#     nothing,
#     Khronos.Volume([0.0, 0.0, 0.0], [6.0, 2.0, 2.0]);
#     plot_geometry=true,
# )


# scene = Khronos.plot_source(sim, sim.sources[1])
# save("waveguide_source.png", scene)

t_end = 40.0;
Khronos.run(sim, until = t_end)

scene = Khronos.plot2D(
    sim,
    Khronos.Ey(),
    Khronos.Volume([0.0, 0.0, 0.0], [6.0, 4.0, 4.0]);
    plot_geometry = true,
)
save("waveguide.png", scene)
