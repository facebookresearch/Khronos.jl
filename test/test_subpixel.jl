# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for subpixel smoothing at dielectric interfaces.

import Khronos
using Test
using GeometryPrimitives

@testset "Subpixel smoothing types" begin
    @test Khronos.NoSmoothing() isa Khronos.SubpixelSmoothing
    @test Khronos.VolumeAveraging() isa Khronos.SubpixelSmoothing
    @test Khronos.AnisotropicSmoothing() isa Khronos.SubpixelSmoothing
end

@testset "Subpixel smoothing default is NoSmoothing" begin
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 5,
        geometry = [
            Khronos.Object(
                Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
                Khronos.Material(ε = 1.0),
            ),
        ],
        sources = [
            Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            ),
        ],
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )
    @test sim.subpixel_smoothing isa Khronos.NoSmoothing
end

@testset "Subpixel smoothing on sphere" begin
    ε_sphere = 12.0
    ε_bg = 1.0
    radius = 1.0
    resolution = 10

    geometry = [
        Khronos.Object(
            Ball([0.0, 0.0, 0.0], radius),
            Khronos.Material(ε = ε_sphere),
        ),
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            Khronos.Material(ε = ε_bg),
        ),
    ]

    source = Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
    )

    function make_sim(smoothing)
        sim = Khronos.Simulation(
            cell_size = [4.0, 4.0, 4.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = resolution,
            geometry = geometry,
            sources = [source],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            subpixel_smoothing = smoothing,
        )
        Khronos.prepare_simulation!(sim)
        return sim
    end

    T = Khronos.backend_number
    inv_sphere = T(1.0 / ε_sphere)
    inv_bg = T(1.0 / ε_bg)

    # --- NoSmoothing: should give binary ε values ---
    sim_none = make_sim(Khronos.NoSmoothing())
    gd_none = sim_none.geometry_data
    ε_inv_x_none = Array(gd_none.ε_inv_x)

    # All values should be either 1/ε_sphere or 1/ε_bg
    for val in ε_inv_x_none
        @test (isapprox(val, inv_sphere, rtol=1e-4) || isapprox(val, inv_bg, rtol=1e-4))
    end

    # --- VolumeAveraging: should have intermediate values at interfaces ---
    sim_vol = make_sim(Khronos.VolumeAveraging())
    gd_vol = sim_vol.geometry_data
    ε_inv_x_vol = Array(gd_vol.ε_inv_x)

    # Count voxels with intermediate ε values
    n_intermediate_vol = count(ε_inv_x_vol) do val
        !isapprox(val, inv_sphere, rtol=1e-4) && !isapprox(val, inv_bg, rtol=1e-4)
    end
    @test n_intermediate_vol > 0

    # Interior voxels should be unchanged
    cx = div(size(ε_inv_x_vol, 1), 2)
    cy = div(size(ε_inv_x_vol, 2), 2)
    cz = div(size(ε_inv_x_vol, 3), 2)
    @test isapprox(ε_inv_x_vol[cx, cy, cz], inv_sphere, rtol=1e-4)
    @test isapprox(ε_inv_x_vol[1, 1, 1], inv_bg, rtol=1e-4)

    # All values should be bounded between 1/ε_sphere and 1/ε_bg
    for val in ε_inv_x_vol
        @test inv_sphere - 1e-5 ≤ val ≤ inv_bg + 1e-5
    end

    # --- AnisotropicSmoothing: component-dependent values at interfaces ---
    sim_aniso = make_sim(Khronos.AnisotropicSmoothing())
    gd_aniso = sim_aniso.geometry_data
    ε_inv_x_aniso = Array(gd_aniso.ε_inv_x)
    ε_inv_y_aniso = Array(gd_aniso.ε_inv_y)
    ε_inv_z_aniso = Array(gd_aniso.ε_inv_z)

    # Should also have intermediate values
    n_intermediate_aniso = count(ε_inv_x_aniso) do val
        !isapprox(val, inv_sphere, rtol=1e-4) && !isapprox(val, inv_bg, rtol=1e-4)
    end
    @test n_intermediate_aniso > 0

    # At interface points, x/y/z components should generally differ
    # (due to direction-dependent normal projection on a sphere)
    # Compare over minimum shared dimensions
    nx = min(size(ε_inv_x_aniso, 1), size(ε_inv_y_aniso, 1), size(ε_inv_z_aniso, 1))
    ny = min(size(ε_inv_x_aniso, 2), size(ε_inv_y_aniso, 2), size(ε_inv_z_aniso, 2))
    nz = min(size(ε_inv_x_aniso, 3), size(ε_inv_y_aniso, 3), size(ε_inv_z_aniso, 3))

    n_different = 0
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        vx = ε_inv_x_aniso[ix, iy, iz]
        vy = ε_inv_y_aniso[ix, iy, iz]
        vz = ε_inv_z_aniso[ix, iy, iz]
        is_interface = !isapprox(vx, inv_sphere, rtol=1e-4) && !isapprox(vx, inv_bg, rtol=1e-4)
        if is_interface && (!isapprox(vx, vy, rtol=1e-4) || !isapprox(vx, vz, rtol=1e-4))
            n_different += 1
        end
    end
    @test n_different > 0

    # Interior and exterior voxels should be unchanged
    @test isapprox(ε_inv_x_aniso[cx, cy, cz], inv_sphere, rtol=1e-4)
    @test isapprox(ε_inv_x_aniso[1, 1, 1], inv_bg, rtol=1e-4)

    # All values should be bounded
    for val in ε_inv_x_aniso
        @test inv_sphere - 1e-5 ≤ val ≤ inv_bg + 1e-5
    end
end
