# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for adiabatic absorber boundary condition.

import Khronos
using Test
using GeometryPrimitives

@testset "Absorber data structure" begin
    # Test default construction
    abs_default = Khronos.Absorber()
    @test abs_default.num_layers == 40
    @test abs_default.sigma_order == 3
    @test abs_default.sigma_max == 0.0

    # Test custom construction
    abs_custom = Khronos.Absorber(num_layers = 60, sigma_order = 4, sigma_max = 1.5)
    @test abs_custom.num_layers == 60
    @test abs_custom.sigma_order == 4
    @test abs_custom.sigma_max == 1.5
end

@testset "Absorber conductivity profile" begin
    # Set up a simulation with an absorber on +y and verify the σ arrays
    geometry = [
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 0.5, 0.22]),
            Khronos.Material(ε = 3.4^2),
        ),
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            Khronos.Material(ε = 1.44^2),
        ),
    ]

    num_layers = 10
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        geometry = geometry,
        sources = [
            Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            ),
        ],
        boundaries = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        absorbers = [nothing, [nothing, Khronos.Absorber(num_layers = num_layers)], nothing],
    )

    Khronos.prepare_simulation!(sim)

    # The σD arrays should exist (forced by absorber)
    gd = sim.geometry_data
    @test !isnothing(gd.σDx) || !isnothing(gd.σDy) || !isnothing(gd.σDz)

    # Check that the absorber side (+y) has non-zero conductivity
    # The +y side corresponds to high y-indices in the σ array
    if !isnothing(gd.σDy)
        σDy_cpu = Array(gd.σDy)
        ny = size(σDy_cpu, 2)

        # Interior should be zero (no material conductivity)
        interior_y = div(ny, 2)
        @test σDy_cpu[1, interior_y, 1] ≈ 0.0

        # Absorber region (last num_layers cells) should have increasing σ
        if ny > num_layers
            # The last cell should have the highest conductivity
            σ_last = σDy_cpu[1, ny, 1]
            σ_second_last = σDy_cpu[1, ny - 1, 1]
            @test σ_last > 0.0
            @test σ_last >= σ_second_last  # monotonically increasing into boundary
        end

        # Left side (-y) should NOT have absorber conductivity
        # (we only put absorber on +y)
        @test σDy_cpu[1, 1, 1] ≈ 0.0 atol=1e-15
    end
end

@testset "Absorber simulation runs without divergence" begin
    # Run a short simulation with absorber and verify it doesn't diverge
    geometry = [
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            Khronos.Material(ε = 1.0),
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        geometry = geometry,
        sources = [
            Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            ),
        ],
        boundaries = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        absorbers = [nothing, [nothing, Khronos.Absorber(num_layers = 20)], nothing],
    )

    # Run a few timesteps
    Khronos.run(sim, until = 2.0)

    # Check fields are not NaN or Inf
    for field in [sim.fields.fEx, sim.fields.fEy, sim.fields.fEz,
                  sim.fields.fHx, sim.fields.fHy, sim.fields.fHz]
        if !isnothing(field)
            @test !any(isnan.(field))
            @test !any(isinf.(field))
        end
    end
end

@testset "Absorber on multiple sides" begin
    # Test absorber on both sides of y-axis
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
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
        boundaries = [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        absorbers = [nothing, [Khronos.Absorber(num_layers = 15), Khronos.Absorber(num_layers = 15)], nothing],
    )

    Khronos.prepare_simulation!(sim)

    gd = sim.geometry_data
    if !isnothing(gd.σDy)
        σDy_cpu = Array(gd.σDy)
        ny = size(σDy_cpu, 2)

        # Both sides should have non-zero conductivity
        @test σDy_cpu[1, 1, 1] > 0.0   # -y side
        @test σDy_cpu[1, ny, 1] > 0.0   # +y side

        # Interior should be zero
        @test σDy_cpu[1, div(ny, 2), 1] ≈ 0.0
    end
end

@testset "Absorber classify_region_physics detects absorber" begin
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
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
        boundaries = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        absorbers = [nothing, [nothing, Khronos.Absorber(num_layers = 15)], nothing],
    )

    # Check that classify_region_physics detects absorber σ
    vol_at_boundary = Khronos.Volume(
        center = [0.0, 2.0, 0.0],  # near +y boundary
        size = [4.0, 0.5, 4.0],
    )
    physics = Khronos.classify_region_physics(sim, vol_at_boundary, sim.geometry, sim.boundaries)
    @test physics.has_sigma_D == true
    @test physics.has_sigma_B == true

    # Interior volume should not have absorber-induced σ (unless material has it)
    vol_interior = Khronos.Volume(
        center = [0.0, 0.0, 0.0],
        size = [1.0, 1.0, 1.0],
    )
    physics_interior = Khronos.classify_region_physics(sim, vol_interior, sim.geometry, sim.boundaries)
    # Interior should not have sigma from absorber (material ε=1 has no σ)
    @test physics_interior.has_sigma_D == false
    @test physics_interior.has_sigma_B == false
end
