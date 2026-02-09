# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Test chunking data structures and physics classification.

import Khronos
using Test
using GeometryPrimitives

@testset "Chunking" begin

    @testset "PhysicsFlags construction" begin
        # Default: all false
        pf = Khronos.PhysicsFlags()
        @test !pf.has_epsilon
        @test !pf.has_mu
        @test !pf.has_sigma_D
        @test !pf.has_sigma_B
        @test !pf.has_pml_x
        @test !pf.has_pml_y
        @test !pf.has_pml_z
        @test !pf.has_sources
        @test !pf.has_monitors
        @test !Khronos.has_any_pml(pf)

        # Selective flags
        pf2 = Khronos.PhysicsFlags(has_epsilon = true, has_pml_x = true)
        @test pf2.has_epsilon
        @test pf2.has_pml_x
        @test !pf2.has_mu
        @test Khronos.has_any_pml(pf2)
    end

    @testset "Single-chunk plan (vacuum with PML)" begin
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
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

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)

        @test plan.total_chunks == 1
        @test length(plan.chunks) == 1
        @test isempty(plan.adjacency)

        chunk = plan.chunks[1]
        @test chunk.id == 1
        @test isempty(chunk.neighbor_ids)
        @test chunk.device_id == 0

        # Vacuum: no epsilon/mu
        @test !chunk.physics.has_epsilon
        @test !chunk.physics.has_mu
        @test !chunk.physics.has_sigma_D
        @test !chunk.physics.has_sigma_B

        # PML present on all axes
        @test chunk.physics.has_pml_x
        @test chunk.physics.has_pml_y
        @test chunk.physics.has_pml_z

        # Source present
        @test chunk.physics.has_sources
    end

    @testset "Single-chunk plan (sphere with epsilon)" begin
        mat = Khronos.Material(ε = 12.0)
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [
                Khronos.UniformSource(
                    time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                    component = Khronos.Ez(),
                    center = [0.0, 0.0, 0.0],
                    size = [0.0, 0.0, 0.0],
                ),
            ],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)
        chunk = plan.chunks[1]

        # Sphere has epsilon
        @test chunk.physics.has_epsilon
        @test !chunk.physics.has_mu
        @test !chunk.physics.has_sigma_D
    end

    @testset "Single-chunk plan (no PML)" begin
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [
                Khronos.UniformSource(
                    time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                    component = Khronos.Ez(),
                    center = [0.0, 0.0, 0.0],
                    size = [0.0, 0.0, 0.0],
                ),
            ],
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)
        chunk = plan.chunks[1]

        # No PML
        @test !chunk.physics.has_pml_x
        @test !chunk.physics.has_pml_y
        @test !chunk.physics.has_pml_z
    end

    @testset "classify_region_physics with lossy material" begin
        mat = Khronos.Material(ε = 4.0, σD = 0.5)
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [
                Khronos.UniformSource(
                    time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                    component = Khronos.Ez(),
                    center = [0.0, 0.0, 0.0],
                    size = [0.0, 0.0, 0.0],
                ),
            ],
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)
        chunk = plan.chunks[1]

        @test chunk.physics.has_epsilon
        @test chunk.physics.has_sigma_D
    end

    @testset "Volumes overlap" begin
        vol_a = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [2.0, 2.0, 2.0])
        vol_b = Khronos.Volume(center = [1.0, 1.0, 1.0], size = [2.0, 2.0, 2.0])
        vol_c = Khronos.Volume(center = [5.0, 5.0, 5.0], size = [1.0, 1.0, 1.0])

        @test Khronos._volumes_overlap(vol_a, vol_b)
        @test !Khronos._volumes_overlap(vol_a, vol_c)
    end

    @testset "Monitor detection" begin
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [
                Khronos.UniformSource(
                    time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                    component = Khronos.Ez(),
                    center = [0.0, 0.0, 0.0],
                    size = [0.0, 0.0, 0.0],
                ),
            ],
            monitors = [
                Khronos.DFTMonitor(
                    component = Khronos.Ez(),
                    center = [0.0, 0.0, 0.0],
                    size = [10.0, 10.0, 0.0],
                    frequencies = [1.0],
                ),
            ],
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)
        chunk = plan.chunks[1]

        @test chunk.physics.has_monitors
    end

    @testset "ChunkPlan after prepare_simulation!" begin
        sim = Khronos.Simulation(
            cell_size = [10.0, 10.0, 10.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
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

        Khronos.prepare_simulation!(sim)

        @test !isnothing(sim.chunk_plan)
        @test sim.chunk_plan.total_chunks == 1
    end
end
