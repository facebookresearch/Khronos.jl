# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Test chunking data structures, physics classification, BSP splitting,
# and halo exchange.

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

    # -------------------------------------------------------- #
    # Phase 2+3: BSP splitting, cost model, adjacency, halo exchange
    # -------------------------------------------------------- #

    @testset "Cost model" begin
        pf_vacuum = Khronos.PhysicsFlags()
        pf_pml = Khronos.PhysicsFlags(has_pml_x = true)
        pf_eps = Khronos.PhysicsFlags(has_epsilon = true)
        pf_lossy = Khronos.PhysicsFlags(has_epsilon = true, has_sigma_D = true)

        cost_vacuum = Khronos.chunk_cost(pf_vacuum, 1000)
        cost_pml = Khronos.chunk_cost(pf_pml, 1000)
        cost_eps = Khronos.chunk_cost(pf_eps, 1000)
        cost_lossy = Khronos.chunk_cost(pf_lossy, 1000)

        # PML is more expensive than vacuum
        @test cost_pml > cost_vacuum
        # Epsilon adds cost
        @test cost_eps > cost_vacuum
        # Lossy adds more cost
        @test cost_lossy > cost_eps
        # PML is more expensive than epsilon alone
        @test cost_pml > cost_eps
    end

    @testset "Volume splitting" begin
        vol = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [10.0, 10.0, 10.0])

        # Split at midpoint along axis 1
        left, right = Khronos._split_volume(vol, 1, 0.5)
        @test left.size[1] ≈ 5.0
        @test right.size[1] ≈ 5.0
        @test left.center[1] ≈ -2.5
        @test right.center[1] ≈ 2.5

        # Other dimensions unchanged
        @test left.size[2] ≈ 10.0
        @test right.size[2] ≈ 10.0
        @test left.size[3] ≈ 10.0
        @test right.size[3] ≈ 10.0
    end

    @testset "Material boundary detection" begin
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
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        )

        vol = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [10.0, 10.0, 10.0])
        fracs = Khronos._find_material_boundaries(sim, vol, 1)

        # Should find material boundaries (sphere edges + PML)
        @test length(fracs) > 0
        # All fractions should be in (0, 1)
        @test all(0 .< fracs .< 1)
    end

    @testset "BSP splitting produces valid chunks" begin
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
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        vol = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [10.0, 10.0, 10.0])
        splits = Khronos._bsp_split(sim, vol, 2)

        # Should produce 2 chunks
        @test length(splits) == 2

        # Each split should have a Volume and PhysicsFlags
        for (v, pf) in splits
            @test v isa Khronos.Volume
            @test pf isa Khronos.PhysicsFlags
            @test all(v.size .> 0)
        end
    end

    @testset "Multi-chunk plan with num_chunks=2" begin
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
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = 2,
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)

        @test plan.total_chunks == 2
        @test length(plan.chunks) == 2

        # Should have adjacency
        @test length(plan.adjacency) >= 1

        # Each chunk should have valid grid volumes
        for chunk in plan.chunks
            @test chunk.grid_volume.Nx > 0
            @test chunk.grid_volume.Ny > 0
            @test chunk.grid_volume.Nz > 0
        end

        # Check that chunk IDs are 1 and 2
        @test Set(c.id for c in plan.chunks) == Set([1, 2])

        # Neighbor IDs should reference each other
        for chunk in plan.chunks
            @test length(chunk.neighbor_ids) >= 1
        end
    end

    @testset "Adjacency computation" begin
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
            num_chunks = 2,
        )

        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)

        plan = Khronos.plan_chunks(sim)

        # Two adjacent chunks should produce one adjacency entry
        @test length(plan.adjacency) == 1

        # Adjacency should be (1, 2, axis) for some axis
        adj = plan.adjacency[1]
        @test adj[1] == 1
        @test adj[2] == 2
        @test 1 <= adj[3] <= 3
    end

    @testset "num_chunks=nothing defaults to single chunk" begin
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
        @test Khronos._resolve_num_chunks(sim) == 1
    end

    @testset "num_chunks=:auto" begin
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
            num_chunks = :auto,
        )
        n = Khronos._resolve_num_chunks(sim)
        @test n >= 1
    end

    @testset "Halo connection setup" begin
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
            num_chunks = 2,
        )

        Khronos.prepare_simulation!(sim)

        @test !isnothing(sim.chunk_data)
        @test length(sim.chunk_data) == 2

        # Check halo connections exist
        total_sends = sum(length(c.halo_send) for c in sim.chunk_data)
        total_recvs = sum(length(c.halo_recv) for c in sim.chunk_data)

        # Each adjacency creates 2 send + 2 recv connections
        @test total_sends == 2
        @test total_recvs == 2
    end

    @testset "Halo exchange no-op for single chunk" begin
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

        # Should not error
        Khronos.exchange_halos!(sim, :E)
        Khronos.exchange_halos!(sim, :H)
        Khronos.exchange_halos!(sim, :B)
        Khronos.exchange_halos!(sim, :D)
        @test true  # no error
    end

    @testset "Field component group mapping" begin
        @test Khronos._field_components_for_group(:B) == (Khronos.Bx(), Khronos.By(), Khronos.Bz())
        @test Khronos._field_components_for_group(:H) == (Khronos.Hx(), Khronos.Hy(), Khronos.Hz())
        @test Khronos._field_components_for_group(:D) == (Khronos.Dx(), Khronos.Dy(), Khronos.Dz())
        @test Khronos._field_components_for_group(:E) == (Khronos.Ex(), Khronos.Ey(), Khronos.Ez())
    end

    @testset "Multi-chunk simulation runs without error" begin
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
            num_chunks = 2,
        )

        Khronos.prepare_simulation!(sim)

        # Run a few timesteps to verify no runtime errors
        for _ in 1:3
            Khronos.step!(sim)
        end
        @test sim.timestep == 3
    end

    @testset "Multi-chunk with geometry runs without error" begin
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
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 2.0), mat)],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = 2,
        )

        Khronos.prepare_simulation!(sim)

        # Run a few timesteps
        for _ in 1:3
            Khronos.step!(sim)
        end
        @test sim.timestep == 3
    end

    @testset "Chunk-local field accessors" begin
        pf = Khronos.PhysicsFlags(has_sources = true)
        gv = Khronos.GridVolume(Khronos.Center(), [1, 1, 1], [10, 10, 10], 10, 10, 10)
        vol = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [10.0, 10.0, 10.0])
        spec = Khronos.ChunkSpec(1, vol, gv, pf, Int[], 0)

        # Create a minimal chunk with some fields
        fields = Khronos.Fields{AbstractArray}(
            fEx = zeros(10, 10, 10),
            fEy = zeros(10, 10, 10),
            fEz = zeros(10, 10, 10),
        )
        geom = Khronos.GeometryData{Float64,Array}(ε_inv = 1.0, μ_inv = 1.0)
        bnd = Khronos.BoundaryData{Array}()
        chunk = Khronos.ChunkData{Float64,Array,Array{ComplexF64},AbstractArray}(
            spec, fields, geom, bnd,
            Khronos.SourceData{Array{ComplexF64}}[],
            Khronos.MonitorData[],
            Khronos.HaloConnection[], Khronos.HaloConnection[],
            AbstractArray[], AbstractArray[],
            (10, 10, 10),
        )

        @test Khronos._get_chunk_field(chunk, Khronos.Ex()) === fields.fEx
        @test Khronos._get_chunk_field(chunk, Khronos.Ey()) === fields.fEy
        @test Khronos._get_chunk_field(chunk, Khronos.Ez()) === fields.fEz
    end

    @testset "Source overlap by grid volume" begin
        src_gv = Khronos.GridVolume(Khronos.Ez(), [45, 45, 45], [55, 55, 55], 11, 11, 11)
        chunk_gv = Khronos.GridVolume(Khronos.Center(), [1, 1, 1], [50, 100, 100], 50, 100, 100)

        @test Khronos._source_overlaps_volume_by_gv(src_gv, chunk_gv)

        # Non-overlapping
        chunk_gv2 = Khronos.GridVolume(Khronos.Center(), [60, 1, 1], [100, 100, 100], 41, 100, 100)
        @test !Khronos._source_overlaps_volume_by_gv(src_gv, chunk_gv2)
    end
end
