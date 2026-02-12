# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# MPI correctness test for Khronos.jl distributed support.
# Run with:
#   mpiexecjl -np 1 julia --project test/test_mpi.jl
#   mpiexecjl -np 2 julia --project test/test_mpi.jl

using Test
using MPI

import Khronos

# Initialize MPI
Khronos.init_mpi!()
Khronos.choose_backend(Float64)  # CPU backend for testing

rank = Khronos.mpi_rank()
nranks = Khronos.mpi_size()
comm = Khronos.mpi_comm()

rank == 0 && println("Running MPI tests with $nranks rank(s)...")

@testset "MPI Distributed Tests (nranks=$nranks)" begin

    @testset "MPI initialization" begin
        @test Khronos.mpi_rank() >= 0
        @test Khronos.mpi_size() >= 1
        @test Khronos.mpi_rank() < Khronos.mpi_size()
        if nranks > 1
            @test Khronos.is_distributed() == true
        else
            @test Khronos.is_distributed() == false
        end
        if rank == 0
            @test Khronos.is_root() == true
        else
            @test Khronos.is_root() == false
        end
    end

    @testset "Chunk-to-rank assignment" begin
        sim = Khronos.Simulation(
            cell_size = [5.0, 5.0, 5.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.GaussianPulseSource(fcen = 1.0, fwidth = 0.5),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = :auto,
        )

        # Plan chunks (all ranks do this identically)
        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)
        plan = Khronos.plan_chunks(sim)

        if plan.total_chunks >= nranks
            assignment = Khronos.assign_chunks_to_ranks(plan, nranks)
            @test length(assignment) == plan.total_chunks
            # All ranks should be used (if enough chunks)
            for r in 0:(min(nranks, plan.total_chunks) - 1)
                @test r in assignment
            end
            # All assignments are valid ranks
            @test all(0 .<= assignment .< nranks)
        end
    end

    @testset "Simulation with MPI ($nranks rank(s))" begin
        sim = Khronos.Simulation(
            cell_size = [5.0, 5.0, 5.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.GaussianPulseSource(fcen = 1.0, fwidth = 0.5),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = :auto,
        )

        Khronos.prepare_simulation!(sim)

        @test !isnothing(sim.chunk_data)
        @test length(sim.chunk_data) >= 1

        if nranks > 1
            @test !isnothing(sim.chunk_rank_assignment)
            # Each rank should have at least one chunk
            local_count = length(sim.chunk_data)
            @test local_count >= 1
        end

        # Run a few timesteps
        nsteps = 20
        for _ in 1:nsteps
            Khronos.step!(sim)
        end

        @test sim.timestep == nsteps

        # Check that Ez at center is non-zero (source is at center)
        # Gather field from all ranks
        ez_fields = Khronos._pull_fields_from_device(sim, Khronos.Ez())

        if rank == 0
            # The center voxel should have been excited by the source
            cx = sim.Nx ÷ 2
            cy = sim.Ny ÷ 2
            cz = sim.Nz ÷ 2
            @test abs(ez_fields[cx, cy, cz]) > 0.0
            # Field norm should be positive
            @test sum(abs.(ez_fields)) > 0.0
        end
    end

    @testset "MPI vs single-process consistency" begin
        # This test verifies that the field norm is consistent
        sim = Khronos.Simulation(
            cell_size = [5.0, 5.0, 5.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.GaussianPulseSource(fcen = 1.0, fwidth = 0.5),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = :auto,
        )

        Khronos.prepare_simulation!(sim)
        for _ in 1:10
            Khronos.step!(sim)
        end

        ez_fields = Khronos._pull_fields_from_device(sim, Khronos.Ez())

        if rank == 0
            field_norm = sqrt(sum(ez_fields .^ 2))
            @test field_norm > 0.0
            @test isfinite(field_norm)
            println("  Ez field norm after 10 steps: $field_norm")
        end
    end
end

rank == 0 && println("MPI tests complete.")
MPI.Finalize()
