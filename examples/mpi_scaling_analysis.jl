#!/usr/bin/env julia
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# MPI scaling analysis for Khronos.jl
# Compares single-GPU vs multi-GPU performance across problem sizes.
#
# Usage:
#   Single GPU baseline (:auto PML chunks):
#       julia --project examples/mpi_scaling_analysis.jl baseline
#   Single GPU BSP-8 (fair comparison for MPI):
#       julia --project examples/mpi_scaling_analysis.jl bsp8
#   MPI (N ranks, BSP-8 chunks):
#       mpiexecjl -np N julia --project examples/mpi_scaling_analysis.jl mpi
#
# Notes:
#   - :auto mode produces 27 PML-grid chunks optimized for single-GPU via
#     Nothing-dispatch (PML-free chunks skip PML computation at compile time).
#   - BSP with num_chunks=8 produces 8 equal-sized chunks for balanced
#     multi-GPU distribution (4 chunks/rank with 2 GPUs).
#   - Comparing MPI results to BSP-8 baseline gives the true MPI overhead.
#     Comparing to :auto baseline shows the total cost including loss of
#     Nothing-dispatch optimization.

import Khronos
using MPI
using CUDA
using Printf

const RUN_MODE = length(ARGS) >= 1 ? ARGS[1] : "baseline"
const USE_MPI = (RUN_MODE == "mpi")
const USE_BSP8 = (RUN_MODE == "bsp8" || USE_MPI)

if USE_MPI
    Khronos.init_mpi!()
end

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

if USE_MPI
    Khronos.select_device_for_rank!()
end

const RANK = USE_MPI ? Khronos.mpi_rank() : 0
const NRANKS = USE_MPI ? Khronos.mpi_size() : 1

# Problem sizes: (cell_size, resolution, label)
const PROBLEM_SIZES = [
    (6.4,  10, "64^3"),       # 262K voxels
    (12.8, 10, "128^3"),      # 2.1M voxels
    (25.6, 10, "256^3"),      # 16.8M voxels
    (38.4, 10, "384^3"),      # 56.6M voxels
    (51.2, 10, "512^3"),      # 134.2M voxels
]

const NSTEPS_WARMUP = 20
const NSTEPS_BENCH = 50
const NUM_CHUNKS = USE_BSP8 ? 8 : :auto

function run_benchmark_suite()
    if RANK == 0
        println("=" ^ 80)
        println("Khronos.jl MPI Scaling Analysis")
        println("  Mode: $RUN_MODE | Ranks: $NRANKS | GPU(s): $(length(CUDA.devices()))")
        println("  Chunks: $NUM_CHUNKS | Warmup: $NSTEPS_WARMUP | Bench: $NSTEPS_BENCH steps")
        println("=" ^ 80)
        @printf("%-8s %12s %8s %8s %10s %12s %8s\n",
                "Grid", "Voxels", "Chunks", "Local", "Time (s)", "MVoxels/s", "ms/step")
        println("-" ^ 80)
    end

    for (cell_sz, res, label) in PROBLEM_SIZES
        sim = Khronos.Simulation(
            cell_size = [cell_sz, cell_sz, cell_sz],
            cell_center = [0.0, 0.0, 0.0],
            resolution = res,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.GaussianPulseSource(fcen = 1.0, fwidth = 0.5),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = NUM_CHUNKS,
        )

        Khronos.prepare_simulation!(sim)
        num_voxels = sim.Nx * sim.Ny * sim.Nz
        nlocal = length(sim.chunk_data)
        total_chunks = sim.chunk_plan.total_chunks

        # Warmup
        for _ in 1:NSTEPS_WARMUP
            Khronos.step!(sim)
        end
        CUDA.synchronize()

        if USE_MPI
            MPI.Barrier(Khronos.mpi_comm())
        end

        CUDA.synchronize()
        t_start = time()
        for _ in 1:NSTEPS_BENCH
            Khronos.step!(sim)
        end
        CUDA.synchronize()

        if USE_MPI
            MPI.Barrier(Khronos.mpi_comm())
        end
        t_end = time()

        elapsed = t_end - t_start
        mcells = num_voxels * NSTEPS_BENCH / elapsed / 1e6
        ms_step = elapsed / NSTEPS_BENCH * 1000

        if RANK == 0
            @printf("%-8s %12d %8d %8d %10.3f %12.1f %8.2f\n",
                    label, num_voxels, total_chunks, nlocal, elapsed, mcells, ms_step)
        end

        sim = nothing
        GC.gc()
        CUDA.reclaim()
    end

    if RANK == 0
        println("=" ^ 80)
    end
end

run_benchmark_suite()

if USE_MPI
    MPI.Finalize()
end
