#!/usr/bin/env julia
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# MPI two-GPU benchmark for Khronos.jl.
# Usage: ~/.julia/bin/mpiexecjl -np 2 julia --project examples/mpi_two_gpu.jl

import Khronos
using MPI
using CUDA
using KernelAbstractions

Khronos.init_mpi!()
Khronos.choose_backend(Khronos.CUDADevice(), Float64)
Khronos.select_device_for_rank!()

rank = Khronos.mpi_rank()
nranks = Khronos.mpi_size()

rank == 0 && println("MPI two-GPU benchmark: $nranks ranks, $(length(CUDA.devices())) GPUs")

sim = Khronos.Simulation(
    cell_size = [25.6, 25.6, 25.6],
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

# Warmup
Khronos.prepare_simulation!(sim)
for _ in 1:10
    Khronos.step!(sim)
end
if Khronos.backend_engine isa KernelAbstractions.CUDABackend
    CUDA.synchronize()
end

# Benchmark
nsteps = 100
MPI.Barrier(Khronos.mpi_comm())
t_start = MPI.Wtime()
for _ in 1:nsteps
    Khronos.step!(sim)
end
if Khronos.backend_engine isa KernelAbstractions.CUDABackend
    CUDA.synchronize()
end
MPI.Barrier(Khronos.mpi_comm())
t_end = MPI.Wtime()

elapsed = t_end - t_start
num_voxels = sim.Nx * sim.Ny * sim.Nz
mcells = num_voxels * nsteps / elapsed / 1e6
nlocal = length(sim.chunk_data)

rank == 0 && println("  Grid: $(sim.Nx)×$(sim.Ny)×$(sim.Nz) = $num_voxels voxels")
println("  Rank $rank: $nlocal local chunks on GPU $(CUDA.device())")
rank == 0 && println("  $nsteps steps in $(round(elapsed, digits=3))s = $(round(mcells, digits=1)) MVoxels/s")

MPI.Finalize()
