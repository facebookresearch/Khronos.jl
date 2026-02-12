#!/usr/bin/env julia
# Profile MPI halo exchange overhead
import Khronos
using MPI, CUDA

Khronos.init_mpi!()
Khronos.choose_backend(Khronos.CUDADevice(), Float64)
Khronos.select_device_for_rank!()

rank = Khronos.mpi_rank()
sim = Khronos.Simulation(cell_size=[25.6,25.6,25.6], cell_center=[0.0,0.0,0.0], resolution=10,
    sources=[Khronos.UniformSource(time_profile=Khronos.GaussianPulseSource(fcen=1.0, fwidth=0.5),
        component=Khronos.Ez(), center=[0.0,0.0,0.0], size=[0.0,0.0,0.0])],
    boundaries=[[1.0,1.0],[1.0,1.0],[1.0,1.0]], num_chunks=8)
Khronos.prepare_simulation!(sim)
for _ in 1:20; Khronos.step!(sim); end
CUDA.synchronize()

nlocal = length(sim.chunk_data)
assignment = sim.chunk_rank_assignment

function count_conns(sim, assignment, rank, remote)
    n = 0
    for chunk in sim.chunk_data
        for c in chunk.halo_send
            if isnothing(assignment)
                if !remote; n += 1; end
            elseif remote == (assignment[c.dst_chunk_id] != rank)
                n += 1
            end
        end
    end
    return n
end
nremote = count_conns(sim, assignment, rank, true)
nlocal_send = count_conns(sim, assignment, rank, false)

# Measure total step time
CUDA.synchronize()
MPI.Barrier(Khronos.mpi_comm())
t0 = time()
for _ in 1:50; Khronos.step!(sim); end
CUDA.synchronize()
MPI.Barrier(Khronos.mpi_comm())
total = time() - t0

# Measure buffer sizes
function compute_buf_bytes(sim, assignment, rank)
    isnothing(assignment) && return 0
    total = 0
    for chunk in sim.chunk_data
        for (ci, conn) in enumerate(chunk.halo_send)
            if assignment[conn.dst_chunk_id] != rank
                total += length(chunk.halo_send_buffers[ci]) * 8
            end
        end
    end
    return total
end
total_buf_bytes = compute_buf_bytes(sim, assignment, rank)

if rank == 0
    nvox = sim.Nx * sim.Ny * sim.Nz
    println("Grid: $(sim.Nx)x$(sim.Ny)x$(sim.Nz) = $nvox voxels")
    println("Chunks: $(sim.chunk_plan.total_chunks) total, $nlocal local")
    println("Halo: $nremote remote sends, $nlocal_send local sends")
    println("Buffer: $(total_buf_bytes) bytes/exchange ($(round(total_buf_bytes/1024, digits=1)) KB)")
    println("Perf: $(round(total, digits=3))s / 50 steps = $(round(total/50*1000, digits=2)) ms/step")
    println("Rate: $(round(nvox * 50 / total / 1e6, digits=1)) MVoxels/s")
end

MPI.Finalize()
