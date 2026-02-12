# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# MPI-based distributed multi-GPU support for Khronos.jl.
# Uses an SPMD model: every rank builds the full chunk plan, then each rank
# only allocates and computes its assigned chunks. Halo exchange between ranks
# uses MPI with host-staging buffers; local halo exchange remains on-device.

export init_mpi!, mpi_rank, mpi_size, is_distributed, is_root

# -------------------------------------------------------- #
# MPI state management
# -------------------------------------------------------- #

const _mpi_initialized = Ref(false)
const _mpi_comm = Ref{MPI.Comm}(MPI.COMM_WORLD)
const _mpi_rank = Ref(0)
const _mpi_size = Ref(1)

"""
    init_mpi!(; comm = MPI.COMM_WORLD)

Initialize MPI state for distributed simulation. Calls `MPI.Init()` if MPI
has not been initialized yet. Safe to call multiple times (idempotent).
"""
function init_mpi!(; comm = MPI.COMM_WORLD)
    if !_mpi_initialized[]
        if !MPI.Initialized()
            MPI.Init()
        end
        _mpi_comm[] = comm
        _mpi_rank[] = MPI.Comm_rank(comm)
        _mpi_size[] = MPI.Comm_size(comm)
        _mpi_initialized[] = true
    end
end

mpi_rank() = _mpi_rank[]
mpi_size() = _mpi_size[]
mpi_comm() = _mpi_comm[]
is_distributed() = _mpi_initialized[] && _mpi_size[] > 1
is_root() = mpi_rank() == 0

# -------------------------------------------------------- #
# Device assignment
# -------------------------------------------------------- #

"""
    select_device_for_rank!()

Assign the current MPI rank to a GPU device (round-robin over available devices).
No-op on CPU backend.
"""
function select_device_for_rank!()
    if backend_engine isa CUDABackend
        ndevices = length(CUDA.devices())
        CUDA.device!(mpi_rank() % ndevices)
    end
end

# -------------------------------------------------------- #
# Chunk-to-rank assignment
# -------------------------------------------------------- #

"""
    assign_chunks_to_ranks(plan::ChunkPlan, nranks::Int)::Vector{Int}

Greedy load-balanced assignment of chunks to MPI ranks using `chunk_cost`.
Returns a vector mapping chunk_id -> rank (0-indexed).
Deterministic: all ranks get the same result for the same input.
"""
function assign_chunks_to_ranks(plan::ChunkPlan, nranks::Int)::Vector{Int}
    costs = [chunk_cost(spec.physics,
                spec.grid_volume.Nx * spec.grid_volume.Ny * max(1, spec.grid_volume.Nz))
             for spec in plan.chunks]
    assignment = Vector{Int}(undef, plan.total_chunks)
    rank_load = zeros(Float64, nranks)
    for chunk_id in sortperm(costs, rev=true)
        target = argmin(rank_load) - 1  # 0-indexed rank
        assignment[chunk_id] = target
        rank_load[target + 1] += costs[chunk_id]
    end
    return assignment
end

# -------------------------------------------------------- #
# MPI halo exchange
# -------------------------------------------------------- #

"""
    allocate_mpi_halo_buffers!(sim::SimulationData)

Allocate host-side staging buffers and GPU-side contiguous staging buffers
for MPI halo exchange. Each remote connection gets:
  - A flat host `Vector` (for MPI send/recv)
  - A flat GPU `CuVector` (for contiguous GPU↔host DMA transfers)
Local connections get zero-length placeholder entries.
"""
function allocate_mpi_halo_buffers!(sim::SimulationData)
    rank = mpi_rank()
    assignment = sim.chunk_rank_assignment

    h_comps = (Hx(), Hy(), Hz())
    e_comps = (Ex(), Ey(), Ez())

    for chunk in sim.chunk_data
        gv = chunk.spec.grid_volume

        # Send buffers: host + GPU staging per connection
        sbufs = AbstractArray[]
        sbufs_gpu = AbstractArray[]
        for conn in chunk.halo_send
            if assignment[conn.dst_chunk_id] != rank
                max_size = 0
                for comp in (h_comps..., e_comps...)
                    comp_dims = Tuple(_get_chunk_component_voxel_count(sim, comp, gv))
                    sr = _component_send_range(comp_dims, conn.axis, conn.src_range)
                    face_size = length(sr[1]) * length(sr[2]) * length(sr[3])
                    max_size = max(max_size, face_size)
                end
                buf_size = max_size * 3
                push!(sbufs, zeros(backend_number, buf_size))
                # GPU staging: contiguous CuVector for fast DMA
                if backend_engine isa CUDABackend
                    push!(sbufs_gpu, CUDA.zeros(backend_number, buf_size))
                else
                    push!(sbufs_gpu, zeros(backend_number, 0))
                end
            else
                push!(sbufs, zeros(backend_number, 0))
                push!(sbufs_gpu, zeros(backend_number, 0))
            end
        end
        chunk.halo_send_buffers = sbufs
        chunk.halo_send_gpu_staging = sbufs_gpu

        # Recv buffers: host + GPU staging per connection
        rbufs = AbstractArray[]
        rbufs_gpu = AbstractArray[]
        for conn in chunk.halo_recv
            if assignment[conn.src_chunk_id] != rank
                max_size = 0
                for comp in (h_comps..., e_comps...)
                    comp_dims = Tuple(_get_chunk_component_voxel_count(sim, comp, gv))
                    dr = _component_recv_range(comp_dims, conn.axis, conn.dst_range)
                    face_size = length(dr[1]) * length(dr[2]) * length(dr[3])
                    max_size = max(max_size, face_size)
                end
                buf_size = max_size * 3
                push!(rbufs, zeros(backend_number, buf_size))
                if backend_engine isa CUDABackend
                    push!(rbufs_gpu, CUDA.zeros(backend_number, buf_size))
                else
                    push!(rbufs_gpu, zeros(backend_number, 0))
                end
            else
                push!(rbufs, zeros(backend_number, 0))
                push!(rbufs_gpu, zeros(backend_number, 0))
            end
        end
        chunk.halo_recv_buffers = rbufs
        chunk.halo_recv_gpu_staging = rbufs_gpu
    end
end

"""
    _pack_halo_to_host!(host_buf, offset, gpu_parent, sr_shifted, gpu_staging)

Copy a GPU halo face into a host staging buffer via a contiguous GPU staging
buffer. Two-step process avoids scalar indexing on non-contiguous GPU SubArrays:
  1. GPU view → contiguous GPU staging (on-device kernel, fast)
  2. Contiguous GPU staging → host buffer (single DMA transfer)
Returns the number of elements written.
"""
function _pack_halo_to_host!(host_buf::Vector, offset::Int, gpu_parent::CuArray,
                              sr_shifted, gpu_staging::CuArray)
    src_view = @view gpu_parent[sr_shifted[1], sr_shifted[2], sr_shifted[3]]
    n = length(src_view)
    # Step 1: Non-contiguous GPU view → contiguous GPU staging (GPU kernel)
    staging_view = reshape(@view(gpu_staging[1:n]), size(src_view))
    copyto!(staging_view, src_view)
    # Step 2: Contiguous GPU staging → host buffer (DMA)
    unsafe_copyto!(host_buf, offset + 1, gpu_staging, 1, n)
    return n
end

function _pack_halo_to_host!(host_buf::Vector, offset::Int, cpu_parent::Array,
                              sr_shifted, _gpu_staging=nothing)
    src_view = @view cpu_parent[sr_shifted[1], sr_shifted[2], sr_shifted[3]]
    n = length(src_view)
    dst_view = reshape(@view(host_buf[offset+1:offset+n]), size(src_view))
    copyto!(dst_view, src_view)
    return n
end

"""
    _unpack_halo_from_host!(gpu_parent, dr_shifted, host_buf, offset, shape, gpu_staging)

Copy from host staging buffer into GPU ghost cells via a contiguous GPU staging
buffer. Two-step process:
  1. Host buffer → contiguous GPU staging (single DMA transfer)
  2. Contiguous GPU staging → GPU view (on-device kernel, fast)
Returns the number of elements written.
"""
function _unpack_halo_from_host!(gpu_parent::CuArray, dr_shifted, host_buf::Vector,
                                  offset::Int, shape, gpu_staging::CuArray)
    n = prod(shape)
    # Step 1: Host buffer → contiguous GPU staging (DMA)
    unsafe_copyto!(gpu_staging, 1, host_buf, offset + 1, n)
    # Step 2: Contiguous GPU staging → non-contiguous GPU view (GPU kernel)
    staging_view = reshape(@view(gpu_staging[1:n]), shape)
    dst_view = @view gpu_parent[dr_shifted[1], dr_shifted[2], dr_shifted[3]]
    copyto!(dst_view, staging_view)
    return n
end

function _unpack_halo_from_host!(cpu_parent::Array, dr_shifted, host_buf::Vector,
                                  offset::Int, shape, _gpu_staging=nothing)
    n = prod(shape)
    src_view = reshape(@view(host_buf[offset+1:offset+n]), shape)
    dst_view = @view cpu_parent[dr_shifted[1], dr_shifted[2], dr_shifted[3]]
    copyto!(dst_view, src_view)
    return n
end

"""
    _mpi_halo_tag(src_id, dst_id, comp_idx, total_chunks)

Compute a unique MPI tag for a halo exchange message.
"""
function _mpi_halo_tag(src_id::Int, dst_id::Int, comp_idx::Int, total_chunks::Int)
    return (src_id - 1) * total_chunks * 3 + (dst_id - 1) * 3 + comp_idx
end

"""
    _exchange_halos_mpi!(sim::SimulationData, components)

MPI halo exchange protocol (5 phases):
  Phase 1: GPU->host: copy halo faces into host staging buffers
  Phase 2: MPI_Isend/Irecv: post non-blocking sends and receives
  Phase 3: Local copies: do local (same-rank) halo copies while MPI is in flight
  Phase 4: MPI_Waitall: block until all sends/receives complete
  Phase 5: Host->GPU: copy received buffers into GPU ghost cells
"""
function _exchange_halos_mpi!(sim::SimulationData, components)
    rank = mpi_rank()
    assignment = sim.chunk_rank_assignment
    comm = mpi_comm()
    total_chunks = sim.chunk_plan.total_chunks

    # Phase 1: GPU -> host for remote sends
    for chunk in sim.chunk_data
        gv = chunk.spec.grid_volume
        for (ci, conn) in enumerate(chunk.halo_send)
            assignment[conn.dst_chunk_id] == rank && continue
            sbuf = chunk.halo_send_buffers[ci]
            gpu_staging = chunk.halo_send_gpu_staging[ci]
            offset = 0
            for comp in components
                src_f = _get_chunk_field(chunk, comp)
                isnothing(src_f) && continue
                src_parent = parent(src_f)
                src_dims = size(src_parent) .- 2
                sr = _component_send_range(src_dims, conn.axis, conn.src_range)
                sr_shifted = (sr[1] .+ 1, sr[2] .+ 1, sr[3] .+ 1)
                n = _pack_halo_to_host!(sbuf, offset, src_parent, sr_shifted, gpu_staging)
                offset += n
            end
        end
    end
    if backend_engine isa CUDABackend
        CUDA.synchronize()
    end

    # Phase 2: Post non-blocking MPI sends and receives
    requests = MPI.Request[]
    for chunk in sim.chunk_data
        for (ci, conn) in enumerate(chunk.halo_send)
            assignment[conn.dst_chunk_id] == rank && continue
            dst_rank = assignment[conn.dst_chunk_id]
            tag = _mpi_halo_tag(conn.src_chunk_id, conn.dst_chunk_id, 0, total_chunks)
            push!(requests, MPI.Isend(chunk.halo_send_buffers[ci], comm; dest=dst_rank, tag=tag))
        end
        for (ci, conn) in enumerate(chunk.halo_recv)
            assignment[conn.src_chunk_id] == rank && continue
            src_rank = assignment[conn.src_chunk_id]
            tag = _mpi_halo_tag(conn.src_chunk_id, conn.dst_chunk_id, 0, total_chunks)
            push!(requests, MPI.Irecv!(chunk.halo_recv_buffers[ci], comm; source=src_rank, tag=tag))
        end
    end

    # Phase 3: Local copies while MPI is in flight
    _exchange_halos_local!(sim, components)

    # Phase 4: Wait for all MPI operations
    if !isempty(requests)
        MPI.Waitall(requests)
    end

    # Phase 5: Host -> GPU for remote receives
    for chunk in sim.chunk_data
        gv = chunk.spec.grid_volume
        for (ci, conn) in enumerate(chunk.halo_recv)
            assignment[conn.src_chunk_id] == rank && continue
            rbuf = chunk.halo_recv_buffers[ci]
            gpu_staging = chunk.halo_recv_gpu_staging[ci]
            offset = 0
            for comp in components
                dst_f = _get_chunk_field(chunk, comp)
                isnothing(dst_f) && continue
                dst_parent = parent(dst_f)
                dst_dims = size(dst_parent) .- 2
                dr = _component_recv_range(dst_dims, conn.axis, conn.dst_range)
                dr_shifted = (dr[1] .+ 1, dr[2] .+ 1, dr[3] .+ 1)
                shape = (length(dr_shifted[1]), length(dr_shifted[2]), length(dr_shifted[3]))
                n = _unpack_halo_from_host!(dst_parent, dr_shifted, rbuf, offset, shape, gpu_staging)
                offset += n
            end
        end
    end
    if backend_engine isa CUDABackend
        CUDA.synchronize()
    end
end
