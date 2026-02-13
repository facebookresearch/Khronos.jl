# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Memory estimation for Khronos.jl simulations.
# Predicts total GPU memory required before allocation, mirroring the
# conditional allocation logic in Chunking.jl:805-882.

export estimate_memory

"""
    _count_field_arrays(pf::PhysicsFlags, source_components)

Count the number of real-valued field arrays needed for a chunk with the
given PhysicsFlags. Mirrors `_allocate_chunk_fields` in Chunking.jl.

Returns (primary, pml_auxiliary, source_fields) counts.
"""
function _count_field_arrays(pf::PhysicsFlags, source_components::Union{Vector{<:Field},Nothing})
    # Primary fields: E, H, B, D (always allocated, 12 arrays)
    primary = 12

    # PML auxiliary fields: C, U, W for both B and D
    has_pml = [pf.has_pml_x, pf.has_pml_y, pf.has_pml_z]

    pml_aux = 0
    for dir in 1:3
        # B auxiliary
        needs_c_b = pf.has_sigma_B && (has_pml[mod1(dir+1,3)] || has_pml[mod1(dir-1,3)])
        needs_u_b = has_pml[mod1(dir+1,3)]
        needs_w_b = has_pml[dir]
        pml_aux += needs_c_b + needs_u_b + needs_w_b

        # D auxiliary
        needs_c_d = pf.has_sigma_D && (has_pml[mod1(dir+1,3)] || has_pml[mod1(dir-1,3)])
        needs_u_d = has_pml[mod1(dir+1,3)]
        needs_w_d = has_pml[dir]
        pml_aux += needs_c_d + needs_u_d + needs_w_d
    end

    # Source fields (SB, SD)
    src_count = 0
    if pf.has_sources && !isnothing(source_components)
        e_comps = (Ex(), Ey(), Ez())
        h_comps = (Hx(), Hy(), Hz())
        for c in h_comps
            if c ∈ source_components
                src_count += 1
            end
        end
        for c in e_comps
            if c ∈ source_components
                src_count += 1
            end
        end
    end

    return (primary, pml_aux, src_count)
end

"""
    _count_geometry_arrays(pf::PhysicsFlags)

Count the number of geometry arrays (ε_inv, σ_D, μ_inv, σ_B) needed.
Each is per-component (x, y, z = 3 arrays per property).
"""
function _count_geometry_arrays(pf::PhysicsFlags)
    count = 0
    if pf.has_epsilon
        count += 3  # ε_inv_x, ε_inv_y, ε_inv_z
    end
    if pf.has_sigma_D
        count += 3  # σDx, σDy, σDz
    end
    if pf.has_mu
        count += 3  # μ_inv_x, μ_inv_y, μ_inv_z
    end
    if pf.has_sigma_B
        count += 3  # σBx, σBy, σBz
    end
    return count
end

"""
    _pml_sigma_bytes(pf::PhysicsFlags, Nx, Ny, Nz, elem_size)

Estimate bytes for 1D PML sigma profile arrays. These are small (O(N) per axis),
but included for completeness. Each enabled axis has 2 sigma arrays (for B and D),
each of length 2*N+1.
"""
function _pml_sigma_bytes(pf::PhysicsFlags, Nx::Int, Ny::Int, Nz::Int, elem_size::Int)
    bytes = 0
    if pf.has_pml_x
        bytes += 2 * (2 * Nx + 1) * elem_size  # σBx, σDx
    end
    if pf.has_pml_y
        bytes += 2 * (2 * Ny + 1) * elem_size  # σBy, σDy
    end
    if pf.has_pml_z
        bytes += 2 * (2 * Nz + 1) * elem_size  # σBz, σDz
    end
    return bytes
end

"""
    estimate_memory(sim::SimulationData; verbose::Bool=false)

Predict the total GPU memory required for a simulation **before** allocation.
Uses the chunk plan (or creates one) and PhysicsFlags to mirror the
conditional allocation logic in `_allocate_chunk_fields`.

Returns a NamedTuple:
  - `total_bytes`: Total estimated GPU memory
  - `per_chunk_bytes`: Vector of bytes per chunk
  - `field_bytes`: Total field array memory
  - `geometry_bytes`: Total geometry array memory
  - `monitor_bytes`: Total DFT monitor memory
  - `halo_bytes`: Total halo buffer memory
  - `recommended_gpus`: Minimum GPU count (at 85% utilization)
  - `device_memory`: Available memory per GPU (0 if no GPU)
"""
function estimate_memory(sim::SimulationData; verbose::Bool=false)
    # Get or create chunk plan
    chunk_plan = if !isnothing(sim.chunk_plan)
        sim.chunk_plan
    else
        plan_chunks(sim)
    end

    elem_size = sizeof(backend_number)
    complex_elem_size = 2 * elem_size

    # Determine source components (need this for source field allocation)
    source_components = sim.source_components
    if isnothing(source_components) && !isnothing(sim.sources)
        # Pre-compute source components like add_sources would
        source_components = Field[]
        for src in sim.sources
            for comp in get_source_components(src)
                if !(comp ∈ source_components)
                    push!(source_components, comp)
                end
            end
        end
    end

    total_field_bytes = 0
    total_geometry_bytes = 0
    total_halo_bytes = 0
    per_chunk_bytes = Int[]

    for spec in chunk_plan.chunks
        gv = spec.grid_volume
        pf = spec.physics
        Nx, Ny, Nz = gv.Nx, gv.Ny, max(1, gv.Nz)

        # Field arrays: each is (Nx+2) × (Ny+2) × (Nz+2) with ghost cells
        # Component voxel counts vary due to Yee stagger, but for estimation
        # we use the center count + 2 ghost cells as a representative size
        field_voxels = (Nx + 2) * (Ny + 2) * (Nz + 2)
        primary, pml_aux, src_count = _count_field_arrays(pf, source_components)
        chunk_field_bytes = (primary + pml_aux + src_count) * field_voxels * elem_size

        # Geometry arrays: sized to chunk without ghost cells
        geom_voxels = Nx * Ny * Nz
        geom_count = _count_geometry_arrays(pf)
        chunk_geom_bytes = geom_count * geom_voxels * elem_size

        # PML sigma profiles (1D, small)
        chunk_pml_bytes = _pml_sigma_bytes(pf, Nx, Ny, Nz, elem_size)

        chunk_total = chunk_field_bytes + chunk_geom_bytes + chunk_pml_bytes

        total_field_bytes += chunk_field_bytes
        total_geometry_bytes += chunk_geom_bytes + chunk_pml_bytes
        push!(per_chunk_bytes, chunk_total)
    end

    # Monitor memory: DFT monitors store complex arrays
    total_monitor_bytes = 0
    if !isnothing(sim.monitors)
        for mon in sim.monitors
            if mon isa DFTMonitor
                # Estimate monitor grid size from physical dimensions
                mon_Nx = max(1, floor(Int, mon.size[1] * sim.resolution))
                mon_Ny = max(1, floor(Int, mon.size[2] * sim.resolution))
                mon_Nz = max(1, (sim.ndims == 3 && mon.size[3] > 0) ?
                             floor(Int, mon.size[3] * sim.resolution) : 1)
                n_freq = length(mon.frequencies)
                # Complex field array + real scale array
                total_monitor_bytes += mon_Nx * mon_Ny * mon_Nz * n_freq * complex_elem_size
                total_monitor_bytes += mon_Nx * mon_Ny * mon_Nz * elem_size  # scale array
            end
        end
    end

    # Halo buffer memory (for multi-chunk or MPI)
    if chunk_plan.total_chunks > 1
        for (i, j, axis) in chunk_plan.adjacency
            gv_i = chunk_plan.chunks[i].grid_volume
            gv_j = chunk_plan.chunks[j].grid_volume

            # Face size along the split axis
            dims_i = [gv_i.Nx, gv_i.Ny, max(1, gv_i.Nz)]
            dims_j = [gv_j.Nx, gv_j.Ny, max(1, gv_j.Nz)]
            dims_i[axis] = 1
            dims_j[axis] = 1
            face_i = prod(dims_i)
            face_j = prod(dims_j)

            # Each connection needs send+recv buffers for 3 components × 2 directions
            total_halo_bytes += 2 * max(face_i, face_j) * 3 * elem_size
        end
    end

    total_bytes = total_field_bytes + total_geometry_bytes + total_monitor_bytes + total_halo_bytes

    # Query GPU memory
    device_memory = 0
    try
        if backend_engine isa CUDABackend
            dev = CUDA.device()
            device_memory = CUDA.totalmem(dev)
        end
    catch
        # No GPU available
    end

    recommended_gpus = if device_memory > 0
        ceil(Int, total_bytes / (0.85 * device_memory))
    else
        0
    end

    if verbose || (!is_distributed() || is_root())
        _print_memory_summary(
            total_bytes, per_chunk_bytes, total_field_bytes,
            total_geometry_bytes, total_monitor_bytes, total_halo_bytes,
            device_memory, recommended_gpus, chunk_plan,
        )
    end

    return (
        total_bytes = total_bytes,
        per_chunk_bytes = per_chunk_bytes,
        field_bytes = total_field_bytes,
        geometry_bytes = total_geometry_bytes,
        monitor_bytes = total_monitor_bytes,
        halo_bytes = total_halo_bytes,
        recommended_gpus = recommended_gpus,
        device_memory = device_memory,
    )
end

"""
    _print_memory_summary(...)

Print a formatted memory estimation summary.
"""
function _print_memory_summary(
    total_bytes, per_chunk_bytes, field_bytes,
    geometry_bytes, monitor_bytes, halo_bytes,
    device_memory, recommended_gpus, chunk_plan,
)
    _gb(b) = round(b / 1e9, digits=2)
    _mb(b) = round(b / 1e6, digits=1)

    @info("Memory Estimate:")
    @info("  Fields:       $(_gb(field_bytes)) GB")
    @info("  Geometry:     $(_gb(geometry_bytes)) GB")
    @info("  Monitors:     $(_gb(monitor_bytes)) GB")
    @info("  Halo buffers: $(_gb(halo_bytes)) GB")
    @info("  ─────────────────────────")
    @info("  Total:        $(_gb(total_bytes)) GB")

    if chunk_plan.total_chunks > 1
        max_chunk = maximum(per_chunk_bytes)
        min_chunk = minimum(per_chunk_bytes)
        @info("  Chunks:       $(chunk_plan.total_chunks) (max=$(_mb(max_chunk)) MB, min=$(_mb(min_chunk)) MB)")
    end

    if device_memory > 0
        @info("  GPU memory:   $(_gb(device_memory)) GB per device")
        @info("  Recommended:  $(recommended_gpus) GPU(s) (at 85% utilization)")
        if total_bytes > 0.85 * device_memory && recommended_gpus > 1
            @warn("Simulation requires $(recommended_gpus) GPUs. " *
                  "Use MPI with `mpirun -np $(recommended_gpus) julia ...`")
        end
    end
end
