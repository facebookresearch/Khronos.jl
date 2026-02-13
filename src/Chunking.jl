# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Material-based domain chunking for Khronos.jl.
# Splits the simulation domain into rectangular chunks based on material type
# and physics characteristics. Each chunk gets its own specialized kernel launch
# (leveraging the existing Nothing-dispatch pattern), its own field arrays, and
# halo regions for inter-chunk communication.
#
# Type definitions (PhysicsFlags, ChunkSpec, ChunkPlan, HaloConnection, ChunkData)
# live in DataStructures.jl so they are available to SimulationData.

# -------------------------------------------------------- #
# classify_region_physics -- determine PhysicsFlags for a region
# -------------------------------------------------------- #

"""
    _volumes_overlap(vol_a::Volume, vol_b::Volume)

Check if two axis-aligned volumes overlap.
"""
function _volumes_overlap(vol_a::Volume, vol_b::Volume)
    min_a = get_min_corner(vol_a)
    max_a = get_max_corner(vol_a)
    min_b = get_min_corner(vol_b)
    max_b = get_max_corner(vol_b)
    for d in 1:3
        if max_a[d] <= min_b[d] || max_b[d] <= min_a[d]
            return false
        end
    end
    return true
end

"""
    _shape_overlaps_volume(shape, vol::Volume)

Check if a GeometryPrimitives shape's bounding box overlaps a Volume.
"""
function _shape_overlaps_volume(shape, vol::Volume)
    b = bounds(shape)
    shape_min = b[1]
    shape_max = b[2]
    vol_min = get_min_corner(vol)
    vol_max = get_max_corner(vol)
    N = length(shape_min)
    for d in 1:N
        if shape_max[d] <= vol_min[d] || vol_max[d] <= shape_min[d]
            return false
        end
    end
    return true
end

"""
    _source_overlaps_volume(source, vol::Volume)

Check if a source's spatial volume overlaps a chunk volume.
For point sources (size=[0,0,0]), we check if the center is inside the volume.
"""
function _source_overlaps_volume(source, vol::Volume)
    src_vol = get_source_volume(source)
    # Point source: check if center is inside the volume
    if all(src_vol.size .== 0.0)
        min_v = get_min_corner(vol)
        max_v = get_max_corner(vol)
        c = src_vol.center
        for d in 1:3
            if c[d] < min_v[d] || c[d] > max_v[d]
                return false
            end
        end
        return true
    end
    return _volumes_overlap(src_vol, vol)
end

"""
    _monitor_overlaps_volume(monitor, vol::Volume)

Check if a monitor's spatial volume overlaps a chunk volume.
"""
function _monitor_overlaps_volume(monitor, vol::Volume)
    mon_center = monitor.center
    mon_size = monitor.size
    mon_vol = Volume(center = Real[mon_center[1], mon_center[2], mon_center[3]],
                     size = Real[mon_size[1], mon_size[2], mon_size[3]])
    # For zero-size dimensions, check point containment
    if all(mon_vol.size .== 0.0)
        min_v = get_min_corner(vol)
        max_v = get_max_corner(vol)
        c = mon_vol.center
        for d in 1:3
            if c[d] < min_v[d] || c[d] > max_v[d]
                return false
            end
        end
        return true
    end
    return _volumes_overlap(mon_vol, vol)
end

"""
    _pml_overlaps_chunk_axis(sim, chunk_gv::GridVolume, axis::Int)

Check if PML regions along a given axis overlap the chunk's grid indices.
PML regions are at the low and high ends of the domain along each axis.
"""
function _pml_overlaps_chunk_axis(sim::SimulationData, chunk_gv::GridVolume, axis::Int)
    isnothing(sim.boundaries) && return false
    length(sim.boundaries) < axis && return false

    pml_left = sim.boundaries[axis][1]
    pml_right = sim.boundaries[axis][2]
    (pml_left == 0.0 && pml_right == 0.0) && return false

    # Get grid spacings and domain sizes
    Δ = [sim.Δx, sim.Δy, sim.Δz]
    N_grid = [sim.Nx, sim.Ny, sim.Nz]

    # PML occupies voxels at the domain edges
    pml_left_end = (pml_left > 0.0) ? ceil(Int, pml_left / Δ[axis]) : 0
    pml_right_start = (pml_right > 0.0) ? N_grid[axis] - ceil(Int, pml_right / Δ[axis]) + 1 : N_grid[axis] + 1

    chunk_start = chunk_gv.start_idx[axis]
    chunk_end = chunk_gv.end_idx[axis]

    # Check if chunk overlaps left PML region
    if pml_left_end > 0 && chunk_start <= pml_left_end
        return true
    end

    # Check if chunk overlaps right PML region
    if pml_right_start <= N_grid[axis] && chunk_end >= pml_right_start
        return true
    end

    return false
end

"""
    classify_region_physics(sim, vol, geometry, boundaries; chunk_gv=nothing)

Determine `PhysicsFlags` for a rectangular sub-domain.
Uses bounding-box intersection for O(num_objects) fast rejection.

If `chunk_gv` is provided, it is used directly for PML overlap checks
instead of converting the volume to grid indices (avoiding float roundtrip errors).
"""
function classify_region_physics(
    sim::SimulationData,
    vol::Volume,
    geometry::Union{Vector{Object},Nothing},
    boundaries::Union{Vector{Vector{T}},Nothing};
    chunk_gv::Union{GridVolume,Nothing} = nothing,
) where {T}
    has_epsilon = false
    has_mu = false
    has_sigma_D = false
    has_sigma_B = false

    # Check geometry objects for material properties
    if !isnothing(geometry)
        for obj in geometry
            if !_shape_overlaps_volume(obj.shape, vol)
                continue
            end
            mat = obj.material
            if !isnothing(mat.ε) || !isnothing(mat.εx) || !isnothing(mat.εy) || !isnothing(mat.εz)
                has_epsilon = true
            end
            if !isnothing(mat.μ) || !isnothing(mat.μx) || !isnothing(mat.μy) || !isnothing(mat.μz)
                has_mu = true
            end
            if !isnothing(mat.σD) || !isnothing(mat.σDx) || !isnothing(mat.σDy) || !isnothing(mat.σDz)
                has_sigma_D = true
            end
            if !isnothing(mat.σB) || !isnothing(mat.σBx) || !isnothing(mat.σBy) || !isnothing(mat.σBz)
                has_sigma_B = true
            end
        end
    end

    # Check PML overlap (use provided GridVolume or compute from Volume)
    pml_gv = isnothing(chunk_gv) ? GridVolume(sim, vol, Center()) : chunk_gv
    has_pml_x = _pml_overlaps_chunk_axis(sim, pml_gv, 1)
    has_pml_y = _pml_overlaps_chunk_axis(sim, pml_gv, 2)
    has_pml_z = (sim.ndims == 3) ? _pml_overlaps_chunk_axis(sim, pml_gv, 3) : false

    # Check sources
    has_sources = false
    if !isnothing(sim.sources)
        for src in sim.sources
            if _source_overlaps_volume(src, vol)
                has_sources = true
                break
            end
        end
    end

    # Check monitors
    has_monitors = false
    if !isnothing(sim.monitors)
        for mon in sim.monitors
            if _monitor_overlaps_volume(mon, vol)
                has_monitors = true
                break
            end
        end
    end

    return PhysicsFlags(
        has_epsilon = has_epsilon,
        has_mu = has_mu,
        has_sigma_D = has_sigma_D,
        has_sigma_B = has_sigma_B,
        has_pml_x = has_pml_x,
        has_pml_y = has_pml_y,
        has_pml_z = has_pml_z,
        has_sources = has_sources,
        has_monitors = has_monitors,
    )
end

# -------------------------------------------------------- #
# Cost model (inspired by Meep's fragment_stats)
# -------------------------------------------------------- #

"""
    chunk_cost(physics::PhysicsFlags, num_voxels::Int)::Float64

Estimate computational cost for a chunk based on its physics and size.
"""
function chunk_cost(physics::PhysicsFlags, num_voxels::Int)::Float64
    cost = Float64(num_voxels)
    if has_any_pml(physics)
        cost *= 1.6   # PML ~60% more expensive
    end
    if physics.has_epsilon; cost *= 1.1; end
    if physics.has_sigma_D || physics.has_sigma_B; cost *= 1.2; end
    return cost
end

# -------------------------------------------------------- #
# Volume splitting utilities
# -------------------------------------------------------- #

"""
    _split_volume(vol::Volume, axis::Int, split_frac::Float64)

Split a volume into two halves along the given axis at the specified fraction
(0 to 1) of the volume's extent in that direction.
Returns (left_vol, right_vol).
"""
function _split_volume(vol::Volume, axis::Int, split_frac::Float64)
    min_c = get_min_corner(vol)
    max_c = get_max_corner(vol)
    split_coord = min_c[axis] + split_frac * vol.size[axis]

    left_center = copy(vol.center)
    left_size = copy(vol.size)
    left_center[axis] = (min_c[axis] + split_coord) / 2
    left_size[axis] = split_coord - min_c[axis]

    right_center = copy(vol.center)
    right_size = copy(vol.size)
    right_center[axis] = (split_coord + max_c[axis]) / 2
    right_size[axis] = max_c[axis] - split_coord

    return (
        Volume(center = left_center, size = left_size),
        Volume(center = right_center, size = right_size),
    )
end

"""
    _volume_voxels(sim::SimulationData, vol::Volume)::Int

Estimate the number of voxels in a volume.
"""
function _volume_voxels(sim::SimulationData, vol::Volume)::Int
    gv = GridVolume(sim, vol, Center())
    return max(1, gv.Nx * gv.Ny * max(1, gv.Nz))
end

"""
    _find_material_boundaries(sim::SimulationData, vol::Volume, axis::Int)

Find candidate split points along an axis where material boundaries occur.
Returns fractions (0 to 1) within the volume's extent along that axis.
"""
function _find_material_boundaries(sim::SimulationData, vol::Volume, axis::Int)
    fracs = Float64[]
    isnothing(sim.geometry) && return fracs

    vol_min = get_min_corner(vol)[axis]
    vol_max = get_max_corner(vol)[axis]
    vol_extent = vol_max - vol_min
    vol_extent <= 0 && return fracs

    for obj in sim.geometry
        b = bounds(obj.shape)
        shape_min = b[1][axis]
        shape_max = b[2][axis]
        # Candidate splits at shape boundaries that fall within the volume
        for coord in (shape_min, shape_max)
            if vol_min < coord < vol_max
                frac = (coord - vol_min) / vol_extent
                push!(fracs, frac)
            end
        end
    end

    # Also add PML boundaries as candidates
    if !isnothing(sim.boundaries) && length(sim.boundaries) >= axis
        Δ = [sim.Δx, sim.Δy, sim.Δz]
        pml_left = sim.boundaries[axis][1]
        pml_right = sim.boundaries[axis][2]
        if pml_left > 0
            coord = vol_min + pml_left
            if vol_min < coord < vol_max
                push!(fracs, (coord - vol_min) / vol_extent)
            end
        end
        if pml_right > 0
            coord = vol_max - pml_right
            if vol_min < coord < vol_max
                push!(fracs, (coord - vol_min) / vol_extent)
            end
        end
    end

    sort!(unique!(fracs))
    return fracs
end

"""
    _min_chunk_voxels(sim::SimulationData)::Int

Minimum number of voxels per chunk for GPU efficiency.
"""
_min_chunk_voxels(sim::SimulationData)::Int = 32^3

# -------------------------------------------------------- #
# BSP splitting
# -------------------------------------------------------- #

"""
    _find_best_split(sim::SimulationData, vol::Volume)

Find the best axis and split fraction that minimizes max(cost_left, cost_right).
Returns (best_axis, best_frac) or nothing if no valid split exists.
"""
function _find_best_split(sim::SimulationData, vol::Volume)
    best_cost = Inf
    best_axis = 0
    best_frac = 0.5
    min_voxels = _min_chunk_voxels(sim)
    ndims = sim.ndims

    for axis in 1:ndims
        # Skip axes with zero size
        vol.size[axis] <= 0 && continue

        # Candidate split fractions: material boundaries + midpoint
        candidates = _find_material_boundaries(sim, vol, axis)
        push!(candidates, 0.5)
        sort!(unique!(candidates))

        # 30% preference for longest axis (Meep convention)
        axis_weight = (axis == argmax(vol.size[1:ndims])) ? 0.7 : 1.0

        for frac in candidates
            left_vol, right_vol = _split_volume(vol, axis, frac)
            left_voxels = _volume_voxels(sim, left_vol)
            right_voxels = _volume_voxels(sim, right_vol)

            # Enforce minimum chunk size
            (left_voxels < min_voxels || right_voxels < min_voxels) && continue

            left_physics = classify_region_physics(sim, left_vol, sim.geometry, sim.boundaries)
            right_physics = classify_region_physics(sim, right_vol, sim.geometry, sim.boundaries)

            left_cost = chunk_cost(left_physics, left_voxels)
            right_cost = chunk_cost(right_physics, right_voxels)
            cost = max(left_cost, right_cost) * axis_weight

            if cost < best_cost
                best_cost = cost
                best_axis = axis
                best_frac = frac
            end
        end
    end

    best_axis == 0 && return nothing
    return (best_axis, best_frac)
end

"""
    _bsp_split(sim::SimulationData, vol::Volume, remaining::Int)

Recursively split a volume into chunks using BSP. Returns a vector of
(volume, physics) tuples.
"""
function _bsp_split(sim::SimulationData, vol::Volume, remaining::Int)
    if remaining <= 1
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
        return [(vol, physics)]
    end

    result = _find_best_split(sim, vol)
    if isnothing(result)
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
        return [(vol, physics)]
    end

    best_axis, best_frac = result
    left_vol, right_vol = _split_volume(vol, best_axis, best_frac)

    left_count = remaining >> 1
    right_count = remaining - left_count
    return vcat(
        _bsp_split(sim, left_vol, left_count),
        _bsp_split(sim, right_vol, right_count),
    )
end

"""
    _bsp_split_grid(sim, start_idx, end_idx, remaining)

Recursively split a grid index range into non-overlapping chunks using BSP.
Works entirely in grid index space to guarantee non-overlapping partitions.
Returns a vector of (start_idx, end_idx, Volume, PhysicsFlags) tuples.
"""
function _bsp_split_grid(
    sim::SimulationData,
    start_idx::Vector{Int},
    end_idx::Vector{Int},
    remaining::Int,
)
    # Convert grid range to continuous volume for physics classification
    vol = _grid_range_to_volume(sim, start_idx, end_idx)

    if remaining <= 1
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
        return [(copy(start_idx), copy(end_idx), vol, physics)]
    end

    # Find best split using continuous-space cost model
    result = _find_best_split(sim, vol)
    if isnothing(result)
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
        return [(copy(start_idx), copy(end_idx), vol, physics)]
    end

    best_axis, best_frac = result

    # Convert split fraction to grid index
    range_size = end_idx[best_axis] - start_idx[best_axis] + 1
    split_offset = max(1, round(Int, best_frac * range_size))
    split_idx = start_idx[best_axis] + split_offset - 1

    # Clamp to ensure at least 1 cell on each side
    split_idx = clamp(split_idx, start_idx[best_axis], end_idx[best_axis] - 1)

    # Left: start_idx to split_idx, Right: split_idx+1 to end_idx
    left_start = copy(start_idx)
    left_end = copy(end_idx)
    left_end[best_axis] = split_idx

    right_start = copy(start_idx)
    right_start[best_axis] = split_idx + 1
    right_end = copy(end_idx)

    left_count = remaining >> 1
    right_count = remaining - left_count

    return vcat(
        _bsp_split_grid(sim, left_start, left_end, left_count),
        _bsp_split_grid(sim, right_start, right_end, right_count),
    )
end

"""
    _grid_range_to_volume(sim, start_idx, end_idx)

Convert a grid index range to a continuous Volume.
"""
function _grid_range_to_volume(sim::SimulationData, start_idx::Vector{Int}, end_idx::Vector{Int})
    Δ = [sim.Δx, sim.Δy, sim.Δz]
    origin = get_component_origin(sim)

    min_corner = collect(origin) .+ (start_idx .- 1) .* Δ .- Δ ./ 2
    max_corner = collect(origin) .+ (end_idx .- 1) .* Δ .+ Δ ./ 2

    center = (min_corner .+ max_corner) ./ 2
    sz = max_corner .- min_corner

    return Volume(center = center, size = sz)
end

"""
    _compute_adjacency(chunks::Vector{ChunkSpec}, sim::SimulationData)

Compute adjacency between chunks. Two chunks are adjacent if they share
a face along one axis (their ranges touch in that axis and overlap in
the other two axes).
"""
function _compute_adjacency(chunks::Vector{ChunkSpec}, sim::SimulationData)
    adjacency = Tuple{Int,Int,Int}[]
    n = length(chunks)
    for i in 1:n
        for j in (i+1):n
            gv_i = chunks[i].grid_volume
            gv_j = chunks[j].grid_volume
            for axis in 1:sim.ndims
                # Check if chunks touch along this axis
                touches = (gv_i.end_idx[axis] == gv_j.start_idx[axis] - 1) ||
                          (gv_j.end_idx[axis] == gv_i.start_idx[axis] - 1)
                if !touches
                    continue
                end
                # Check overlap in other axes
                overlaps_in_other = true
                for other_axis in 1:sim.ndims
                    other_axis == axis && continue
                    if gv_i.end_idx[other_axis] < gv_j.start_idx[other_axis] ||
                       gv_j.end_idx[other_axis] < gv_i.start_idx[other_axis]
                        overlaps_in_other = false
                        break
                    end
                end
                if overlaps_in_other
                    push!(adjacency, (i, j, axis))
                    push!(chunks[i].neighbor_ids, j)
                    push!(chunks[j].neighbor_ids, i)
                end
            end
        end
    end
    return adjacency
end

# -------------------------------------------------------- #
# PML grid splitting (Meep-style)
# -------------------------------------------------------- #

"""
    _pml_grid_regions(sim::SimulationData)

Compute PML boundary grid indices and generate up to 3^ndims regions by taking
the Cartesian product of per-axis intervals:
  [1, pml_left_end], [pml_left_end+1, pml_right_start-1], [pml_right_start, N]

Follows Meep's `add_to_effort_volumes` approach: splits the domain at PML
boundaries first, producing isotropic regions where each has uniform per-direction
PML characteristics.

Returns Vector of (start_idx::Vector{Int}, end_idx::Vector{Int}) tuples.
"""
function _pml_grid_regions(sim::SimulationData)
    ndims = sim.ndims
    Δ = [sim.Δx, sim.Δy, sim.Δz]
    N_grid = [sim.Nx, sim.Ny, max(sim.Nz, 1)]

    # For each axis, compute the intervals
    # intervals[axis] is a Vector of (start, end) tuples
    intervals = Vector{Vector{Tuple{Int,Int}}}(undef, ndims)

    for axis in 1:ndims
        axis_intervals = Tuple{Int,Int}[]

        if !isnothing(sim.boundaries) && length(sim.boundaries) >= axis
            pml_left = sim.boundaries[axis][1]
            pml_right = sim.boundaries[axis][2]
        else
            pml_left = 0.0
            pml_right = 0.0
        end

        pml_left_end = (pml_left > 0.0) ? ceil(Int, pml_left / Δ[axis]) : 0
        pml_right_start = (pml_right > 0.0) ? N_grid[axis] - ceil(Int, pml_right / Δ[axis]) + 1 : N_grid[axis] + 1

        # Left PML interval: [1, pml_left_end]
        if pml_left_end >= 1
            push!(axis_intervals, (1, pml_left_end))
        end

        # Interior interval: [pml_left_end+1, pml_right_start-1]
        interior_start = pml_left_end + 1
        interior_end = pml_right_start - 1
        if interior_start <= interior_end
            push!(axis_intervals, (interior_start, interior_end))
        end

        # Right PML interval: [pml_right_start, N]
        if pml_right_start <= N_grid[axis]
            push!(axis_intervals, (pml_right_start, N_grid[axis]))
        end

        # Fallback: if no intervals were created (e.g., PML covers entire axis)
        if isempty(axis_intervals)
            push!(axis_intervals, (1, N_grid[axis]))
        end

        intervals[axis] = axis_intervals
    end

    # Take Cartesian product across axes
    regions = Tuple{Vector{Int},Vector{Int}}[]

    if ndims == 2
        for (sx, ex) in intervals[1]
            for (sy, ey) in intervals[2]
                push!(regions, ([sx, sy, 1], [ex, ey, max(sim.Nz, 1)]))
            end
        end
    else  # 3D
        for (sx, ex) in intervals[1]
            for (sy, ey) in intervals[2]
                for (sz, ez) in intervals[3]
                    push!(regions, ([sx, sy, sz], [ex, ey, ez]))
                end
            end
        end
    end

    return regions
end

"""
    _plan_chunks_pml_grid(sim::SimulationData)::ChunkPlan

Build a ChunkPlan from PML grid regions. Each region has uniform per-direction
PML characteristics, enabling Nothing-dispatch for PML auxiliary fields.
"""
function _plan_chunks_pml_grid(sim::SimulationData)::ChunkPlan
    regions = _pml_grid_regions(sim)

    specs = ChunkSpec[]
    for (id, (s_idx, e_idx)) in enumerate(regions)
        vol = _grid_range_to_volume(sim, s_idx, e_idx)
        Nx = e_idx[1] - s_idx[1] + 1
        Ny = e_idx[2] - s_idx[2] + 1
        Nz = e_idx[3] - s_idx[3] + 1
        gv = GridVolume(Center(), s_idx, e_idx, Nx, Ny, Nz)
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries; chunk_gv = gv)
        push!(specs, ChunkSpec(id, vol, gv, physics, Int[], 0))
    end

    adjacency = _compute_adjacency(specs, sim)
    return ChunkPlan(specs, adjacency, length(specs))
end

# -------------------------------------------------------- #
# plan_chunks -- main entry point
# -------------------------------------------------------- #

"""
    plan_chunks(sim::SimulationData)::ChunkPlan

Generate a chunk plan for the simulation.

If `sim.num_chunks` is `nothing`, returns a single-chunk plan (backward compat).
If `:auto` with PML boundaries, uses PML grid splitting for isotropic regions.
If `:auto` without PML, returns a single chunk.
If an `Int`, uses that number of chunks via BSP splitting.
"""
function plan_chunks(sim::SimulationData)::ChunkPlan
    nc = sim.num_chunks

    # :auto with PML -> PML grid split
    if nc === :auto && !isnothing(sim.boundaries)
        return _plan_chunks_pml_grid(sim)
    end

    target = _resolve_num_chunks(sim)

    if target <= 1
        # Single-chunk baseline
        vol = Volume(center = sim.cell_center, size = sim.cell_size)
        gv = GridVolume(sim, Center())
        physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
        spec = ChunkSpec(1, vol, gv, physics, Int[], 0)
        return ChunkPlan([spec], Tuple{Int,Int,Int}[], 1)
    end

    # BSP splitting in grid index space (guarantees non-overlapping partitions)
    global_start = [1, 1, 1]
    global_end = [sim.Nx, sim.Ny, max(sim.Nz, 1)]
    splits = _bsp_split_grid(sim, global_start, global_end, target)

    specs = ChunkSpec[]
    for (id, (s_idx, e_idx, vol, physics)) in enumerate(splits)
        Nx = e_idx[1] - s_idx[1] + 1
        Ny = e_idx[2] - s_idx[2] + 1
        Nz = e_idx[3] - s_idx[3] + 1
        gv = GridVolume(Center(), s_idx, e_idx, Nx, Ny, Nz)
        push!(specs, ChunkSpec(id, vol, gv, physics, Int[], 0))
    end

    adjacency = _compute_adjacency(specs, sim)
    return ChunkPlan(specs, adjacency, length(specs))
end

"""
    _resolve_num_chunks(sim::SimulationData)::Int

Resolve the target number of chunks from the user specification.
"""
function _resolve_num_chunks(sim::SimulationData)::Int
    nc = sim.num_chunks
    isnothing(nc) && return 1
    nc isa Int && return max(1, nc)
    if nc === :auto
        return _auto_chunk_count(sim)
    end
    return 1
end

"""
    _auto_chunk_count(sim::SimulationData)::Int

Automatically determine chunk count based on domain size and materials.
"""
function _auto_chunk_count(sim::SimulationData)::Int
    total_voxels = sim.Nx * sim.Ny * max(sim.Nz, 1)
    min_per_chunk = _min_chunk_voxels(sim)
    max_from_size = max(1, total_voxels ÷ min_per_chunk)
    num_regions = _count_distinct_material_regions(sim)
    return clamp(num_regions + 1, 1, max_from_size)
end

"""
    _count_distinct_material_regions(sim::SimulationData)::Int

Count the number of distinct geometry objects (rough proxy for material regions).
"""
function _count_distinct_material_regions(sim::SimulationData)::Int
    isnothing(sim.geometry) && return 0
    return length(sim.geometry)
end

# -------------------------------------------------------- #
# Per-chunk field allocation
# -------------------------------------------------------- #

"""
    _get_chunk_component_voxel_count(sim, component, chunk_gv)

Get the voxel count for a field component within a chunk's grid volume,
accounting for Yee stagger. The chunk_gv uses Center() component; we compute
the appropriate counts for the requested field component.
"""
function _get_chunk_component_voxel_count(sim::SimulationData, component::Field, chunk_gv::GridVolume)
    # Full-domain component counts
    full = get_component_voxel_count(sim, component)
    # Center counts = [Nx, Ny, Nz]
    center = [chunk_gv.Nx, chunk_gv.Ny, chunk_gv.Nz]

    # The stagger offset: difference between component's full-domain count and Nx/Ny/Nz
    offset = full .- [sim.Nx, sim.Ny, sim.Nz]
    # Apply same offset to chunk
    return center .+ offset
end

"""
    _get_chunk_component_gv(sim, chunk_gv, component)

Construct a component-specific GridVolume for a chunk, directly from the chunk's
Center grid_volume. This avoids float-to-int roundtrip errors from converting
physical coordinates back to grid indices.
"""
function _get_chunk_component_gv(sim::SimulationData, chunk_gv::GridVolume, component::Field)
    dims = _get_chunk_component_voxel_count(sim, component, chunk_gv)
    return GridVolume(
        component = component,
        start_idx = copy(chunk_gv.start_idx),
        end_idx = chunk_gv.start_idx .+ dims .- 1,
        Nx = dims[1],
        Ny = dims[2],
        Nz = dims[3],
    )
end

"""
    _initialize_chunk_field_array(sim, component, chunk_gv)

Allocate a field array sized for a chunk, with +2 ghost cells per dimension.
"""
function _initialize_chunk_field_array(sim::SimulationData, component::Field, chunk_gv::GridVolume)
    # P.5: Allocate raw GPU array without OffsetArray wrapping.
    # Ghost cells are at raw indices 1 and N+2; interior at 2:(N+1).
    # Kernels use shifted indices (ix+1, iy+1, iz+1) to access interior cells.
    dims = _get_chunk_component_voxel_count(sim, component, chunk_gv) .+ 2
    return KernelAbstractions.zeros(backend_engine, backend_number, dims...)
end

"""
    _allocate_chunk_fields(sim, chunk_spec)

Allocate field arrays for a chunk based on its PhysicsFlags.
Returns a Fields struct with arrays sized to the chunk.
"""
function _allocate_chunk_fields(sim::SimulationData, spec::ChunkSpec)
    gv = spec.grid_volume
    pf = spec.physics

    # Helper: allocate if condition is true
    alloc(comp, cond) = cond ? _initialize_chunk_field_array(sim, comp, gv) : nothing

    # Primary fields are always allocated
    fEx = _initialize_chunk_field_array(sim, Ex(), gv)
    fEy = _initialize_chunk_field_array(sim, Ey(), gv)
    fEz = _initialize_chunk_field_array(sim, Ez(), gv)
    fHx = _initialize_chunk_field_array(sim, Hx(), gv)
    fHy = _initialize_chunk_field_array(sim, Hy(), gv)
    fHz = _initialize_chunk_field_array(sim, Hz(), gv)
    fBx = _initialize_chunk_field_array(sim, Bx(), gv)
    fBy = _initialize_chunk_field_array(sim, By(), gv)
    fBz = _initialize_chunk_field_array(sim, Bz(), gv)
    fDx = _initialize_chunk_field_array(sim, Dx(), gv)
    fDy = _initialize_chunk_field_array(sim, Dy(), gv)
    fDz = _initialize_chunk_field_array(sim, Dz(), gv)

    # B auxiliary fields: PML-driven
    has_pml = [pf.has_pml_x, pf.has_pml_y, pf.has_pml_z]
    # C fields: need conductivity AND PML in next or prev direction
    needs_c_b(dir) = pf.has_sigma_B && (has_pml[mod1(dir+1,3)] || has_pml[mod1(dir-1,3)])
    # U fields: need PML in next direction
    needs_u_b(dir) = has_pml[mod1(dir+1,3)]
    # W fields: need PML in own direction
    needs_w_b(dir) = has_pml[dir]

    fCBx = alloc(Bx(), needs_c_b(1))
    fCBy = alloc(By(), needs_c_b(2))
    fCBz = alloc(Bz(), needs_c_b(3))
    fUBx = alloc(Bx(), needs_u_b(1))
    fUBy = alloc(By(), needs_u_b(2))
    fUBz = alloc(Bz(), needs_u_b(3))
    fWBx = alloc(Bx(), needs_w_b(1))
    fWBy = alloc(By(), needs_w_b(2))
    fWBz = alloc(Bz(), needs_w_b(3))

    # D auxiliary fields: PML-driven
    needs_c_d(dir) = pf.has_sigma_D && (has_pml[mod1(dir+1,3)] || has_pml[mod1(dir-1,3)])
    needs_u_d(dir) = has_pml[mod1(dir+1,3)]
    needs_w_d(dir) = has_pml[dir]

    fCDx = alloc(Dx(), needs_c_d(1))
    fCDy = alloc(Dy(), needs_c_d(2))
    fCDz = alloc(Dz(), needs_c_d(3))
    fUDx = alloc(Dx(), needs_u_d(1))
    fUDy = alloc(Dy(), needs_u_d(2))
    fUDz = alloc(Dz(), needs_u_d(3))
    fWDx = alloc(Dx(), needs_w_d(1))
    fWDy = alloc(Dy(), needs_w_d(2))
    fWDz = alloc(Dz(), needs_w_d(3))

    # Source fields
    fSBx = alloc(Hx(), pf.has_sources && Hx() ∈ sim.source_components)
    fSBy = alloc(Hy(), pf.has_sources && Hy() ∈ sim.source_components)
    fSBz = alloc(Hz(), pf.has_sources && Hz() ∈ sim.source_components)
    fSDx = alloc(Ex(), pf.has_sources && Ex() ∈ sim.source_components)
    fSDy = alloc(Ey(), pf.has_sources && Ey() ∈ sim.source_components)
    fSDz = alloc(Ez(), pf.has_sources && Ez() ∈ sim.source_components)

    return Fields{AbstractArray}(
        fEx=fEx, fEy=fEy, fEz=fEz,
        fHx=fHx, fHy=fHy, fHz=fHz,
        fBx=fBx, fBy=fBy, fBz=fBz,
        fDx=fDx, fDy=fDy, fDz=fDz,
        fCBx=fCBx, fCBy=fCBy, fCBz=fCBz,
        fUBx=fUBx, fUBy=fUBy, fUBz=fUBz,
        fWBx=fWBx, fWBy=fWBy, fWBz=fWBz,
        fSBx=fSBx, fSBy=fSBy, fSBz=fSBz,
        fCDx=fCDx, fCDy=fCDy, fCDz=fCDz,
        fUDx=fUDx, fUDy=fUDy, fUDz=fUDz,
        fWDx=fWDx, fWDy=fWDy, fWDz=fWDz,
        fSDx=fSDx, fSDy=fSDy, fSDz=fSDz,
    )
end

# -------------------------------------------------------- #
# Per-chunk geometry voxelization
# -------------------------------------------------------- #

"""
    _init_chunk_geometry(sim, spec, geometry)

Voxelize geometry within a chunk's bounds. Chunks without epsilon/mu get
scalar GeometryData.
"""
function _init_chunk_geometry(sim::SimulationData, spec::ChunkSpec, geometry::Union{Vector{Object},Nothing})
    pf = spec.physics
    if !pf.has_epsilon && !pf.has_mu && !pf.has_sigma_D && !pf.has_sigma_B
        return GeometryData{backend_number,backend_array}(ε_inv = one(backend_number), μ_inv = one(backend_number))
    end

    if isnothing(geometry) || isempty(geometry)
        return GeometryData{backend_number,backend_array}(ε_inv = one(backend_number), μ_inv = one(backend_number))
    end

    chunk_gv = spec.grid_volume
    ndims = sim.ndims

    # Allocate arrays sized to the chunk
    alloc(comp, need) = need ? zeros(backend_number, _get_chunk_component_voxel_count(sim, comp, chunk_gv)[1:ndims]...) : nothing

    ε_inv_x = alloc(Ex(), needs_perm(geometry, Ex()))
    ε_inv_y = alloc(Ey(), needs_perm(geometry, Ey()))
    ε_inv_z = alloc(Ez(), needs_perm(geometry, Ez()))
    σDx = alloc(Ex(), needs_conductivities(geometry, Dx()))
    σDy = alloc(Ey(), needs_conductivities(geometry, Dy()))
    σDz = alloc(Ez(), needs_conductivities(geometry, Dz()))
    μ_inv_x = alloc(Hx(), needs_perm(geometry, Hx()))
    μ_inv_y = alloc(Hy(), needs_perm(geometry, Hy()))
    μ_inv_z = alloc(Hz(), needs_perm(geometry, Hz()))
    σBx = alloc(Hx(), needs_conductivities(geometry, Bx()))
    σBy = alloc(Hy(), needs_conductivities(geometry, By()))
    σBz = alloc(Hz(), needs_conductivities(geometry, Bz()))

    # Voxelize within chunk bounds
    # Use grid-index-derived GridVolumes to avoid float roundtrip errors
    components = (
        (_get_chunk_component_gv(sim, chunk_gv, Ex()), Dx(), ε_inv_x, σDx),
        (_get_chunk_component_gv(sim, chunk_gv, Hx()), Bx(), μ_inv_x, σBx),
        (_get_chunk_component_gv(sim, chunk_gv, Ey()), Dy(), ε_inv_y, σDy),
        (_get_chunk_component_gv(sim, chunk_gv, Hy()), By(), μ_inv_y, σBy),
        (_get_chunk_component_gv(sim, chunk_gv, Ez()), Dz(), ε_inv_z, σDz),
        (_get_chunk_component_gv(sim, chunk_gv, Hz()), Bz(), μ_inv_z, σBz),
    )

    tasks = Vector{Task}(undef, length(components))
    for (ci, (gv, f, perm_arr, σ_arr)) in enumerate(components)
        xs, ys, zs = _precompute_coords(sim, gv)
        tasks[ci] = Threads.@spawn begin
            if ndims == 3
                _rasterize_geometry_3d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)
            else
                _rasterize_geometry_2d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)
            end
        end
    end
    for t in tasks
        wait(t)
    end

    return GeometryData{backend_number,backend_array}(
        ε_inv_x = isnothing(ε_inv_x) ? ε_inv_x : backend_array(ε_inv_x),
        ε_inv_y = isnothing(ε_inv_y) ? ε_inv_y : backend_array(ε_inv_y),
        ε_inv_z = isnothing(ε_inv_z) ? ε_inv_z : backend_array(ε_inv_z),
        σDx = isnothing(σDx) ? σDx : backend_array(σDx),
        σDy = isnothing(σDy) ? σDy : backend_array(σDy),
        σDz = isnothing(σDz) ? σDz : backend_array(σDz),
        μ_inv = one(backend_number),
        σBx = isnothing(σBx) ? σBx : backend_array(σBx),
        σBy = isnothing(σBy) ? σBy : backend_array(σBy),
        σBz = isnothing(σBz) ? σBz : backend_array(σBz),
    )
end

# -------------------------------------------------------- #
# Per-chunk boundary data
# -------------------------------------------------------- #

"""
    _slice_pml_sigma(σ_global, N_local, start_idx)

Create a chunk-local PML sigma array from the global sigma array.
The kernel accesses sigma via `get_σ(σ, ix) = σ[2*ix-1]` using chunk-local
indices. For a chunk starting at global index `start_idx`, local index `i`
corresponds to global index `i + start_idx - 1`, so we need:
    σ_local[2*i - 1] = σ_global[2*(i + start_idx - 1) - 1]
"""
function _slice_pml_sigma(σ_global, N_local::Int, start_idx::Int)
    isnothing(σ_global) && return nothing

    σ_cpu = Array(σ_global)
    σ_local = zeros(eltype(σ_cpu), 2 * N_local + 1)
    for i = 1:N_local
        global_sigma_idx = 2 * (i + start_idx - 1) - 1
        if 1 <= global_sigma_idx <= length(σ_cpu)
            σ_local[2*i-1] = σ_cpu[global_sigma_idx]
        end
    end
    return backend_array(σ_local)
end

"""
    _init_chunk_boundaries(sim, spec)

Create boundary data for a chunk. Non-PML chunks get all-nothing BoundaryData.
For PML chunks, create chunk-local sigma arrays so that kernel-local indices
map to the correct global PML conductivity values.
"""
function _init_chunk_boundaries(sim::SimulationData, spec::ChunkSpec)
    pf = spec.physics
    if !has_any_pml(pf)
        return BoundaryData{backend_array}()
    end

    gv = spec.grid_volume

    return BoundaryData{backend_array}(
        σBx = pf.has_pml_x ? _slice_pml_sigma(sim.boundary_data.σBx, gv.Nx, gv.start_idx[1]) : nothing,
        σBy = pf.has_pml_y ? _slice_pml_sigma(sim.boundary_data.σBy, gv.Ny, gv.start_idx[2]) : nothing,
        σBz = pf.has_pml_z ? _slice_pml_sigma(sim.boundary_data.σBz, gv.Nz, gv.start_idx[3]) : nothing,
        σDx = pf.has_pml_x ? _slice_pml_sigma(sim.boundary_data.σDx, gv.Nx, gv.start_idx[1]) : nothing,
        σDy = pf.has_pml_y ? _slice_pml_sigma(sim.boundary_data.σDy, gv.Ny, gv.start_idx[2]) : nothing,
        σDz = pf.has_pml_z ? _slice_pml_sigma(sim.boundary_data.σDz, gv.Nz, gv.start_idx[3]) : nothing,
    )
end

# -------------------------------------------------------- #
# Per-chunk source assignment
# -------------------------------------------------------- #

"""
    _assign_sources_to_chunk(sim, spec)

Find sources that overlap this chunk and create chunk-local copies with
amplitude data clipped to the chunk boundaries.
For single-chunk, this returns all sources unchanged.
For multi-chunk, source amplitude arrays are sliced to only cover the
overlap region, and the GridVolume is adjusted accordingly.
"""
function _assign_sources_to_chunk(sim::SimulationData{N,T,CN,CT,BT}, spec::ChunkSpec) where {N,T,CN,CT,BT}
    isnothing(sim.source_data) && return SourceData{CT}[]

    if sim.chunk_plan.total_chunks == 1
        return sim.source_data
    end

    chunk_sources = SourceData{CT}[]
    for src in sim.source_data
        if !_source_overlaps_volume_by_gv(src.gv, spec.grid_volume)
            continue
        end

        # Compute the chunk's component-specific end index.
        # The chunk's grid_volume uses Center(), but the source uses a specific
        # component (e.g., Ex) which may have +1 voxels in staggered dimensions.
        chunk_comp_dims = _get_chunk_component_voxel_count(sim, src.gv.component, spec.grid_volume)
        chunk_comp_end = [
            spec.grid_volume.start_idx[1] + chunk_comp_dims[1] - 1,
            spec.grid_volume.start_idx[2] + chunk_comp_dims[2] - 1,
            spec.grid_volume.start_idx[3] + chunk_comp_dims[3] - 1,
        ]

        # Compute the overlap region in global grid indices
        overlap_start = [
            max(src.gv.start_idx[1], spec.grid_volume.start_idx[1]),
            max(src.gv.start_idx[2], spec.grid_volume.start_idx[2]),
            max(src.gv.start_idx[3], spec.grid_volume.start_idx[3]),
        ]
        overlap_end = [
            min(src.gv.end_idx[1], chunk_comp_end[1]),
            min(src.gv.end_idx[2], chunk_comp_end[2]),
            min(src.gv.end_idx[3], chunk_comp_end[3]),
        ]

        # Verify overlap is valid (could become invalid after stagger adjustment)
        any(overlap_end .< overlap_start) && continue

        # Convert global overlap indices to source-local array indices
        # Source amplitude_data is indexed 1:Nx, 1:Ny, 1:Nz
        # where index 1 corresponds to src.gv.start_idx
        local_start = overlap_start .- src.gv.start_idx .+ 1
        local_end = overlap_end .- src.gv.start_idx .+ 1

        # Extract the clipped amplitude sub-array
        clipped_amplitude = src.amplitude_data[
            local_start[1]:local_end[1],
            local_start[2]:local_end[2],
            local_start[3]:local_end[3],
        ]

        # Create a new GridVolume for the clipped region
        clipped_gv = GridVolume(
            component = src.gv.component,
            start_idx = overlap_start,
            end_idx = overlap_end,
            Nx = overlap_end[1] - overlap_start[1] + 1,
            Ny = overlap_end[2] - overlap_start[2] + 1,
            Nz = overlap_end[3] - overlap_start[3] + 1,
        )

        # Create a new SourceData with the clipped data
        clipped_src = SourceData{CT}(
            amplitude_data = clipped_amplitude,
            time_src = src.time_src,
            gv = clipped_gv,
            component = src.component,
        )
        push!(chunk_sources, clipped_src)
    end
    return chunk_sources
end

"""
    _source_overlaps_volume_by_gv(src_gv, chunk_gv)

Check if a source's grid volume overlaps a chunk's grid volume.
"""
function _source_overlaps_volume_by_gv(src_gv::GridVolume, chunk_gv::GridVolume)
    for d in 1:3
        if src_gv.end_idx[d] < chunk_gv.start_idx[d] || chunk_gv.end_idx[d] < src_gv.start_idx[d]
            return false
        end
    end
    return true
end

# -------------------------------------------------------- #
# create_all_chunks -- build ChunkData from sim data
# -------------------------------------------------------- #

"""
    create_all_chunks(sim::SimulationData)

Create ChunkData objects for each chunk in the plan. For the single-chunk
case, the chunk wraps the sim's existing data. For multi-chunk, each chunk
gets its own allocated fields, geometry, and boundaries.
"""
function create_all_chunks(sim::SimulationData{N,T,CN,CT,BT}) where {N,T,CN,CT,BT}
    chunks = ChunkData{N,T,CT,BT}[]
    single_chunk = sim.chunk_plan.total_chunks == 1

    for spec in sim.chunk_plan.chunks
        ndrange = (spec.grid_volume.Nx, spec.grid_volume.Ny, max(1, spec.grid_volume.Nz))

        if single_chunk
            # Single chunk: wrap existing data (no copy)
            fields = sim.fields
            geom = sim.geometry_data
            bnd = sim.boundary_data
            src = isnothing(sim.source_data) ? SourceData{CT}[] : sim.source_data
            mon = sim.monitor_data
        else
            # Multi-chunk: allocate per-chunk data
            fields = _allocate_chunk_fields(sim, spec)
            geom = _init_chunk_geometry(sim, spec, sim.geometry)
            bnd = _init_chunk_boundaries(sim, spec)
            src = _assign_sources_to_chunk(sim, spec)
            mon = MonitorData[]  # Monitors stay on sim for now
        end

        chunk = ChunkData{N,T,CT,BT}(
            spec, fields, geom, bnd, src, mon,
            HaloConnection[], HaloConnection[],
            AbstractArray[], AbstractArray[],
            AbstractArray[], AbstractArray[],
            ndrange,
        )
        push!(chunks, chunk)
    end
    return chunks
end

# -------------------------------------------------------- #
# create_local_chunks -- MPI: only allocate chunks for this rank
# -------------------------------------------------------- #

"""
    create_local_chunks(sim::SimulationData)

Like `create_all_chunks` but only allocates chunks assigned to the current
MPI rank. Used in distributed mode.
"""
function create_local_chunks(sim::SimulationData{N,T,CN,CT,BT}) where {N,T,CN,CT,BT}
    rank = mpi_rank()
    assignment = sim.chunk_rank_assignment
    chunks = ChunkData{N,T,CT,BT}[]

    for spec in sim.chunk_plan.chunks
        assignment[spec.id] != rank && continue
        ndrange = (spec.grid_volume.Nx, spec.grid_volume.Ny, max(1, spec.grid_volume.Nz))

        fields = _allocate_chunk_fields(sim, spec)
        geom = _init_chunk_geometry(sim, spec, sim.geometry)
        bnd = _init_chunk_boundaries(sim, spec)
        src = _assign_sources_to_chunk(sim, spec)
        mon = MonitorData[]

        chunk = ChunkData{N,T,CT,BT}(
            spec, fields, geom, bnd, src, mon,
            HaloConnection[], HaloConnection[],
            AbstractArray[], AbstractArray[],
            AbstractArray[], AbstractArray[],
            ndrange,
        )
        push!(chunks, chunk)
    end
    return chunks
end

# -------------------------------------------------------- #
# Halo exchange
# -------------------------------------------------------- #

"""
    _compute_interior_face_range(chunk_gv, axis, side)

Compute the range of indices for the interior face of a chunk along an axis.
:upper = last layer, :lower = first layer.
"""
function _compute_interior_face_range(chunk_gv::GridVolume, axis::Int, side::Symbol)
    ranges = [1:chunk_gv.Nx, 1:chunk_gv.Ny, 1:max(1, chunk_gv.Nz)]
    if side == :upper
        ranges[axis] = chunk_gv.end_idx[axis]:chunk_gv.end_idx[axis]
    else
        ranges[axis] = chunk_gv.start_idx[axis]:chunk_gv.start_idx[axis]
    end
    return (ranges[1], ranges[2], ranges[3])
end

"""
    _local_chunk_index(sim::SimulationData, chunk_id::Int)

Find the index of a chunk in sim.chunk_data by its spec id.
Returns nothing if the chunk is not local (e.g., on another MPI rank).
"""
function _local_chunk_index(sim::SimulationData, chunk_id::Int)
    for (i, chunk) in enumerate(sim.chunk_data)
        chunk.spec.id == chunk_id && return i
    end
    return nothing
end

"""
    connect_chunks!(sim::SimulationData)

Build halo connections between adjacent chunks and allocate send/recv buffers.
No-op for single-chunk plans.
"""
function connect_chunks!(sim::SimulationData)
    isnothing(sim.chunk_data) && return
    length(sim.chunk_data) <= 1 && !is_distributed() && return

    for (i, j, axis) in sim.chunk_plan.adjacency
        gv_i = sim.chunk_plan.chunks[i].grid_volume
        gv_j = sim.chunk_plan.chunks[j].grid_volume

        # Forward connection: i -> j (i's upper face -> j's ghost at lower)
        src_fwd = _make_send_range(gv_i, axis, :upper)
        dst_fwd = _make_recv_range(gv_j, axis, :lower)
        conn_fwd = HaloConnection(i, j, axis, src_fwd, dst_fwd)

        # Reverse connection: j -> i
        src_rev = _make_send_range(gv_j, axis, :lower)
        dst_rev = _make_recv_range(gv_i, axis, :upper)
        conn_rev = HaloConnection(j, i, axis, src_rev, dst_rev)

        # In distributed mode, only add connections where at least one
        # endpoint is local
        local_i = _local_chunk_index(sim, i)
        local_j = _local_chunk_index(sim, j)

        if !isnothing(local_i)
            push!(sim.chunk_data[local_i].halo_send, conn_fwd)
            push!(sim.chunk_data[local_i].halo_recv, conn_rev)
        end
        if !isnothing(local_j)
            push!(sim.chunk_data[local_j].halo_recv, conn_fwd)
            push!(sim.chunk_data[local_j].halo_send, conn_rev)
        end
    end
    return
end

"""
    _make_send_range(gv, axis, side)

Compute the NTuple{3,UnitRange} for extracting a 1-cell-thick face from a
chunk's interior. Uses local indices (1-based within the chunk).
"""
function _make_send_range(gv::GridVolume, axis::Int, side::Symbol)
    dims = [gv.Nx, gv.Ny, max(1, gv.Nz)]
    ranges = [1:dims[1], 1:dims[2], 1:dims[3]]
    if side == :upper
        ranges[axis] = dims[axis]:dims[axis]
    else  # :lower
        ranges[axis] = 1:1
    end
    return (ranges[1], ranges[2], ranges[3])
end

"""
    _make_recv_range(gv, axis, side)

Compute the NTuple{3,UnitRange} for writing into a chunk's ghost cells.
Ghost cells are at index 0 (lower) or N+1 (upper) in the OffsetArray.
"""
function _make_recv_range(gv::GridVolume, axis::Int, side::Symbol)
    dims = [gv.Nx, gv.Ny, max(1, gv.Nz)]
    ranges = [1:dims[1], 1:dims[2], 1:dims[3]]
    if side == :lower
        ranges[axis] = 0:0  # ghost cell below
    else  # :upper
        n = dims[axis]
        ranges[axis] = (n+1):(n+1)  # ghost cell above
    end
    return (ranges[1], ranges[2], ranges[3])
end

"""
    _get_chunk_field(chunk, component)

Get a specific field array from a chunk's Fields struct.
"""
_get_chunk_field(chunk::ChunkData, ::Ex) = chunk.fields.fEx
_get_chunk_field(chunk::ChunkData, ::Ey) = chunk.fields.fEy
_get_chunk_field(chunk::ChunkData, ::Ez) = chunk.fields.fEz
_get_chunk_field(chunk::ChunkData, ::Hx) = chunk.fields.fHx
_get_chunk_field(chunk::ChunkData, ::Hy) = chunk.fields.fHy
_get_chunk_field(chunk::ChunkData, ::Hz) = chunk.fields.fHz
_get_chunk_field(chunk::ChunkData, ::Bx) = chunk.fields.fBx
_get_chunk_field(chunk::ChunkData, ::By) = chunk.fields.fBy
_get_chunk_field(chunk::ChunkData, ::Bz) = chunk.fields.fBz
_get_chunk_field(chunk::ChunkData, ::Dx) = chunk.fields.fDx
_get_chunk_field(chunk::ChunkData, ::Dy) = chunk.fields.fDy
_get_chunk_field(chunk::ChunkData, ::Dz) = chunk.fields.fDz

"""
    _field_components_for_group(group::Symbol)

Return the field component instances for a given field group.
"""
function _field_components_for_group(group::Symbol)
    if group == :B
        return (Bx(), By(), Bz())
    elseif group == :H
        return (Hx(), Hy(), Hz())
    elseif group == :D
        return (Dx(), Dy(), Dz())
    elseif group == :E
        return (Ex(), Ey(), Ez())
    else
        error("Unknown field group: $group")
    end
end

"""
    _halo_copy!(dst_parent, dr, src_parent, sr)

Copy a halo region from src_parent to dst_parent. On GPU (CuArray), uses
CUDA.unsafe_copy3d! which maps to cuMemcpy3DAsync (DMA engine, no kernel launch).
On CPU (Array), falls back to @views broadcast.

dr and sr are tuples of UnitRange{Int} indexing into the parent arrays.
"""
function _halo_copy!(dst_parent::CuArray{T}, dr, src_parent::CuArray{T}, sr) where T
    width  = length(sr[1])
    height = length(sr[2])
    depth  = length(sr[3])

    src_Nx = size(src_parent, 1)
    src_Ny = size(src_parent, 2)
    dst_Nx = size(dst_parent, 1)
    dst_Ny = size(dst_parent, 2)

    CUDA.unsafe_copy3d!(
        pointer(dst_parent), CUDA.DeviceMemory,
        pointer(src_parent), CUDA.DeviceMemory,
        width, height, depth;
        srcPos = (first(sr[1]), first(sr[2]), first(sr[3])),
        dstPos = (first(dr[1]), first(dr[2]), first(dr[3])),
        srcPitch = src_Nx * sizeof(T),
        srcHeight = src_Ny,
        dstPitch = dst_Nx * sizeof(T),
        dstHeight = dst_Ny,
        async = true,
    )
end

function _halo_copy!(dst_parent::Array, dr, src_parent::Array, sr)
    @views dst_parent[dr[1], dr[2], dr[3]] .= src_parent[sr[1], sr[2], sr[3]]
end

# -------------------------------------------------------- #
# Precomputed halo exchange for fast runtime execution
# -------------------------------------------------------- #

"""
    HaloCopyOp{T}

Precomputed halo copy operation. Stores all parameters needed for CUDA.unsafe_copy3d!
so that runtime exchange avoids field lookups, size computations, and range calculations.
"""
struct HaloCopyOp{T}
    dst_ptr::CUDA.CuPtr{T}
    src_ptr::CUDA.CuPtr{T}
    width::Int
    height::Int
    depth::Int
    src_pos::NTuple{3,Int}
    dst_pos::NTuple{3,Int}
    src_pitch::Int
    src_height::Int
    dst_pitch::Int
    dst_height::Int
end

"""
    precompute_halo_ops!(sim::SimulationData)

Precompute all halo copy operations for H and E field groups.
Called once during prepare_simulation! to avoid per-step field lookups.
"""
function precompute_halo_ops!(sim::SimulationData)
    isnothing(sim.chunk_data) && return
    length(sim.chunk_data) <= 1 && return

    sim._halo_ops_H = _precompute_halo_ops_for_group(sim, :H)
    sim._halo_ops_E = _precompute_halo_ops_for_group(sim, :E)
end

function _precompute_halo_ops_for_group(sim::SimulationData, field_group::Symbol)
    components = _field_components_for_group(field_group)
    T = eltype(sim.chunk_data[1].fields.fEx)
    ops = HaloCopyOp{T}[]
    rank = is_distributed() ? mpi_rank() : 0
    assignment = sim.chunk_rank_assignment

    for chunk in sim.chunk_data
        for conn in chunk.halo_send
            if !isnothing(assignment) && assignment[conn.dst_chunk_id] != rank
                continue
            end
            idx = _local_chunk_index(sim, conn.dst_chunk_id)
            isnothing(idx) && continue
            dst = sim.chunk_data[idx]
            for comp in components
                src_f = _get_chunk_field(chunk, comp)
                dst_f = _get_chunk_field(dst, comp)
                if !isnothing(src_f) && !isnothing(dst_f)
                    src_parent = parent(src_f)
                    dst_parent = parent(dst_f)
                    src_dims = size(src_parent) .- 2
                    dst_dims = size(dst_parent) .- 2
                    sr = _component_send_range(src_dims, conn.axis, conn.src_range)
                    dr = _component_recv_range(dst_dims, conn.axis, conn.dst_range)
                    sr_shifted = (sr[1] .+ 1, sr[2] .+ 1, sr[3] .+ 1)
                    dr_shifted = (dr[1] .+ 1, dr[2] .+ 1, dr[3] .+ 1)

                    width  = length(sr_shifted[1])
                    height = length(sr_shifted[2])
                    depth  = length(sr_shifted[3])
                    src_Nx = size(src_parent, 1)
                    src_Ny = size(src_parent, 2)
                    dst_Nx = size(dst_parent, 1)
                    dst_Ny = size(dst_parent, 2)

                    push!(ops, HaloCopyOp{T}(
                        pointer(dst_parent), pointer(src_parent),
                        width, height, depth,
                        (first(sr_shifted[1]), first(sr_shifted[2]), first(sr_shifted[3])),
                        (first(dr_shifted[1]), first(dr_shifted[2]), first(dr_shifted[3])),
                        src_Nx * sizeof(T), src_Ny,
                        dst_Nx * sizeof(T), dst_Ny,
                    ))
                end
            end
        end
    end
    return ops
end

"""
    _exchange_halos_precomputed!(ops)

Execute precomputed halo copy operations. Minimal CPU overhead per operation.
"""
function _exchange_halos_precomputed!(ops)
    @inbounds for op in ops
        CUDA.unsafe_copy3d!(
            op.dst_ptr, CUDA.DeviceMemory,
            op.src_ptr, CUDA.DeviceMemory,
            op.width, op.height, op.depth;
            srcPos = op.src_pos,
            dstPos = op.dst_pos,
            srcPitch = op.src_pitch,
            srcHeight = op.src_height,
            dstPitch = op.dst_pitch,
            dstHeight = op.dst_height,
            async = true,
        )
    end
end

"""
    exchange_halos!(sim, field_group::Symbol)

Exchange ghost cells between chunks for a given field group (:B, :H, :D, :E).
Dispatches to local-only or MPI paths as appropriate.
Uses precomputed copy operations when available for minimum CPU overhead.
"""
function exchange_halos!(sim::SimulationData, field_group::Symbol)
    isnothing(sim.chunk_data) && return
    length(sim.chunk_data) <= 1 && !is_distributed() && return

    if !is_distributed()
        # Use precomputed fast path if available
        ops = field_group == :H ? sim._halo_ops_H : sim._halo_ops_E
        if !isempty(ops)
            _exchange_halos_precomputed!(ops)
        else
            components = _field_components_for_group(field_group)
            _exchange_halos_local!(sim, components)
        end
        # No explicit synchronize needed: all halo copies (CUDA.unsafe_copy3d!)
        # and subsequent kernel launches use the same default CUDA stream,
        # so stream ordering guarantees copies complete before kernels execute.
        return
    end

    components = _field_components_for_group(field_group)
    _exchange_halos_mpi!(sim, components)
end

"""
    _exchange_halos_local!(sim, components)

Perform local (same-rank) halo copies between chunks. In distributed mode,
skips connections that cross rank boundaries.
"""
function _exchange_halos_local!(sim::SimulationData, components)
    assignment = sim.chunk_rank_assignment
    rank = is_distributed() ? mpi_rank() : 0

    for chunk in sim.chunk_data
        for conn in chunk.halo_send
            # Skip remote connections
            if !isnothing(assignment) && assignment[conn.dst_chunk_id] != rank
                continue
            end
            idx = _local_chunk_index(sim, conn.dst_chunk_id)
            isnothing(idx) && continue
            dst = sim.chunk_data[idx]
            for comp in components
                src_f = _get_chunk_field(chunk, comp)
                dst_f = _get_chunk_field(dst, comp)
                if !isnothing(src_f) && !isnothing(dst_f)
                    src_parent = parent(src_f)
                    dst_parent = parent(dst_f)
                    src_dims = size(src_parent) .- 2
                    dst_dims = size(dst_parent) .- 2
                    sr = _component_send_range(src_dims, conn.axis, conn.src_range)
                    dr = _component_recv_range(dst_dims, conn.axis, conn.dst_range)
                    sr_shifted = (sr[1] .+ 1, sr[2] .+ 1, sr[3] .+ 1)
                    dr_shifted = (dr[1] .+ 1, dr[2] .+ 1, dr[3] .+ 1)
                    _halo_copy!(dst_parent, dr_shifted, src_parent, sr_shifted)
                end
            end
        end
    end
end

"""
    _component_send_range(comp_dims, axis, center_range)

Compute the send range for a specific component, extending non-split dimensions
to cover the full component extent (which may be larger than Center due to Yee stagger).
The split axis uses the Center-based range because:
- For non-staggered components (Nz_comp = Nz_center): Nz_center IS the last interior cell
- For staggered components (Nz_comp = Nz_center+1): Nz_center is the last COMPUTED cell;
  Nz_center+1 is the boundary cell (filled by recv, not computed by kernel)
"""
function _component_send_range(comp_dims::NTuple{3,Int}, axis::Int, center_range::NTuple{3,UnitRange{Int}})
    ranges = [1:comp_dims[1], 1:comp_dims[2], 1:comp_dims[3]]
    # Keep the split axis range from center_range (correct for both staggered and non-staggered)
    ranges[axis] = center_range[axis]
    return (ranges[1], ranges[2], ranges[3])
end

"""
    _component_recv_range(comp_dims, axis, center_range)

Compute the recv range for a specific component, extending non-split dimensions
to cover the full component extent.
The split axis uses the Center-based range because:
- For non-staggered (Nz_comp = Nz_center): Nz_center+1 IS the ghost cell
- For staggered (Nz_comp = Nz_center+1): Nz_center+1 IS the boundary cell
  (the uncomputed last interior cell that needs to be filled from the neighbor)
"""
function _component_recv_range(comp_dims::NTuple{3,Int}, axis::Int, center_range::NTuple{3,UnitRange{Int}})
    ranges = [1:comp_dims[1], 1:comp_dims[2], 1:comp_dims[3]]
    # Keep the split axis range from center_range (correct for both staggered and non-staggered)
    ranges[axis] = center_range[axis]
    return (ranges[1], ranges[2], ranges[3])
end
