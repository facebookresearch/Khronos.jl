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
    classify_region_physics(sim, vol, geometry, boundaries)

Determine `PhysicsFlags` for a rectangular sub-domain.
Uses bounding-box intersection for O(num_objects) fast rejection.
"""
function classify_region_physics(
    sim::SimulationData,
    vol::Volume,
    geometry::Union{Vector{Object},Nothing},
    boundaries::Union{Vector{Vector{T}},Nothing},
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

    # Check PML overlap
    chunk_gv = GridVolume(sim, vol, Center())
    has_pml_x = _pml_overlaps_chunk_axis(sim, chunk_gv, 1)
    has_pml_y = _pml_overlaps_chunk_axis(sim, chunk_gv, 2)
    has_pml_z = (sim.ndims == 3) ? _pml_overlaps_chunk_axis(sim, chunk_gv, 3) : false

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
# plan_chunks -- single-chunk baseline
# -------------------------------------------------------- #

"""
    plan_chunks(sim::SimulationData)::ChunkPlan

Generate a chunk plan for the simulation. Currently returns a single-chunk
plan spanning the entire domain as a baseline.
"""
function plan_chunks(sim::SimulationData)::ChunkPlan
    vol = Volume(center = sim.cell_center, size = sim.cell_size)
    gv = GridVolume(sim, Center())
    physics = classify_region_physics(sim, vol, sim.geometry, sim.boundaries)
    spec = ChunkSpec(1, vol, gv, physics, Int[], 0)
    return ChunkPlan([spec], Tuple{Int,Int,Int}[], 1)
end
