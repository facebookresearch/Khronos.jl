# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Contains all relevant functions for manipulating geometry.
#
# NOTE [smartalecH] Don't use Inf to specify bounds in gometric shapes, as it's
# not compatible with `findfirst`.

"""
    Base.findfirst()

    Search for the first object at the current point.

Right now, this function is going to be slow for two reasons: (1) we are
interating through the list in sequential fashion. We can accelerate this by
using a binary search (e.g. with a KDTree). GeometryPrimitives already supports
this... we just need to implement the proper methods with our new `Object` data
structure. (2) julia has to rely on _dynamic dispatch_ to resolve the members of
the geometry list. This is because each shape inherits from a common type
(`Shape`) and the elements of this abstract vector aren't known at compile time.
Luckily, we can get around this too, but it will require some modification to
GeometryPrimitives itself. Best to avoid tackling those issues until we really
need to...
"""


function Base.findfirst(p::Vector{T}, s::Vector{Object}) where {T<:Real}
    for i in eachindex(s)
        b = bounds(s[i].shape)
        if all(b[1] .< p .< b[2]) && p ∈ s[i].shape  # check if p is within bounding box is faster
            return i # return the _index_ of the shape
        end
    end
    return nothing
end

# Non-allocating SVector overload for the hot path in init_geometry
function Base.findfirst(p::SVector{N}, s::Vector{Object}) where {N}
    for i in eachindex(s)
        if _check_point_in_shape(p, s[i].shape)
            return i
        end
    end
    return nothing
end

# Function barrier: Julia specializes on the concrete Shape type at dispatch,
# so bounds() and ∈ run without dynamic dispatch inside each call.
@inline function _check_point_in_shape(p::SVector{N}, shape::Shape{N}) where {N}
    b = bounds(shape)
    @inbounds for d = 1:N
        (b[1][d] < p[d] < b[2][d]) || return false
    end
    return p ∈ shape
end


"""
For now, let's just assume diagonal materials, so that the inverse is easy to
compute!

TODO: better input checking and sanitizing...
"""

function get_perm_inv(obj::Object, f::Field)
    perm = get_material_perm_comp(obj.material, f)
    if isnothing(perm)
        perm = get_material_perm(obj.material, f)
    end
    return one(backend_number) / backend_number(perm)
end

function get_σ(obj::Object, f::Field)
    σ = get_material_conductivity_comp(obj.material, f)
    if isnothing(σ)
        σ = get_material_conductivity(obj.material, f)
    end
    if isnothing(σ)
        return zero(backend_number)
    end
    return backend_number(σ)
end

"""
TODO: Eventually we want to set up a "default material" to call...
"""
function get_perm_inv(obj::Nothing, ::Field)
    one(backend_number)
end

function get_σ(obj::Nothing, ::Field)
    zero(backend_number)
end

# ---------------------------------------------------------- #
# Geometry data initialization
# ---------------------------------------------------------- #

function pull_geometry(
    sim::SimulationData,
    geometry::Vector{Object},
    index,
    f::Field,
    idx,
    ε_inv,
    σD,
)
    if isnothing(index)
        obj = nothing
    else
        obj = geometry[index]
    end

    if !isnothing(ε_inv)
        ε_inv[idx...] = get_perm_inv(obj, f)
    end

    if !isnothing(σD)
        σD[idx...] = get_σ(obj, f)
    end

    return
end

# ---------------------------------------------------------- #
# Spatial acceleration for geometry initialization
# ---------------------------------------------------------- #

"""
    _build_kdtree(geometry::Vector{Object})

Build a KDTree from geometry objects for accelerated spatial queries.
Returns the tree when all shapes share a single concrete type (avoiding
dynamic dispatch inside the tree), or `nothing` for heterogeneous shapes
(falls back to linear search).
"""
function _build_kdtree(geometry::Vector{Object})
    isempty(geometry) && return nothing
    ShapeType = typeof(geometry[1].shape)
    for i in 2:length(geometry)
        typeof(geometry[i].shape) === ShapeType || return nothing
    end
    shapes = ShapeType[obj.shape for obj in geometry]
    return KDTree(shapes)
end

# Dispatch on KDTree vs Nothing for type stability in the hot loop.
@inline _find_shape(::Nothing, point, geometry) = findfirst(point, geometry)
@inline _find_shape(kdtree::KDTree, point, ::Vector{Object}) = findfirst(point, kdtree)

# ── Rasterization approach ──────────────────────────────────────────
# Instead of querying "which object is at this voxel?" for every voxel
# (O(voxels × log(objects)) with KDTree), iterate over objects and paint
# their voxels: O(sum of bounding_box_volumes). For scenes with many small
# objects (like metalens pillars), this is orders of magnitude faster.

"""
    _rasterize_object!(shape, obj, f, perm_arr, σ_arr, xs, ys, zs)

Paint the voxels belonging to a single object into the permittivity/conductivity
arrays. Uses a function barrier (`where {S<:Shape}`) so Julia specializes on the
concrete shape type, avoiding dynamic dispatch for `bounds()` and `∈` calls.
"""
@inline function _rasterize_object!(
    shape::S, obj::Object, f::F,
    perm_arr, σ_arr,
    xs::Vector{Float64}, ys::Vector{Float64}, zs::Vector{Float64},
) where {S<:Shape, F<:Field}
    _rasterize_object_yrange!(shape, obj, f, perm_arr, σ_arr,
        xs, ys, zs, 1, length(ys))
end

"""
    _rasterize_object_yrange!(shape, obj, f, perm_arr, σ_arr, xs, ys, zs, iy_start, iy_end)

Paint voxels of a single object within a restricted y-index range [iy_start, iy_end].
Used by the y-range parallel rasterization path. Each y-range writes to disjoint
array slices, so concurrent calls with different y-ranges are data-race free.
"""
@inline function _rasterize_object_yrange!(
    shape::S, obj::Object, f::F,
    perm_arr, σ_arr,
    xs::Vector{Float64}, ys::Vector{Float64}, zs::Vector{Float64},
    iy_start::Int, iy_end::Int,
) where {S<:Shape, F<:Field}
    b = bounds(shape)
    # Map bounding box to grid index ranges via binary search on sorted coords
    ix_lo = max(searchsortedfirst(xs, b[1][1]), 1)
    ix_hi = min(searchsortedlast(xs, b[2][1]), length(xs))
    iy_lo = max(searchsortedfirst(ys, b[1][2]), iy_start)
    iy_hi = min(searchsortedlast(ys, b[2][2]), iy_end)
    iz_lo = max(searchsortedfirst(zs, b[1][3]), 1)
    iz_hi = min(searchsortedlast(zs, b[2][3]), length(zs))

    # Skip if no overlap with this y-range or bounding box is empty
    (ix_lo > ix_hi || iy_lo > iy_hi || iz_lo > iz_hi) && return

    # Pre-compute material values (only when the corresponding array exists)
    has_perm = !isnothing(perm_arr)
    has_σ = !isnothing(σ_arr)
    perm_val = has_perm ? get_perm_inv(obj, f) : zero(backend_number)
    σ_val = has_σ ? get_σ(obj, f) : zero(backend_number)

    for iz = iz_lo:iz_hi, iy = iy_lo:iy_hi, ix = ix_lo:ix_hi
        point = SVector(xs[ix], ys[iy], zs[iz])
        if point ∈ shape
            if has_perm
                @inbounds perm_arr[ix, iy, iz] = perm_val
            end
            if has_σ
                @inbounds σ_arr[ix, iy, iz] = σ_val
            end
        end
    end
end

"""
    _rasterize_geometry_3d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)

Rasterize all geometry objects into the permittivity/conductivity arrays.
Objects are painted in reverse order so that earlier objects (higher priority
per `findfirst` convention) overwrite later ones.
"""
function _rasterize_geometry_3d!(
    geometry::Vector{Object},
    f::F,
    perm_arr,
    σ_arr,
    xs::Vector{Float64},
    ys::Vector{Float64},
    zs::Vector{Float64},
) where {F<:Field}
    # Nothing to do if both arrays are absent
    isnothing(perm_arr) && isnothing(σ_arr) && return

    # Fill with free-space defaults
    if !isnothing(perm_arr)
        fill!(perm_arr, one(backend_number))
    end
    # σ_arr is already zeros from allocation

    # Paint objects in reverse order (last → first) so earlier objects take priority
    for gi in length(geometry):-1:1
        obj = geometry[gi]
        # Function barrier: obj.shape is abstractly typed (Shape), but
        # _rasterize_object! dispatches on the concrete type via `where {S<:Shape}`.
        # This costs one dispatch per object (~23K) rather than per voxel (~billions).
        _rasterize_object!(obj.shape, obj, f, perm_arr, σ_arr, xs, ys, zs)
    end
end

"""
    _rasterize_object_2d!(shape, obj, f, perm_arr, σ_arr, xs, ys, zs)

2D version of _rasterize_object! for 2D simulations.
"""
@inline function _rasterize_object_2d!(
    shape::S, obj::Object, f::F,
    perm_arr, σ_arr,
    xs::Vector{Float64}, ys::Vector{Float64}, zs::Vector{Float64},
) where {S<:Shape, F<:Field}
    b = bounds(shape)
    ix_lo = max(searchsortedfirst(xs, b[1][1]), 1)
    ix_hi = min(searchsortedlast(xs, b[2][1]), length(xs))
    iy_lo = max(searchsortedfirst(ys, b[1][2]), 1)
    iy_hi = min(searchsortedlast(ys, b[2][2]), length(ys))

    has_perm = !isnothing(perm_arr)
    has_σ = !isnothing(σ_arr)
    perm_val = has_perm ? get_perm_inv(obj, f) : zero(backend_number)
    σ_val = has_σ ? get_σ(obj, f) : zero(backend_number)

    for iy = iy_lo:iy_hi, ix = ix_lo:ix_hi
        point = SVector(xs[ix], ys[iy], zs[1])
        if point ∈ shape
            if has_perm
                @inbounds perm_arr[ix, iy, 1] = perm_val
            end
            if has_σ
                @inbounds σ_arr[ix, iy, 1] = σ_val
            end
        end
    end
end

function _rasterize_geometry_2d!(
    geometry::Vector{Object},
    f::F,
    perm_arr,
    σ_arr,
    xs::Vector{Float64},
    ys::Vector{Float64},
    zs::Vector{Float64},
) where {F<:Field}
    isnothing(perm_arr) && isnothing(σ_arr) && return

    if !isnothing(perm_arr)
        fill!(perm_arr, one(backend_number))
    end
    for gi in length(geometry):-1:1
        obj = geometry[gi]
        _rasterize_object_2d!(obj.shape, obj, f, perm_arr, σ_arr, xs, ys, zs)
    end
end

get_material_perm_comp(mat::Material, ::Union{Dx,Ex}) = mat.εx
get_material_perm_comp(mat::Material, ::Union{Dy,Ey}) = mat.εy
get_material_perm_comp(mat::Material, ::Union{Dz,Ez}) = mat.εz
get_material_perm_comp(mat::Material, ::Union{Bx,Hx}) = mat.μx
get_material_perm_comp(mat::Material, ::Union{By,Hy}) = mat.μy
get_material_perm_comp(mat::Material, ::Union{Bz,Hz}) = mat.μz
get_material_perm(mat::Material, ::Electric) = mat.ε
get_material_perm(mat::Material, ::Magnetic) = mat.μ

function needs_perm(obj::Object, f::Field)
    return !isnothing(get_material_perm(obj.material, f)) ||
           !isnothing(get_material_perm_comp(obj.material, f))
end

function needs_perm(geometry::Vector{Object}, f::Field)
    needs_ε = false
    for obj in geometry
        needs_ε = needs_ε || needs_perm(obj, f)
    end
    return needs_ε
end

get_material_conductivity_comp(mat::Material, ::Union{Dx,Ex}) = mat.σDx
get_material_conductivity_comp(mat::Material, ::Union{Dy,Ey}) = mat.σDy
get_material_conductivity_comp(mat::Material, ::Union{Dz,Ez}) = mat.σDz
get_material_conductivity_comp(mat::Material, ::Union{Bx,Hx}) = mat.σBx
get_material_conductivity_comp(mat::Material, ::Union{By,Hy}) = mat.σBy
get_material_conductivity_comp(mat::Material, ::Union{Bz,Hz}) = mat.σBz
get_material_conductivity(mat::Material, ::Electric) = mat.σD
get_material_conductivity(mat::Material, ::Magnetic) = mat.σB

function needs_conductivities(obj::Object, f::Field)
    return !isnothing(get_material_conductivity(obj.material, f)) ||
           !isnothing(get_material_conductivity_comp(obj.material, f))
end

function needs_conductivities(geometry::Vector{Object}, f::Field)
    needs_cond = false
    for obj in geometry
        needs_cond = needs_cond || needs_conductivities(obj, f)
    end
    return needs_cond
end

# ---------------------------------------------------------- #
# Geometry initialization helpers (module-level for type stability)
# ---------------------------------------------------------- #

function _precompute_coords(sim::SimulationData, gv::GridVolume)
    origin = get_component_origin(sim, gv.component)
    gv_origin = get_min_corner(gv)
    Δx, Δy, Δz = _scalar_spacing(sim.Δx), _scalar_spacing(sim.Δy), _scalar_spacing(sim.Δz)
    xs = [origin[1] + (ix + gv_origin[1] - 2) * Δx for ix = 1:gv.Nx]
    ys = [origin[2] + (iy + gv_origin[2] - 2) * Δy for iy = 1:gv.Ny]
    pz_base = origin[3] + (gv_origin[3] - 2) * Δz
    zs = Vector{Float64}(undef, gv.Nz)
    for iz = 1:gv.Nz
        pz_raw = pz_base + iz * Δz
        zs[iz] = (isinf(pz_raw) || isnan(pz_raw)) ? sim.cell_center[3] : pz_raw
    end
    return xs, ys, zs
end

# Coordinate builders: uniform (scalar Δ) vs non-uniform (vector Δ)
function _build_coords(Δ::Real, origin, gv_offset, N; default_val=nothing)
    coords = Vector{Float64}(undef, N)
    for i in 1:N
        val = origin + (i + gv_offset - 2) * Δ
        coords[i] = (!isnothing(default_val) && (isinf(val) || isnan(val))) ? default_val : val
    end
    return coords
end

function _build_coords(Δ::AbstractVector, origin, gv_offset, N; default_val=nothing)
    # For non-uniform grids, compute cumulative position from spacing vector
    # Δ[k] = spacing of cell k. Position of cell index i = origin + sum(Δ[1:i+offset-2])
    coords = Vector{Float64}(undef, N)
    cumpos = 0.0
    start_idx = gv_offset - 1  # first cell index (0-based in the spacing array)
    for i in 1:N
        cell_idx = i + gv_offset - 2  # 0-based global cell index
        if cell_idx <= 0
            cumpos = cell_idx * (length(Δ) > 0 ? Δ[1] : 0.0)
        elseif cell_idx <= length(Δ)
            cumpos = sum(Δ[1:cell_idx])
        else
            cumpos = sum(Δ) + (cell_idx - length(Δ)) * Δ[end]
        end
        val = origin + cumpos
        coords[i] = (!isnothing(default_val) && (isinf(val) || isnan(val))) ? default_val : val
    end
    return coords
end

function _write_geometry_3d!(
    kdtree::KD,
    sim::SimulationData,
    geometry::Vector{Object},
    gv::GridVolume,
    f::F,
    perm_arr,
    σ_arr,
    xs::Vector{Float64},
    ys::Vector{Float64},
    zs::Vector{Float64},
) where {F<:Field, KD}
    for iz = 1:gv.Nz, iy = 1:gv.Ny, ix = 1:gv.Nx
        point = SVector(xs[ix], ys[iy], zs[iz])
        index = _find_shape(kdtree, point, geometry)
        obj = isnothing(index) ? nothing : geometry[index]
        if !isnothing(perm_arr)
            @inbounds perm_arr[ix, iy, iz] = get_perm_inv(obj, f)
        end
        if !isnothing(σ_arr)
            @inbounds σ_arr[ix, iy, iz] = get_σ(obj, f)
        end
    end
end

function _write_geometry_2d!(
    kdtree::KD,
    sim::SimulationData,
    geometry::Vector{Object},
    gv::GridVolume,
    f::F,
    perm_arr,
    σ_arr,
    xs::Vector{Float64},
    ys::Vector{Float64},
    zs::Vector{Float64},
) where {F<:Field, KD}
    for iy = 1:gv.Ny, ix = 1:gv.Nx
        point = SVector(xs[ix], ys[iy], zs[1])
        index = _find_shape(kdtree, point, geometry)
        obj = isnothing(index) ? nothing : geometry[index]
        if !isnothing(perm_arr)
            @inbounds perm_arr[ix, iy, 1] = get_perm_inv(obj, f)
        end
        if !isnothing(σ_arr)
            @inbounds σ_arr[ix, iy, 1] = get_σ(obj, f)
        end
    end
end

# isotropic materials
function init_geometry(sim::SimulationData, geometry::Vector{Object})
    if length(geometry) == 0
        init_geometry(sim, nothing)
        return
    end

    T = backend_number
    # Always allocate 3D arrays even for 2D simulations, because the GPU
    # kernels use 3D indexing.  get_component_voxel_count already clamps
    # Nz >= 1, so the z-dimension is at least 1 voxel.
    _alloc(need, comp) = need ?
        Array{T}(undef, get_component_voxel_count(sim, comp)...) : nothing

    ε_inv_x = _alloc(needs_perm(geometry, Ex()), Ex())
    ε_inv_y = _alloc(needs_perm(geometry, Ey()), Ey())
    ε_inv_z = _alloc(needs_perm(geometry, Ez()), Ez())

    σDx = _alloc(needs_conductivities(geometry, Dx()), Ex())
    σDy = _alloc(needs_conductivities(geometry, Dy()), Ey())
    σDz = _alloc(needs_conductivities(geometry, Dz()), Ez())

    μ_inv_x = _alloc(needs_perm(geometry, Hx()), Hx())
    μ_inv_y = _alloc(needs_perm(geometry, Hy()), Hy())
    μ_inv_z = _alloc(needs_perm(geometry, Hz()), Hz())

    σBx = _alloc(needs_conductivities(geometry, Bx()), Hx())
    σBy = _alloc(needs_conductivities(geometry, By()), Hy())
    σBz = _alloc(needs_conductivities(geometry, Bz()), Hz())

    # If absorbers are specified, ensure σD and σB arrays exist (absorbers
    # add conductivity additively to the geometry arrays).
    if !isnothing(sim.absorbers)
        for (axis, axis_absorbers) in enumerate(sim.absorbers)
            isnothing(axis_absorbers) && continue
            for side_abs in axis_absorbers
                isnothing(side_abs) && continue
                # Force allocation of all conductivity arrays
                if isnothing(σDx)
                    σDx = _alloc(true, Ex())
                    if !isnothing(σDx); fill!(σDx, zero(T)); end
                end
                if isnothing(σDy)
                    σDy = _alloc(true, Ey())
                    if !isnothing(σDy); fill!(σDy, zero(T)); end
                end
                if isnothing(σDz)
                    σDz = _alloc(true, Ez())
                    if !isnothing(σDz); fill!(σDz, zero(T)); end
                end
                if isnothing(σBx)
                    σBx = _alloc(true, Hx())
                    if !isnothing(σBx); fill!(σBx, zero(T)); end
                end
                if isnothing(σBy)
                    σBy = _alloc(true, Hy())
                    if !isnothing(σBy); fill!(σBy, zero(T)); end
                end
                if isnothing(σBz)
                    σBz = _alloc(true, Hz())
                    if !isnothing(σBz); fill!(σBz, zero(T)); end
                end
                break  # only need to allocate once
            end
        end
    end

    # Pre-compute 1D coordinate arrays for each component to avoid
    # per-voxel allocations. Each coordinate is separable:
    #   coord[i] = origin[d] + (i + gv_origin[d] - 2) * Δ[d]

    # Build per-component data: (GridVolume, field_type, perm_array, σ_array)
    components = (
        (GridVolume(sim, Ex()), Dx(), ε_inv_x, σDx),
        (GridVolume(sim, Hx()), Bx(), μ_inv_x, σBx),
        (GridVolume(sim, Ey()), Dy(), ε_inv_y, σDy),
        (GridVolume(sim, Hy()), By(), μ_inv_y, σBy),
        (GridVolume(sim, Ez()), Dz(), ε_inv_z, σDz),
        (GridVolume(sim, Hz()), Bz(), μ_inv_z, σBz),
    )

    # Use rasterization for geometry init: iterate over objects and paint their
    # voxels, rather than iterating over all voxels and searching for objects.
    # This is O(sum of bounding_box_volumes) vs O(voxels × log(objects)) with
    # KDTree, which is dramatically faster for scenes with many small objects
    # (e.g., metalens: 23K pillars × ~350 voxels each = 8M tests vs 600M+ tree
    # traversals). Uses function barriers to avoid dynamic dispatch.
    # For 3D: spawn tasks across (component, y-range) pairs for better thread
    # utilization. The original 6-task approach (one per Yee component) uses at
    # most 6 threads; splitting each component's y-axis into chunks gives up to
    # 6 × nchunks tasks, utilizing all available threads. Each y-range writes to
    # disjoint array slices, so concurrent tasks are data-race free.
    nthreads = Threads.nthreads()
    tasks = Task[]

    for (ci, (gv, f, perm_arr, σ_arr)) in enumerate(components)
        # Skip components that don't need any arrays
        isnothing(perm_arr) && isnothing(σ_arr) && continue

        xs, ys, zs = _precompute_coords(sim, gv)

        if sim.ndims == 3
            # Use actual array dimensions for fill loops and y-range chunking.
            # GridVolume coordinate counts (length(xs/ys/zs)) can differ from
            # array dimensions (get_component_voxel_count) due to Yee staggering.
            ref_arr = something(perm_arr, σ_arr)
            arr_nx, arr_ny, arr_nz = size(ref_arr)

            # Truncate coordinate arrays to match array dimensions so that
            # _rasterize_object_yrange! indices stay within array bounds.
            xs_t = length(xs) > arr_nx ? xs[1:arr_nx] : xs
            ys_t = length(ys) > arr_ny ? ys[1:arr_ny] : ys
            zs_t = length(zs) > arr_nz ? zs[1:arr_nz] : zs

            # Split y into chunks; at least 8 y-values per chunk to avoid overhead
            nchunks = max(1, min(nthreads, arr_ny ÷ 8))
            chunk_size = cld(arr_ny, nchunks)

            for ch in 1:nchunks
                iy_start = (ch - 1) * chunk_size + 1
                iy_end = min(ch * chunk_size, arr_ny)
                push!(tasks, Threads.@spawn begin
                    # Fill this y-range with free-space defaults (distributes
                    # page faults across threads instead of single-threaded fill!)
                    if !isnothing(perm_arr)
                        for iz in 1:arr_nz, iy in iy_start:iy_end, ix in 1:arr_nx
                            perm_arr[ix, iy, iz] = one(T)
                        end
                    end
                    if !isnothing(σ_arr)
                        for iz in 1:arr_nz, iy in iy_start:iy_end, ix in 1:size(σ_arr, 1)
                            σ_arr[ix, iy, iz] = zero(T)
                        end
                    end
                    # Rasterize objects within this y-range
                    for gi in length(geometry):-1:1
                        obj = geometry[gi]
                        _rasterize_object_yrange!(obj.shape, obj, f,
                            perm_arr, σ_arr, xs_t, ys_t, zs_t, iy_start, iy_end)
                    end
                end)
            end
        else
            # 2D path: single task per component
            push!(tasks, Threads.@spawn begin
                _rasterize_geometry_2d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)
            end)
        end
    end
    for t in tasks
        wait(t)
    end

    # Apply subpixel smoothing to ε_inv arrays at material interfaces
    _apply_subpixel_smoothing!(sim, geometry, ε_inv_x, ε_inv_y, ε_inv_z)

    # Apply absorber conductivity profiles to σD/σB arrays
    if !isnothing(sim.absorbers)
        _apply_absorbers!(sim, σDx, σDy, σDz, σBx, σBy, σBz)
    end

    # Allocate and rasterize chi3 if any material has it
    chi3_arr = nothing
    if any(obj -> !isnothing(obj.material.chi3), geometry)
        chi3_arr = zeros(backend_number, sim.Nx, sim.Ny, max(1, sim.Nz))
        for gi in length(geometry):-1:1
            obj = geometry[gi]
            chi3_val = isnothing(obj.material.chi3) ? zero(backend_number) : backend_number(obj.material.chi3)
            if chi3_val != 0
                # Simple rasterization: paint chi3 for all voxels inside the shape
                gv = GridVolume(sim, Center())
                xs, ys, zs = _precompute_coords(sim, gv)
                b = bounds(obj.shape)
                ix_lo = max(searchsortedfirst(xs, b[1][1]), 1)
                ix_hi = min(searchsortedlast(xs, b[2][1]), length(xs))
                iy_lo = max(searchsortedfirst(ys, b[1][2]), 1)
                iy_hi = min(searchsortedlast(ys, b[2][2]), length(ys))
                iz_lo = max(searchsortedfirst(zs, b[1][3]), 1)
                iz_hi = min(searchsortedlast(zs, b[2][3]), length(zs))
                for iz in iz_lo:iz_hi, iy in iy_lo:iy_hi, ix in ix_lo:ix_hi
                    point = SVector(xs[ix], ys[iy], zs[iz])
                    if point ∈ obj.shape
                        chi3_arr[ix, iy, iz] = chi3_val
                    end
                end
            end
        end
    end

    # Construct geometry data structure. Importantly, this will transfer all
    # data from the host to the device.
    sim.geometry_data = GeometryData{backend_number,backend_array}(
        ε_inv_x = isnothing(ε_inv_x) ? ε_inv_x : backend_array(ε_inv_x),
        ε_inv_y = isnothing(ε_inv_y) ? ε_inv_y : backend_array(ε_inv_y),
        ε_inv_z = isnothing(ε_inv_z) ? ε_inv_z : backend_array(ε_inv_z),
        σDx = isnothing(σDx) ? σDx : backend_array(σDx),
        σDy = isnothing(σDy) ? σDy : backend_array(σDy),
        σDz = isnothing(σDz) ? σDz : backend_array(σDz),
        μ_inv = one(backend_number), # todo add μ support
        # right now it's either zeros or nothing...
        σBx = isnothing(σBx) ? σBx : backend_array(σBx),
        σBy = isnothing(σBy) ? σBy : backend_array(σBy),
        σBz = isnothing(σBz) ? σBz : backend_array(σBz),
        chi3 = isnothing(chi3_arr) ? nothing : backend_array(chi3_arr),
    )

end

function init_geometry(sim::SimulationData, ::Nothing)
    sim.geometry_data = GeometryData{backend_number,backend_array}(ε_inv = one(backend_number), μ_inv = one(backend_number))
end

""" get_mat_conductivity_from_field()

"""
get_mat_conductivity_from_field(sim::SimulationData, ::Union{Bx,Hx}) = sim.geometry_data.σBx
get_mat_conductivity_from_field(sim::SimulationData, ::Union{By,Hy}) = sim.geometry_data.σBy
get_mat_conductivity_from_field(sim::SimulationData, ::Union{Bz,Hz}) = sim.geometry_data.σBz
get_mat_conductivity_from_field(sim::SimulationData, ::Union{Dx,Ex}) = sim.geometry_data.σDx
get_mat_conductivity_from_field(sim::SimulationData, ::Union{Dy,Ey}) = sim.geometry_data.σDy
get_mat_conductivity_from_field(sim::SimulationData, ::Union{Dz,Ez}) = sim.geometry_data.σDz
get_mat_conductivity_from_field(sim::SimulationData, ::Magnetic, ::X) =
    sim.geometry_data.σBx
get_mat_conductivity_from_field(sim::SimulationData, ::Magnetic, ::Y) =
    sim.geometry_data.σBy
get_mat_conductivity_from_field(sim::SimulationData, ::Magnetic, ::Z) =
    sim.geometry_data.σBz
get_mat_conductivity_from_field(sim::SimulationData, ::Electric, ::X) =
    sim.geometry_data.σDx
get_mat_conductivity_from_field(sim::SimulationData, ::Electric, ::Y) =
    sim.geometry_data.σDy
get_mat_conductivity_from_field(sim::SimulationData, ::Electric, ::Z) =
    sim.geometry_data.σDz

# ---------------------------------------------------------- #
# Absorber conductivity application
# ---------------------------------------------------------- #

"""
    _apply_absorbers!(sim, σDx, σDy, σDz, σBx, σBy, σBz)

Apply adiabatic absorber conductivity profiles to the per-voxel σD/σB arrays.
The absorber conductivity is additive (stacks with any material conductivity).

σ(d) = σ_max * (d / L)^p

where d = distance from domain interior into absorber, L = absorber thickness,
p = polynomial order.

For impedance matching: σ_B = σ_D * μ₀/ε₀ = σ_D (in natural units where μ₀=ε₀=1).
"""
function _apply_absorbers!(sim::SimulationData, σDx, σDy, σDz, σBx, σBy, σBz)
    T = backend_number
    absorbers = sim.absorbers
    isnothing(absorbers) && return

    for (axis, axis_absorbers) in enumerate(absorbers)
        isnothing(axis_absorbers) && continue

        for (side_idx, abs_spec) in enumerate(axis_absorbers)
            isnothing(abs_spec) && continue

            num_layers = abs_spec.num_layers
            p = abs_spec.sigma_order

            # Auto-compute σ_max if not specified
            # Target reflection R ≈ exp(-2 * σ_max * L / (p+1)) ≈ 1e-6
            # => σ_max = -(p+1) * log(1e-6) / (2 * L)
            # Note: σ_max is stored as a raw physical conductivity value (no Δt
            # pre-factor) because get_σD() in the kernel already applies the
            # 0.5*Δt scaling needed for the FDTD update equation.
            Δ = axis == 1 ? sim.Δx : (axis == 2 ? sim.Δy : sim.Δz)
            L = num_layers * Δ
            σ_max = if abs_spec.sigma_max > 0
                abs_spec.sigma_max
            else
                T(-(p + 1) * log(1e-6) / (2.0 * L))
            end

            # side_idx == 1 is left (-), side_idx == 2 is right (+)
            is_right = (side_idx == 2)

            # Apply conductivity to all σD/σB arrays
            # The absorber occupies the outermost num_layers cells on this side
            _all_σ_pairs = [
                (σDx, :x), (σDy, :y), (σDz, :z),
                (σBx, :x), (σBy, :y), (σBz, :z),
            ]

            for (σ_arr, comp_label) in _all_σ_pairs
                isnothing(σ_arr) && continue

                N_axis = size(σ_arr, axis)

                for layer in 1:min(num_layers, N_axis)
                    # layer counts from the boundary inward:
                    #   layer 1 = outermost cell (at boundary) → max σ
                    #   layer num_layers = innermost cell (at interior edge) → min σ
                    # d_normalized = distance from interior edge into absorber
                    d_normalized = (num_layers - layer + 1) / num_layers

                    σ_val = T(σ_max * d_normalized^p)

                    if is_right
                        # Right side: layer 1 at N_axis (boundary), layer num_layers at N_axis - num_layers + 1 (interior)
                        idx = N_axis - layer + 1
                    else
                        # Left side: layer 1 at index 1 (boundary), layer num_layers at num_layers (interior)
                        idx = layer
                    end

                    # Apply to all voxels in the plane perpendicular to axis
                    if axis == 1
                        for iz in 1:size(σ_arr, 3), iy in 1:size(σ_arr, 2)
                            @inbounds σ_arr[idx, iy, iz] += σ_val
                        end
                    elseif axis == 2
                        for iz in 1:size(σ_arr, 3), ix in 1:size(σ_arr, 1)
                            @inbounds σ_arr[ix, idx, iz] += σ_val
                        end
                    else  # axis == 3
                        for iy in 1:size(σ_arr, 2), ix in 1:size(σ_arr, 1)
                            @inbounds σ_arr[ix, iy, idx] += σ_val
                        end
                    end
                end
            end

            @info("  Applied absorber on axis=$axis, side=$(is_right ? "+" : "-"), " *
                  "layers=$num_layers, σ_max=$(round(σ_max, sigdigits=4)), order=$p")
        end
    end
end

# ---------------------------------------------------------- #
# Subpixel smoothing
# ---------------------------------------------------------- #

"""
    _apply_subpixel_smoothing!(sim, geometry, ε_inv_x, ε_inv_y, ε_inv_z)

Apply subpixel smoothing to ε_inv arrays at material interfaces.
Called after rasterization and before absorber/GPU transfer.

Supports two modes:
- `VolumeAveraging()`: isotropic ε̄ = f·ε₁ + (1-f)·ε₂
- `AnisotropicSmoothing()`: Farjadpour et al. 2006 anisotropic tensor
"""
function _apply_subpixel_smoothing!(
    sim::SimulationData,
    geometry::Vector{Object},
    ε_inv_x, ε_inv_y, ε_inv_z,
)
    sim.subpixel_smoothing isa NoSmoothing && return

    if sim.ndims != 3
        @warn "Subpixel smoothing is currently only supported for 3D simulations"
        return
    end

    is_anisotropic = sim.subpixel_smoothing isa AnisotropicSmoothing
    Δx, Δy, Δz = sim.Δx, sim.Δy, sim.Δz

    # E-field components: (component type, ε_inv array, comp_idx for normal projection)
    ε_entries = (
        (Ex(), ε_inv_x, 1),
        (Ey(), ε_inv_y, 2),
        (Ez(), ε_inv_z, 3),
    )

    nthreads = Threads.nthreads()
    all_tasks = Vector{Tuple{Task, Int}}()

    for (comp, ε_inv, comp_idx) in ε_entries
        isnothing(ε_inv) && continue

        gv = GridVolume(sim, comp)
        xs, ys, zs = _precompute_coords(sim, gv)

        # Truncate coordinates to match array dimensions
        nx, ny, nz = size(ε_inv)
        xs = xs[1:min(length(xs), nx)]
        ys = ys[1:min(length(ys), ny)]
        zs = zs[1:min(length(zs), nz)]

        # Make a read-only copy to avoid data races at y-range boundaries
        ε_inv_orig = copy(ε_inv)

        # Split y into chunks for threading
        nchunks = max(1, min(nthreads, ny ÷ 8))
        chunk_size = cld(ny, nchunks)

        for ch in 1:nchunks
            iy_start = (ch - 1) * chunk_size + 1
            iy_end = min(ch * chunk_size, ny)
            t = Threads.@spawn _smooth_component_yrange!(
                ε_inv, ε_inv_orig, comp_idx, xs, ys, zs,
                Δx, Δy, Δz, geometry, is_anisotropic, iy_start, iy_end)
            push!(all_tasks, (t, comp_idx))
        end
    end

    # Collect results per component
    counts = Dict{Int, Int}()
    for (t, ci) in all_tasks
        n = fetch(t)
        counts[ci] = get(counts, ci, 0) + n
    end

    mode = is_anisotropic ? "anisotropic" : "volume averaging"
    total = sum(values(counts), init=0)
    @info("  Subpixel smoothing ($mode): smoothed $total interface voxels " *
          "(x=$(get(counts, 1, 0)), y=$(get(counts, 2, 0)), z=$(get(counts, 3, 0)))")
end

"""
    _smooth_component_yrange!(ε_inv, ε_inv_orig, comp_idx, xs, ys, zs,
                               Δx, Δy, Δz, geometry, is_anisotropic, iy_start, iy_end)

Smooth interface voxels in a y-range slice for one ε_inv component.
Reads neighbor values from `ε_inv_orig` (unmodified copy) and writes to `ε_inv`.
Returns the number of smoothed voxels.
"""
function _smooth_component_yrange!(
    ε_inv::Array{T,3},
    ε_inv_orig::Array{T,3},
    comp_idx::Int,
    xs::Vector{Float64}, ys::Vector{Float64}, zs::Vector{Float64},
    Δx, Δy, Δz,
    geometry::Vector{Object},
    is_anisotropic::Bool,
    iy_start::Int, iy_end::Int,
) where {T}
    nx, ny, nz = size(ε_inv)
    rtol = T(1e-6)
    half_Δ = SVector(Float64(Δx)/2, Float64(Δy)/2, Float64(Δz)/2)
    n_smoothed = 0

    for iz in 1:nz, iy in iy_start:iy_end, ix in 1:nx
        ε_inv_c = ε_inv_orig[ix, iy, iz]

        # Fast interface detection: check 6 face-sharing neighbors
        is_interface = false
        ε_inv_nbr = ε_inv_c
        for (dx, dy, dz) in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
            jx, jy, jz = ix+dx, iy+dy, iz+dz
            (1 ≤ jx ≤ nx && 1 ≤ jy ≤ ny && 1 ≤ jz ≤ nz) || continue
            nbr = ε_inv_orig[jx, jy, jz]
            if abs(nbr - ε_inv_c) > rtol * max(abs(nbr), abs(ε_inv_c))
                is_interface = true
                ε_inv_nbr = nbr
                break
            end
        end
        is_interface || continue

        # --- Interface voxel: compute smoothed ε_inv ---
        p = SVector(xs[ix], ys[iy], zs[iz])
        ε_c = Float64(one(T) / ε_inv_c)
        ε_n = Float64(one(T) / ε_inv_nbr)

        # Find closest shape surface for normal and fill fraction
        min_dist_sq = typemax(Float64)
        best_nout = SVector(0.0, 0.0, 1.0)
        best_surfpt = p
        best_shape_idx = 0

        for (gi, obj) in enumerate(geometry)
            # Quick bounding box proximity check
            b = bounds(obj.shape)
            min_bb_dist_sq = 0.0
            for d in 1:3
                if p[d] < b[1][d]
                    min_bb_dist_sq += (b[1][d] - p[d])^2
                elseif p[d] > b[2][d]
                    min_bb_dist_sq += (p[d] - b[2][d])^2
                end
            end
            min_bb_dist_sq ≥ min_dist_sq && continue

            sp, nout = surfpt_nearby(p, obj.shape)
            d2 = sum(abs2, sp - p)
            if d2 < min_dist_sq
                min_dist_sq = d2
                best_nout = nout
                best_surfpt = sp
                best_shape_idx = gi
            end
        end

        best_shape_idx == 0 && continue

        # Normalize the outward normal
        nrm = norm(best_nout)
        n̂ = nrm > 0 ? best_nout / nrm : SVector(0.0, 0.0, 1.0)

        # Compute fill fraction using volfrac from GeometryPrimitives
        # volfrac returns the fraction of the voxel INSIDE the half-space
        # (opposite to nout), i.e., inside the shape when nout is outward.
        vxl = (SVector{3}(p - half_Δ), SVector{3}(p + half_Δ))
        f_in_shape = volfrac(vxl, SVector{3}(n̂), SVector{3}(best_surfpt))

        # Determine which ε is the shape material and which is the background
        shape = geometry[best_shape_idx].shape
        if level(p, shape) ≥ 0  # center is inside shape
            ε_shape = ε_c
            ε_bg = ε_n
        else
            ε_shape = ε_n
            ε_bg = ε_c
        end

        # Compute averages
        ε_avg = f_in_shape * ε_shape + (1 - f_in_shape) * ε_bg            # ⟨ε⟩
        ε_inv_harm = f_in_shape / ε_shape + (1 - f_in_shape) / ε_bg       # ⟨ε⁻¹⟩

        if is_anisotropic
            # Farjadpour: component-dependent effective ε_inv
            n_comp_sq = n̂[comp_idx]^2
            @inbounds ε_inv[ix, iy, iz] = T((1 - n_comp_sq) * ε_inv_harm + n_comp_sq / ε_avg)
        else
            # Volume averaging: isotropic
            @inbounds ε_inv[ix, iy, iz] = T(1.0 / ε_avg)
        end
        n_smoothed += 1
    end

    return n_smoothed
end

# ---------------------------------------------------------- #
# Material functions
# ---------------------------------------------------------- #

function get_ε_at_frequency(material, frequency)
    ε = zeros(ComplexF64, 3, 3)

    # Account for all the DC terms first
    ε .+= isnothing(material.ε) ? 0 : material.ε * I(3)
    ε[1, 1] += isnothing(material.εx) ? 0 : material.εx
    ε[2, 2] += isnothing(material.εy) ? 0 : material.εy
    ε[3, 3] += isnothing(material.εz) ? 0 : material.εz
    ε[1, 2] += isnothing(material.εxy) ? 0 : material.εxy
    ε[2, 1] += isnothing(material.εxy) ? 0 : material.εxy
    ε[1, 3] += isnothing(material.εxz) ? 0 : material.εxz
    ε[3, 1] += isnothing(material.εxz) ? 0 : material.εxz
    ε[2, 3] += isnothing(material.εyz) ? 0 : material.εyz
    ε[3, 2] += isnothing(material.εyz) ? 0 : material.εyz

    # Now handle all the conductivities
    ω = 2 * π * frequency
    if !isnothing(material.σD)
        ε = (1 .+ im * material.σD / ω .* I(3)) .* ε
    end
    if !isnothing(material.σDx)
        ε[1, 1] = (1 + im * material.σDx / ω) * ε[1, 1]
    end
    if !isnothing(material.σDy)
        ε[2, 2] = (1 + im * material.σDy / ω) * ε[2, 2]
    end
    if !isnothing(material.σDx)
        ε[3, 3] = (1 + im * material.σDz / ω) * ε[3, 3]
    end

    return ε
end

function fit_complex_material(ε::Number, frequency::Number)
    return Material{Float64}(ε = real(ε), σD = 2 * π * frequency * imag(ε) / real(ε))
end

"""
    transform_material(material::AbstractArray)::AbstractArray

Transforms a material tensor `material` by a specified `transformation_matrix`.

More generally, the susceptibilities χ are transformed to MχMᵀ/|det M|, which
corresponds to [transformation
optics](http://math.mit.edu/~stevenj/18.369/coordinate-transform.pdf) for an
arbitrary curvilinear coordinate transformation with Jacobian matrix M. The
absolute value of the determinant is to prevent inadvertent construction of
left-handed materials
"""
function transform_material(
    material::AbstractArray,
    transformation_matrix::AbstractArray,
)::AbstractArray
    return transformation_matrix * material * transformation_matrix' /
           det(transformation_matrix)
end

# ---------------------------------------------------------- #
# Dispersive material polarization initialization
# ---------------------------------------------------------- #

"""
    _collect_unique_poles(geometry) -> Vector{LorentzianSusceptibility}

Collect all unique susceptibility poles from the geometry. Two poles are
considered the same if they have identical (omega_0, gamma) parameters
(the sigma field varies per object/voxel).
"""
function _collect_unique_poles(geometry)
    poles = LorentzianSusceptibility[]
    seen = Set{Tuple{Float64,Float64}}()
    for obj in geometry
        has_susceptibilities(obj.material) || continue
        for s in obj.material.susceptibilities
            key = (s.omega_0, s.gamma)
            if key ∉ seen
                push!(seen, key)
                push!(poles, s)
            end
        end
    end
    return poles
end

"""
    _rasterize_pole_sigma!(sigma_arr, geometry, pole, xs, ys, zs)

Paint the per-voxel sigma (oscillator strength) for a single susceptibility pole
into the 3D array. Objects are painted in reverse order (last = lowest priority).
"""
function _rasterize_pole_sigma!(
    sigma_arr::Array{T,3},
    geometry::Vector{Object},
    pole::LorentzianSusceptibility,
    xs::Vector{Float64},
    ys::Vector{Float64},
    zs::Vector{Float64},
) where {T}
    fill!(sigma_arr, zero(T))
    pole_key = (pole.omega_0, pole.gamma)

    for gi in length(geometry):-1:1
        obj = geometry[gi]
        has_susceptibilities(obj.material) || continue

        # Find matching pole in this object's susceptibilities
        sigma_val = zero(T)
        found = false
        for s in obj.material.susceptibilities
            if (s.omega_0, s.gamma) == pole_key
                sigma_val = T(s.sigma)
                found = true
                break
            end
        end
        found || continue

        shape = obj.shape
        b = bounds(shape)
        ix_lo = max(searchsortedfirst(xs, b[1][1]), 1)
        ix_hi = min(searchsortedlast(xs, b[2][1]), length(xs))
        iy_lo = max(searchsortedfirst(ys, b[1][2]), 1)
        iy_hi = min(searchsortedlast(ys, b[2][2]), length(ys))
        iz_lo = max(searchsortedfirst(zs, b[1][3]), 1)
        iz_hi = min(searchsortedlast(zs, b[2][3]), length(zs))

        for iz in iz_lo:iz_hi, iy in iy_lo:iy_hi, ix in ix_lo:ix_hi
            point = SVector(xs[ix], ys[iy], zs[iz])
            if point ∈ shape
                @inbounds sigma_arr[ix, iy, iz] = sigma_val
            end
        end
    end
end

"""
    init_polarization!(sim::SimulationData)

Initialize dispersive polarization data for all chunks. Creates per-pole
sigma arrays by rasterizing the geometry, allocates P/P_prev field arrays,
and computes ADE coefficients.

Must be called after `create_all_chunks` / `create_local_chunks` and after
`init_geometry`.
"""
function init_polarization!(sim::SimulationData)
    isnothing(sim.chunk_data) && return
    !any_material_has_susceptibilities(sim.geometry) && return

    geometry = sim.geometry
    poles = _collect_unique_poles(geometry)
    isempty(poles) && return

    @info("  Initializing $(length(poles)) dispersive polarization pole(s)...")

    # Pre-compute coordinate arrays for center-positioned Yee component (Ex used as reference)
    gv_ref = GridVolume(sim, Ex())
    origin = get_component_origin(sim, Ex())
    gv_origin = get_min_corner(gv_ref)

    xs = _build_coords(sim.Δx, origin[1], gv_origin[1], gv_ref.Nx)
    ys = _build_coords(sim.Δy, origin[2], gv_origin[2], gv_ref.Ny)
    zs = if sim.ndims == 3
        _build_coords(sim.Δz, origin[3], gv_origin[3], gv_ref.Nz)
    else
        [0.0]
    end

    # Get the array dimensions (same as used in field allocation)
    ref_dims = get_component_voxel_count(sim, Ex())
    arr_nx, arr_ny = ref_dims[1], ref_dims[2]
    arr_nz = sim.ndims == 3 ? ref_dims[3] : 1

    # Truncate coordinate arrays to match array dimensions
    xs = length(xs) > arr_nx ? xs[1:arr_nx] : xs
    ys = length(ys) > arr_ny ? ys[1:arr_ny] : ys
    zs = length(zs) > arr_nz ? zs[1:arr_nz] : zs

    T = backend_number

    # Rasterize sigma arrays for each pole on CPU
    pole_sigma_cpu = Array{T,3}[]
    for pole in poles
        sigma = Array{T}(undef, length(xs), length(ys), length(zs))
        _rasterize_pole_sigma!(sigma, geometry, pole, xs, ys, zs)
        push!(pole_sigma_cpu, sigma)
    end

    # Zero out ADE sigma inside PML regions.
    # Dispersive materials inside PML cause exponential instability because the
    # PML stretched-coordinate damping conflicts with the ADE polarization dynamics.
    # This is a well-known issue in FDTD: dispersive materials must not overlap PML.
    if !isnothing(sim.boundaries)
        half_cell = sim.cell_size ./ 2
        cc = Float64.(sim.cell_center)
        pml_zeroed = 0
        for sigma in pole_sigma_cpu
            for ix in eachindex(xs), iy in eachindex(ys), iz in eachindex(zs)
                sigma[ix, iy, iz] == 0 && continue
                in_pml = false
                # Check x PML
                if length(sim.boundaries) >= 1
                    x_lo = cc[1] - half_cell[1]
                    x_hi = cc[1] + half_cell[1]
                    pml_left_x = sim.boundaries[1][1]
                    pml_right_x = sim.boundaries[1][2]
                    if (pml_left_x > 0 && xs[ix] < x_lo + pml_left_x) ||
                       (pml_right_x > 0 && xs[ix] > x_hi - pml_right_x)
                        in_pml = true
                    end
                end
                # Check y PML
                if !in_pml && length(sim.boundaries) >= 2
                    y_lo = cc[2] - half_cell[2]
                    y_hi = cc[2] + half_cell[2]
                    pml_left_y = sim.boundaries[2][1]
                    pml_right_y = sim.boundaries[2][2]
                    if (pml_left_y > 0 && ys[iy] < y_lo + pml_left_y) ||
                       (pml_right_y > 0 && ys[iy] > y_hi - pml_right_y)
                        in_pml = true
                    end
                end
                # Check z PML
                if !in_pml && length(sim.boundaries) >= 3
                    z_lo = cc[3] - half_cell[3]
                    z_hi = cc[3] + half_cell[3]
                    pml_left_z = sim.boundaries[3][1]
                    pml_right_z = sim.boundaries[3][2]
                    if (pml_left_z > 0 && zs[iz] < z_lo + pml_left_z) ||
                       (pml_right_z > 0 && zs[iz] > z_hi - pml_right_z)
                        in_pml = true
                    end
                end
                if in_pml
                    sigma[ix, iy, iz] = zero(T)
                    pml_zeroed += 1
                end
            end
        end
        if pml_zeroed > 0
            @warn("  Zeroed $pml_zeroed dispersive sigma values inside PML regions " *
                  "(dispersive materials in PML cause instability)")
        end
    end

    # Compute chi1 (semi-implicit ADE correction) per voxel.
    # chi1 = Σ_poles γ₁⁻¹ * C_k * σ_k / 2, where C_k is the per-sigma driving
    # coefficient. This stabilizes the coupled ADE+FDTD system by modifying
    # ε_inv → ε_inv / (1 + ε_inv * chi1). Reference: meep step_generic.cpp.
    chi1_cpu = zeros(T, length(xs), length(ys), length(zs))
    for (pi, pole) in enumerate(poles)
        coeffs = compute_ade_coefficients(pole, sim.Δt)
        c = if coeffs.is_drude
            T(coeffs.gamma1_inv * coeffs.drude_coeff / 2)
        else
            T(coeffs.gamma1_inv * coeffs.sigma_omega0_dt_sq / 2)
        end
        @info("  chi1 pole $pi: c=$c, sigma_max=$(maximum(pole_sigma_cpu[pi])), dt=$(sim.Δt), drude_coeff=$(coeffs.drude_coeff), gamma1_inv=$(coeffs.gamma1_inv)")
        chi1_cpu .+= pole_sigma_cpu[pi] .* c
    end

    # For each chunk, create PolarizationData
    for chunk in sim.chunk_data
        !chunk.spec.physics.has_polarizability && continue

        chunk_gv = chunk.spec.grid_volume
        nr = chunk.ndrange

        # Allocate fPDx/fPDy/fPDz on the chunk's fields if not already done
        if isnothing(chunk.fields.fPDx)
            field_dims = (nr[1] + 2, nr[2] + 2, nr[3] + 2)
            chunk.fields = Fields{AbstractArray}(;
                # Copy all existing fields
                [(fn, getfield(chunk.fields, fn)) for fn in fieldnames(Fields)
                 if fn ∉ (:fPDx, :fPDy, :fPDz)]...,
                fPDx = KernelAbstractions.zeros(backend_engine, T, field_dims...),
                fPDy = KernelAbstractions.zeros(backend_engine, T, field_dims...),
                fPDz = KernelAbstractions.zeros(backend_engine, T, field_dims...),
            )
        end

        # Create per-pole state
        pole_states = PolarizationPoleState[]
        for (pi, pole) in enumerate(poles)
            coeffs = compute_ade_coefficients(pole, sim.Δt)

            field_dims = (nr[1] + 2, nr[2] + 2, nr[3] + 2)

            # Slice the global sigma array to this chunk's region
            # chunk_gv.start_idx gives the offset in the global array
            si = chunk_gv.start_idx
            chunk_sigma_cpu = pole_sigma_cpu[pi][
                si[1]:min(si[1]+nr[1]-1, size(pole_sigma_cpu[pi], 1)),
                si[2]:min(si[2]+nr[2]-1, size(pole_sigma_cpu[pi], 2)),
                si[3]:min(si[3]+nr[3]-1, size(pole_sigma_cpu[pi], 3)),
            ]

            # Transfer to GPU
            sigma_gpu = backend_array(chunk_sigma_cpu)

            push!(pole_states, PolarizationPoleState(
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Px
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Py
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Pz
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Px_prev
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Py_prev
                KernelAbstractions.zeros(backend_engine, T, field_dims...),  # Pz_prev
                sigma_gpu,  # sigma_x (using same sigma for all components; isotropic)
                sigma_gpu,  # sigma_y
                sigma_gpu,  # sigma_z
                coeffs,
            ))
        end

        chunk.polarization_data = PolarizationData(pole_states)

        # Apply chi1 semi-implicit correction to ε_inv arrays.
        # This modifies ε_inv → ε_inv / (1 + ε_inv * chi1) to stabilize
        # the coupled ADE+FDTD system for dispersive materials.
        chi1_si = chunk_gv.start_idx
        chunk_chi1 = chi1_cpu[
            chi1_si[1]:min(chi1_si[1]+nr[1]-1, size(chi1_cpu, 1)),
            chi1_si[2]:min(chi1_si[2]+nr[2]-1, size(chi1_cpu, 2)),
            chi1_si[3]:min(chi1_si[3]+nr[3]-1, size(chi1_cpu, 3)),
        ]
        if maximum(abs, chunk_chi1) > 0
            g = chunk.geometry_data
            @info("  chi1 max: $(maximum(chunk_chi1)), nonzero: $(count(x->x!=0, chunk_chi1))")
            # Download ε_inv from GPU (or create from scalar)
            if isnothing(g.ε_inv_x)
                # Scalar ε_inv — expand to per-voxel arrays
                eps_val = isnothing(g.ε_inv) ? one(T) : T(g.ε_inv)
                eps_x = fill(eps_val, nr...)
                eps_y = fill(eps_val, nr...)
                eps_z = fill(eps_val, nr...)
                @info("  ε_inv: scalar $(eps_val)")
            else
                eps_x = Array(g.ε_inv_x)
                eps_y = Array(g.ε_inv_y)
                eps_z = Array(g.ε_inv_z)
                @info("  ε_inv_x before chi1: min=$(minimum(eps_x)) max=$(maximum(eps_x))")
            end

            # Apply: ε_inv_eff = ε_inv / (1 + ε_inv * chi1)
            for idx in eachindex(chunk_chi1)
                c1 = chunk_chi1[idx]
                c1 == 0 && continue
                eps_x[idx] /= (1 + eps_x[idx] * c1)
                eps_y[idx] /= (1 + eps_y[idx] * c1)
                eps_z[idx] /= (1 + eps_z[idx] * c1)
            end

            # Re-upload and replace geometry_data (use existing array type parameter)
            AT = typeof(g).parameters[2]  # array type from GeometryData{T,A}
            chunk.geometry_data = GeometryData{T,AT}(;
                [(fn, getfield(g, fn)) for fn in fieldnames(GeometryData)
                 if fn ∉ (:ε_inv, :ε_inv_x, :ε_inv_y, :ε_inv_z)]...,
                ε_inv = nothing,  # now per-voxel
                ε_inv_x = backend_array(eps_x),
                ε_inv_y = backend_array(eps_y),
                ε_inv_z = backend_array(eps_z),
            )
        end
    end
end
