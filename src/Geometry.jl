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
    b = bounds(shape)
    # Map bounding box to grid index ranges via binary search on sorted coords
    ix_lo = max(searchsortedfirst(xs, b[1][1]), 1)
    ix_hi = min(searchsortedlast(xs, b[2][1]), length(xs))
    iy_lo = max(searchsortedfirst(ys, b[1][2]), 1)
    iy_hi = min(searchsortedlast(ys, b[2][2]), length(ys))
    iz_lo = max(searchsortedfirst(zs, b[1][3]), 1)
    iz_hi = min(searchsortedlast(zs, b[2][3]), length(zs))

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
                @inbounds perm_arr[ix, iy] = perm_val
            end
            if has_σ
                @inbounds σ_arr[ix, iy] = σ_val
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
    Δx, Δy, Δz = sim.Δx, sim.Δy, sim.Δz
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
            @inbounds perm_arr[ix, iy] = get_perm_inv(obj, f)
        end
        if !isnothing(σ_arr)
            @inbounds σ_arr[ix, iy] = get_σ(obj, f)
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
    ε_inv_x =
        needs_perm(geometry, Ex()) ?
        zeros(T, get_component_voxel_count(sim, Ex())[1:sim.ndims]...) : nothing
    ε_inv_y =
        needs_perm(geometry, Ey()) ?
        zeros(T, get_component_voxel_count(sim, Ey())[1:sim.ndims]...) : nothing
    ε_inv_z =
        needs_perm(geometry, Ez()) ?
        zeros(T, get_component_voxel_count(sim, Ez())[1:sim.ndims]...) : nothing

    σDx =
        needs_conductivities(geometry, Dx()) ?
        zeros(T, get_component_voxel_count(sim, Ex())[1:sim.ndims]...) : nothing
    σDy =
        needs_conductivities(geometry, Dy()) ?
        zeros(T, get_component_voxel_count(sim, Ey())[1:sim.ndims]...) : nothing
    σDz =
        needs_conductivities(geometry, Dz()) ?
        zeros(T, get_component_voxel_count(sim, Ez())[1:sim.ndims]...) : nothing

    μ_inv_x =
        needs_perm(geometry, Hx()) ?
        zeros(T, get_component_voxel_count(sim, Hx())[1:sim.ndims]...) : nothing
    μ_inv_y =
        needs_perm(geometry, Hy()) ?
        zeros(T, get_component_voxel_count(sim, Hy())[1:sim.ndims]...) : nothing
    μ_inv_z =
        needs_perm(geometry, Hz()) ?
        zeros(T, get_component_voxel_count(sim, Hz())[1:sim.ndims]...) : nothing

    σBx =
        needs_conductivities(geometry, Bx()) ?
        zeros(T, get_component_voxel_count(sim, Hx())[1:sim.ndims]...) : nothing
    σBy =
        needs_conductivities(geometry, By()) ?
        zeros(T, get_component_voxel_count(sim, Hy())[1:sim.ndims]...) : nothing
    σBz =
        needs_conductivities(geometry, Bz()) ?
        zeros(T, get_component_voxel_count(sim, Hz())[1:sim.ndims]...) : nothing

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
    @info("  Rasterizing $(length(geometry)) geometry objects...")

    # Parallelize the 6 independent component loops via Threads.@spawn.
    # Each writes to separate arrays and reads from shared immutable geometry.
    # Gracefully degrades to serial when Threads.nthreads() == 1.
    tasks = Vector{Task}(undef, length(components))
    for (ci, (gv, f, perm_arr, σ_arr)) in enumerate(components)
        xs, ys, zs = _precompute_coords(sim, gv)
        tasks[ci] = Threads.@spawn begin
            if sim.ndims == 3
                _rasterize_geometry_3d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)
            else
                _rasterize_geometry_2d!(geometry, f, perm_arr, σ_arr, xs, ys, zs)
            end
        end
    end
    for t in tasks
        wait(t)
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
