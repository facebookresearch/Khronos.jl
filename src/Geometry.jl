
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


"""
For now, let's just assume diagonal materials, so that the inverse is easy to
compute!

TODO: better input checking and sanitizing...
"""

function get_perm_inv(obj::Object, f::Field)
    perm = get_material_perm_comp(obj.material,f)
    if isnothing(perm)
        perm = get_material_perm(obj.material,f)
    end
    return 1.0 / perm
end

function get_σ(obj::Object, f::Field)
    σ = get_material_conductivity_comp(obj.material,f)
    if isnothing(σ)
        σ = get_material_conductivity(obj.material,f)
    end
    if isnothing(σ)
        σ = 0.0
    end
    return σ
end

"""
TODO: Eventually we want to set up a "default material" to call...
"""
function get_perm_inv(obj::Nothing, ::Field)
    1.0
end

function get_σ(obj::Nothing, ::Field)
    0.0
end

# ---------------------------------------------------------- #
# Geometry data initialization
# ---------------------------------------------------------- #

function pull_geometry(sim::SimulationData, geometry::Vector{Object}, index, f::Field, idx, ε_inv, σD)
    if isnothing(index)
        obj = nothing
    else
        obj = geometry[index]
    end

    if !isnothing(ε_inv)
        ε_inv[idx...] = get_perm_inv(obj,f)
    end

    if !isnothing(σD)
        σD[idx...] = get_σ(obj,f)
    end

    return
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
    return !isnothing(get_material_perm(obj.material,f)) || !isnothing(get_material_perm_comp(obj.material,f))
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
    return !isnothing(get_material_conductivity(obj.material, f)) || !isnothing(get_material_conductivity_comp(obj.material, f))
end

function needs_conductivities(geometry::Vector{Object}, f::Field)
    needs_cond = false
    for obj in geometry
        needs_cond = needs_cond || needs_conductivities(obj, f)
    end
    return needs_cond
end

# isotropic materials
function init_geometry(sim::SimulationData, geometry::Vector{Object})
    if length(geometry) == 0
        init_geometry(sim, nothing)
        return
    end

    ε_inv_x = needs_perm(geometry,Ex()) ? zeros(get_component_voxel_count(sim, Ex())[1:sim.ndims]...) : nothing
    ε_inv_y = needs_perm(geometry,Ey()) ? zeros(get_component_voxel_count(sim, Ey())[1:sim.ndims]...) : nothing
    ε_inv_z = needs_perm(geometry,Ez()) ? zeros(get_component_voxel_count(sim, Ez())[1:sim.ndims]...) : nothing

    σDx = needs_conductivities(geometry,Dx()) ? zeros(get_component_voxel_count(sim, Ex())[1:sim.ndims]...) : nothing
    σDy = needs_conductivities(geometry,Dy()) ? zeros(get_component_voxel_count(sim, Ey())[1:sim.ndims]...) : nothing
    σDz = needs_conductivities(geometry,Dz()) ? zeros(get_component_voxel_count(sim, Ez())[1:sim.ndims]...) : nothing

    μ_inv_x = needs_perm(geometry,Hx()) ? zeros(get_component_voxel_count(sim, Hx())[1:sim.ndims]...) : nothing
    μ_inv_y = needs_perm(geometry,Hy()) ? zeros(get_component_voxel_count(sim, Hy())[1:sim.ndims]...) : nothing
    μ_inv_z = needs_perm(geometry,Hz()) ? zeros(get_component_voxel_count(sim, Hz())[1:sim.ndims]...) : nothing

    σBx = needs_conductivities(geometry,Bx()) ? zeros(get_component_voxel_count(sim, Hx())[1:sim.ndims]...) : nothing
    σBy = needs_conductivities(geometry,By()) ? zeros(get_component_voxel_count(sim, Hy())[1:sim.ndims]...) : nothing
    σBz = needs_conductivities(geometry,Bz()) ? zeros(get_component_voxel_count(sim, Hz())[1:sim.ndims]...) : nothing

    """
    there are a lot of ways we could loop over the all the proper points.
    Ideally, we would do a _single_ loop and just be careful with how we update
    the different regions of each voxel. That requires some thought (and
    additional consideration w.r.t. branching) and probably isn't necessary
    until it really _is_ a performance bottleneck. So we'll just flag this as a
    potential performance improvement, and do the easy thing for now.
    TODO: make faster
    """

    # loop over x
    gv_x = GridVolume(sim,Ex())
    for iz=1:gv_x.Nz, iy=1:gv_x.Ny, ix=1:gv_x.Nx
        point = grid_volume_idx_to_point(sim, gv_x, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), Dx(), idx,
            ε_inv_x, σDx)
    end

    gv_x = GridVolume(sim,Hx())
    for iz=1:gv_x.Nz, iy=1:gv_x.Ny, ix=1:gv_x.Nx
        point = grid_volume_idx_to_point(sim, gv_x, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), Bx(), idx,
            μ_inv_x, σBx)
    end

    # loop over y
    gv_y = GridVolume(sim,Ey())
    for iz=1:gv_y.Nz, iy=1:gv_y.Ny, ix=1:gv_y.Nx
        point = grid_volume_idx_to_point(sim, gv_y, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), Dy(), idx,
            ε_inv_y, σDy)
    end

    gv_y = GridVolume(sim,Hy())
    for iz=1:gv_y.Nz, iy=1:gv_y.Ny, ix=1:gv_y.Nx
        point = grid_volume_idx_to_point(sim, gv_y, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), By(), idx,
            μ_inv_y, σBy)
    end

    # loop over z
    gv_z = GridVolume(sim,Ez())
    for iz=1:gv_z.Nz, iy=1:gv_z.Ny, ix=1:gv_z.Nx
        point = grid_volume_idx_to_point(sim, gv_z, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), Dz(), idx,
            ε_inv_z, σDz)
    end

    gv_z = GridVolume(sim,Hz())
    for iz=1:gv_z.Nz, iy=1:gv_z.Ny, ix=1:gv_z.Nx
        point = grid_volume_idx_to_point(sim, gv_z, [ix, iy, iz])
        idx = [ix,iy,iz][1:sim.ndims]
        pull_geometry(sim, geometry, findfirst(point, geometry), Bz(), idx,
            μ_inv_z, σBz)
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

        μ_inv = 1.0, # todo add μ support
        # right now it's either zeros or nothing...
        σBx = isnothing(σBx) ? σBx : backend_array(σBx),
        σBy = isnothing(σBy) ? σBy : backend_array(σBy),
        σBz = isnothing(σBz) ? σBz : backend_array(σBz),
    )

end

function init_geometry(sim::SimulationData, ::Nothing)
    sim.geometry_data = GeometryData{backend_number,backend_array}(ε_inv = 1.0, μ_inv = 1.0)
end

""" get_mat_conductivity_from_field()

"""
get_mat_conductivity_from_field(sim::SimulationData,::Union{Bx,Hx}) = sim.geometry_data.σBx
get_mat_conductivity_from_field(sim::SimulationData,::Union{By,Hy}) = sim.geometry_data.σBy
get_mat_conductivity_from_field(sim::SimulationData,::Union{Bz,Hz}) = sim.geometry_data.σBz
get_mat_conductivity_from_field(sim::SimulationData,::Union{Dx,Ex}) = sim.geometry_data.σDx
get_mat_conductivity_from_field(sim::SimulationData,::Union{Dy,Ey}) = sim.geometry_data.σDy
get_mat_conductivity_from_field(sim::SimulationData,::Union{Dz,Ez}) = sim.geometry_data.σDz
get_mat_conductivity_from_field(sim::SimulationData,::Magnetic,::X) = sim.geometry_data.σBx
get_mat_conductivity_from_field(sim::SimulationData,::Magnetic,::Y) = sim.geometry_data.σBy
get_mat_conductivity_from_field(sim::SimulationData,::Magnetic,::Z) = sim.geometry_data.σBz
get_mat_conductivity_from_field(sim::SimulationData,::Electric,::X) = sim.geometry_data.σDx
get_mat_conductivity_from_field(sim::SimulationData,::Electric,::Y) = sim.geometry_data.σDy
get_mat_conductivity_from_field(sim::SimulationData,::Electric,::Z) = sim.geometry_data.σDz
