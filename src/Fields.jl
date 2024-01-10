# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
    next_dir(direction)

Return the next direction from the current direction.
"""
next_dir(::X) = Y()
next_dir(::Y) = Z()
next_dir(::Z) = X()

"""
    prev_dir(direction)

Return the previous direction from the current direction.
"""
prev_dir(::X) = Z()
prev_dir(::Y) = X()
prev_dir(::Z) = Y()

"""
    is_electric(component)

Check if a particular field component is "electric".
"""
is_electric(::Electric) = true
is_electric(::Magnetic) = false

"""
    is_magnetic(component)

Check if a particular field component is "magnetic".
"""
is_magnetic(::Electric) = false
is_magnetic(::Magnetic) = true

"""
    is_TE()

Check if a particular field component corresponds to a TE mode.
"""
is_TE(::Union{Ez,Dz,Hx,Bx,Hy,By}) = true
is_TE(::Union{Ex,Dx,Ey,Dy,Hz,Bz}) = false

"""
    is_TM()

Check if a particular field component corresponds to a TE mode.
"""
is_TM(::Union{Ez,Dz,Hx,Bx,Hy,By}) = false
is_TM(::Union{Ex,Dx,Ey,Dy,Hz,Bz}) = true

"""
    direction_from_field(component)

Infer the direction from a particular field component.
"""
direction_from_field(::Union{Ex,Dx,Hx,Bx}) = X()
direction_from_field(::Union{Ey,Dy,Hy,By}) = Y()
direction_from_field(::Union{Ez,Dz,Hz,Bz}) = Z()

"""
    direction_from_int(dir::Int)::Direction

Provide the direction that maps to an integer.
"""
function direction_from_int(dir::Int)::Direction
    mapping = Dict(
        0 => X(),
        1 => Y(),
        2 => Z(),
    )
    return mapping[dir]
end

int_from_direction(dir::Field)::Int = int_from_direction(direction_from_field(dir))
int_from_direction(dir::X)::Int = 1
int_from_direction(dir::Y)::Int = 2
int_from_direction(dir::Z)::Int = 3

"""
    get_fields_from_component(sim, component)

Given a field component, return the simulation fields array that corresponds to
that component.
"""
get_fields_from_component(sim::SimulationData,::Ex) = sim.fields.fEx
get_fields_from_component(sim::SimulationData,::Ey) = sim.fields.fEy
get_fields_from_component(sim::SimulationData,::Ez) = sim.fields.fEz
get_fields_from_component(sim::SimulationData,::Hx) = sim.fields.fHx
get_fields_from_component(sim::SimulationData,::Hy) = sim.fields.fHy
get_fields_from_component(sim::SimulationData,::Hz) = sim.fields.fHz
# FIXME: once we have offdiag compoents, the inverse is more complicated...
get_fields_from_component(sim::SimulationData,::εx) = inv.(sim.geometry_data.ε_inv_x)
get_fields_from_component(sim::SimulationData,::εy) = inv.(sim.geometry_data.ε_inv_y)
get_fields_from_component(sim::SimulationData,::εz) = inv.(sim.geometry_data.ε_inv_z)

"""
    get_fields_args_TE(sim)

Pull the proper fields arguments for a 2D TE simulation.
"""
function get_fields_args_TE(sim::SimulationData)
    return (
        fEz=zeros(get_component_voxel_count(sim, Ez())[1:sim.ndims]...),
        fDz=zeros(get_component_voxel_count(sim, Dz())[1:sim.ndims]...),
        fCDz=needs_C(sim, Dz()) ? zeros(get_component_voxel_count(sim, Dz())[1:sim.ndims]...) : nothing,
        fUDz=needs_U(sim, Dz()) ? zeros(get_component_voxel_count(sim, Dz())[1:sim.ndims]...) : nothing,
        fWDz=needs_W(sim, Dz()) ? zeros(get_component_voxel_count(sim, Dz())[1:sim.ndims]...) : nothing,
        fHx=zeros(get_component_voxel_count(sim, Hx())[1:sim.ndims]...),
        fBx=zeros(get_component_voxel_count(sim, Bx())[1:sim.ndims]...),
        fCBx=needs_C(sim, Bx()) ? zeros(get_component_voxel_count(sim, Bx())[1:sim.ndims]...) : nothing,
        fUBx=needs_U(sim, Bx()) ? zeros(get_component_voxel_count(sim, Bx())[1:sim.ndims]...) : nothing,
        fWBx=needs_W(sim, Bx()) ? zeros(get_component_voxel_count(sim, Bx())[1:sim.ndims]...) : nothing,
        fHy=zeros(get_component_voxel_count(sim, Hy())[1:sim.ndims]...),
        fBy=zeros(get_component_voxel_count(sim, By())[1:sim.ndims]...),
        fCBy=needs_C(sim, By()) ? zeros(get_component_voxel_count(sim, By())[1:sim.ndims]...) : nothing,
        fUBy=needs_U(sim, By()) ? zeros(get_component_voxel_count(sim, By())[1:sim.ndims]...) : nothing,
        fWBy=needs_W(sim, By()) ? zeros(get_component_voxel_count(sim, By())[1:sim.ndims]...) : nothing,
        fSBx=Hx() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Hx())[1:sim.ndims]...) : nothing,
        fSBy=Hy() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Hy())[1:sim.ndims]...) : nothing,
        fSDz=Ez() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Ez())[1:sim.ndims]...) : nothing,
     )
end

"""
    get_fields_args_TM(sim)

Pull the proper fields arguments for a 2D TM simulation.
"""
function get_fields_args_TM(sim::SimulationData)
    return (
        fHz=zeros(get_component_voxel_count(sim, Hz())[1:sim.ndims]...),
        fBz=zeros(get_component_voxel_count(sim, Bz())[1:sim.ndims]...),
        fCBz=needs_C(sim, Bz()) ? zeros(get_component_voxel_count(sim, Bz())[1:sim.ndims]...) : nothing,
        fUBz=needs_U(sim, Bz()) ? zeros(get_component_voxel_count(sim, Bz())[1:sim.ndims]...) : nothing,
        fWBz=needs_W(sim, Bz()) ? zeros(get_component_voxel_count(sim, Bz())[1:sim.ndims]...) : nothing,
        fEx=zeros(get_component_voxel_count(sim, Ex())[1:sim.ndims]...),
        fDx=zeros(get_component_voxel_count(sim, Dx())[1:sim.ndims]...),
        fCDx=needs_C(sim, Dx()) ? zeros(get_component_voxel_count(sim, Dx())[1:sim.ndims]...) : nothing,
        fUDx=needs_U(sim, Dx()) ? zeros(get_component_voxel_count(sim, Dx())[1:sim.ndims]...) : nothing,
        fWDx=needs_W(sim, Dx()) ? zeros(get_component_voxel_count(sim, Dx())[1:sim.ndims]...) : nothing,
        fEy=zeros(get_component_voxel_count(sim, Ey())[1:sim.ndims]...),
        fDy=zeros(get_component_voxel_count(sim, Dy())[1:sim.ndims]...),
        fCDy=needs_C(sim, Dy()) ? zeros(get_component_voxel_count(sim, Dy())[1:sim.ndims]...) : nothing,
        fUDy=needs_U(sim, Dy()) ? zeros(get_component_voxel_count(sim, Dy())[1:sim.ndims]...) : nothing,
        fWDy=needs_W(sim, Dy()) ? zeros(get_component_voxel_count(sim, Dy())[1:sim.ndims]...) : nothing,
        fSDx=Ex() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Ex())[1:sim.ndims]...) : nothing,
        fSDy=Ey() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Ey())[1:sim.ndims]...) : nothing,
        fSBz=Hz() ∈ sim.source_components ? zeros(get_component_voxel_count(sim, Hz())[1:sim.ndims]...) : nothing,
    )
end

function init_fields(sim::SimulationData, ::Union{Type{TwoD},Type{ThreeD}})
    sim.fields = Fields{backend_array}(; get_fields_args_TE(sim)..., get_fields_args_TM(sim)...)
end

"""
    needs_C(sim::SimulationData, f::Field)

We need `C` auxilliary fields if we have PML _and_ we have conductivites.
Specifically, whether we have PML or not can be a bit tricky, as we don't just
look at the same direction as the current component...rather we need to look at
the direction in the _next_ component. Intuitively, this is because

`dC_k/dt +σ_{D}C = K.`
`dU_k/dt +σ_{k+1}U = dC_k/dt.`

So if `σ_{k+1} = 0``, then `U = C`` and is redundant.
"""
function needs_C(sim::SimulationData, f::Field)
    dir = direction_from_field(f)
    return (!isnothing(get_mat_conductivity_from_field(sim, f)) &&
        (   !isnothing(get_pml_conductivity_from_field(sim, f, next_dir(dir))) ||
            !isnothing(get_pml_conductivity_from_field(sim, f, prev_dir(dir)))))
end

"""
    needs_U(sim::SimulationData, f::Field)

We only need `U` auxilliary fields if we have PML. Specifically, whether we have
PML or not can be a bit tricky, as we don't just look at the same direction
as the current component...rather we need to look at the direction in the
_next_ component. Intuitively, this is because

`dU_k/dt +σ_{k+1}U = dC_k/dt.`

So if `σ_{k-1} = 0``, then `D = U`` and is redundant.
"""
function needs_U(sim::SimulationData, f::Field)
    dir = direction_from_field(f)
    return !isnothing(get_pml_conductivity_from_field(sim, f, next_dir(dir)))
end

"""
    needs_W(sim::SimulationData, f::Field)

We only need `W` auxilliary fields if we have PML.

`dE_k/dt = dW/dt + σ_kW.`

So if `σ_{k} = 0``, then `E = W`` and is redundant.
"""
function needs_W(sim::SimulationData, f::Field)
    dir = direction_from_field(f)
    return !isnothing(get_pml_conductivity_from_field(sim, f, dir))
end


"""
    get_fields_for_planewave(
    sim::SimulationData, 
    k_vector::Vector{<:Real}, 
    polarization::Vector{<:Real}, 
    vol::Volume,
    )

TBW
"""
function get_fields_for_planewave(
    sim::SimulationData, 
    k_vector::Vector{<:Real}, 
    polarization::Vector{<:Real}, 
    vol::Volume,
    )
    # Normalized direction of propagation
    check_3d_vector(k_vector)
    norm_k_vector = normalize_vector(k_vector)
    
    # polarization weighting vectors for E and H
    check_3d_vector(polarization)
    e_vector = normalize_vector(polarization)
    h_vector = cross(norm_k_vector, e_vector)
    
    normal_direction = plane_normal_direction(vol)
    transverse_components = get_plane_transverse_fields(vol)
    grid_vols = [GridVolume(sim, vol, c) for c in transverse_components]
    #fields = [create_array_from_gridvolume(sim, gv) for gv in grid_vols]
    fields = Dict(gv.component => create_array_from_gridvolume(sim, gv) for gv in grid_vols)
    
    # The E field will be defined by the polarization vector
    for component in transverse_components
        pol_factor = is_electric(component) ? e_vector[int_from_direction(component)] : h_vector[int_from_direction(component)]
        for idx = CartesianIndices(grid_vols[k])
            pos = point_from_grid_index(sim, gv, idx)
            fields[component][idx] = pol_factor * real(exp(im*(sum(pos.*norm_k_vector))))
        end
    end
    return fields
end

struct FieldProfile{T}
    Ex::Vector{T}
    Ey::Vector{T}
    Ez::Vector{T}
    Hx::Vector{T}
    Hy::Vector{T}
    Hz::Vector{T}
    gv_Ex::GridVolume
    gv_Ey::GridVolume
    gv_Ez::GridVolume
    gv_Hx::GridVolume
    gv_Hy::GridVolume
    gv_Hz::GridVolume
end

function overlap_integral(field_profile_1::FieldProfile{T}, field_profile_2::FieldProfile{T}) where T
    # interpolate fields to center of yee cell

    # perform the integral

end
