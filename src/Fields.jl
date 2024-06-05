# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# -------------------------------------------------------------------------- #
# General field array utility functions
# -------------------------------------------------------------------------- #

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
    mapping = Dict(0 => X(), 1 => Y(), 2 => Z())
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
get_fields_from_component(sim::SimulationData, ::Ex) = sim.fields.fEx
get_fields_from_component(sim::SimulationData, ::Ey) = sim.fields.fEy
get_fields_from_component(sim::SimulationData, ::Ez) = sim.fields.fEz
get_fields_from_component(sim::SimulationData, ::Hx) = sim.fields.fHx
get_fields_from_component(sim::SimulationData, ::Hy) = sim.fields.fHy
get_fields_from_component(sim::SimulationData, ::Hz) = sim.fields.fHz
# FIXME: once we have offdiag compoents, the inverse is more complicated...
get_fields_from_component(sim::SimulationData, ::εx) = inv.(sim.geometry_data.ε_inv_x)
get_fields_from_component(sim::SimulationData, ::εy) = inv.(sim.geometry_data.ε_inv_y)
function get_fields_from_component(sim::SimulationData, ::εz)
    inv.(sim.geometry_data.ε_inv_z)
end

# -------------------------------------------------------------------------- #
# General simulation field allocation functions.
#
#
# [smartalecH] Currently, Khronos.jl only supports full 3D simulations. In the
# future, however, it would be nice to support 2D simulations (and
# cylindrically-symmetric simulations). In this case, one could specify sources
# and materials that only require half as many computational resources as a full
# 2D simulation. Specifically, they only require a TE or a TM configuration. We
# can create independent functions that set up field arrays for these disjoint
# cases, and then rely on dispatch to pick the proper configuration once Khronos
# is ready for it. Since 3D is just a superset of the two configurations, we'll
# just call both.
# -------------------------------------------------------------------------- #

"""
    initialize_field_component_array(sim::SimulationData, component::Field)::AbstractArray

Initialize an electromagnetic field array for a given `component` over the
entire simulation (`sim`) domain.

We add a "ghost" pixel to the boundary of our domain to more easily facilitate
boundary conditions with the GPU. In the future, we can use this same formalism
to enable MPI communication between "chunks" of data.

To ensure that these ghost pixels serve as a "drop-in" with the existing stencils,
we use offset arrays and start 1-based indexing after the first ghost pixel.
"""
function initialize_field_component_array(sim::SimulationData, component::Field)::AbstractArray
    array_dims = get_component_voxel_count(sim, component) .+ 2
    return OffsetArray(KernelAbstractions.zeros(backend_engine, backend_number, array_dims...), -1, -1, -1)
end

"""
    allocate_fields_TE(sim)

Allocate the proper field components for a 2D TE simulation (where the E-field is
"out-of-plane"). This includes Dz/Ez, Bx/Hx, By/Hy, and their corresponding
current sources. 
"""
function allocate_fields_TE(sim::SimulationData)
    return (
        fEz = initialize_field_component_array(sim, Ez()),
        fDz = initialize_field_component_array(sim, Dz()),
        # Only allocate PML auxilliary fields if needed
        fCDz = needs_C(sim, Dz()) ? initialize_field_component_array(sim, Dz()) : nothing,
        fUDz = needs_U(sim, Dz()) ? initialize_field_component_array(sim, Dz()) : nothing,
        fWDz = needs_W(sim, Dz()) ? initialize_field_component_array(sim, Dz()) : nothing,
        
        fHx = initialize_field_component_array(sim, Hx()),
        fBx = initialize_field_component_array(sim, Bx()),
        # Only allocate PML auxilliary fields if needed
        fCBx = needs_C(sim, Bx()) ? initialize_field_component_array(sim, Bx()) : nothing,
        fUBx = needs_U(sim, Bx()) ? initialize_field_component_array(sim, Bx()) : nothing,
        fWBx = needs_W(sim, Bx()) ? initialize_field_component_array(sim, Bx()) : nothing,
        
        fHy = initialize_field_component_array(sim, Hy()),
        fBy = initialize_field_component_array(sim, By()),
        # Only allocate PML auxilliary fields if needed
        fCBy = needs_C(sim, By()) ? initialize_field_component_array(sim, By()) : nothing,
        fUBy = needs_U(sim, By()) ? initialize_field_component_array(sim, By()) : nothing,
        fWBy = needs_W(sim, By()) ? initialize_field_component_array(sim, By()) : nothing,
        
        # Only allocate source arrays if we specify sources for that component
        fSBx = Hx() ∈ sim.source_components ? initialize_field_component_array(sim, Hx()) : nothing,
        fSBy = Hy() ∈ sim.source_components ? initialize_field_component_array(sim, Hy()) : nothing,
        fSDz = Ez() ∈ sim.source_components ? initialize_field_component_array(sim, Ez()) : nothing,
    )
end

"""
    allocate_fields_TM(sim::SimulationData)

Allocate the proper field components for a 2D TM simulation (where the E-field
is "in-plane"). This includes Bz/Hz, Dx/Ex, Dy/Ey, and their corresponding
current sources. 
"""
function allocate_fields_TM(sim::SimulationData)
    return (
        fHz = initialize_field_component_array(sim, Hz()),
        fBz = initialize_field_component_array(sim, Bz()),
        # Only allocate PML auxilliary fields if needed
        fCBz = needs_C(sim, Bz()) ? initialize_field_component_array(sim, Bz()) : nothing,
        fUBz = needs_U(sim, Bz()) ? initialize_field_component_array(sim, Bz()) : nothing,
        fWBz = needs_W(sim, Bz()) ? initialize_field_component_array(sim, Bz()) : nothing,
        
        fEx = initialize_field_component_array(sim, Ex()),
        fDx = initialize_field_component_array(sim, Dx()),
        # Only allocate PML auxilliary fields if needed
        fCDx = needs_C(sim, Dx()) ? initialize_field_component_array(sim, Dx()) : nothing,
        fUDx = needs_U(sim, Dx()) ? initialize_field_component_array(sim, Dx()) : nothing,
        fWDx = needs_W(sim, Dx()) ? initialize_field_component_array(sim, Dx()) : nothing,
        
        fEy = initialize_field_component_array(sim, Ey()),
        fDy = initialize_field_component_array(sim, Dy()),
        # Only allocate PML auxilliary fields if needed
        fCDy = needs_C(sim, Dy()) ? initialize_field_component_array(sim, Dy()) : nothing,
        fUDy = needs_U(sim, Dy()) ? initialize_field_component_array(sim, Dy()) : nothing,
        fWDy = needs_W(sim, Dy()) ? initialize_field_component_array(sim, Dy()) : nothing,

        # Only allocate source arrays if we specify sources for that component
        fSDx = Ex() ∈ sim.source_components ? initialize_field_component_array(sim, Ex()) : nothing,
        fSDy = Ey() ∈ sim.source_components ? initialize_field_component_array(sim, Ey()) : nothing,
        fSBz = Hz() ∈ sim.source_components ? initialize_field_component_array(sim, Hz()) : nothing,
    )
end

"""
    init_fields(sim::SimulationData, ::Union{Type{TwoD},Type{ThreeD}})

Allocate all the necessary field components for a full 3D simulation.
"""
function init_fields(sim::SimulationData, ::Union{Type{ThreeD}})
    sim.fields =
        Fields{AbstractArray}(; allocate_fields_TE(sim)..., allocate_fields_TM(sim)...)
end

# -------------------------------------------------------------------------- #
# PML auxilliary field utility functions
# -------------------------------------------------------------------------- #

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
    return (
        !isnothing(get_mat_conductivity_from_field(sim, f)) && (
            !isnothing(get_pml_conductivity_from_field(sim, f, next_dir(dir))) ||
            !isnothing(get_pml_conductivity_from_field(sim, f, prev_dir(dir)))
        )
    )
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

# -------------------------------------------------------------------------- #
# Specific usecase field utility functions
# -------------------------------------------------------------------------- #

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
        pol_factor =
            is_electric(component) ? e_vector[int_from_direction(component)] :
            h_vector[int_from_direction(component)]
        for idx in CartesianIndices(grid_vols[k])
            pos = point_from_grid_index(sim, gv, idx)
            fields[component][idx] =
                pol_factor * real(exp(im * (sum(pos .* norm_k_vector))))
        end
    end
    return fields
end
