# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
    get_direction_index(direction)

Given a direction, compute the index for that direction.
"""
get_direction_index(::X) = 1
get_direction_index(::Y) = 2
get_direction_index(::Z) = 3

"""
    get_component_voxel_count(sim, component)

For a given component, get the nnumber of grid voxels throughout the domain.
"""
get_component_voxel_count(sim::SimulationData,::Union{Ex,Dx}) = [sim.Nx, sim.Ny + 1, sim.Nz + 1]
get_component_voxel_count(sim::SimulationData,::Union{Ey,Dy}) = [sim.Nx + 1, sim.Ny, sim.Nz + 1]
get_component_voxel_count(sim::SimulationData,::Union{Ez,Dz}) = [sim.Nx + 1, sim.Ny + 1, sim.Nz]
get_component_voxel_count(sim::SimulationData,::Union{Hx,Bx}) = [sim.Nx + 1, sim.Ny, sim.Nz]
get_component_voxel_count(sim::SimulationData,::Union{Hy,By}) = [sim.Nx, sim.Ny + 1, sim.Nz]
get_component_voxel_count(sim::SimulationData,::Union{Hz,Bz}) = [sim.Nx, sim.Ny, sim.Nz + 1]
get_component_voxel_count(sim::SimulationData,::Center) = [sim.Nx, sim.Ny, sim.Nz]

"""create_array_from_volume

To create a discrete array from a continuous volume, we need three things:
    1. The array itself
    2. The weights for the array and how it maps to the volume
    3. The array location in grid coordinates

"""

function create_array_from_volume(sim::SimulationData, volume::Volume, component::Field)
    gv = GridVolume(sim, volume, component)
    create_array_from_gridvolume(sim, gv)
end

# TODO extend to 3D
function create_array_from_gridvolume(sim::SimulationData, gv::GridVolume)
    return zeros(gv.Nx, gv.Ny)
end

function get_gridvolume_dims(gv::GridVolume)
    return [gv.Nx, gv.Ny, gv.Nz]
end

Base.IteratorsMD.CartesianIndices(gv::GridVolume) = CartesianIndices((1:gv.Nx,1:gv.Ny,1:gv.Nz))

function create_arrays_from_surface()

end

function point_from_grid_index(sim::SimulationData, gv::GridVolume, idx_point::Vector{<:Real})
    Δ = [sim.Δx, sim.Δy, sim.Δz]
    origin = get_component_origin(sim, gv.component)
    gv_origin = get_min_corner(gv)
    return origin .+ (gv_origin .+ idx_point - [1, 1, 1]) .* Δ
end

function point_from_grid_index(sim::SimulationData, gv::GridVolume, idx_point::CartesianIndex)
    point_from_grid_index(sim,gv,[Tuple(idx_point)...])
end

function loop_over_grid_volume(gv::GridVolume)
    pos = 
    idx = Base.IteratorsMD.CartesianIndices(gv)
    return (pos, idx)
end

function loop_over_volume()

end

function loop_over_surface()

end

"""
Create a GridVolume (coordinates on a grid) from a 
continuous Volume.
"""

function GridVolume(sim::SimulationData, volume::Volume, component::Field)
    start_idx = get_lower_grid_idx(sim, get_min_corner(volume), component)
    end_idx = get_upper_grid_idx(sim, get_max_corner(volume), component)
    Nx, Ny, Nz = end_idx .- start_idx .+ [1, 1, 1] # index-1 math
    GridVolume(component, start_idx, end_idx, Nx, Ny, Nz)
end

function GridVolume(sim::SimulationData, center::Vector{Real}, size::Vector{Real}, component::Field)
    GridVolume(sim, Volume(center=center,size=size), component)
end

function GridVolume(sim::SimulationData, component::Field)
    GridVolume(sim, Volume(center=sim.cell_center,size=sim.cell_size), component)
end

function GridVolume(sim::SimulationData)
    # Create a grid volume spanning the simulation domain for voxel centers
    GridVolume(sim, Volume(center=sim.cell_center,size=sim.cell_size), Center())
end

"""get_yee_shift(sim, component)

Return a displacement vector to go from a cell origin to that particular component.

Make sure the simulation is prepared first...
"""
get_yee_shift(sim::SimulationData, ::Center) = [0., 0., 0.]
get_yee_shift(sim::SimulationData, ::Union{Dx,Ex,εx}) = [0, -sim.Δy / 2, -sim.Δz / 2]
get_yee_shift(sim::SimulationData, ::Union{Dy,Ey,εy}) = [-sim.Δx / 2.0, 0.0, -sim.Δz / 2.0]
get_yee_shift(sim::SimulationData, ::Union{Dz,Ez,εz}) = [-sim.Δx / 2.0, -sim.Δy / 2.0, 0.0]
get_yee_shift(sim::SimulationData, ::Union{Bx,Hx,μx}) = [-sim.Δx / 2.0, 0.0, 0.0]
get_yee_shift(sim::SimulationData, ::Union{By,Hy,μy}) = [0.0, -sim.Δy / 2, 0.0]
get_yee_shift(sim::SimulationData, ::Union{Bz,Hz,μz}) = [0.0, 0.0, -sim.Δz / 2.0]

"""get_component_origin()

The origin of a component array is defined as the coordinate in the lowest
corner of the cell (-Lx/2-Cx,-Ly/2-Cx,-Lz/2-Cx).

By definition, let's set the extent of the cell size to match the edges
of the edge voxels themselves. This way the resolution specifies how many
voxels are in the domain.
"""

function get_component_origin(sim::SimulationData)
    return -sim.cell_size / 2 .- sim.cell_center .+ [sim.Δx/2, sim.Δy/2, sim.Δz/2] 
end

function get_component_origin(sim::SimulationData, component::Field)
    return get_component_origin(sim) .+ get_yee_shift(sim, component)
end

"""get_grid_idx()

Get the grid index of a point nearest a real location (no rounding)
"""
function get_grid_idx(sim::SimulationData, point::Vector{<:Real}, component::Field)
    Δ = [sim.Δx, sim.Δy, sim.Δz]

    sim_vol = Volume(center=sim.cell_center,size=sim.cell_size)
    corner_to_center = [sim.Δx, sim.Δy, sim.Δz] ./ 2
    min_corner = get_min_corner(sim_vol) + corner_to_center + get_yee_shift(sim, component)
    max_corner = get_max_corner(sim_vol) - corner_to_center - get_yee_shift(sim, component)
    
    point_trimmed = min.(point,max_corner)
    point_trimmed = max.(point_trimmed,min_corner)
    point_idx = (point_trimmed-min_corner) ./ Δ .+ 1 # add 1 due to 1-indexing

    # This is simply a matter of convention. In the case of a 2D
    # simulation, then we result in ∞/∞. In reality, it doesn't matter
    # what we do here, so we'll just return a number that can be operated
    # on down the road, but will throw an error if indexed upon.
    replace!(point_idx, NaN=>0)

    return point_idx
end

"""get_lower_grid_idx()

Get the grid index of a point nearest a real location (rounding down)
"""
function get_lower_grid_idx(sim::SimulationData, point::Vector{<:Real}, component::Field)
    return Int.(floor.( get_grid_idx(sim, point, component)))
end

"""get_upper_grid_idx()

Get the grid index of a point nearest a real location (rounding up)
"""
function get_upper_grid_idx(sim::SimulationData, point::Vector{<:Real}, component::Field)
    return Int.(ceil.( get_grid_idx(sim, point, component)))
end

"""
    grid_volume_idx_to_point(sim,gv,idx_point)

From a `GridVolume` (`gv`) compute the point in continuous coordinates.
"""
function grid_volume_idx_to_point(sim::SimulationData, gv::GridVolume, idx_point)
    # FIXME
    if length(idx_point) < 3
        idx_point = CartesianIndex(idx_point,(0))
    end
    idx_point = collect(Tuple(idx_point))
    origin = get_component_origin(sim, gv.component)
    gv_origin = get_min_corner(gv)
    
    # FIXME
    # Our current convention, combined with Julia's 1-based indexing, means
    # we need to compensate by `2`, not just `1` like we ould expect. We should
    # revisit this, as it's prone to errors in the future...
    offset = (idx_point + gv_origin .- 2) .* [sim.Δx, sim.Δy, sim.Δz]
    point = origin .+ offset
    # note that if our simulation is 2D, we will get a 0×∞ op, resulting in a
    # NaN. We need to account for that manually (*sigh*). The point we choose to
    # fill it with doesn't actually matter (one might prefer to leave it as NaN)
    # but for usability later, we'll replace it with the simulation center.
    point[3] = (isinf(point[3]) || isnan(point[3])) ? sim.cell_center[3] : point[3]
    return point
end

"""
    sim_idx_to_point()

Return the point in real coordinates that corresponds to a simulation voxel index.
"""
@inline function sim_idx_to_point(sim::SimulationData, idx_point::Vector{<:Real})
    Δ = [sim.Δx, sim.Δy, sim.Δz]
    return get_component_origin(sim) .+ (idx_point - [1, 1, 1]) .* Δ
end

@inline function get_min_corner(gv::GridVolume)
    # get the corner index relative to the global grid index
    return gv.start_idx
end

@inline function get_min_corner(volume::Volume)
    return volume.center .- volume.size ./ 2
end

@inline function get_max_corner(volume::Volume)
    return volume.center .+ volume.size ./ 2
end

@inline function point_in_volume(point::Vector{Real}, volume::Volume)
    inside = true
    max_corner = get_max_corner(volume)
    min_corner = get_min_corner(volume)
    for k in range(sim.ndims)
        inside = inside && (point[k] < max_corner[k]) && (point[k] > min_corner[k])
    end
    return inside
end


"""
    get_volume_dimensionality(vol::Volume)::Int

Return the dimensionality of a volume (either 0, 1, 2, or 3).
"""
function get_volume_dimensionality(vol::Volume)::Int
    return 3 - count(==(0.0), vol.size)
end

function check_if_plane(vol::Volume)::Bool
    return get_volume_dimensionality(vol) == 2
end

"""
plane_normal_direction(vol::Volume)::Int

TBW
"""
function plane_normal_direction(vol::Volume)::Int
    if !check_if_plane(vol)
        error("The specified volume is not a plane (size=$(vol.size))")
    end

    return argmin(vol.size)
end

function get_normal_vector(vol::Volume)::Vector{<:Real}
    normal_direction = plane_normal_direction(vol)
    normal_vector = [0.0, 0.0, 0.0]
    normal_vector[normal_direction] = 1.0

    return normal_vector
end

"""
    get_plane_transverse_fields(vol::Volume)

Identify the transverse field components for a particular planar volume.
"""
function get_plane_transverse_fields(vol::Volume)
    if get_volume_dimensionality(vol) != 2
        error("The specified volume is not a plane (size=$(vol.size))")
    end
    
    normal_dir = plane_normal_direction(vol)

    transverse_field_mapping = Dict(
        1 => [Ey(), Ez(), Hy(), Hz()],
        2 => [Ex(), Ez(), Ex(), Hz()],
        3 => [Ex(), Ey(), Hx(), Hy()],
    )

    return transverse_field_mapping[normal_dir]
end

function normalize_vector(vec::Vector{<:Number})
    return vec / sum(abs.(vec).^2).^2
end

"""
    compute_interpolation_weight(point, volume, ndims, Δx, Δy, Δz)

Calculates the interpolation weight for a continuous `volume` at a `point`.

TODO: In the future, we may want to refactor such that a new function accepts a
`GridVolume` and the actual `Volume`. From these two quantites alone, we can
determine 4 unique weighting factor (in each direction) which perfectly
characterize the weighting routine.

Then, we have another routine that takes a `point` and a `Volume` (the current
version of this), and creates a `GridVolume` from the `Volume`. This is a bit
more expensive, however, and requires constructing a Simulation object in order
to snap the `GridVolume` to the right grid. We usually need to do this anyway,
however, since we are evaluating these points on a particular field grid...

This will get tricky when we have nonuninform gridding. We'll have to think of a
new way to perform the interpolation and restriction, and how to efficiently
pull weights.
"""
function compute_interpolation_weight(point::Vector{<:Real}, volume::Volume,
    ndims::Int, Δx::Union{Real,Nothing}, Δy::Union{Real,Nothing}, Δz::Union{Real,Nothing})
    Δ = [Δx, Δy, Δz]
    weight = 1
    max_corner = get_max_corner(volume)
    min_corner = get_min_corner(volume)

    # The problem is separable by dimension
    for dim in 1:ndims

        if (point[dim] <= (min_corner[dim] - Δ[dim])) || (point[dim] >= (max_corner[dim] + Δ[dim]))
            # If we are further than a pixel away from the boundary, then no weighting
            weight = 0
        else

            if volume.size[dim] == 0.0
                # point source in this dimension, easy
                weight *= 1 - min(abs(point[dim] - volume.center[dim]) / Δ[dim], 1)

            elseif volume.size[dim] < Δ[dim]
                # not a point source, but still subpixel, 5 total cases:
                if (point[dim] >= min_corner[dim]) && (point[dim] <= max_corner[dim])
                    # the point is _inside_ the subpixel volume.
                    # subtract two quadratics, one for each side.
                    weight *= 1 -
                              0.5 * (1.0 - (point[dim] - min_corner[dim]) / Δ[dim])^2 -
                              0.5 * (1.0 - (max_corner[dim] - point[dim]) / Δ[dim])^2
                elseif (point[dim] <= min_corner[dim]) &&
                       (abs(point[dim] - min_corner[dim]) < Δ[dim])
                    # current point is left of left corner...
                    if (max_corner[dim] < (point[dim] + Δ[dim]))
                        # ...and the next point is to the right of the right corner
                        # (e.g. not inside the volume)
                        # one quadratic subtracted from another
                        # the first quadratic should increase as the distance between left point and left corner decreases
                        # the second quadratic should increase as the distance between left point and right corner increases
                        weight *= 0.5 * (1.0 - (min_corner[dim] - point[dim]) / Δ[dim])^2 -
                                  0.5 * (1.0 - (max_corner[dim] - point[dim]) / Δ[dim])^2

                    else
                        # ...and the next point is to the left of the volume corner (inside it)
                        # range should be 0 to 0.5
                        weight *= 0.5 * (1.0 - (min_corner[dim] - point[dim]) / Δ[dim])^2
                    end
                elseif (point[dim] >= max_corner[dim]) && (abs(point[dim] - max_corner[dim]) < Δ[dim])
                    # point is right of right corner...
                    if (min_corner[dim] > (point[dim] - Δ[dim]))
                        # ... and the point before is before the left corner
                        # (e.g. not inside the volume)
                        # one quadratic subtracted from another
                        # the first quadratic should increase as the distance between right point and right corner decreases
                        # the second quadratic should increase as the distance between right point and left corner increases
                        weight *= 0.5 * (1.0 - (point[dim] - max_corner[dim]) / Δ[dim])^2 -
                                  0.5 * (1.0 - (point[dim] - min_corner[dim]) / Δ[dim])^2
                    else
                        # ... and the point before is after the left corner
                        # (e.g. inside the volume)
                        # range should be from 0 to 0.5
                        weight *= 0.5 * (1.0 - (point[dim] - max_corner[dim]) / Δ[dim])^2
                    end
                end
            else
                # region spans beyond one pixel, four cases
                if (point[dim] < min_corner[dim]) && (abs(point[dim] - min_corner[dim]) < Δ[dim])
                    # point left of left corner, but within 1 pixel radius
                    # range is quadratic from 0 to 0.5
                    weight *= 0.5 * (1.0 - (min_corner[dim] - point[dim]) / Δ[dim])^2
                elseif (point[dim] >= min_corner[dim]) && (abs(point[dim] - min_corner[dim]) < Δ[dim])
                    # point right of left corner, but within 1 pixel radius
                    # range is quadratic from 0.5 to 1
                    weight *= 1 - 0.5 * (1.0 - (point[dim] - min_corner[dim]) / Δ[dim])^2
                elseif (point[dim] <= max_corner[dim]) && (abs(point[dim] - max_corner[dim]) < Δ[dim])
                    # point left of right corner, but within 1 pixel radius
                    # range is quadratic from 0.5 to 1
                    weight *= 1 - 0.5 * (1.0 - (max_corner[dim] - point[dim]) / Δ[dim])^2
                elseif (point[dim] > max_corner[dim]) && (abs(point[dim] - max_corner[dim]) < Δ[dim])
                    # point right of right corner, but within 1 pixel radius
                    # range is quadratic from 0 to 0.5
                    weight *= 0.5 * (1.0 - (point[dim] - max_corner[dim]) / Δ[dim])^2
                end
            end
        end
    end
    return weight
end

"""
    check_3d_vector(vec::Vector{<:Real})

TBW
"""
function check_3d_vector(vec::Vector{<:Real})
    if length(vec) != 3
        error("Supplied vector is not the right length (length=$(length(vec)))")
    end
end

"""
    DTFT(t, x, f)

Computes the DTFT for a signal `x` at frequencies `f` using `t` as a time sequence.
"""
function DTFT(t, x, f)
    if length(t) != length(x)
        error("The time (t) and signal (x) vectors must be the same length (lengths $(length(t)) and $(length(f)))")
    end

    N = length(t)
    Nf = length(f)
    X = zeros(ComplexF64, Nf)
    Δt = t[2] - t[1]

    for n in 1:Nf
        X[n] = sum(Δt / sqrt(2.0 * π) * exp.(-im * 2 * π * f[n] * t) .* x)
    end

    return X
end

# ---------------------------------------------- #
# Interpolation routines
# ---------------------------------------------- #

"""
    get_neighbors(x::AbstractArray, point::Float64)

TBW
"""
function get_neighbors(x::AbstractArray, point::Float64)
    # get closest point
    closest_idx = argmin(abs.(x-point))
    # determine which side and get the other point
    if (x[closest_idx] == point) || (closest_idx) == 1 || (closest_idx == length(x))
        left_idx = right_idx = closest_idx
    elseif x[closest_idx] > point
        left_idx = x[closest_idx-1]
        right_idx = x[closest_idx]
    else
        left_idx = x[closest_idx]
        right_idx = x[closest_idx+1]
    end

    return (left_idx, right_idx)
end

"""
    linear_interpolator(x::AbstractArray, y::AbstractArray, point::Float64)

TBW
"""
function linear_interpolator(x::AbstractArray, y::AbstractArray, point::Float64)    
    left_idx, right_idx = get_neighbors(x, point)
    
    if left_idx == right_idx
        return y[left_idx]
    end
    
    Δx = x[right_idx] - x[left_idx]
    left_weight = (point - x[left_idx]) / Δx
    right_weight = (x[right_idx] - point) / Δx

    return left_weight * y[left_idx] + right_weight * y[right_idx]
end

"""
    bilinear_interpolator(x1::AbstractArray, x2::AbstractArray, y::AbstractArray, point::AbstractArray)

TBW
"""
function bilinear_interpolator(x1::AbstractArray, x2::AbstractArray, y::AbstractArray, point::AbstractArray)
    bottom_idx, top_idx = get_neighbors(x2, point[2])
    
    # linearly interpolate top x1
    top_val = linear_interpolator(x1, y[:,top_idx], point[1])

    # linearly interpolate bottom x1
    bottom_val = linear_interpolator(x1, y[:,bottom_idx], point[1])

    # manually linearly interpolate x2
    Δy = x2[top_idx] - x2[bottom_idx]
    bottom_weight = (point[2] - x2[bottom_idx]) / Δy
    top_weight = (x2[top_idx] - point[2]) / Δy
    
    return bottom_weight * bottom_val + top_weight * top_val
end

"""
    trilinear_interpolator(x1::AbstractArray, x2::AbstractArray, x3::AbstractArray, y::AbstractArray, point::AbstractArray)

TBW
"""
function trilinear_interpolator(x1::AbstractArray, x2::AbstractArray, x3::AbstractArray, y::AbstractArray, point::AbstractArray)
    bottom_idx, top_idx = get_neighbors(x3, point[3])
    
    # bilinearly interpolate top (x1,x2)
    top_val = bilinear_interpolator(x1, x2, y[:,:,top_idx], point[1:2])

    # bilinearly interpolate bottom (x1,x2)
    bottom_val = bilinear_interpolator(x1, x2, y[:,:,bottom_idx], point[1:2])

    # manually linearly interpolate x3
    Δz = x3[top_idx] - x3[bottom_idx]
    bottom_weight = (point[3] - x3[bottom_idx]) / Δz
    top_weight = (x3[top_idx] - point[3]) / Δz
    
    return bottom_weight * bottom_val + top_weight * top_val
end

"""
    gen_interpolator_from_array(data::AbstractArray, vol::Volume)

TBW
"""
function gen_interpolator_from_array(data::AbstractArray, vol::Volume)
    
    # make sure array is 3D
    if ndims(data) != 3
        error("The supplied data array must be 3D (supplied $(ndims(data)) dimensions).")
    end

    # pull the grid from the array
    ranges = [
        collect(range(vol.center[k] - vol.size[k]/2, vol.center[k] + vol.size[k]/2, length=size(data)[k])) for k in 1:3
    ]

    function interpolator(point)
        return trilinear_interpolator(ranges[1], ranges[2], ranges[3], data, point)
    end

    return interpolator
end