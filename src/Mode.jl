# (c) Meta Platforms, Inc. and affiliates.
#
# Functions involving mode calculations (e.g. modes sources, mode overlaps)


"""
    get_mode_profiles(;
        frequency::Number,
        mode_solver_resolution::Int,
        mode_index::Int,
        center::Vector{<:Real},
        size::Vector{<:Real},
        solver_tolerance::Number,
        geometry::Vector{Object},
        boundary_conditions::Tuple{Int,Int,Int,Int} = (1,1,1,1),
    )::VectorModesolver.Mode

Computes mode profiles for a given `geometry` configuration.

The mode solver handles all solves in the XY plane. In order to take an
arbitrary cross section in 3D (e.g. an XZ or YZ plane), we need to properly
transform the coordinates, materials, _and_ the fields. This is subtle, because
the H-field is a pseudovector, meaning the coordinate handedness is opposite
that of a normal vector system (it uses the left-hand rule). _Luckily_, the
rotation matrices we use are _proper_ rotations (they have a determinant of +1)
so we can use the same transformation for _all_ quantities.

To do so, we establish the following conventions:
* Forward propagation is in the  +X, +Y, or +Z direction (whichever is aligned
    with the monitor's normal direction)
* XY monitors are unchanged (easy)
* XZ monitors are to be rotated 90 degrees around the x-axis, such that the
    "new" forward direction is +Z. In this case, the orignal x-axis maps to the
    new x-axis, and the original z-axis maps to the new y-axis.
* YZ monitors are first rotated 90 degrees around the z-axis, and then 90
    degrees around the x-axis (a repeat of XZ). In this case, the original
    y-axis maps to the new x-axis, and the original z-axis maps to the new
    y-axis.
"""
function get_mode_profiles(;
    frequency::Number,
    mode_solver_resolution::Int,
    mode_index::Int,
    center::Vector{<:Real},
    size::Vector{<:Real},
    solver_tolerance::Number,
    geometry::Vector{Object},
    boundary_conditions::Tuple{Int,Int,Int,Int} = (1, 1, 1, 1),
)::VectorModesolver.Mode
    # Note that the volume specified must be a plane, so we need 2 nonzero dimensions
    @assert count(x -> x != 0, size) >= 2 "The mode region must be a plane (2D)"

    # Extract the proper grid and transformation function depending on the
    # configuration of the monitor.
    if size[1] == 0
        # YZ plane
        idx_x = 2 # Khronos y maps to VectorModesolver x
        idx_y = 3 # Khronos z maps to VectorModesolver y
        new_dims = (3, 1, 2, 4)
        rotate_material = rotate_YZ_to_XZ
        rotate_fields = rotate_XY_to_YZ
    elseif size[2] == 0
        # XZ plane
        idx_x = 1 # Khronos x maps to VectorModesolver x
        idx_y = 3 # Khronos z maps to VectorModesolver y
        new_dims = (1, 3, 2, 4)
        rotate_material = rotate_XZ_to_XY
        rotate_fields = rotate_XY_to_XZ
    elseif size[3] == 0
        # XY plane
        idx_x = 1 # Khronos x maps to VectorModesolver x
        idx_y = 2 # Khronos y maps to VectorModesolver y
        new_dims = (1, 2, 3, 4)
        rotate_material = x -> x
        rotate_fields = x -> x
    end

    # Build the domain for the mode solver
    step_size = 1.0 / mode_solver_resolution
    function _custom_range(idx_dim)
        return (-size[idx_dim]/2.0+center[idx_dim]):step_size:(size[idx_dim]/2.0+center[idx_dim])
    end
    x_range = _custom_range(idx_x)
    y_range = _custom_range(idx_y)

    current_point = copy(center)

    # Build a function that queries the geometry
    function ε_profile(x::Float64, y::Float64)
        # Since the mode solver is some 2D plane, we need to figure out where we
        # are oriented and map back to the full 3D space.
        current_point[idx_x] = x
        current_point[idx_y] = y

        material_idx = Base.findfirst(current_point, geometry)
        if isnothing(material_idx)
            # no material specified at current point; return vacuum
            return (1.0, 0.0, 0.0, 1.0, 1.0)
        else
            material_tensor = get_ε_at_frequency(geometry[material_idx].material, frequency)

            # Rotate the material tensor appropriately
            material_tensor = rotate_material(material_tensor)

            # Pull the supported tensor elements (εxx, εxy, εyx, εyy, εzz)
            return (
                material_tensor[1, 1],
                material_tensor[1, 2],
                material_tensor[2, 1],
                material_tensor[2, 2],
                material_tensor[3, 3],
            )
        end
    end

    mode_solver = VectorModesolver.VectorialModesolver(
        λ = 1.0 / frequency,
        x = x_range,
        y = y_range,
        ε = ε_profile,
        boundary = boundary_conditions,
    )
    mode_solve_results =
        VectorModesolver.solve(mode_solver, mode_index, solver_tolerance)[mode_index]

    # rotate the fields back appropriately, and swap the coordinate axes
    E_fields =
        cat(mode_solve_results.Ex, mode_solve_results.Ey, mode_solve_results.Ez, dims = 4)
    H_fields =
        cat(mode_solve_results.Hx, mode_solve_results.Hy, mode_solve_results.Hz, dims = 4)
    E_fields = permutedims(rotate_fields(E_fields), new_dims)
    H_fields = permutedims(rotate_fields(H_fields), new_dims)

    # Build a new mode object with the rotated fields
    return VectorModesolver.Mode(
        λ = mode_solve_results.λ,
        neff = mode_solve_results.neff,
        x = x_range,
        y = y_range,
        Ex = E_fields[:, :, :, 1],
        Ey = E_fields[:, :, :, 2],
        Ez = E_fields[:, :, :, 3],
        Hx = H_fields[:, :, :, 1],
        Hy = H_fields[:, :, :, 2],
        Hz = H_fields[:, :, :, 3],
    )
end

# ------------------------------------------------------------------- #
# vector field rotation functions
# ------------------------------------------------------------------- #

"""
    _rotate_vector_field(rotation_matrix::AbstractArray, vector_field::AbstractArray)::AbstractArray

Perform a batch matrix-vector multiplication on the `vector_field` given a `rotation_matrix`.
"""
function _rotate_vector_field(
    rotation_matrix::AbstractArray,
    vector_field::AbstractArray,
)::AbstractArray
    if ndims(vector_field) == 2
        rotated_vector_field = transform_material(vector_field, rotation_matrix)
    else
        rotated_vector_field = similar(vector_field)
        @einsum rotated_vector_field[k, l, m, i] =
            rotation_matrix[i, j] * vector_field[k, l, m, j]
    end

    return rotated_vector_field
end

function rotate_XY_to_XZ(vector_field::AbstractArray)
    rotation_matrix = [1 0 0; 0 cos(-pi / 2) -sin(-pi / 2); 0 sin(-pi / 2) cos(-pi / 2)]
    return _rotate_vector_field(rotation_matrix, vector_field)
end

function rotate_XZ_to_XY(vector_field::AbstractArray)
    rotation_matrix = [1 0 0; 0 cos(-pi / 2) -sin(-pi / 2); 0 sin(-pi / 2) cos(-pi / 2)]'
    return _rotate_vector_field(rotation_matrix, vector_field)
end

function rotate_XZ_to_YZ(vector_field::AbstractArray)
    rotation_matrix = [cos(-pi / 2) -sin(-pi / 2) 0; sin(-pi / 2) cos(-pi / 2) 0; 0 0 1]
    return _rotate_vector_field(rotation_matrix, vector_field)
end

function rotate_YZ_to_XZ(vector_field::AbstractArray)
    rotation_matrix = [cos(-pi / 2) -sin(-pi / 2) 0; sin(-pi / 2) cos(-pi / 2) 0; 0 0 1]'
    return _rotate_vector_field(rotation_matrix, vector_field)
end

function rotate_XY_to_YZ(vector_field::AbstractArray)
    return vector_field |> rotate_XY_to_XZ |> rotate_XZ_to_YZ
end

function rotate_YZ_to_XY(vector_field::AbstractArray)
    return vector_field |> rotate_YZ_to_XZ |> rotate_XZ_to_XY
end
