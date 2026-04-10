# Copyright (c) Meta Platforms, Inc. and affiliates.
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

    # Build a memoized function that queries the geometry.
    # The Dict cache avoids redundant O(N_objects) findfirst scans when the
    # mode solver queries the same (x,y) positions multiple times (e.g. during
    # assembly and again during E-field post-processing in getE).
    _eps_cache = Dict{Tuple{Float64,Float64}, NTuple{5,Float64}}()

    function ε_profile(x::Float64, y::Float64)
        key = (x, y)
        cached = get(_eps_cache, key, nothing)
        if !isnothing(cached)
            return cached
        end

        # Since the mode solver is some 2D plane, we need to figure out where we
        # are oriented and map back to the full 3D space.
        current_point[idx_x] = x
        current_point[idx_y] = y

        material_idx = Base.findfirst(current_point, geometry)
        result = if isnothing(material_idx)
            # no material specified at current point; return vacuum
            (1.0, 0.0, 0.0, 1.0, 1.0)
        else
            material_tensor = get_ε_at_frequency(geometry[material_idx].material, frequency)

            # Rotate the material tensor appropriately
            material_tensor = rotate_material(material_tensor)

            # Pull the supported tensor elements (εxx, εxy, εyx, εyy, εzz)
            (
                material_tensor[1, 1],
                material_tensor[1, 2],
                material_tensor[2, 1],
                material_tensor[2, 2],
                material_tensor[3, 3],
            )
        end
        _eps_cache[key] = result
        return result
    end

    mode_solver = VectorModesolver.VectorialModesolver(
        λ = 1.0 / frequency,
        x = collect(x_range),
        y = collect(y_range),
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

    # Build a new mode object with the rotated fields.
    # dropdims squeezes the singleton normal dimension so fields are 2D matrices.
    _drop(f, i) = dropdims(f[:, :, :, i]; dims=_singleton_dim(f[:, :, :, i]))
    return VectorModesolver.Mode(
        λ = mode_solve_results.λ,
        neff = mode_solve_results.neff,
        x = x_range,
        y = y_range,
        Ex = _drop(E_fields, 1),
        Ey = _drop(E_fields, 2),
        Ez = _drop(E_fields, 3),
        Hx = _drop(H_fields, 1),
        Hy = _drop(H_fields, 2),
        Hz = _drop(H_fields, 3),
    )
end

# Find the singleton dimension in a 3D array (for squeezing after field rotation)
function _singleton_dim(arr::AbstractArray{T,3}) where T
    for d in 1:3
        if Base.size(arr, d) == 1
            return d
        end
    end
    error("No singleton dimension found in 3D array of size $(Base.size(arr))")
end

"""
    get_mode_profiles_sweep(;
        frequencies::Vector{Float64},
        mode_solver_resolution::Int,
        max_mode_index::Int,
        center::Vector{<:Real},
        size::Vector{<:Real},
        solver_tolerance::Number,
        geometry::Vector{Object},
        boundary_conditions::Tuple{Int,Int,Int,Int} = (1,1,1,1),
        target_neff::Float64 = 0.0,
    )::Vector{Vector{VectorModesolver.Mode}}

Compute mode profiles at multiple frequencies efficiently using
`VectorModesolver.solve_sweep`.  The geometry-dependent operator matrix is
assembled **once** and reused across all wavelengths, with warm-starting of
the eigensolver between consecutive wavelengths.

Returns `all_modes[freq_idx][mode_idx]` — a vector of mode lists, one per
input frequency, each containing up to `max_mode_index` modes sorted by
descending effective index.

Performance optimisations applied here:
* **Sweep solver** (`solve_sweep`): geometry assembly done once, not per-λ.
* **Memoised ε callback**: geometry lookups cached in a Dict so that the
  `getE` post-processing step (which re-queries ε at cell centres) is O(1).
* **Shift-invert** (optional): when `target_neff > 0`, the eigensolver uses
  shift-invert mode (`σ = n_eff² k₀²`) for faster convergence on interior
  eigenvalues.
"""
function get_mode_profiles_sweep(;
    frequencies::Vector{Float64},
    mode_solver_resolution::Int,
    max_mode_index::Int,
    center::Vector{<:Real},
    size::Vector{<:Real},
    solver_tolerance::Number,
    geometry::Vector{Object},
    boundary_conditions::Tuple{Int,Int,Int,Int} = (1, 1, 1, 1),
    target_neff::Float64 = 0.0,
)::Vector{Vector{VectorModesolver.Mode}}
    @assert count(x -> x != 0, size) >= 2 "The mode region must be a plane (2D)"

    # ---- coordinate mapping (identical to get_mode_profiles) ----
    if size[1] == 0
        idx_x = 2; idx_y = 3
        new_dims = (3, 1, 2, 4)
        rotate_material = rotate_YZ_to_XZ
        rotate_fields = rotate_XY_to_YZ
    elseif size[2] == 0
        idx_x = 1; idx_y = 3
        new_dims = (1, 3, 2, 4)
        rotate_material = rotate_XZ_to_XY
        rotate_fields = rotate_XY_to_XZ
    elseif size[3] == 0
        idx_x = 1; idx_y = 2
        new_dims = (1, 2, 3, 4)
        rotate_material = x -> x
        rotate_fields = x -> x
    end

    step_size = 1.0 / mode_solver_resolution
    function _custom_range(idx_dim)
        return (-size[idx_dim]/2.0+center[idx_dim]):step_size:(size[idx_dim]/2.0+center[idx_dim])
    end
    x_range = _custom_range(idx_x)
    y_range = _custom_range(idx_y)

    # Use central frequency for ε evaluation (constant across sweep).
    # For non-dispersive materials this is exact; for weakly-dispersive
    # materials it is a good approximation over narrow bandwidths.
    central_freq = frequencies[max(1, (length(frequencies) + 1) ÷ 2)]
    current_point = copy(center)

    # Memoised ε callback: geometry lookups are cached so that the VectorModesolver's
    # getE post-processing (which re-queries ε at cell centres) hits the cache
    # instead of doing O(N_objects) findfirst scans again.
    _eps_cache = Dict{Tuple{Float64,Float64}, NTuple{5,Float64}}()

    function ε_profile(x::Float64, y::Float64)
        key = (x, y)
        cached = get(_eps_cache, key, nothing)
        if !isnothing(cached)
            return cached
        end

        current_point[idx_x] = x
        current_point[idx_y] = y
        material_idx = Base.findfirst(current_point, geometry)
        result = if isnothing(material_idx)
            (1.0, 0.0, 0.0, 1.0, 1.0)
        else
            material_tensor = get_ε_at_frequency(geometry[material_idx].material, central_freq)
            material_tensor = rotate_material(material_tensor)
            (
                material_tensor[1, 1],
                material_tensor[1, 2],
                material_tensor[2, 1],
                material_tensor[2, 2],
                material_tensor[3, 3],
            )
        end
        _eps_cache[key] = result
        return result
    end

    # Convert frequencies → wavelengths for VectorModesolver (λ = 1/f in natural units)
    wavelengths = [1.0 / f for f in frequencies]

    # Build mode solver struct (λ is set to the first wavelength; solve_sweep
    # varies λ internally while reusing the assembled geometric operator).
    mode_solver = VectorModesolver.VectorialModesolver(
        λ = wavelengths[1],
        x = collect(x_range),
        y = collect(y_range),
        ε = ε_profile,
        boundary = boundary_conditions,
    )

    # Shift-invert: if target_neff is provided, compute σ = (n_eff · k₀)²
    # at the central wavelength for faster eigenvalue convergence.
    sigma = if target_neff > 0
        λ_mid = wavelengths[max(1, (length(wavelengths) + 1) ÷ 2)]
        (target_neff * 2π / λ_mid)^2
    else
        nothing
    end

    # Solve all wavelengths at once with warm-starting
    all_results = VectorModesolver.solve_sweep(
        mode_solver, wavelengths, max_mode_index, Float64(solver_tolerance);
        sigma = sigma,
    )

    # Post-process: rotate fields from mode-solver coordinates back to
    # Khronos coordinates (same transform as get_mode_profiles).
    all_modes = Vector{Vector{VectorModesolver.Mode}}(undef, length(frequencies))

    for (fi, modes_at_λ) in enumerate(all_results)
        rotated = VectorModesolver.Mode[]
        for m in modes_at_λ
            E_fields = cat(m.Ex, m.Ey, m.Ez, dims = 4)
            H_fields = cat(m.Hx, m.Hy, m.Hz, dims = 4)
            E_fields = permutedims(rotate_fields(E_fields), new_dims)
            H_fields = permutedims(rotate_fields(H_fields), new_dims)

            _drop(f, i) = dropdims(f[:, :, :, i]; dims=_singleton_dim(f[:, :, :, i]))
            push!(rotated, VectorModesolver.Mode(
                λ = m.λ,
                neff = m.neff,
                x = x_range,
                y = y_range,
                Ex = _drop(E_fields, 1),
                Ey = _drop(E_fields, 2),
                Ez = _drop(E_fields, 3),
                Hx = _drop(H_fields, 1),
                Hy = _drop(H_fields, 2),
                Hz = _drop(H_fields, 3),
            ))
        end
        all_modes[fi] = rotated
    end

    return all_modes
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
