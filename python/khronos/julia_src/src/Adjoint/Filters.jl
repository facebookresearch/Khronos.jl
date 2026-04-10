# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    Filters.jl

Topology optimization filters: convolution filters (density filters),
projection operators, and morphological transforms.

These are applied in the design parameter pipeline:
    raw ρ → filter(ρ) → project(ρ̃) → ε(ρ̄)

Ported from meep/python/adjoint/filters.py
"""

# ---------------------------------------------------------- #
# FFT-based convolution infrastructure
# ---------------------------------------------------------- #

"""
    _centered(arr, newshape)

Extract the center portion of an array after FFT-based convolution.
"""
function _centered(arr::AbstractArray, newshape::Tuple{Int,Int})
    currshape = size(arr)
    startind = div.(currshape .- newshape, 2) .+ 1
    endind = startind .+ newshape .- 1
    return arr[startind[1]:endind[1], startind[2]:endind[2]]
end

"""
    _edge_pad(arr, pad)

Border-pad the edges of a 2D array. Each padded pixel gets the value
of the nearest boundary pixel (preserves edge features).
"""
function _edge_pad(arr::AbstractMatrix, pad::Tuple{Tuple{Int,Int},Tuple{Int,Int}})
    (px_left, px_right) = pad[1]
    (py_left, py_right) = pad[2]
    nx, ny = size(arr)

    # Compute total output size
    out_nx = px_left + nx + px_right
    out_ny = py_left + ny + py_right
    out = similar(arr, out_nx, out_ny)

    # Fill center
    out[px_left+1:px_left+nx, py_left+1:py_left+ny] .= arr

    # Fill edges
    for ix in 1:px_left
        out[ix, py_left+1:py_left+ny] .= arr[1, :]
    end
    for ix in px_left+nx+1:out_nx
        out[ix, py_left+1:py_left+ny] .= arr[end, :]
    end
    for iy in 1:py_left
        out[:, iy] .= out[:, py_left+1]
    end
    for iy in py_left+ny+1:out_ny
        out[:, iy] .= out[:, py_left+ny]
    end

    return out
end

"""
    _quarter_to_full_kernel(h_quarter, pad_to)

Construct the full 2D kernel from its nonnegative quadrant (exploiting C4v symmetry).
Zero-pads to size `pad_to`.
"""
function _quarter_to_full_kernel(h_quarter::AbstractMatrix, pad_to::Tuple{Int,Int})
    kx, ky = size(h_quarter)
    full_nx, full_ny = pad_to

    full = zeros(eltype(h_quarter), full_nx, full_ny)

    # Place the four quadrants
    # Top-left: h_quarter itself
    for iy in 1:ky, ix in 1:kx
        full[ix, iy] = h_quarter[ix, iy]
    end
    # Top-right: flip in x
    for iy in 1:ky, ix in 2:kx
        full[full_nx - ix + 2, iy] = h_quarter[ix, iy]
    end
    # Bottom-left: flip in y
    for iy in 2:ky, ix in 1:kx
        full[ix, full_ny - iy + 2] = h_quarter[ix, iy]
    end
    # Bottom-right: flip both
    for iy in 2:ky, ix in 2:kx
        full[full_nx - ix + 2, full_ny - iy + 2] = h_quarter[ix, iy]
    end

    return full
end

"""
    convolve_design_weights(x, h)

Convolve 2D design weights `x` with kernel quarter `h` using FFT.
The design weights are border-padded to preserve edge features.
"""
function convolve_design_weights(x::AbstractMatrix, h::AbstractMatrix)
    sx, sy = size(x)

    # Build full kernel and pad design weights
    h_full = _quarter_to_full_kernel(h, (3 * sx, 3 * sy))
    x_padded = _edge_pad(x, ((sx, sx), (sy, sy)))

    # Normalize kernel
    h_full ./= sum(h_full)

    # FFT convolution
    result = real.(ifft(fft(x_padded) .* fft(h_full)))

    return _centered(result, (sx, sy))
end

"""
    _mesh_grid(radius, Lx, Ly, resolution)

Compute the design grid mesh and kernel coordinate arrays.
"""
function _mesh_grid(radius::Real, Lx::Real, Ly::Real, resolution::Real)
    Nx = round(Int, Lx * resolution) + 1
    Ny = round(Int, Ly * resolution) + 1

    xv = resolution > 0 ? collect(range(0, stop=Lx/2, step=1/resolution)) : [0.0]
    yv = resolution > 0 ? collect(range(0, stop=Ly/2, step=1/resolution)) : [0.0]

    return Nx, Ny, xv, yv
end

# ---------------------------------------------------------- #
# Filter functions
# ---------------------------------------------------------- #

"""
    conic_filter(x, radius, Lx, Ly, resolution)

Linear conic ("hat") convolution filter.
"""
function conic_filter(
    x::AbstractVector,
    radius::Real,
    Lx::Real,
    Ly::Real,
    resolution::Real,
)
    Nx, Ny, xv, yv = _mesh_grid(radius, Lx, Ly, resolution)
    x2d = reshape(x, Nx, Ny)

    # Build conic kernel (quarter)
    h = zeros(length(xv), length(yv))
    for (iy, y) in enumerate(yv), (ix, xc) in enumerate(xv)
        r2 = xc^2 + y^2
        if r2 < radius^2
            h[ix, iy] = 1 - sqrt(r2) / radius
        end
    end

    return vec(convolve_design_weights(x2d, h))
end

"""
    cylindrical_filter(x, radius, Lx, Ly, resolution)

Cylindrical (top-hat) convolution filter.
"""
function cylindrical_filter(
    x::AbstractVector,
    radius::Real,
    Lx::Real,
    Ly::Real,
    resolution::Real,
)
    Nx, Ny, xv, yv = _mesh_grid(radius, Lx, Ly, resolution)
    x2d = reshape(x, Nx, Ny)

    h = zeros(length(xv), length(yv))
    for (iy, y) in enumerate(yv), (ix, xc) in enumerate(xv)
        if xc^2 + y^2 < radius^2
            h[ix, iy] = 1.0
        end
    end

    return vec(convolve_design_weights(x2d, h))
end

"""
    gaussian_filter(x, sigma, Lx, Ly, resolution)

Gaussian convolution filter with standard deviation `sigma`.
"""
function gaussian_filter(
    x::AbstractVector,
    sigma::Real,
    Lx::Real,
    Ly::Real,
    resolution::Real,
)
    Nx, Ny, xv, yv = _mesh_grid(3 * sigma, Lx, Ly, resolution)
    x2d = reshape(x, Nx, Ny)

    h = zeros(length(xv), length(yv))
    for (iy, y) in enumerate(yv), (ix, xc) in enumerate(xv)
        h[ix, iy] = exp(-(xc^2 + y^2) / sigma^2)
    end

    return vec(convolve_design_weights(x2d, h))
end

# ---------------------------------------------------------- #
# Projection functions
# ---------------------------------------------------------- #

"""
    tanh_projection(x, beta, eta)

Sigmoid projection filter using hyperbolic tangent.
Maps filtered design parameters toward binary (0 or 1) values.

Arguments:
- `x`: design parameters (filtered)
- `beta`: thresholding strength [0, ∞]. Higher = more binary.
- `eta`: threshold point [0, 1]
"""
function tanh_projection(x::AbstractArray, beta::Real, eta::Real)
    if isinf(beta)
        return @. ifelse(x > eta, 1.0, 0.0)
    else
        return @. (tanh(beta * eta) + tanh(beta * (x - eta))) /
                  (tanh(beta * eta) + tanh(beta * (1 - eta)))
    end
end

# ---------------------------------------------------------- #
# Morphological operations
# ---------------------------------------------------------- #

"""
    exponential_erosion(x, radius, beta, Lx, Ly, resolution)

Morphological erosion using exponential projection.
"""
function exponential_erosion(
    x::AbstractVector, radius::Real, beta::Real,
    Lx::Real, Ly::Real, resolution::Real,
)
    x_hat = exp.(beta .* (1 .- x))
    return 1 .- log.(cylindrical_filter(x_hat, radius, Lx, Ly, resolution)) ./ beta
end

"""
    exponential_dilation(x, radius, beta, Lx, Ly, resolution)

Morphological dilation using exponential projection.
"""
function exponential_dilation(
    x::AbstractVector, radius::Real, beta::Real,
    Lx::Real, Ly::Real, resolution::Real,
)
    x_hat = exp.(beta .* x)
    return log.(cylindrical_filter(x_hat, radius, Lx, Ly, resolution)) ./ beta
end

"""
    heaviside_erosion(x, radius, beta, Lx, Ly, resolution)

Morphological erosion using Heaviside projection.
"""
function heaviside_erosion(
    x::AbstractVector, radius::Real, beta::Real,
    Lx::Real, Ly::Real, resolution::Real,
)
    x_hat = cylindrical_filter(x, radius, Lx, Ly, resolution)
    return exp.(-beta .* (1 .- x_hat)) .+ exp(-beta) .* (1 .- x_hat)
end

"""
    heaviside_dilation(x, radius, beta, Lx, Ly, resolution)

Morphological dilation using Heaviside projection.
"""
function heaviside_dilation(
    x::AbstractVector, radius::Real, beta::Real,
    Lx::Real, Ly::Real, resolution::Real,
)
    x_hat = cylindrical_filter(x, radius, Lx, Ly, resolution)
    return 1 .- exp.(-beta .* x_hat) .+ exp(-beta) .* x_hat
end

"""
    gray_indicator(x)

Measure of "grayness" (non-binarization) of design parameters.
Lower values (<2%) indicate good binarization.
"""
function gray_indicator(x::AbstractVector)
    return mean(4 .* x .* (1 .- x)) * 100
end

# ---------------------------------------------------------- #
# Length scale utilities
# ---------------------------------------------------------- #

"""
    get_eta_from_conic(b, R)

Compute the eroded threshold point for a conic filter given the desired
minimum lengthscale `b` and filter radius `R`.
"""
function get_eta_from_conic(b::Real, R::Real)
    norm_length = b / R
    if norm_length < 0
        return 0.0
    elseif norm_length < 1
        return 0.25 * norm_length^2 + 0.5
    elseif norm_length < 2
        return -0.25 * norm_length^2 + norm_length
    else
        return 1.0
    end
end

"""
    get_conic_radius_from_eta_e(b, eta_e)

Compute the conic filter radius from the minimum lengthscale and erosion threshold.
"""
function get_conic_radius_from_eta_e(b::Real, eta_e::Real)
    if eta_e >= 0.5 && eta_e < 0.75
        return b / (2 * sqrt(eta_e - 0.5))
    elseif eta_e >= 0.75 && eta_e <= 1.0
        return b / (2 - 2 * sqrt(1 - eta_e))
    else
        error("eta_e must be between 0.5 and 1.0, got $eta_e")
    end
end
