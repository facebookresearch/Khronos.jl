# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    FilteredSource.jl

Implements the multi-frequency adjoint source using Nuttall-windowed basis
functions. Each frequency of interest gets a basis function centered at that
frequency. The basis weights are solved via a Vandermonde-like system to
match the desired frequency response at the specified frequencies.

This enables broadband adjoint optimization with a single adjoint FDTD run.

Reference: Hammond et al., Optics Express (2022), Section 3.2 (Frequency
parallelism) and Figure 4 (Nuttall basis functions).
"""

"""
    FilteredSourceData

Multi-frequency adjoint source constructed from Nuttall window basis functions.

Fields:
- `center_frequency`: center frequency of the source
- `frequencies`: target frequencies
- `nodes`: weight matrix (n_basis × n_spatial_points)
- `T_duration`: duration of each basis function
- `N_steps`: number of timesteps
- `dt`: half-timestep (staggered E,H grids)
- `basis_sources`: vector of CustomSourceData, one per basis function
"""
struct FilteredSourceData
    center_frequency::Float64
    frequencies::Vector{Float64}
    nodes::Matrix{ComplexF64}         # (n_basis, n_pts) or (n_basis, n_freq)
    err::Float64
    T_duration::Float64
    N_steps::Int
    dt::Float64
    basis_sources::Vector{<:TimeSource}
end

# ---------------------------------------------------------- #
# Nuttall window functions
# ---------------------------------------------------------- #

const NUTTALL_COEFFS = [0.355768, 0.4873960, 0.144232, 0.012604]

"""
    nuttall_td(t, f0, T_duration)

Nuttall window in the time domain, centered at frequency f0.
Returns a complex scalar (or vector if f0 is a vector).
"""
function nuttall_td(t::Real, f0, T_duration::Real)
    cos_sum = sum(
        (-1)^k * NUTTALL_COEFFS[k+1] * cos(2π * t * k / T_duration)
        for k in 0:3
    )
    return exp.(-im .* 2π .* f0 .* t) .* cos_sum
end

"""
    nuttall_sinc(f, f0, N, dt)

The DTFT of a rectangular window (geometric sum), used as a building block
for the Nuttall window DTFT.
"""
function nuttall_sinc(f, f0, N::Int, dt::Real)
    Δf = f .- f0
    # Handle the f == f0 case to avoid 0/0
    num = @. ifelse(
        abs(Δf) < 1e-15,
        complex(N + 1),
        (1 - exp(im * (N + 1) * 2π * Δf * dt)),
    )
    den = @. ifelse(
        abs(Δf) < 1e-15,
        complex(1.0),
        (1 - exp(im * 2π * Δf * dt)),
    )
    return num ./ den
end

"""
    nuttall_dtft(f, f0, N, dt)

Nuttall window in the frequency domain (DTFT).
"""
function nuttall_dtft(f, f0, N::Int, dt::Real)
    df = 1 / (N * dt)
    result = NUTTALL_COEFFS[1] * nuttall_sinc(f, f0, N, dt)
    for k in 1:3
        result .+= (-1)^k * NUTTALL_COEFFS[k+1] / 2 .* (
            nuttall_sinc(f, f0 .- k * df, N, dt) .+
            nuttall_sinc(f, f0 .+ k * df, N, dt)
        )
    end
    return result
end

"""
    nuttall_bandwidth(N, dt)

Compute the bandwidth of the Nuttall window by fitting the asymptotic
decay to a power law C/f^3 and finding where it drops below tolerance.
"""
function nuttall_bandwidth(N::Int, dt::Real)
    tol = 1e-7
    fwidth = 1 / (N * dt)
    frq_inf = 10000 * fwidth
    na_dtft = nuttall_dtft([frq_inf], 0.0, N, dt)[1]
    coeff = frq_inf^3 * abs(na_dtft)
    na_dtft_max = nuttall_dtft([0.0], 0.0, N, dt)[1]
    bw = 2 * (coeff / (tol * abs(na_dtft_max)))^(1/3)
    return real(bw)
end

# ---------------------------------------------------------- #
# FilteredSource constructor
# ---------------------------------------------------------- #

"""
    FilteredSource(center_frequency, frequencies, frequency_response, dt)

Construct a FilteredSource that produces time-domain adjoint sources
matching the desired `frequency_response` at the specified `frequencies`.

Arguments:
- `center_frequency`: center frequency of the envelope
- `frequencies`: vector of target frequencies
- `frequency_response`: matrix or vector of desired frequency-domain amplitudes.
  Shape: (n_freq,) for scalar sources or (n_freq, n_pts) for spatially-varying.
- `dt`: simulation timestep (will be halved for staggered E,H)
"""
function FilteredSource(
    center_frequency::Float64,
    frequencies::Vector{Float64},
    frequency_response::AbstractArray{<:Number},
    dt::Float64,
)
    # Halve dt to account for staggered E, H time stepping
    dt_half = dt / 2

    # Duration: set by minimum frequency spacing (narrower spacing -> longer source)
    if length(frequencies) > 1
        T_duration = maximum(abs.(1.0 ./ diff(frequencies)))
    else
        T_duration = 1.0 / (0.1 * center_frequency)  # fallback for single freq
    end
    N_steps = round(Int, T_duration / dt_half)

    fwidth = nuttall_bandwidth(N_steps, dt_half)

    # Ensure frequency_response is a matrix: (n_freq, n_pts)
    H = if ndims(frequency_response) == 1
        reshape(frequency_response, :, 1)
    else
        Matrix(frequency_response)
    end

    # Construct Vandermonde matrix: V[i,j] = nuttall_dtft(freq[i], center_freq[j])
    n_freq = length(frequencies)
    vandermonde = zeros(ComplexF64, n_freq, n_freq)
    for j in 1:n_freq
        vandermonde[:, j] = nuttall_dtft(frequencies, frequencies[j], N_steps, dt_half)
    end

    # Solve for weights: nodes = pinv(V) * H
    nodes = pinv(vandermonde) * H

    # Estimate reconstruction error
    H_hat = vandermonde * nodes
    l2_err = sum(abs2.(H .- H_hat) ./ max.(abs2.(H), 1e-30))

    # Create basis CustomSources
    basis_sources = TimeSource[]
    for i in 1:n_freq
        bf = let idx = i, freqs = frequencies, Td = T_duration, dth = dt_half
            function (t)
                if t > Td
                    return 0.0 + 0.0im
                end
                return nuttall_td(t, freqs[idx], Td) / (dth / sqrt(2π))
            end
        end

        push!(basis_sources, CustomSource(
            src_func = bf,
            fcen = center_frequency,
            fwidth = fwidth,
            end_time = T_duration,
        ))
    end

    return FilteredSourceData(
        center_frequency,
        frequencies,
        nodes,
        l2_err,
        T_duration,
        N_steps,
        dt_half,
        basis_sources,
    )
end
