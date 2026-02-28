# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    AdjointSourceScale.jl

Computes the scaling factors for adjoint sources to ensure correct gradient
computation in the hybrid time-/frequency-domain adjoint method.

The key factors are:
1. Corrected FDTD frequency: ω̂ = (1 - exp(-iωΔt)) / (-iΔt)
2. DTFT of the forward source waveform
3. Resolution scaling (dV = 1/resolution^ndims)
4. Real-field correction factor (×2 for real fields)

Reference: Hammond et al., Optics Express (2022), Appendix A.
"""

"""
    corrected_fdtd_frequency(ω, Δt)

Compute the corrected FDTD frequency that accounts for the finite-difference
time derivative approximation: ω̂ = (1 - exp(-iωΔt)) / (-iΔt).

This correction ensures second-order accuracy in Δt and is essential for
matching FDTD frequency-domain results to true frequency-domain solvers.
"""
function corrected_fdtd_frequency(ω::Real, Δt::Real)
    return (1 - exp(-im * ω * Δt)) / (-im * Δt)
end

"""
    create_adjoint_time_profile(frequencies; fwidth_frac=0.1, cutoff=5.0)

Create a Gaussian time profile for the adjoint source normalization.
Returns a GaussianPulseSource centered at the mean frequency.

For single-frequency optimization, this Gaussian envelope modulates the
adjoint eigenmode source.
"""
function create_adjoint_time_profile(
    frequencies::Vector{Float64};
    fwidth_frac::Float64 = 0.1,
    cutoff::Float64 = 5.0,
)
    fcen = mean(frequencies)
    fwidth = fwidth_frac * fcen
    return GaussianPulseSource(fcen=fcen, fwidth=fwidth, cutoff_scale=cutoff)
end

"""
    adj_src_scale(sim, frequencies, time_profile; include_resolution=true)

Compute the adjoint source scaling factor for each frequency.

This implements the formula from meep's `_adj_src_scale()`:
    scale = dV * iomega / fwd_dtft / adj_src_phase

where:
- dV = 1/resolution^ndims (resolution factor)
- iomega = (1 - exp(-iωΔt))/Δt  (corrected frequency with discrete derivative)
- fwd_dtft = DTFT of the forward source at each frequency
- adj_src_phase = phase correction from center frequency

For multi-frequency, the forward source DTFT division is handled by the
FilteredSource and is omitted from the scale.
"""
function adj_src_scale(
    sim::SimulationData,
    frequencies::Vector{Float64},
    time_profile::TimeSource;
    include_resolution::Bool = true,
    fwidth_frac::Float64 = 0.1,
)
    T_sim = Float64(round_time(sim))  # total simulation time
    dt = Float64(sim.Δt)

    # Resolution factor
    if include_resolution
        ndims = sim.ndims
        dV = 1.0 / Float64(sim.resolution)^ndims
    else
        dV = 1.0
    end

    # Corrected iomega: discrete time derivative factor
    iomega = (1.0 .- exp.(-im .* (2π .* frequencies) .* dt)) ./ dt

    # DTFT of the forward source
    # Evaluate the source at each timestep and compute the DTFT
    n_steps = floor(Int, T_sim / dt)
    t_vals = collect(0:n_steps-1) .* dt
    y = [Complex{Float64}(eval_time_source(time_profile, t)) for t in t_vals]

    fwd_dtft = zeros(ComplexF64, length(frequencies))
    for (fi, f) in enumerate(frequencies)
        fwd_dtft[fi] = sum(
            dt / sqrt(2π) .* exp.(im .* 2π .* f .* t_vals) .* y
        )
    end

    # Phase correction from center frequency
    fcen = get_frequency(time_profile)
    src_center_dtft = sum(
        dt / sqrt(2π) .* exp.(im .* 2π .* fcen .* t_vals) .* y
    )
    cutoff = 5.0
    fwidth_scale = exp(-2im * π * cutoff / fwidth_frac)
    adj_src_phase = exp(im * angle(src_center_dtft)) * fwidth_scale

    if length(frequencies) == 1
        # Single-frequency: divide by forward source DTFT
        scale = dV .* iomega ./ fwd_dtft ./ adj_src_phase
    else
        # Multi-frequency: FilteredSource handles the source normalization
        scale = dV .* iomega ./ adj_src_phase
    end

    # Real-field correction: Re[J] = (J + J*)/2 halves the positive-freq amplitude
    # Khronos uses complex fields by default, so no correction needed unless
    # the simulation explicitly uses real fields.
    # TODO: add real-field detection if Khronos supports it

    return scale
end

"""
    forward_source_dtft(sim, time_profile, frequencies)

Compute the DTFT of the forward source time profile at the given frequencies.
"""
function forward_source_dtft(
    sim::SimulationData,
    time_profile::TimeSource,
    frequencies::Vector{Float64},
)
    dt = Float64(sim.Δt)
    T_sim = Float64(round_time(sim))
    n_steps = floor(Int, T_sim / dt)
    t_vals = collect(0:n_steps-1) .* dt

    y = [Complex{Float64}(eval_time_source(time_profile, t)) for t in t_vals]

    dtft = zeros(ComplexF64, length(frequencies))
    for (fi, f) in enumerate(frequencies)
        dtft[fi] = sum(dt / sqrt(2π) .* exp.(im .* 2π .* f .* t_vals) .* y)
    end

    return dtft
end
