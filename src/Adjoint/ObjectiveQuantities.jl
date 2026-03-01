# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    ObjectiveQuantities.jl

Defines abstract and concrete objective quantity types used in the adjoint
optimization pipeline. Each objective quantity knows how to:
1. Register monitors in the forward simulation
2. Evaluate itself from monitor data
3. Place adjoint sources for the adjoint simulation

Ported from meep/python/adjoint/objective.py
"""

# ---------------------------------------------------------- #
# Abstract type
# ---------------------------------------------------------- #

"""
    ObjectiveQuantity

Abstract base type for differentiable objective quantities.
Concrete subtypes must implement:
- `register_monitors!(oq, sim, frequencies)`
- `evaluate(oq, sim)`
- `place_adjoint_source(oq, sim, dJ, frequencies)`
"""
abstract type ObjectiveQuantity end

# ---------------------------------------------------------- #
# EigenmodeCoefficient
# ---------------------------------------------------------- #

"""
    EigenmodeCoefficient

A differentiable frequency-dependent eigenmode coefficient (S-parameter).
Measures the overlap of simulation fields with a waveguide eigenmode.

Fields:
- `volume`: Volume specifying the mode monitor cross-section
- `mode_num`: eigenmode number (1-indexed)
- `forward`: whether to measure forward (+) or backward (-) propagating mode
- `mode_spec`: ModeSpec for the eigenmode solver configuration
- `monitor`: the ModeMonitor (set during register_monitors!)
- `eval_data`: cached evaluation result
- `cscale`: normalization constant from mode decomposition
"""
@with_kw mutable struct EigenmodeCoefficient <: ObjectiveQuantity
    volume::Volume
    mode_num::Int = 1
    forward::Bool = true
    mode_spec::ModeSpec = ModeSpec()
    monitor::Union{Nothing, ModeMonitor} = nothing
    eval_data::Union{Nothing, Vector{ComplexF64}} = nothing
    cscale::Union{Nothing, ComplexF64} = nothing
    frequencies::Union{Nothing, Vector{Float64}} = nothing
end

"""
    register_monitors!(oq::EigenmodeCoefficient, sim, frequencies)

Install a ModeMonitor for this eigenmode coefficient.
"""
function register_monitors!(
    oq::EigenmodeCoefficient,
    sim::SimulationData,
    frequencies::Vector{Float64},
)
    oq.frequencies = frequencies
    oq.monitor = ModeMonitor(
        center = oq.volume.center,
        size = oq.volume.size,
        frequencies = frequencies,
        mode_spec = oq.mode_spec,
    )
    # The ModeMonitor will be initialized when added to sim.monitors
    # and prepare_simulation! is called
    return oq.monitor
end

"""
    evaluate(oq::EigenmodeCoefficient, sim)

Evaluate the eigenmode coefficient after a forward simulation.
Returns a vector of complex mode amplitudes, one per frequency.
"""
function evaluate(oq::EigenmodeCoefficient, sim::SimulationData)
    if isnothing(oq.monitor) || isnothing(oq.monitor.monitor_data)
        error("Monitor not initialized. Run prepare_simulation! first.")
    end

    md = oq.monitor.monitor_data

    # Compute mode amplitudes using existing infrastructure
    amplitudes = compute_mode_amplitudes(sim, md)

    # Select forward or backward coefficient
    # amplitudes is typically a Dict or array with forward/backward keys
    if oq.forward
        oq.eval_data = amplitudes[:forward]
    else
        oq.eval_data = amplitudes[:backward]
    end

    return oq.eval_data
end

"""
    place_adjoint_source(oq::EigenmodeCoefficient, sim, dJ, frequencies)

Construct the adjoint source for an EigenmodeCoefficient.
The adjoint source is an eigenmode source propagating in the opposite direction,
scaled by dJ and the adjoint source scale factor.
"""
function place_adjoint_source(
    oq::EigenmodeCoefficient,
    sim::SimulationData,
    dJ::AbstractVector{<:Number},
    frequencies::Vector{Float64},
)
    # Get the adjoint source scaling
    time_profile = create_adjoint_time_profile(frequencies)
    scale = adj_src_scale(sim, frequencies, get_time_profile(sim.sources[1]))

    # The cscale normalizes the mode overlap integral
    da_dE = isnothing(oq.cscale) ? 0.5 : 0.5 * oq.cscale

    if length(frequencies) == 1
        # Single frequency: use Gaussian pulse with scaled amplitude
        amp = da_dE * dJ[1] * scale[1]

        # The adjoint eigenmode source propagates in the opposite direction
        adjoint_source = _create_eigenmode_adjoint_source(
            oq, sim, time_profile, amp, frequencies,
        )
        return [adjoint_source]
    else
        # Multi-frequency: use FilteredSource
        freq_scale = da_dE .* dJ .* scale
        filtered_src = FilteredSource(
            get_frequency(time_profile),
            frequencies,
            freq_scale,
            Float64(sim.Δt),
        )

        sources = Source[]
        for (bi, bf) in enumerate(filtered_src.basis_sources)
            src = _create_eigenmode_adjoint_source(
                oq, sim, bf, filtered_src.nodes[bi, :], frequencies,
            )
            push!(sources, src)
        end
        return sources
    end
end

"""
    _create_eigenmode_adjoint_source(oq, sim, time_src, amplitude, frequencies)

Create an eigenmode adjoint source using the mode profiles from the forward
run's ModeMonitor.  The adjoint source propagates in the opposite direction
by negating the H-field components of the mode profile.

For single-frequency: `amplitude` is a scalar dJ * scale, and the spatial
profile is the single mode profile at that frequency.

For multi-frequency: `amplitude` is a vector of per-frequency weights
(FilteredSource nodes row).  The spatial profile is the weighted sum
of per-frequency mode profiles, accounting for modal dispersion.
"""
function _create_eigenmode_adjoint_source(
    oq::EigenmodeCoefficient,
    sim::SimulationData,
    time_src::TimeSource,
    amplitude,
    frequencies::Vector{Float64},
)
    vol = oq.volume
    normal_axis = findfirst(x -> x == 0.0, vol.size)
    isnothing(normal_axis) && error("EigenmodeCoefficient volume must have one zero-size dimension")

    md = oq.monitor.monitor_data

    # Expand 2D mode fields to 3D with singleton along normal axis
    function _expand(f)
        if ndims(f) == 2
            shape = collect(Base.size(f))
            insert!(shape, normal_axis, 1)
            return reshape(f, shape...)
        end
        return f
    end

    # H-field sign: negate to reverse propagation direction.
    # For backward mode (oq.forward == false), the forward mode profile is
    # already backward-propagating, so the adjoint should be forward → no negate.
    h_sign = oq.forward ? -1.0 : 1.0

    if amplitude isa AbstractVector && length(amplitude) == length(md.mode_profiles)
        # Multi-frequency: weighted sum of per-frequency mode profiles.
        # This accounts for modal dispersion (profile shape varies with frequency).
        weights = ComplexF64.(amplitude)
        mode0 = md.mode_profiles[1]
        fields = Dict{Field, AbstractArray}(
            Ex() => _expand(zeros(ComplexF64, size(mode0.Ex))),
            Ey() => _expand(zeros(ComplexF64, size(mode0.Ey))),
            Ez() => _expand(zeros(ComplexF64, size(mode0.Ez))),
            Hx() => _expand(zeros(ComplexF64, size(mode0.Hx))),
            Hy() => _expand(zeros(ComplexF64, size(mode0.Hy))),
            Hz() => _expand(zeros(ComplexF64, size(mode0.Hz))),
        )
        for (fi, mode) in enumerate(md.mode_profiles)
            w = weights[fi]
            w == 0 && continue
            fields[Ex()] .+= w .* _expand(ComplexF64.(mode.Ex))
            fields[Ey()] .+= w .* _expand(ComplexF64.(mode.Ey))
            fields[Ez()] .+= w .* _expand(ComplexF64.(mode.Ez))
            fields[Hx()] .+= (w * h_sign) .* _expand(ComplexF64.(mode.Hx))
            fields[Hy()] .+= (w * h_sign) .* _expand(ComplexF64.(mode.Hy))
            fields[Hz()] .+= (w * h_sign) .* _expand(ComplexF64.(mode.Hz))
        end
        amp = 1.0  # per-frequency weighting is baked into the spatial profile
    else
        # Single frequency: use the one mode profile directly
        mode = md.mode_profiles[1]
        fields = Dict{Field, AbstractArray}(
            Ex() => _expand(ComplexF64.(mode.Ex)),
            Ey() => _expand(ComplexF64.(mode.Ey)),
            Ez() => _expand(ComplexF64.(mode.Ez)),
            Hx() => _expand(ComplexF64.(h_sign .* mode.Hx)),
            Hy() => _expand(ComplexF64.(h_sign .* mode.Hy)),
            Hz() => _expand(ComplexF64.(h_sign .* mode.Hz)),
        )
        amp = amplitude isa AbstractVector ? amplitude[1] : amplitude
    end

    return EquivalentSource(
        time_profile = time_src,
        fields = fields,
        center = collect(Float64.(vol.center)),
        size = collect(Float64.(vol.size)),
        amplitude = amp,
    )
end

# ---------------------------------------------------------- #
# FourierFieldsObjective
# ---------------------------------------------------------- #

"""
    FourierFieldsObjective

A differentiable objective based on Fourier-transformed (DFT) fields
at specified spatial locations.
"""
@with_kw mutable struct FourierFieldsObjective <: ObjectiveQuantity
    volume::Volume
    component::Field
    monitor::Union{Nothing, DFTMonitor} = nothing
    eval_data::Union{Nothing, AbstractArray} = nothing
    frequencies::Union{Nothing, Vector{Float64}} = nothing
end

function register_monitors!(
    oq::FourierFieldsObjective,
    sim::SimulationData,
    frequencies::Vector{Float64},
)
    oq.frequencies = frequencies
    oq.monitor = DFTMonitor(
        component = oq.component,
        center = oq.volume.center,
        size = oq.volume.size,
        frequencies = frequencies,
    )
    return oq.monitor
end

function evaluate(oq::FourierFieldsObjective, sim::SimulationData)
    if isnothing(oq.monitor) || isnothing(oq.monitor.monitor_data)
        error("Monitor not initialized.")
    end
    oq.eval_data = Array(oq.monitor.monitor_data.fields)
    return oq.eval_data
end

function place_adjoint_source(
    oq::FourierFieldsObjective,
    sim::SimulationData,
    dJ::AbstractArray{<:Number},
    frequencies::Vector{Float64},
)
    time_profile = create_adjoint_time_profile(frequencies)
    scale = adj_src_scale(
        sim, frequencies, get_time_profile(sim.sources[1]);
        include_resolution=false,
    )

    # For FourierFields, the adjoint source is a uniform source at the monitor
    # volume, weighted by dJ * scale. The amplitude at each spatial point
    # comes from the derivative of the objective w.r.t. the DFT field at that point.
    #
    # For a scalar objective like f = sum(|E|^2), dJ is a vector (one per freq).
    # The adjoint source amplitude at each frequency is dJ[f] * scale[f].

    dJ_vec = vec(dJ)

    # For single frequency: amplitude is dJ * scale, use Gaussian time profile
    if length(frequencies) == 1
        amp = dJ_vec[1] * scale[1]
        src = UniformSource(
            time_profile = time_profile,
            component = oq.component,
            center = oq.volume.center,
            size = oq.volume.size,
            amplitude = amp,
        )
        return Source[src]
    else
        # Multi-frequency: use FilteredSource
        freq_scale = dJ_vec .* scale
        filtered_src = FilteredSource(
            get_frequency(time_profile),
            frequencies,
            freq_scale,
            Float64(sim.Δt),
        )
        sources = Source[]
        for (bi, bf) in enumerate(filtered_src.basis_sources)
            amp = filtered_src.nodes[bi, 1]  # scalar amplitude for uniform source
            src = UniformSource(
                time_profile = bf,
                component = oq.component,
                center = oq.volume.center,
                size = oq.volume.size,
                amplitude = amp,
            )
            push!(sources, src)
        end
        return sources
    end
end
