# (c) Meta Platforms, Inc. and affiliates.

export ContinuousWaveSource, GaussianPulseSource

# ---------------------------------------------------------- #
# Interface functions
# ---------------------------------------------------------- #

"""
    eval_time_source(src::TimeSource, t::Real)

TBW
"""
function eval_time_source(src::TimeSource, t::Real)
    error("`eval_time_source` not yet implemented for this type of source...")
end

"""
    get_cutoff(src::TimeSource)

TBW
"""
function get_cutoff(src::TimeSource)
    error("`get_cutoff` not yet implemented for this type of source...")
end

function get_frequency(src::TimeSource)
    error("`get_frequency` not yet implemented for this type of source...")
end

# ---------------------------------------------------------- #
# Helper routines
# ---------------------------------------------------------- #

"""
    last_source_time(src::Source)

TBW
"""
last_source_time(src::Source) = get_cutoff(get_time_profile(src))
"""
    last_source_time(sim::SimulationData)

TBW
"""
last_source_time(sim::SimulationData) =
    maximum([last_source_time(src) for src in sim.sources])

# ---------------------------------------------------------- #
# CW source
# ---------------------------------------------------------- #

@with_kw struct ContinuousWaveData{N<:Number} <: TimeSource
    fcen::N
end

function ContinuousWaveSource(; fcen::Number)::ContinuousWaveData
    return ContinuousWaveData{backend_number}(fcen)
end

function eval_time_source(src::ContinuousWaveData{N}, t::Real) where {N<:Number}
    return exp(-im * 2 * π * src.fcen * t)
end

function get_cutoff(src::ContinuousWaveData)
    error("Cutoff not possible with CW source...")
end

function get_frequency(src::ContinuousWaveData)
    return src.fcen
end

# ---------------------------------------------------------- #
# Gaussian pulse source
# ---------------------------------------------------------- #

@with_kw struct GaussianPulseData{N<:Number} <: TimeSource
    fcen::N
    fwidth::N
    width::N
    peak_time::N
    cutoff::N
end

"""
    GaussianPulseSource(;fcen, fwidth, start_time = 0.0, cutoff_scale = 5.0)

Create a Gaussian pulse time source with center frequency `fcen` and width `fwidth`.
"""
function GaussianPulseSource(; fcen, fwidth, start_time = 0.0, cutoff_scale = 5.0)
    #TODO right now the cutoff doesn't make much sense...
    width = 1.0 / fwidth
    cutoff = width * cutoff_scale + start_time
    fwidth = _gaussian_bandwidth(width)

    while (exp(-cutoff * cutoff / (2 * width * width)) < 1e-100)
        cutoff *= 0.9
    end

    peak_time = (cutoff) / 2

    return GaussianPulseData{backend_number}(fcen, fwidth, width, peak_time, cutoff)
end

"""
    _gaussian_bandwidth(width)

bandwidth (in frequency units, not angular frequency) of the
continuous Fourier transform of the Gaussian source function
when it has decayed by a tolerance tol below its peak value
"""
function _gaussian_bandwidth(width)
    tol = 1e-7
    return sqrt(-2.0 * log(tol)) / (width * pi)
end

function eval_time_source(src::GaussianPulseData{numType}, t::Real) where {numType<:Number}
    tt = t - src.peak_time
    if tt > src.cutoff
        return 0.0
    end
    amp = inv((-2 * pi * src.fcen * im))
    return exp(-tt * tt / (2 * src.width * src.width)) *
           exp(-2 * pi * im * src.fcen * tt) *
           amp
end

get_cutoff(src::GaussianPulseData) = src.cutoff

function get_frequency(src::GaussianPulseData)
    return src.fcen
end

# ---------------------------------------------------------- #
# Custom source
# ---------------------------------------------------------- #

# TODO
