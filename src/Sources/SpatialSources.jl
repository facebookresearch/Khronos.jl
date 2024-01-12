# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Here live the various functions and routines needed to implement a source that
# is spatially varying. Importantly, all spatially varying sources take as one
# of their parameters the corresponding time profile (temporal dependence).
#
# Each source is defined via the following interface functions. Since the
# initialization of the source is *not* done in the main hot loop, and very
# rarely encompassing the whole domain, we don't need to worry too much about
# dynamic dispatch. So we'll leverage multiple dispatch as much as possible and
# worry about type stability once it's a problem.

export UniformSource, EquivalentSource, PlaneWaveSource, GaussianBeamSource

# ---------------------------------------------------------- #
# Interface functions
# ---------------------------------------------------------- #

"""
    get_time_profile(source::Source)

Returns the time profile associated with a source.
"""
function get_time_profile(source::Source)
    return source.time_profile
end

"""
    get_source_center(source::Source)

Returns the center of the source (in 3D space).
"""
function get_source_center(source::Source)
    return source.center
end

"""
    get_source_size(source::Source)

Returns the size of the source (in 3D space).
"""
function get_source_size(source::Source)
    return source.size
end

"""
    get_source_volume(source::Source)

Returns a `Volume`` object for the source.
"""
function get_source_volume(source::Source)
    return Volume(center = get_source_center(source), size = get_source_size(source))
end

"""
    get_source_components(source::Source)

Returns all of a source's field components.
"""
function get_source_components(source::Source)
    return source.components
end

"""
    get_amplitude(source::Source)

Returns the scalar amplitude of the source.
"""
function get_amplitude(source::Source)
    return source.amplitude
end

"""
    get_source_profile(source::Source, point::Vector{<:Real}, component::Field)

Returns the spatially varying amplitude of the source.
"""
function get_source_profile(source::Source, point::Vector{<:Real}, component::Field)
    error("`source_profile` not yet implemented for this type of source...")
end

# ---------------------------------------------------------- #
# Uniform source
# ---------------------------------------------------------- #

@with_kw struct UniformSourceData <: Source
    time_profile::TimeSource
    component::Field
    amplitude::Number = 1.0
    center::AbstractVector = [0.0, 0.0, 0.0]
    size::AbstractVector = [0.0, 0.0, 0.0]
end

"""
    UniformSource(;
    time_profile::TimeSource,
    component::Field,
    amplitude::Number = 1.0,
    center::AbstractVector = [0.0, 0.0, 0.0],
    size::AbstractVector = [0.0, 0.0, 0.0],
    )

TBW
"""
function UniformSource(;
    time_profile::TimeSource,
    component::Field,
    amplitude::Number = 1.0,
    center::AbstractVector = [0.0, 0.0, 0.0],
    size::AbstractVector = [0.0, 0.0, 0.0],
)
    UniformSourceData(
        time_profile = time_profile,
        component = component,
        amplitude = amplitude,
        center = center,
        size = size,
    )
end

function get_source_profile(::UniformSourceData, ::Vector{<:Real}, ::Field)
    return 1.0
end

function get_source_components(source::UniformSourceData)
    return [source.component]
end

# ---------------------------------------------------------- #
# Equivalent source
# ---------------------------------------------------------- #

@with_kw struct EquivalentSourceData <: Source
    time_profile::TimeSource
    center::AbstractVector = [0.0, 0.0, 0.0]
    size::AbstractVector = [0.0, 0.0, 0.0]
    amplitude::Number = 1.0
    source_profile::Dict{<:Field,<:Function}
end

"""
    EquivalentSource(;
    time_profile::TimeSource,
    fields::Dict{<:Field,<:AbstractArray},
    center::AbstractVector{N} = [0.0, 0.0, 0.0],
    size::AbstractVector{N} = [0.0, 0.0, 0.0],
    amplitude::N = 1.0,
)::EquivalentSourceData where N<:Number

Implements equivalent current sources corresponding to an input dataset
containing `E` and `H` fields.

For the injection to work as expected (i.e. to reproduce the required `E` and
`H` fields), the field data must decay by the edges of the source plane, or the
source plane must span the entire simulation domain and the fields must match
the simulation boundary conditions. The equivalent source currents are fully
defined by the field components tangential to the source plane. For e.g. source
normal along z, the normal components (Ez and Hz) can be provided but will have
no effect on the results, and at least one of the tangential components has to
be in the dataset, i.e. at least one of Ex, Ey, Hx, and Hy.
"""
function EquivalentSource(;
    time_profile::TimeSource,
    fields::Dict{<:Field,<:AbstractArray},
    center::AbstractVector = [0.0, 0.0, 0.0],
    size::AbstractVector = [0.0, 0.0, 0.0],
    amplitude::Number = 1.0,
)::EquivalentSourceData

    # Sanitize the input
    required_field_components = [Ex(), Ey(), Ez(), Hx(), Hy(), Hz()]
    for component in required_field_components
        if !haskey(fields, component)
            error("An equivalent source requires all 6 field components.")
        end
        if ndims(fields[component]) != 3
            error("The supplied field profiles must be 3D.")
        end
    end

    src_vol = Volume(center = center, size = size)
    normal_vector = get_normal_vector(src_vol)
    transverse_fields = plane_normal_direction(src_vol)

    Ex = fields[Ex()]
    Ey = fields[Ey()]
    Ez = fields[Ez()]
    Hx = fields[Hx()]
    Hy = fields[Hy()]
    Hz = fields[Hz()]

    current_sources = dict(
        # Electric current J = nHat x H
        Ex() => gen_interpolator_from_array(
            normal_vector[1] * Hz .- normal_vector[2] * Hy,
            src_vol,
        ),
        Ey() => gen_interpolator_from_array(
            normal_vector[2] * Hx .- normal_vector[0] * Hz,
            src_vol,
        ),
        Ez() => gen_interpolator_from_array(
            normal_vector[0] * Hy .- normal_vector[1] * Hx,
            src_vol,
        ),
        # Magnetic current K = - nHat x E
        Hx() => gen_interpolator_from_array(
            normal_vector[2] * Ey .- normal_vector[1] * Ez,
            src_vol,
        ),
        Hy() => gen_interpolator_from_array(
            normal_vector[0] * Ez .- normal_vector[2] * Ex,
            src_vol,
        ),
        Hz() => gen_interpolator_from_array(
            normal_vector[1] * Ex .- normal_vector[0] * Ey,
            src_vol,
        ),
    )

    # Only create current sources we really need
    for component in required_field_components
        if (iszero(current_sources[component]) || !(component in transverse_fields))
            delete!(current_sources, component)
        end
    end

    return EquivalentSourceData(
        time_profile = time_profile,
        center = center,
        size = size,
        amplitude = amplitude,
        source_profile = current_sources,
    )
end

function get_source_components(source::EquivalentSourceData)
    return collect(keys(source.source_profile))
end

function get_source_profile(
    source::EquivalentSourceData,
    point::Vector{<:Real},
    component::Field,
)
    return source.source_profile[component](point)
end

# ---------------------------------------------------------- #
# Custom current source
# ---------------------------------------------------------- #

"""
    CustomCurrentSource(;
    time_profile::TimeSource,
    fields::Dict{<:Field,<:AbstractArray},
    center::AbstractVector{N} = [0.0, 0.0, 0.0],
    size::AbstractVector{N} = [0.0, 0.0, 0.0],
    amplitude::N = 1.0,
    snap_to_grid::Bool = false,
)

Implements a source corresponding to an input dataset containing `E` and `H` fields.

Injects the specified components of the `E` and `H` dataset directly as `J` and `M`
current distributions in the FDTD solver.
"""
function CustomCurrentSource(;
    time_profile::TimeSource,
    fields::Dict{<:Field,<:AbstractArray},
    center::AbstractVector = [0.0, 0.0, 0.0],
    size::AbstractVector = [0.0, 0.0, 0.0],
    amplitude::Complex = 1.0,
    snap_to_grid::Bool = false,
)
    # TODO
end

# ---------------------------------------------------------- #
# Planewave source
# ---------------------------------------------------------- #

@with_kw struct PlaneWaveSourceData <: Source
    time_profile::TimeSource
    center::AbstractVector{<:Real} = [0.0, 0.0, 0.0]
    size::AbstractVector{<:Real} = [0.0, 0.0, 0.0]
    amplitude::Number = 1.0
    ε::Number = 1.0
    μ::Number = 1.0
    polarization::AbstractVector{<:Number}
    k_vector::AbstractVector{<:Real}
    k::Number
    normal_direction::AbstractVector{<:Real}
end


"""
    PlaneWaveSource(;
    time_profile::TimeSource,
    center::AbstractVector{<:Real} = [0.0, 0.0, 0.0],
    size::AbstractVector{<:Real} = [0.0, 0.0, 0.0],
    amplitude::Number = 1.0,
    ε::Number = 1.0,
    μ::Number = 1.0,
    polarization_angle::Real,
    angle_theta::Union{Real,Nothing} = nothing,
    angle_phi::Union{Real,Nothing} = nothing,
    k_vector::Union{AbstractVector{<:Real}, Nothing} = nothing,
)::PlaneWaveSourceData

TBW
"""
function PlaneWaveSource(;
    time_profile::TimeSource,
    center::AbstractVector{<:Real} = [0.0, 0.0, 0.0],
    size::AbstractVector{<:Real} = [0.0, 0.0, 0.0],
    amplitude::Number = 1.0,
    ε::Number = 1.0,
    μ::Number = 1.0,
    polarization_angle::Real,
    angle_theta::Union{Real,Nothing} = nothing,
    angle_phi::Union{Real,Nothing} = nothing,
    k_vector::Union{AbstractVector{<:Real},Nothing} = nothing,
)::PlaneWaveSourceData

    if isnothing(k_vector) && isnothing(angle_theta) && isnothing(angle_phi)
        error("Must either specify a `k_vector` or `angle_phi` and `angle_theta`.")
    end

    # Compute k-vector from polar angle representation.
    if isnothing(k_vector)
        k_vector = [
            sin(angle_phi) * cos(angle_theta),
            sin(angle_phi) * sin(angle_theta),
            cos(angle_phi),
        ]
    end

    k = Real(2 * π * get_frequency(time_profile) * sqrt(ε * μ))
    k_vector = k_vector / norm(k_vector)

    # Injection plane normal direction
    normal_direction = get_normal_vector(Volume(center = center, size = size))

    # Compute the normal vector of the plane formed by the injection plane
    # and direction of propagation.
    if dot(normal_direction, k_vector) == 1.0
        # Make sure that the vectors are aligned with the cartesian axis at this
        # point.
        if count(==(0.0), normal_direction) != 2
            error(
                "Invalid injection plane normal direction $(normal_direction) and k vector $(k_vector).",
            )
        end

        # If the direction of propogation is inline with the normal, then S and
        # P are degenerate. So we establish the following convention.
        pol_mapping = Dict(1 => [0.0, 1.0, 0.0], 2 => [0.0, 0.0, 1.0], 3 => [1.0, 0.0, 0.0])
        pol_plane_normal = pol_mapping[findfirst(!=(0.0), k_vector)]
    else

        pol_plane_normal = cross(normal_direction, k_vector)
    end
    # Using Rodrigues' formula, we'll rotate this new normal vector around
    # the direction of propagation. Note that we omit the last term, since
    # the polarization direction is always orthogonal to the direction of
    # propagation.
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    polarization_vector = (
        pol_plane_normal * cos(polarization_angle) +
        cross(k_vector, pol_plane_normal) * sin(polarization_angle)
    )

    return PlaneWaveSourceData(
        time_profile = time_profile,
        center = center,
        size = size,
        ε = ε,
        μ = μ,
        amplitude = amplitude,
        polarization = polarization_vector,
        k_vector = k_vector,
        k = k,
        normal_direction = normal_direction,
    )
end

"""
    get_planewave_polarization_scaling(
    source::PlaneWaveSourceData, 
    component::Electric
    )

Computes the electric current-source polarization scaling term (Ĵ₀) for a planewave.
"""
function get_planewave_polarization_scaling(
    source::PlaneWaveSourceData,
    component::Electric,
)::Number
    Z0 = sqrt(source.μ / source.ε)
    # Ĵ = n̂×(k̂×Ê) / Z₀
    return (cross(
        source.normal_direction,
        cross(source.k_vector, 1 / Z0 * source.polarization),
    ))[int_from_direction(component)]
end

"""
    get_planewave_polarization_scaling(source::PlaneWaveSourceData, component::Magnetic)

Computes the magnetic current-source polarization scaling term (M̂₀) for a planewave.
"""
function get_planewave_polarization_scaling(
    source::PlaneWaveSourceData,
    component::Magnetic,
)::Number
    # M̂ = -n̂×Ê
    return -(cross(source.normal_direction, source.polarization))[int_from_direction(
        component,
    )]
end

"""
    get_source_components(source::PlaneWaveSourceData)

Returns the relevant current source components for a planewave source.

In particular, only non-zero, transverse components are relevant.
"""
function get_source_components(source::PlaneWaveSourceData)
    transverse_components =
        get_plane_transverse_fields(Volume(center = source.center, size = source.size))
    source_components = Field[]
    for component in transverse_components
        if get_planewave_polarization_scaling(source, component) != 0.0
            push!(source_components, component)
        end
    end
    return source_components
end

"""
    get_source_profile(source::PlaneWaveSourceData, point::Vector{<:Real}, component::Field)

Computes a plane wave current source for a given component at a point in space.
"""
function get_source_profile(
    source::PlaneWaveSourceData,
    point::Vector{<:Real},
    component::Field,
)
    pol_factor = get_planewave_polarization_scaling(source, component)
    return pol_factor * exp(-im * point ⋅ source.k_vector * source.k)
end

# ---------------------------------------------------------- #
# Mode source
# ---------------------------------------------------------- #

#TODO

@with_kw struct ModeSourceData{N<:Number} <: Source
    amplitude::N = 1.0
end

# ---------------------------------------------------------- #
# Gaussian beam source
# ---------------------------------------------------------- #

@with_kw struct GaussianBeamData <: Source
    time_profile::TimeSource
    center::Vector{<:Real} = [0.0, 0.0, 0.0]
    size::Vector{<:Real} = [0.0, 0.0, 0.0]
    amplitude::Number = 1.0
    ε::Number = 1.0
    μ::Number = 1.0
    beam_center::Vector{<:Number}
    beam_waist::Real
    k_vector::Vector{<:Real}
    polarization::Vector{<:Number}
    normal_direction::Vector{<:Real}
end

"""
    GaussianBeamSource(;
    time_profile::TimeSource,
    center::Vector{<:Real} = [0.0, 0.0, 0.0],
    size::Vector{<:Real} = [0.0, 0.0, 0.0],
    amplitude::Number = 1.0,
    ε::Number = 1.0,
    μ::Number = 1.0,
    beam_center::Vector{<:Number},
    beam_waist::Real,
    k_vector::Vector{<:Real},
    polarization::Vector{<:Number},
)::GaussianBeamData

TBW
"""
function GaussianBeamSource(;
    time_profile::TimeSource,
    center::Vector{<:Real} = [0.0, 0.0, 0.0],
    size::Vector{<:Real} = [0.0, 0.0, 0.0],
    amplitude::Number = 1.0,
    ε::Number = 1.0,
    μ::Number = 1.0,
    beam_center::Vector{<:Number},
    beam_waist::Real,
    k_vector::Vector{<:Real},
    polarization::Vector{<:Number},
)::GaussianBeamData
    return GaussianBeamData(
        time_profile = time_profile,
        center = center,
        size = size,
        amplitude = amplitude,
        ε = ε,
        μ = μ,
        beam_center = beam_center,
        beam_waist = beam_waist,
        k_vector = k_vector,
        polarization = polarization,
        normal_direction = get_normal_vector(Volume(center = center, size = size)),
    )
end

"""
    get_gaussianbeam_field_profiles(
    source::GaussianBeamData, 
    point::Vector{<:Real}
)

TBW
"""
function get_gaussianbeam_field_profiles(source::GaussianBeamData, point::Vector{<:Real})
    # Create useful variables
    ε_R = real(source.ε)
    μ_R = real(source.μ)
    k = ComplexF64(2 * π * get_frequency(source.time_profile) * sqrt(ε_R * μ_R))
    ZR = sqrt(μ_R / ε_R)
    zvE0 = source.polarization
    z0 = k * source.beam_waist^2 / 2.0
    kz0 = k * z0

    # Relative coordinates
    xrel = point - source.beam_center
    zHat = source.k_vector / norm(source.k_vector)
    rho = norm(cross(zHat, xrel))
    z = xrel ⋅ zHat
    use_rescaled_FG = false
    zc = z - im * z0
    Rsq = rho * rho + zc * zc
    R = sqrt(Rsq)
    kR = k * R
    kRsq = kR * kR
    kR3 = kRsq * kR

    # Large kR
    if (abs(kR) > 1e-15)
        # Large imag kR
        if (abs(imag(kR)) > 30.0)
            use_rescaled_FG = true
            ExpI = exp(im * real(kR))
            ExpPlus = exp(imag(kR) - kz0)
            ExpMinus = exp(-imag(kR) - kz0)
            coskR = 0.5 * (ExpI * ExpMinus + conj(ExpI) * ExpPlus)
            sinkR = -im * 0.5 * (ExpI * ExpMinus - conj(ExpI) * ExpPlus)
        else
            coskR = cos(kR)
            sinkR = sin(kR)
        end
        # Calculate f and g
        f = -3 * (coskR / kRsq - sinkR / kR3)
        g = 1.5 * (sinkR / kR + coskR / kRsq - sinkR / kR3)
        fmgbRsq = (f - g) / Rsq

        # Small kR
    else
        # Series expansion
        kR4 = kRsq * kRsq
        f = kR4 / 280.0 - kRsq / 10.0 + 1.0
        g = kR4 * 3.0 / 280.0 - kRsq / 5.0 + 1.0
        fmgbRsq = (kR4 / 5040.0 - kRsq / 140.0 + 0.1) * (k * k)
    end

    # New variables
    i2fk = 0.5 * im * f * k
    E = zeros(ComplexF64, 3)
    H = zeros(ComplexF64, 3)
    rnorm = norm(real.(zvE0))
    inorm = norm(imag.(zvE0))

    # If sufficient field, fill E and H
    if (rnorm > 1e-6)
        xHat = real.(zvE0) ./ rnorm
        yHat = cross(zHat, xHat)
        norm_hat = hcat(xHat, yHat, zHat)

        x = xHat ⋅ xrel
        y = yHat ⋅ xrel

        Ex = g + fmgbRsq * x * x + i2fk * zc
        Ey = fmgbRsq * x * y
        Ez = fmgbRsq * x * zc - i2fk * x
        E_norm = [Ex, Ey, Ez]

        Hx = Ey
        Hy = g + fmgbRsq * y * y + i2fk * zc
        Hz = fmgbRsq * y * zc - i2fk * y
        H_norm = [Hx, Hy, Hz]

        E = E .+ rnorm * (norm_hat * E_norm)
        H = H .+ rnorm * (norm_hat * H_norm)
    end

    if (inorm > 1e-6)
        xHat = imag.(zvE0) / inorm
        yHat = cross(zHat, xHat)
        norm_hat = hcat(xHat, yHat, zHat)

        x = xHat ⋅ xrel
        y = yHat ⋅ xrel

        Ex = g + fmgbRsq * x .* x + i2fk * zc
        Ey = fmgbRsq * x .* y
        Ez = fmgbRsq * x * zc - i2fk * x
        E_norm = [Ex, Ey, Ez]

        Hx = 0.0
        Hy = g + fmgbRsq * y * y + i2fk * zc
        Hz = fmgbRsq * y * zc - i2fk * y
        H_norm = [Hx, Hy, Hz]

        E = E .+ inorm * (norm_hat * E_nor)
        H = H .+ inorm * (norm_hat * H_norm)
    end

    # Rescale as necessary
    if (use_rescaled_FG)
        E_orig =
            3.0 / (2 * kz0 * kz0 * kz0) * (kz0 * (kz0 - 1) + 0.5 * (1.0 - exp(-2.0 * kz0)))
    else
        E_orig = 3.0 / (2 * kz0 * kz0 * kz0) * (exp(kz0) * kz0 * (kz0 - 1) + sinh(kz0))
    end

    # Rescale
    E /= E_orig
    H /= (E_orig * ZR)

    return E, H
end

function get_source_profile(
    source::GaussianBeamData,
    point::Vector{<:Real},
    component::Field,
)
    E, H = get_gaussianbeam_field_profiles(source, point)
    if is_electric(component)
        # Electric current source, Ĵ = n̂×Ĥ
        return cross(source.normal_direction, H)[int_from_direction(component)]
    else
        # Magnetic current source, M̂ = -n̂×Ê
        return -cross(source.normal_direction, E)[int_from_direction(component)]
    end
end

function get_source_components(source::GaussianBeamData)
    transverse_components =
        get_plane_transverse_fields(Volume(center = source.center, size = source.size))
    source_components = Field[]
    for component in transverse_components
        # TODO: it would be great to work out which components are all zero...
        push!(source_components, component)
    end
    return source_components
end
