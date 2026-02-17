# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Susceptibility types and ADE (Auxiliary Differential Equation) coefficients
# for dispersive material models (Drude, Lorentz).
#
# Reference: meep/src/susceptibility.cpp:188-262

export LorentzianSusceptibility, DrudeSusceptibility, ADECoefficients, compute_ade_coefficients

"""
    Susceptibility

Abstract base type for frequency-dependent material susceptibilities.
"""
abstract type Susceptibility end

"""
    LorentzianSusceptibility(omega_0, gamma, sigma)

A Lorentzian oscillator susceptibility pole:

    χ(ω) = sigma * ω₀² / (ω₀² - ω² - iωγ)

where `omega_0` is the resonance frequency, `gamma` is the damping rate,
and `sigma` is the oscillator strength (dimensionless).

Frequencies use Meep convention: angular frequency / 2π (i.e., regular Hz).
"""
struct LorentzianSusceptibility <: Susceptibility
    omega_0::Float64    # resonance frequency (angular freq / 2pi)
    gamma::Float64      # damping rate (same units)
    sigma::Float64      # oscillator strength (dimensionless)
end

"""
    DrudeSusceptibility(gamma, sigma)

A Drude free-electron susceptibility (Lorentzian with ω₀ = 0):

    χ(ω) = -sigma * γ / (ω² + iωγ)

This is equivalent to `LorentzianSusceptibility(0.0, gamma, sigma)`.
"""
DrudeSusceptibility(gamma::Real, sigma::Real) =
    LorentzianSusceptibility(0.0, Float64(gamma), Float64(sigma))

"""
    ADECoefficients

Pre-computed coefficients for the ADE time-stepping update of a single
Lorentzian/Drude pole. The per-voxel sigma (oscillator strength) is stored
separately in the per-voxel sigma arrays; these coefficients do NOT include sigma.

Lorentz update:
    P^{n+1} = γ₁_inv * (P^n * (2 - ω₀²Δt²) - γ₁ * P^{n-1} + ω₀²Δt² * σ_voxel * E^n)

Drude update (ω₀ = 0):
    P^{n+1} = γ₁_inv * (2*P^n - γ₁*P^{n-1} + γ*2π*Δt² * σ_voxel * E^n)
"""
struct ADECoefficients
    gamma1_inv::Float64   # 1 / (1 + gamma*pi*dt)
    gamma1::Float64       # 1 - gamma*pi*dt
    omega0_dt_sq::Float64 # (2*pi*omega_0*dt)^2
    sigma_omega0_dt_sq::Float64  # (2*pi*omega_0*dt)^2 (for Lorentz driving term; sigma from per-voxel array)
    drude_coeff::Float64  # gamma * 2*pi*dt^2 = Γ*Δt² (for Drude driving term; sigma from per-voxel array)
    is_drude::Bool        # true if omega_0 == 0
end

"""
    compute_ade_coefficients(s::LorentzianSusceptibility, dt::Real) -> ADECoefficients

Pre-compute the ADE update coefficients for a given susceptibility and timestep.
"""
function compute_ade_coefficients(s::LorentzianSusceptibility, dt::Real)
    gamma_pi_dt = s.gamma * π * dt
    gamma1 = 1.0 - gamma_pi_dt
    gamma1_inv = 1.0 / (1.0 + gamma_pi_dt)
    omega0_dt_sq = (2π * s.omega_0 * dt)^2
    sigma_omega0_dt_sq = omega0_dt_sq  # sigma applied from per-voxel array, not baked into coefficient
    drude_coeff = s.gamma * 2π * dt * dt  # Γ·Δt² driving coefficient (sigma from per-voxel array)
    is_drude = s.omega_0 == 0.0

    return ADECoefficients(gamma1_inv, gamma1, omega0_dt_sq, sigma_omega0_dt_sq,
                           drude_coeff, is_drude)
end

"""
    eval_susceptibility(s::LorentzianSusceptibility, freq::Real) -> ComplexF64

Evaluate the susceptibility χ(ω) at the given frequency (in Meep convention).
"""
function eval_susceptibility(s::LorentzianSusceptibility, freq::Real)
    ω = 2π * freq
    ω₀ = 2π * s.omega_0
    γ = 2π * s.gamma
    if s.omega_0 == 0  # Drude
        return -s.sigma * γ / (ω^2 + im * ω * γ)  # Note: not ω₀² here
    else  # Lorentz
        return s.sigma * ω₀^2 / (ω₀^2 - ω^2 - im * ω * γ)
    end
end

"""
    has_susceptibilities(m::Material) -> Bool

Check if a material has any dispersive susceptibility poles.
"""
has_susceptibilities(m) = hasfield(typeof(m), :susceptibilities) &&
    !isnothing(m.susceptibilities) && !isempty(m.susceptibilities)

"""
    any_material_has_susceptibilities(geometry) -> Bool

Check if any object in the geometry list has dispersive materials.
"""
function any_material_has_susceptibilities(geometry)
    isnothing(geometry) && return false
    for obj in geometry
        has_susceptibilities(obj.material) && return true
    end
    return false
end
