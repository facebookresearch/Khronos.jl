# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for dispersive material (ADE) implementation.

using Test

# Include Khronos module setup
include("../src/Khronos.jl")
using .Khronos

@testset "Susceptibility types" begin
    # Lorentzian susceptibility
    s = LorentzianSusceptibility(1.0, 0.1, 2.0)
    @test s.omega_0 == 1.0
    @test s.gamma == 0.1
    @test s.sigma == 2.0

    # Drude susceptibility (omega_0 = 0)
    d = DrudeSusceptibility(0.5, 3.0)
    @test d isa LorentzianSusceptibility
    @test d.omega_0 == 0.0
    @test d.gamma == 0.5
    @test d.sigma == 3.0
end

@testset "ADE coefficients" begin
    dt = 0.01
    s = LorentzianSusceptibility(1.0, 0.1, 2.0)
    c = compute_ade_coefficients(s, dt)

    @test c.gamma1 ≈ 1.0 - 0.1 * π * dt
    @test c.gamma1_inv ≈ 1.0 / (1.0 + 0.1 * π * dt)
    @test c.omega0_dt_sq ≈ (2π * 1.0 * dt)^2
    @test c.sigma_omega0_dt_sq ≈ 2.0 * (2π * 1.0 * dt)^2
    @test c.is_drude == false

    # Drude coefficients
    d = DrudeSusceptibility(0.5, 3.0)
    cd = compute_ade_coefficients(d, dt)
    @test cd.is_drude == true
    @test cd.omega0_dt_sq == 0.0
    @test cd.drude_coeff ≈ 3.0 * 0.5 * 2π * dt
end

@testset "Susceptibility evaluation" begin
    # Lorentzian: χ(ω) = sigma * ω₀² / (ω₀² - ω² - iωγ)
    s = LorentzianSusceptibility(1.0, 0.1, 2.0)
    freq = 0.5  # test frequency

    χ = Khronos.eval_susceptibility(s, freq)
    ω = 2π * freq
    ω₀ = 2π * 1.0
    γ = 2π * 0.1
    expected = 2.0 * ω₀^2 / (ω₀^2 - ω^2 - im * ω * γ)
    @test χ ≈ expected

    # Drude: χ(ω) = -sigma * γ / (ω² + iωγ)
    d = DrudeSusceptibility(0.5, 3.0)
    χd = Khronos.eval_susceptibility(d, freq)
    γd = 2π * 0.5
    expected_d = -3.0 * γd / (ω^2 + im * ω * γd)
    @test χd ≈ expected_d
end

@testset "Material with susceptibilities" begin
    # Create a material with dispersive poles
    m = Material{Float64}(
        ε = 1.0,
        susceptibilities = [
            LorentzianSusceptibility(1.0, 0.1, 2.0),
            DrudeSusceptibility(0.5, 3.0),
        ]
    )
    @test Khronos.has_susceptibilities(m)
    @test length(m.susceptibilities) == 2

    # Material without susceptibilities
    m2 = Material{Float64}(ε = 2.25)
    @test !Khronos.has_susceptibilities(m2)
end
