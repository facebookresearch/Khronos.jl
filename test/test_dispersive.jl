# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for dispersive material (ADE) implementation.

import Khronos
using Test
using LinearAlgebra

@testset "Susceptibility types" begin
    # Lorentzian susceptibility
    s = Khronos.LorentzianSusceptibility(1.0, 0.1, 2.0)
    @test s.omega_0 == 1.0
    @test s.gamma == 0.1
    @test s.sigma == 2.0

    # Drude susceptibility (omega_0 = 0)
    d = Khronos.DrudeSusceptibility(0.5, 3.0)
    @test d isa Khronos.LorentzianSusceptibility
    @test d.omega_0 == 0.0
    @test d.gamma == 0.5
    @test d.sigma == 3.0
end

@testset "ADE coefficients" begin
    dt = 0.01
    s = Khronos.LorentzianSusceptibility(1.0, 0.1, 2.0)
    c = Khronos.compute_ade_coefficients(s, dt)

    @test c.gamma1 ≈ 1.0 - 0.1 * π * dt
    @test c.gamma1_inv ≈ 1.0 / (1.0 + 0.1 * π * dt)
    @test c.omega0_dt_sq ≈ (2π * 1.0 * dt)^2
    @test c.sigma_omega0_dt_sq ≈ (2π * 1.0 * dt)^2  # sigma applied per-voxel, not baked in
    @test c.is_drude == false

    # Drude coefficients
    d = Khronos.DrudeSusceptibility(0.5, 3.0)
    cd = Khronos.compute_ade_coefficients(d, dt)
    @test cd.is_drude == true
    @test cd.omega0_dt_sq == 0.0
    @test cd.drude_coeff ≈ 0.5 * 2π * dt^2  # gamma * 2π * dt² (sigma applied per-voxel)
end

@testset "Susceptibility evaluation" begin
    # Lorentzian: χ(ω) = sigma * ω₀² / (ω₀² - ω² - iωγ)
    s = Khronos.LorentzianSusceptibility(1.0, 0.1, 2.0)
    freq = 0.5
    χ = Khronos.eval_susceptibility(s, freq)
    ω = 2π * freq
    ω₀ = 2π * 1.0
    γ = 2π * 0.1
    expected = 2.0 * ω₀^2 / (ω₀^2 - ω^2 - im * ω * γ)
    @test χ ≈ expected

    # Drude: χ(ω) = -sigma * γ / (ω² + iωγ)
    d = Khronos.DrudeSusceptibility(0.5, 3.0)
    χd = Khronos.eval_susceptibility(d, freq)
    γd = 2π * 0.5
    expected_d = -3.0 * γd / (ω^2 + im * ω * γd)
    @test χd ≈ expected_d
end

@testset "Material with susceptibilities" begin
    m = Khronos.Material(
        ε = 1.0,
        susceptibilities = [
            Khronos.LorentzianSusceptibility(1.0, 0.1, 2.0),
            Khronos.DrudeSusceptibility(0.5, 3.0),
        ],
    )
    @test Khronos.has_susceptibilities(m)
    @test length(m.susceptibilities) == 2

    # Material without susceptibilities
    m2 = Khronos.Material(ε = 2.25)
    @test !Khronos.has_susceptibilities(m2)
end

@testset "Drude ADE eigenvalue stability (chi1 correction)" begin
    # Linear stability analysis: verify that the 4×4 ADE update matrix has
    # all eigenvalues |λ| ≤ 1 across the full wavenumber range.
    #
    # Uses the actual silver Drude parameters from the micro-LED tutorial
    # at resolution 40. Tests that WITH the chi1 semi-implicit correction
    # the scheme is stable, and WITHOUT it the scheme is unstable.

    # Silver Drude parameters at 450nm (from uled.jl)
    nm = 1e-3
    lda0 = 450nm
    freq0 = 1.0 / lda0
    eps_ag = (0.028 + 2.88im)^2
    chi_target = eps_ag - 1.0
    w0 = 2π * freq0
    chi_r, chi_i = real(chi_target), imag(chi_target)
    Gamma_ang = -w0 * chi_i / chi_r
    gamma_meep = Gamma_ang / (2π)
    sigma_drude = -chi_r * (w0^2 + Gamma_ang^2) / Gamma_ang

    # Simulation parameters at resolution 40
    resolution = 40
    dx = 1.0 / resolution  # 0.025
    dt = 0.5 * dx / sqrt(3.0)  # Courant-limited Δt

    Co = dt / dx  # Courant number

    # ADE coefficients
    gamma_pi_dt = gamma_meep * π * dt
    g1 = 1 - gamma_pi_dt
    g1i = 1 / (1 + gamma_pi_dt)
    drude_coeff = gamma_meep * 2π * dt^2
    d = drude_coeff * sigma_drude
    chi1 = g1i * d / 2
    eps_eff = 1 + chi1

    # WITH chi1: scan wavenumber space, verify max|λ| ≤ 1
    max_eig_chi1 = 0.0
    for k_frac in range(0.01, 1.0, length = 200)
        k = k_frac * π / dx
        S = 2 * Co * sin(k * dx / 2)

        M = ComplexF64[
            1-(S^2+g1i*d)/eps_eff   im*S/eps_eff   -g1*g1i/eps_eff   g1*g1i/eps_eff;
            im*S                     1               0                  0;
            g1i*d                    0               2g1i              -g1i*g1;
            0                        0               1                  0
        ]

        eigs = eigvals(M)
        max_eig_chi1 = max(max_eig_chi1, maximum(abs.(eigs)))
    end
    @test max_eig_chi1 ≤ 1.0 + 1e-12  # stable with chi1

    # 3D worst case (S² = 3 at the Nyquist corner)
    S_3d = sqrt(3.0)
    M_3d = ComplexF64[
        1-(S_3d^2+g1i*d)/eps_eff   im*S_3d/eps_eff   -g1*g1i/eps_eff   g1*g1i/eps_eff;
        im*S_3d                     1                   0                  0;
        g1i*d                       0                   2g1i              -g1i*g1;
        0                           0                   1                  0
    ]
    eigs_3d = eigvals(M_3d)
    @test maximum(abs.(eigs_3d)) ≤ 1.0 + 1e-12  # stable in 3D worst case

    # WITHOUT chi1: verify scheme is UNSTABLE (proves the correction is necessary)
    eps_no_chi1 = 1.0
    max_eig_no_chi1 = 0.0
    for k_frac in range(0.01, 1.0, length = 200)
        k = k_frac * π / dx
        S = 2 * Co * sin(k * dx / 2)

        M = ComplexF64[
            1-(S^2+g1i*d)/eps_no_chi1   im*S/eps_no_chi1   -g1*g1i/eps_no_chi1   g1*g1i/eps_no_chi1;
            im*S                         1                    0                     0;
            g1i*d                        0                    2g1i                 -g1i*g1;
            0                            0                    1                     0
        ]

        eigs = eigvals(M)
        max_eig_no_chi1 = max(max_eig_no_chi1, maximum(abs.(eigs)))
    end
    @test max_eig_no_chi1 > 1.0  # unstable without chi1
end
