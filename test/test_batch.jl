# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for batch simulation execution and incoherent LEE computation.

import Khronos
using Test
using Statistics

@testset "LEE with uniform power" begin
    # Uniform power: LEE should equal the solid angle fraction
    n_theta = 200
    n_phi = 200
    theta = collect(range(0, π / 2, length = n_theta))
    phi = collect(range(0, 2π, length = n_phi + 1))[1:end-1]

    power = ones(n_theta, length(phi))

    # For uniform power: LEE(α) = (1 - cos(α)) / (1 - cos(π/2)) = 1 - cos(α)
    for alpha_deg in [15.0, 30.0, 45.0, 60.0]
        alpha = deg2rad(alpha_deg)
        lee = Khronos.compute_LEE(power, theta, phi; cone_half_angle = alpha)
        expected = (1 - cos(alpha)) / (1 - cos(π / 2))
        @test lee ≈ expected rtol = 0.02
    end

    # Full hemisphere should give LEE = 1
    lee_full = Khronos.compute_LEE(power, theta, phi; cone_half_angle = π / 2)
    @test lee_full ≈ 1.0 rtol = 0.01
end

@testset "LEE with Lambertian emitter (cosθ)" begin
    # A Lambertian source has P(θ) = cos(θ).
    # Total hemispherical power: ∫₀^{2π} ∫₀^{π/2} cos(θ) sin(θ) dθ dφ = π
    # Cone power: ∫₀^{2π} ∫₀^α cos(θ) sin(θ) dθ dφ = π sin²(α)
    # LEE(α) = sin²(α)
    n_theta = 200
    n_phi = 200
    theta = collect(range(0, π / 2, length = n_theta))
    phi = collect(range(0, 2π, length = n_phi + 1))[1:end-1]

    power = [cos(θ) for θ in theta, _ in phi]

    for alpha_deg in [15.0, 30.0, 45.0, 60.0]
        alpha = deg2rad(alpha_deg)
        lee = Khronos.compute_LEE(power, theta, phi; cone_half_angle = alpha)
        expected = sin(alpha)^2
        @test lee ≈ expected rtol = 0.02
    end
end

@testset "LEE with cos²θ emitter" begin
    # P(θ) = cos²(θ).
    # Total: ∫₀^{2π} ∫₀^{π/2} cos²(θ) sin(θ) dθ dφ = 2π/3
    # Cone: ∫₀^{2π} ∫₀^α cos²(θ) sin(θ) dθ dφ = 2π(1 - cos³(α))/3
    # LEE(α) = 1 - cos³(α)
    n_theta = 200
    n_phi = 200
    theta = collect(range(0, π / 2, length = n_theta))
    phi = collect(range(0, 2π, length = n_phi + 1))[1:end-1]

    power = [cos(θ)^2 for θ in theta, _ in phi]

    for alpha_deg in [15.0, 30.0, 45.0, 60.0]
        alpha = deg2rad(alpha_deg)
        lee = Khronos.compute_LEE(power, theta, phi; cone_half_angle = alpha)
        expected = 1 - cos(alpha)^3
        @test lee ≈ expected rtol = 0.02
    end
end

@testset "LEE boundary conditions" begin
    n_theta = 100
    n_phi = 100
    theta = collect(range(0, π / 2, length = n_theta))
    phi = collect(range(0, 2π, length = n_phi + 1))[1:end-1]

    power = ones(n_theta, length(phi))

    # Full hemisphere → LEE = 1
    @test Khronos.compute_LEE(power, theta, phi; cone_half_angle = π / 2) ≈ 1.0 rtol = 0.01

    # Very small cone → LEE ≈ 0
    lee_tiny = Khronos.compute_LEE(power, theta, phi; cone_half_angle = deg2rad(0.5))
    @test lee_tiny < 0.01
end

@testset "Incoherent power summation preserves LEE" begin
    # Verify that summing power from multiple identical simulations
    # gives the same LEE as a single simulation (LEE is a ratio).
    n_theta = 100
    n_phi = 100
    theta = collect(range(0, π / 2, length = n_theta))
    phi = collect(range(0, 2π, length = n_phi + 1))[1:end-1]

    # Single simulation with cos²(θ) pattern
    power_single = [cos(θ)^2 for θ in theta, _ in phi]
    lee_single = Khronos.compute_LEE(power_single, theta, phi; cone_half_angle = deg2rad(30.0))

    # Sum of 5 identical simulations
    power_sum = 5.0 .* power_single
    lee_sum = Khronos.compute_LEE(power_sum, theta, phi; cone_half_angle = deg2rad(30.0))

    # LEE should be identical (it's a ratio)
    @test lee_single ≈ lee_sum rtol = 1e-10
end
