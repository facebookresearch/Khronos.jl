# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for near-to-far field transformation.

using Test
using LinearAlgebra
using StaticArrays

# Test the Green's function directly (standalone, no simulation needed)
include("../src/Khronos.jl")
using .Khronos

@testset "Green's function 3D" begin
    # Test that green3d! produces non-zero output for a basic case
    EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    x = SVector(1.0, 0.0, 0.0)
    x0 = SVector(0.0, 0.0, 0.0)
    freq = 1.0
    eps = 1.0
    mu = 1.0
    f0 = ComplexF64(1.0)

    Khronos.green3d!(EH, x, freq, eps, mu, x0, 1, f0)  # Jx source
    @test any(abs.(EH) .> 0)

    # Symmetry: Jx source at origin, observed on +x axis
    # By symmetry, Ey and Ez should be zero (source is along x, observation along x)
    # Actually for a Jx dipole observed along x̂: E is perpendicular to r̂
    # E should be zero along r̂ (no radial E in far field)
    # But in near field there IS a radial component
end

@testset "Green's function reciprocity" begin
    # Green's function should satisfy reciprocity: G(x1, x0) = G(x0, x1)^T
    # (up to phase/sign conventions)
    eps = 1.0
    mu = 1.0
    freq = 1.0

    x1 = SVector(2.0, 1.0, 0.5)
    x0 = SVector(0.0, 0.0, 0.0)

    # Forward: Jx at x0 → field at x1
    EH_fwd = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    Khronos.green3d!(EH_fwd, x1, freq, eps, mu, x0, 1, ComplexF64(1.0))

    # Backward: Jx at x1 → field at x0
    EH_bwd = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    Khronos.green3d!(EH_bwd, x0, freq, eps, mu, x1, 1, ComplexF64(1.0))

    # The Ex component should be the same (reciprocity for same component)
    @test EH_fwd[1] ≈ EH_bwd[1] rtol=1e-10
end

@testset "Far-field power computation" begin
    # Create synthetic far-field data: uniform power over upper hemisphere
    n_theta = 10
    n_phi = 20
    theta = range(0, π/2, length=n_theta) |> collect
    phi = range(0, 2π, length=n_phi+1)[1:end-1] |> collect

    n_obs = n_theta * n_phi
    EH = zeros(ComplexF64, n_obs, 6, 1)

    # Set uniform E_θ = 1 everywhere
    Z = 1.0  # free space
    for i in 1:n_obs
        EH[i, 1, 1] = 1.0  # Ex
    end

    power = compute_far_field_power(EH, theta, phi)
    @test size(power) == (n_theta, n_phi)
    @test all(power .≥ 0)  # power should be non-negative
end

@testset "LEE computation" begin
    # Uniform power distribution: LEE should equal the cone fraction
    n_theta = 100
    n_phi = 100
    theta = range(0, π/2, length=n_theta) |> collect
    phi = range(0, 2π, length=n_phi+1)[1:end-1] |> collect

    power = ones(n_theta, n_phi)

    # For uniform power: LEE(α) = (1-cos(α))/(1-cos(π/2)) = (1-cos(α))/1
    alpha = deg2rad(30.0)
    lee = compute_LEE(power, theta, phi; cone_half_angle=alpha)

    expected_lee = (1 - cos(alpha)) / (1 - cos(π/2))
    @test lee ≈ expected_lee rtol=0.05  # within 5% (trapezoidal integration)

    # Full hemisphere should give LEE = 1
    lee_full = compute_LEE(power, theta, phi; cone_half_angle=π/2)
    @test lee_full ≈ 1.0 rtol=0.01
end

@testset "Near2FarMonitor construction" begin
    # Test that a Near2FarMonitor can be constructed
    n2f = Near2FarMonitor(
        center = [0.0, 0.0, 1.0],
        size = [2.0, 2.0, 0.0],  # z-normal surface
        frequencies = [1.0],
        theta = collect(range(0, π/2, length=10)),
        phi = collect(range(0, 2π, length=20)),
        normal_dir = :+,
    )

    @test n2f.center == [0.0, 0.0, 1.0]
    @test n2f.size == [2.0, 2.0, 0.0]
    @test length(n2f.frequencies) == 1
    @test length(n2f.theta) == 10
    @test length(n2f.phi) == 20
end
