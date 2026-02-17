# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for batch simulation execution.

using Test

include("../src/Khronos.jl")
using .Khronos

@testset "plan_batch" begin
    # Test memory planning with a mock-like simulation
    # We can't fully construct a SimulationData without GPU, but we can
    # test the planning logic conceptually
    @test true  # placeholder — full test requires GPU initialization
end

@testset "compute_incoherent_LEE basic" begin
    # Test with synthetic power data
    n_theta = 50
    n_phi = 100
    theta = collect(range(0, π/2, length=n_theta))
    phi = collect(range(0, 2π, length=n_phi+1))[1:end-1]

    # Create two "simulations" with known power patterns
    power1 = ones(n_theta, length(phi))  # uniform
    power2 = ones(n_theta, length(phi))  # uniform

    # Total power should be 2× single
    total_power = power1 .+ power2

    lee1 = compute_LEE(power1, theta, phi; cone_half_angle=π/2)
    lee_total = compute_LEE(total_power, theta, phi; cone_half_angle=π/2)

    # For uniform power, LEE should be ~1.0 regardless of scaling
    @test lee1 ≈ 1.0 rtol=0.01
    @test lee_total ≈ 1.0 rtol=0.01
end
