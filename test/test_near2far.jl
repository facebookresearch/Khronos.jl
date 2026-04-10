# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for near-to-far field transformation.

import Khronos
using Test
using LinearAlgebra
using StaticArrays
using Statistics

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
end

@testset "Green's function reciprocity" begin
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

@testset "Green's function far-field 1/r decay" begin
    # In the far field, E should decay as 1/r.
    # Evaluate at two large distances and verify scaling.
    eps = 1.0
    mu = 1.0
    freq = 1.0
    x0 = SVector(0.0, 0.0, 0.0)

    r1 = 1e4
    r2 = 2e4
    dir = SVector(1.0, 1.0, 1.0) / sqrt(3.0)

    EH1 = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    EH2 = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    Khronos.green3d!(EH1, r1 * dir, freq, eps, mu, x0, 3, ComplexF64(1.0))  # Jz
    Khronos.green3d!(EH2, r2 * dir, freq, eps, mu, x0, 3, ComplexF64(1.0))  # Jz

    # |E(r2)| / |E(r1)| should be r1/r2 = 0.5 in far field
    E_mag1 = sqrt(sum(abs2, EH1[1:3]))
    E_mag2 = sqrt(sum(abs2, EH2[1:3]))
    ratio = E_mag2 / E_mag1
    @test ratio ≈ r1 / r2 rtol=1e-3
end

@testset "Near2far analytical roundtrip (Jz dipole → sin²θ)" begin
    # Pure analytical test: no FDTD, no GPU.
    # Generate near-field from a Jz dipole at origin via green3d! on a z-plane,
    # then project to far field via equivalent currents + green3d!, and verify
    # that the phi-averaged power follows sin²(θ) within the forward cone.
    #
    # A z-oriented electric dipole radiates P(θ) ∝ sin²(θ).
    # This test validates the Green's function + surface integration pipeline.
    #
    # Note: A single planar surface cannot capture radiation at high angles
    # (θ near 90°) because those rays travel nearly parallel to the surface.
    # We restrict validation to θ ≤ 45° where the single-plane approach is accurate.
    freq = 1.0
    eps_m = 1.0
    mu_m = 1.0
    z0 = 2.0      # near-field surface height (2λ from source reduces evanescent contamination)
    r_obs = 1e6   # far-field observation distance

    # Near-field sampling grid (20λ surface, sampled at ~4 pts/λ)
    n_grid = 81
    L_ext = 20.0
    x_coords = collect(range(-L_ext / 2, L_ext / 2, length = n_grid))
    y_coords = collect(range(-L_ext / 2, L_ext / 2, length = n_grid))
    dx = x_coords[2] - x_coords[1]
    dy = y_coords[2] - y_coords[1]
    dA = dx * dy

    # Step 1: Compute exact near-field from Jz at origin using green3d!
    Ex_nf = zeros(ComplexF64, n_grid, n_grid)
    Ey_nf = zeros(ComplexF64, n_grid, n_grid)
    Hx_nf = zeros(ComplexF64, n_grid, n_grid)
    Hy_nf = zeros(ComplexF64, n_grid, n_grid)
    x0_src = SVector(0.0, 0.0, 0.0)

    for i2 in 1:n_grid, i1 in 1:n_grid
        x = SVector(x_coords[i1], y_coords[i2], z0)
        EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
        Khronos.green3d!(EH, x, freq, eps_m, mu_m, x0_src, 3, ComplexF64(1.0))
        Ex_nf[i1, i2] = EH[1]
        Ey_nf[i1, i2] = EH[2]
        Hx_nf[i1, i2] = EH[4]
        Hy_nf[i1, i2] = EH[5]
    end

    # Verify near-field is non-trivial
    @test maximum(abs.(Ex_nf)) > 0
    @test maximum(abs.(Hy_nf)) > 0

    # Step 2: Project to far field via equivalent currents
    # For a z-normal surface with outward normal n̂ = +ẑ:
    #   Jx = +Hy, Jy = -Hx  (J = n̂ × H)
    #   Mx = -Ey, My = +Ex   (M = -n̂ × E)
    # Focus on θ ≤ 45° where single-plane near2far is valid
    theta_arr = collect(range(5.0, 45.0, length = 9) .* (π / 180))
    phi_arr = collect(range(0.0, 2π - 2π / 12, length = 12))
    n_obs = length(theta_arr) * length(phi_arr)
    EH_out = zeros(ComplexF64, n_obs, 6, 1)
    ns = 1.0  # normal sign

    obs_idx = 1
    for φ in phi_arr
        for θ in theta_arr
            x_obs = SVector(
                r_obs * sin(θ) * cos(φ),
                r_obs * sin(θ) * sin(φ),
                r_obs * cos(θ),
            )
            EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)

            for i2 in 1:n_grid, i1 in 1:n_grid
                x0_surf = SVector(x_coords[i1], y_coords[i2], z0)

                ex = Ex_nf[i1, i2]
                ey = Ey_nf[i1, i2]
                hx = Hx_nf[i1, i2]
                hy = Hy_nf[i1, i2]

                # Equivalent currents → green3d! contributions
                Khronos.green3d!(EH, x_obs, freq, eps_m, mu_m, x0_surf, 1, ns * hy * dA)   # Jx
                Khronos.green3d!(EH, x_obs, freq, eps_m, mu_m, x0_surf, 2, -ns * hx * dA)  # Jy
                Khronos.green3d!(EH, x_obs, freq, eps_m, mu_m, x0_surf, 4, -ns * ey * dA)  # Mx
                Khronos.green3d!(EH, x_obs, freq, eps_m, mu_m, x0_surf, 5, ns * ex * dA)   # My
            end

            EH_out[obs_idx, :, 1] .= EH
            obs_idx += 1
        end
    end

    # Step 3: Compute power and compare with sin²(θ) in the forward cone
    power = Khronos.compute_far_field_power(EH_out, theta_arr, phi_arr)
    power_avg = vec(mean(power, dims = 2))  # average over phi
    pmax = maximum(power_avg)
    power_norm = power_avg ./ pmax
    sin2_theta = sin.(theta_arr) .^ 2
    sin2_norm = sin2_theta ./ maximum(sin2_theta)

    # Within the forward cone (θ ≤ 45°), the pattern should track sin²θ
    rms_err = sqrt(mean((power_norm .- sin2_norm) .^ 2))
    @test rms_err < 0.15  # RMS error below 15% for finite-aperture test
end

@testset "Far-field power computation" begin
    # Create synthetic far-field data and verify power is non-negative
    n_theta = 10
    n_phi = 20
    theta = range(0, π / 2, length = n_theta) |> collect
    phi = range(0, 2π, length = n_phi + 1)[1:end-1] |> collect

    n_obs = n_theta * n_phi
    EH = zeros(ComplexF64, n_obs, 6, 1)

    # Set uniform Ex = 1 everywhere
    for i in 1:n_obs
        EH[i, 1, 1] = 1.0
    end

    power = Khronos.compute_far_field_power(EH, theta, phi)
    @test size(power) == (n_theta, n_phi)
    @test all(power .≥ 0)
end

@testset "LEE computation" begin
    # Uniform power distribution: LEE should equal the solid angle fraction
    n_theta = 100
    n_phi = 100
    theta = range(0, π / 2, length = n_theta) |> collect
    phi = range(0, 2π, length = n_phi + 1)[1:end-1] |> collect

    power = ones(n_theta, length(phi))

    # For uniform power: LEE(α) = (1 - cos(α)) / (1 - cos(π/2)) = 1 - cos(α)
    alpha = deg2rad(30.0)
    lee = Khronos.compute_LEE(power, theta, phi; cone_half_angle = alpha)

    expected_lee = (1 - cos(alpha)) / (1 - cos(π / 2))
    @test lee ≈ expected_lee rtol=0.05

    # Full hemisphere should give LEE = 1
    lee_full = Khronos.compute_LEE(power, theta, phi; cone_half_angle = π / 2)
    @test lee_full ≈ 1.0 rtol=0.01
end

@testset "Near2FarMonitor construction" begin
    n2f = Khronos.Near2FarMonitor(
        center = [0.0, 0.0, 1.0],
        size = [2.0, 2.0, 0.0],
        frequencies = [1.0],
        theta = collect(range(0, π / 2, length = 10)),
        phi = collect(range(0, 2π, length = 20)),
        normal_dir = :+,
    )

    @test n2f.center == [0.0, 0.0, 1.0]
    @test n2f.size == [2.0, 2.0, 0.0]
    @test length(n2f.frequencies) == 1
    @test length(n2f.theta) == 10
    @test length(n2f.phi) == 20
end
