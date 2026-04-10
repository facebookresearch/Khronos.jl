# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tests for ModeMonitor: mode overlap integral and S-parameter extraction.

import Khronos
using Test
using GeometryPrimitives

@testset "ModeMonitor data structures" begin
    # Test ModeSpec construction with defaults
    spec = Khronos.ModeSpec()
    @test spec.num_modes == 1
    @test spec.target_neff == 0.0
    @test spec.mode_solver_resolution == 50
    @test spec.solver_tolerance == 1e-6
    @test isempty(spec.geometry)

    # Test ModeMonitor construction
    monitor = Khronos.ModeMonitor(
        center = [0.0, 0.0, 0.0],
        size = [0.0, 2.0, 2.0],
        frequencies = [0.5, 0.6, 0.7],
    )
    @test monitor.center == [0.0, 0.0, 0.0]
    @test monitor.size == [0.0, 2.0, 2.0]
    @test length(monitor.frequencies) == 3
    @test monitor.decimation == 1
    @test isnothing(monitor.monitor_data)
end

@testset "ModeMonitor initialization" begin
    # Set up a simple waveguide simulation with a ModeMonitor
    geometry = [
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 0.5, 0.22]),
            Khronos.Material(ε = 3.4^2),
        ),
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            Khronos.Material(ε = 1.44^2),
        ),
    ]

    freq = 1.0 / 1.55
    mode_monitor = Khronos.ModeMonitor(
        center = [1.0, 0.0, 0.0],
        size = [0.0, 2.0, 2.0],
        frequencies = [freq],
        mode_spec = Khronos.ModeSpec(
            num_modes = 1,
            geometry = geometry,
            mode_solver_resolution = 30,
        ),
    )

    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        geometry = geometry,
        sources = [
            Khronos.ModeSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = freq),
                frequency = freq,
                mode_solver_resolution = 30,
                mode_index = 1,
                center = [-1.0, 0.0, 0.0],
                size = [0.0, 2.0, 2.0],
                solver_tolerance = 1e-6,
                geometry = geometry,
            ),
        ],
        monitors = Khronos.Monitor[mode_monitor],
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )

    Khronos.prepare_simulation!(sim)

    # Check that the ModeMonitor was initialized properly
    @test !isnothing(mode_monitor.monitor_data)
    md = mode_monitor.monitor_data
    @test md isa Khronos.ModeMonitorData
    @test md.normal_axis == 1  # x-normal since size[1] == 0
    @test length(md.tangential_E_monitors) == 2
    @test length(md.tangential_H_monitors) == 2
    @test length(md.mode_profiles) == 1
    @test length(md.frequencies) == 1

    # Check that mode profile was solved
    mode = md.mode_profiles[1]
    @test mode.neff > 1.0  # effective index should be > 1 for guided mode

    # Check base positions were computed
    @test length(md.e1_base) == 3
    @test length(md.e2_base) == 3
    @test length(md.h1_base) == 3
    @test length(md.h2_base) == 3
end

@testset "ModeMonitor mode field helpers" begin
    import VectorModesolver

    # Create a dummy mode
    nx, ny = 10, 10
    x = collect(range(-1.0, 1.0, length=nx))
    y = collect(range(-1.0, 1.0, length=ny))
    mode = VectorModesolver.Mode(
        λ = 1.55,
        neff = 2.5,
        x = x,
        y = y,
        Ex = ones(ComplexF64, nx, ny),
        Ey = 2 * ones(ComplexF64, nx, ny),
        Ez = 3 * ones(ComplexF64, nx, ny),
        Hx = 4 * ones(ComplexF64, nx, ny),
        Hy = 5 * ones(ComplexF64, nx, ny),
        Hz = 6 * ones(ComplexF64, nx, ny),
    )

    # Test tangential field extraction for x-normal
    @test Khronos._get_mode_tangential_field(mode, 1, :e1) ≈ 2 * ones(nx, ny)  # Ey
    @test Khronos._get_mode_tangential_field(mode, 1, :e2) ≈ 3 * ones(nx, ny)  # Ez
    @test Khronos._get_mode_tangential_field(mode, 1, :h1) ≈ 5 * ones(nx, ny)  # Hy
    @test Khronos._get_mode_tangential_field(mode, 1, :h2) ≈ 6 * ones(nx, ny)  # Hz

    # Test for y-normal
    @test Khronos._get_mode_tangential_field(mode, 2, :e1) ≈ 1 * ones(nx, ny)  # Ex
    @test Khronos._get_mode_tangential_field(mode, 2, :e2) ≈ 3 * ones(nx, ny)  # Ez
    @test Khronos._get_mode_tangential_field(mode, 2, :h1) ≈ 4 * ones(nx, ny)  # Hx
    @test Khronos._get_mode_tangential_field(mode, 2, :h2) ≈ 6 * ones(nx, ny)  # Hz

    # Test for z-normal
    @test Khronos._get_mode_tangential_field(mode, 3, :e1) ≈ 1 * ones(nx, ny)  # Ex
    @test Khronos._get_mode_tangential_field(mode, 3, :e2) ≈ 2 * ones(nx, ny)  # Ey
    @test Khronos._get_mode_tangential_field(mode, 3, :h1) ≈ 4 * ones(nx, ny)  # Hx
    @test Khronos._get_mode_tangential_field(mode, 3, :h2) ≈ 5 * ones(nx, ny)  # Hy
end

@testset "ModeMonitor squeeze helper" begin
    # 3D array, z-normal → collapse dim 3, then transpose
    f3d_z = rand(ComplexF64, 5, 7, 1)
    result_z = Khronos._squeeze_mode_field(f3d_z, 3)
    @test size(result_z) == (7, 5)  # transposed from (5, 7) to (7, 5)

    # 3D array, x-normal → collapse dim 1, then transpose
    f3d_x = rand(ComplexF64, 1, 5, 7)
    result_x = Khronos._squeeze_mode_field(f3d_x, 1)
    @test size(result_x) == (7, 5)

    # 3D array, y-normal → collapse dim 2, then transpose
    f3d_y = rand(ComplexF64, 5, 1, 7)
    result_y = Khronos._squeeze_mode_field(f3d_y, 2)
    @test size(result_y) == (7, 5)

    # 2D array → transposed
    f2d = rand(ComplexF64, 5, 7)
    result_2d = Khronos._squeeze_mode_field(f2d, 3)
    @test size(result_2d) == (7, 5)

    # 1D array → reshaped to (1, N) row vector
    f1d = rand(ComplexF64, 5)
    result_1d = Khronos._squeeze_mode_field(f1d, 3)
    @test size(result_1d) == (1, 5)
end

@testset "ModeMonitor DFT field extraction" begin
    # Create dummy 4D DFT field arrays (Nx, Ny, Nz, Nfreq)
    n1, n2, nz, nf = 3, 4, 1, 2
    e1 = rand(ComplexF64, n1, n2, nz, nf)
    e2 = rand(ComplexF64, n1, n2, nz, nf)
    h1 = rand(ComplexF64, n1, n2, nz, nf)
    h2 = rand(ComplexF64, n1, n2, nz, nf)

    # z-normal extraction
    et1, et2, ht1, ht2 = Khronos._extract_dft_fields(e1, e2, h1, h2, 3, 2, 3, 1)
    @test et1 ≈ ComplexF64(e1[2, 3, 1, 1])
    @test et2 ≈ ComplexF64(e2[2, 3, 1, 1])
    @test ht1 ≈ ComplexF64(h1[2, 3, 1, 1])
    @test ht2 ≈ ComplexF64(h2[2, 3, 1, 1])

    # x-normal extraction
    e1x = rand(ComplexF64, nz, n1, n2, nf)
    e2x = rand(ComplexF64, nz, n1, n2, nf)
    h1x = rand(ComplexF64, nz, n1, n2, nf)
    h2x = rand(ComplexF64, nz, n1, n2, nf)
    et1, et2, ht1, ht2 = Khronos._extract_dft_fields(e1x, e2x, h1x, h2x, 1, 2, 3, 1)
    @test et1 ≈ ComplexF64(e1x[1, 2, 3, 1])
end
