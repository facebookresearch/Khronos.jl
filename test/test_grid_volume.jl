# Copyright (c) Meta Platforms, Inc. and affiliates.

import Khronos
using Test



@testset "2D GridVolumes" begin
    #TODO
end

@testset "3D GridVolumes" begin
    sim = Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 20,
        sources = [],
    )

    # --------------------------------------------------- #
    # point gv
    # --------------------------------------------------- #
    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0]),
        Khronos.Ex(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 1, 1])
    @test all(gv.start_idx .≈ [40, 41, 41])
    @test all(gv.end_idx .≈ [41, 41, 41])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0]),
        Khronos.Ey(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 2, 1])
    @test all(gv.start_idx .≈ [41, 40, 41])
    @test all(gv.end_idx .≈ [41, 41, 41])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0]),
        Khronos.Ez(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 1, 2])
    @test all(gv.start_idx .≈ [41, 41, 40])
    @test all(gv.end_idx .≈ [41, 41, 41])

    # --------------------------------------------------- #
    # line gv
    # --------------------------------------------------- #
    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, 0.0]),
        Khronos.Ex(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 81, 1])
    @test all(gv.start_idx .≈ [40, 1, 41])
    @test all(gv.end_idx .≈ [41, 81, 41])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, 0.0]),
        Khronos.Ey(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 80, 1])
    @test all(gv.start_idx .≈ [41, 1, 41])
    @test all(gv.end_idx .≈ [41, 80, 41])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, 0.0]),
        Khronos.Ez(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 81, 2])
    @test all(gv.start_idx .≈ [41, 1, 40])
    @test all(gv.end_idx .≈ [41, 81, 41])

    # --------------------------------------------------- #
    # plane gv
    # --------------------------------------------------- #

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, Inf]),
        Khronos.Ex(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 81, 81])
    @test all(gv.start_idx .≈ [40, 1, 1])
    @test all(gv.end_idx .≈ [41, 81, 81])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, Inf]),
        Khronos.Ey(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 80, 81])
    @test all(gv.start_idx .≈ [41, 1, 1])
    @test all(gv.end_idx .≈ [41, 80, 81])

    gv = Khronos.GridVolume(
        sim,
        Khronos.Volume(center = [0.0, 0.0, 0.0], size = [0.0, Inf, Inf]),
        Khronos.Ez(),
    )

    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 81, 80])
    @test all(gv.start_idx .≈ [41, 1, 1])
    @test all(gv.end_idx .≈ [41, 81, 80])

end
