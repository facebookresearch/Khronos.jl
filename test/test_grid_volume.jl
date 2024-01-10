# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using Test



@testset "2D GridVolumes" begin
    fdtd.environment!(false, Float64, 2)

end

@testset "3D GridVolumes" begin
    fdtd.environment!(false, Float64, 3)
    sim = fdtd.Simulation{fdtd.Data.Array}(
        cell_size=[4.0, 4.0, 4.0],
        cell_center=[0.0, 0.0, 0.0],
        resolution=20,
        sources=[]
    )
    # --------------------------------------------------- #
    # point gv
    # --------------------------------------------------- #
    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , 0. , 0. ]), fdtd.Ex())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 1, 1])
    @test all(gv.start_idx .≈ [40, 41, 41])
    @test all(gv.end_idx .≈ [41, 41, 41])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , 0. , 0. ]), fdtd.Ey())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 2, 1])
    @test all(gv.start_idx .≈ [41, 40, 41])
    @test all(gv.end_idx .≈ [41, 41, 41])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , 0. , 0. ]), fdtd.Ez())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 1, 2])
    @test all(gv.start_idx .≈ [41, 41, 40])
    @test all(gv.end_idx .≈ [41, 41, 41])

    # --------------------------------------------------- #
    # line gv
    # --------------------------------------------------- #
    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , 0. ]), fdtd.Ex())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 81, 1])
    @test all(gv.start_idx .≈ [40, 1, 41])
    @test all(gv.end_idx .≈ [41, 81, 41])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , 0. ]), fdtd.Ey())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 80, 1])
    @test all(gv.start_idx .≈ [41, 1, 41])
    @test all(gv.end_idx .≈ [41, 80, 41])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , 0. ]), fdtd.Ez())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 81, 2])
    @test all(gv.start_idx .≈ [41, 1, 40])
    @test all(gv.end_idx .≈ [41, 81, 41])

    # --------------------------------------------------- #
    # plane gv
    # --------------------------------------------------- #

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , Inf ]), fdtd.Ex())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [2, 81, 81])
    @test all(gv.start_idx .≈ [40, 1, 1])
    @test all(gv.end_idx .≈ [41, 81, 81])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , Inf ]), fdtd.Ey())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 80, 81])
    @test all(gv.start_idx .≈ [41, 1, 1])
    @test all(gv.end_idx .≈ [41, 80, 81])

    gv = fdtd.GridVolume(sim, fdtd.Volume(center=[0. , 0. , 0.],size=[0. , Inf , Inf ]), fdtd.Ez())
    
    @test all([gv.Nx, gv.Ny, gv.Nz] .≈ [1, 81, 80])
    @test all(gv.start_idx .≈ [41, 1, 1])
    @test all(gv.end_idx .≈ [41, 81, 80])

end