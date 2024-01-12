# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# test the Volume() and GridVolume() functionality.

import fdtd
using Test

const USE_GPU = false;
fdtd.environment!(USE_GPU, Float64, 2)

sim = fdtd.Simulation{fdtd.Data.Array}(
    cell_size = [10.0, 10.0, 0.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 1,
    sources = [],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
)

@testset "Simulation-sized GridVolumes" begin
    gv_sim = fdtd.GridVolume(sim)
    @test gv_sim.Nx == sim.Nx
    @test gv_sim.Ny == sim.Ny

    gv_sim_ex = fdtd.GridVolume(sim, fdtd.Ex())
    @test gv_sim_ex.Nx == sim.Nx
    @test gv_sim_ex.Ny == sim.Ny + 1

    gv_sim_ey = fdtd.GridVolume(sim, fdtd.Ey())
    @test gv_sim_ey.Nx == sim.Nx + 1
    @test gv_sim_ey.Ny == sim.Ny

    gv_sim_ez = fdtd.GridVolume(sim, fdtd.Ez())
    @test gv_sim_ez.Nx == sim.Nx + 1
    @test gv_sim_ez.Ny == sim.Ny + 1

    gv_sim_hx = fdtd.GridVolume(sim, fdtd.Hx())
    @test gv_sim_hx.Nx == sim.Nx + 1
    @test gv_sim_hx.Ny == sim.Ny

    gv_sim_hy = fdtd.GridVolume(sim, fdtd.Hy())
    @test gv_sim_hy.Nx == sim.Nx
    @test gv_sim_hy.Ny == sim.Ny + 1

    gv_sim_hz = fdtd.GridVolume(sim, fdtd.Hz())
    @test gv_sim_hz.Nx == sim.Nx
    @test gv_sim_hz.Ny == sim.Ny
end
