# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# test the Volume() and GridVolume() functionality.

import Khronos
using Test

sim = Khronos.Simulation(
    cell_size = [10.0, 10.0, 10.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 1,
    sources = [],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
)

@testset "Simulation-sized GridVolumes" begin
    gv_sim = Khronos.GridVolume(sim)
    @test gv_sim.Nx == sim.Nx
    @test gv_sim.Ny == sim.Ny
    @test gv_sim.Nz == sim.Nz

    # Here we test that the size of each of the field-compoenent arrays are
    # initialized properly. They all follow a simply pattern. Namely, if we are
    # looking at E fields, then the current component has the same axis size as
    # the total number of grid cells in that dimension. All other axes are
    # incremented by one. The H field is simply the opposite of this. (We could
    # do similar analysis with the B and D fields).
    gv_sim_ex = Khronos.GridVolume(sim, Khronos.Ex())
    @test gv_sim_ex.Nx == sim.Nx
    @test gv_sim_ex.Ny == sim.Ny + 1
    @test gv_sim_ex.Nz == sim.Nz + 1

    gv_sim_ey = Khronos.GridVolume(sim, Khronos.Ey())
    @test gv_sim_ey.Nx == sim.Nx + 1
    @test gv_sim_ey.Ny == sim.Ny
    @test gv_sim_ey.Nz == sim.Nz + 1

    gv_sim_ez = Khronos.GridVolume(sim, Khronos.Ez())
    @test gv_sim_ez.Nx == sim.Nx + 1
    @test gv_sim_ez.Ny == sim.Ny + 1
    @test gv_sim_ez.Nz == sim.Nz

    gv_sim_hx = Khronos.GridVolume(sim, Khronos.Hx())
    @test gv_sim_hx.Nx == sim.Nx + 1
    @test gv_sim_hx.Ny == sim.Ny
    @test gv_sim_hx.Nz == sim.Nz

    gv_sim_hy = Khronos.GridVolume(sim, Khronos.Hy())
    @test gv_sim_hy.Nx == sim.Nx
    @test gv_sim_hy.Ny == sim.Ny + 1
    @test gv_sim_hy.Nz == sim.Nz

    gv_sim_hz = Khronos.GridVolume(sim, Khronos.Hz())
    @test gv_sim_hz.Nx == sim.Nx
    @test gv_sim_hz.Ny == sim.Ny
    @test gv_sim_hz.Nz == sim.Nz + 1
end
