import fdtd
using CairoMakie

fdtd.choose_backend(fdtd.CPUDevice(), Float32)

sources = [
    fdtd.UniformSource(
        time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
        component = fdtd.Ez(),
        center = [0.0, 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
    ),
]

sim = fdtd.Simulation(
    cell_size = [10.0, 10.0, 10.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 20,
    sources = sources,
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
)

t_end = 10.0;
fdtd.run(sim, until = t_end)

scene = fdtd.plot2D(sim, fdtd.Ez(), fdtd.Volume([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]))
save("dipole.png", scene)
