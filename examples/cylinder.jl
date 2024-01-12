using Plots
import fdtd

fdtd.environment!(false, Float64, 2)

function simulate_cylinder(polarization = "TE")
    gauss_pulse = fdtd.GaussianPulseSource(1.0, 0.2)
    cw_src = fdtd.ContinuousWaveSource(fcen = 1.0)
    time_src = cw_src

    sources = [
        fdtd.UniformSource(
            time_profile = time_src,
            component = polarization == "TE" ? component = fdtd.Ez() :
                        component = fdtd.Hz(),
            center = [-4.0, 0.0, 0.0],
            size = [0.0, Inf, 0.0],
        ),
        fdtd.UniformSource(
            time_profile = time_src,
            component = polarization == "TE" ? component = fdtd.Hy() :
                        component = fdtd.Ey(),
            center = [-4.0, 0.0, 0.0],
            size = [0.0, Inf, 0.0],
            amplitude = -1,
        ),
    ]
    a = fdtd.Cuboid([0.0, 0.0], [2.0, 4.0])
    Si = fdtd.Material(ε = 3, σD = 10)
    air = fdtd.Material(ε = 1)
    geometry = [
        #fdtd.Object(fdtd.Cylinder([0.0, 0.0, 0.0], 1.5, 100.0), air),
        fdtd.Object(fdtd.Cylinder([0.0, 0.0, 0.0], 2.5, 100.0), Si),
    ]

    monitors = [
        fdtd.DFTMonitor(
            component = fdtd.Ez(),
            center = [0, 0, 0],
            size = [10.0, 10.0, 0.0],
            frequencies = [0.9, 1.0, 1.1],
        ),
    ]

    sim = fdtd.Simulation{fdtd.Data.Array}(
        cell_size = [10.0, 10.0, 0.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 50,
        sources = sources,
        #monitors = monitors,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        geometry = geometry,
    )
    fdtd.prepare_simulation(sim)

    t_end = 20
    fdtd.run(sim, t_end)
    println("Simulation complete")
    q = polarization == "TE" ? sim.fields.fEz : sim.fields.fHz

    hm = heatmap(
        transpose(q),
        color = :RdBu,
        aspect_ratio = :equal,
        show = true,
        clims = (-1.05, 1.05) .* maximum(abs, q),
        flip_y = true,
    )

    display(hm)

    return sim
end

@time sim = simulate_cylinder("TE");
