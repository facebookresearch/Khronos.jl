# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using Plots
import fdtd

fdtd.environment!(false, Float64, 2)

function simulate_pulse()
    t_end = 40.0

    time_src = fdtd.GaussianPulseSource(1.0,0.2)

    sources = [
        fdtd.UniformSource(
            time_profile=time_src,
            component=fdtd.Ez(),
            center=[-4.0, 0.0, 0.0],
            size=[0.0, Inf, 0.0]
        ),
        fdtd.UniformSource(
            time_profile=time_src,
            component=fdtd.Hy(),
            center=[-4.0, 0.0, 0.0],
            size=[0.0, Inf, 0.0],
            amplitude=-1),
    ]

    monitors = [
        fdtd.TimeMonitor(component=fdtd.Ez(),center=[0,0,0],size=[0,0,0],length=t_end)
    ]

    sim = fdtd.Simulation{fdtd.Data.Array}(
        cell_size=[10.0, 10.0, 0.0],
        cell_center=[0.0, 0.0, 0.0],
        resolution=50,
        sources=sources,
        monitors=monitors,
        boundaries=[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
    )
    fdtd.prepare_simulation(sim)

    fdtd.run(sim, t_end)
    println("Simulation complete")

    q = sim.fields.fEz

    hm = heatmap(transpose(q), color=:RdBu, aspect_ratio=:equal, show=true,
        clims=(-1.05, 1.05) .* maximum(abs, q), flip_y=true)

    display(hm)
    return sim
end

@time sim = simulate_pulse()

