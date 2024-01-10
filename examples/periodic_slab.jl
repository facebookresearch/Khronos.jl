# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)

"""
    slab_analytical()

Computes transmision as a function of slab thickness (d), refractive index (n), and wavelength (λ).
"""
function slab_analytical(d, n, λ)
    rho = (n-1) ./ (n+1)
    t = (
        ((1 + rho) * (1 - rho) * exp.( -2 * im * π * n * d ./ λ)) ./ 
        (1 .- rho^2 * exp.(-4 * im * π * n * d ./ λ))
        )
    return abs.(t).^2
end

"""
    build_simulation(d, n, λ)

Builds simulation to compute transmision as a function of slab thickness (d), refractive index (n), and wavelength (λ).
"""
function build_simulation(d, n, λ, resolution; plot_geometry = false)

    dpml = 0.5
    monitor_xy = 0.5
    monitor_height = 0.5

    sxy = 2 * dpml + monitor_xy
    sz = 2 * dpml + 4 * monitor_height
    cell_size = [sxy, sxy, sz]
    
    boundary_layers = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

    sources = [
        fdtd.PlaneWaveSource(
            time_profile = fdtd.GaussianPulseSource(
                fcen = 1 / 1.5, fwidth = 0.4),
            center = [0, 0, -monitor_height],
            size = [Inf, Inf, 0.],
            polarization = [1.0, 0.0, 0.0],
            k_vector = [0.0, 0.0, 1.0]
        )
    ]

    geometry = []

    if plot_geometry
        geometry = [
            fdtd.Object(
            Cuboid([0,0,0], [sxy,sxy,d]), 
            fdtd.Material(ε = n^2)
        )]
    end

    monitors = [
        fdtd.DFTMonitor(
            component = fdtd.Ex(),
            center = [0, 0, monitor_height],
            size = [monitor_xy, monitor_xy, 0],
            frequencies = 1 ./ λ
        )
    ]
    
    sim = fdtd.Simulation(
        cell_size = cell_size,
        cell_center = [0.0,0.0,0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = boundary_layers,
        monitors = monitors
    )

    return sim

end

function run_simulation!(sim::fdtd.SimulationData)
    fdtd.run(
        sim,
        until_after_sources = fdtd.stop_when_dft_decayed(
            tolerance = 1e-11,
            minimum_runtime = 0.0,
            maximum_runtime = 300.0,
            )
    ) 
end

function compute_power(sim::fdtd.SimulationData)
    return sum(abs.(sim.monitors[1].monitor_data.fields).^2, dims=[1,2])[1,1,1,:]
end

function main()
    d = 0.5
    n = 3.5
    λ = range(1.0,2.0,length=100)
    T = slab_analytical(d, n, λ)

    resolution = 20
    sim = build_simulation(d, n, λ, resolution; plot_geometry = false)

    run_simulation!(sim)

    baseline = compute_power(sim)

    sim = build_simulation(d, n, λ, resolution; plot_geometry = true)

    run_simulation!(sim)

    slab = compute_power(sim)

    T_fdtd = slab ./ baseline

    f = Figure()
    ax =  Axis(f[1, 1])
    scatterlines!(ax, λ,T)
    scatterlines!(ax, λ,T_fdtd)
    display(f)
    return sim
end


