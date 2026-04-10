# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Using DFT monitors, compute the transmission of a planewave in 3D through a
# simple slab structure. Compare the computed response to the analytic response.

import Khronos
using CairoMakie
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

"""
    slab_analytical()

Computes the analytic transmision as a function of slab thickness (d), refractive index (n), and wavelength (λ).
"""
function slab_analytical(d, n, λ)
    rho = (n - 1) ./ (n + 1)
    t = (
        ((1 + rho) * (1 - rho) * exp.(-2 * im * π * n * d ./ λ)) ./
        (1 .- rho^2 * exp.(-4 * im * π * n * d ./ λ))
    )
    return abs.(t) .^ 2
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
        Khronos.PlaneWaveSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 1 / 1.5, fwidth = 0.4),
            center = [0, 0, -monitor_height],
            size = [Inf, Inf, 0.0],
            polarization_angle = 0.0,
            k_vector = [0.0, 0.0, 1.0],
        ),
    ]

    geometry = []

    if plot_geometry
        geometry =
            [Khronos.Object(Cuboid([0, 0, 0], [sxy, sxy, d]), Khronos.Material(ε = n^2))]
    end

    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0, 0, monitor_height],
            size = [monitor_xy, monitor_xy, 0],
            frequencies = 1 ./ λ,
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = cell_size,
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = boundary_layers,
        monitors = monitors,
    )

    return sim

end

"""
    run_simulation!(sim::Khronos.SimulationData)

Run a simulation (`sim`) until the DFT fields have converged.
"""
function run_simulation!(sim::Khronos.SimulationData)
    Khronos.run(
        sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance = 1e-11,
            minimum_runtime = 0.0,
            maximum_runtime = 300.0,
        ),
    )
end

"""
    compute_power(sim::Khronos.SimulationData)

Computes the total power of the first DFT monitor in the simulation (`sim`).
"""
function compute_power(sim::Khronos.SimulationData)
    return sum(abs.(sim.monitors[1].monitor_data.fields) .^ 2, dims = [1, 2])[1, 1, 1, :]
end

# Arbitrary simulation parameters that produce at least two maxima
d = 0.5
n = 3.5
λ = range(1.0, 2.0, length = 100)
resolution = 50

function run_periodic_slab(d, n, λ, resolution)
    # Run a normalization simulation
    sim_norm = build_simulation(d, n, λ, resolution; plot_geometry = false)
    run_simulation!(sim_norm)
    baseline = compute_power(sim_norm)

    # Run the actual simulation
    sim_slab = build_simulation(d, n, λ, resolution; plot_geometry = true)
    run_simulation!(sim_slab)
    slab = compute_power(sim_slab)

    T_Khronos = slab ./ baseline
    return T_Khronos
end

# --- Run 1: includes JIT compilation ---
println("=" ^ 60)
println("RUN 1 (cold — includes JIT compilation)")
println("=" ^ 60)
t1 = time()
T_Khronos = run_periodic_slab(d, n, λ, resolution)
println("Run 1 total wall time: $(round(time() - t1, digits=3))s\n")

# --- Run 2: fresh sim, JIT already done ---
println("=" ^ 60)
println("RUN 2 (warm — JIT already compiled)")
println("=" ^ 60)
t2 = time()
T_Khronos = run_periodic_slab(d, n, λ, resolution)
println("Run 2 total wall time: $(round(time() - t2, digits=3))s\n")

# Compute the analytic response for comparison
T = slab_analytical(d, n, λ)

# Plot the comparison
f = Figure()
ax = Axis(f[1, 1], xlabel = "λ (μm)", ylabel = "T (a.u.)")
line1 = scatterlines!(ax, λ, Array(T))
line2 = scatterlines!(ax, λ, Array(T_Khronos))
legend = Legend(f[1, 2], [line1, line2], ["Analytic", "Khronos"])
save("periodic_slab.png", f)
