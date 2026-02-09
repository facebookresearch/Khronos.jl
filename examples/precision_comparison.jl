# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E31: Float32 vs Float64 Precision Comparison
#
# Runs the same dielectric slab transmission problem in both Float32 and
# Float64, comparing throughput (MCells/s), memory usage, and accuracy
# against the analytical Fabry-Perot formula.
#
# Reference: EXAMPLES.md E31

import Khronos
using CairoMakie
using GeometryPrimitives

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution = 40         # pixels/μm
d_slab     = 0.5        # slab thickness (μm)
n_slab     = 3.5        # refractive index
nfreq      = 60         # frequency points
n_steps    = 100        # benchmark steps for throughput measurement
# ─────────────────────────────────────────────────────────────────────────────

function slab_analytical(d, n, λ)
    rho = (n - 1) / (n + 1)
    t = ((1 + rho) * (1 - rho) .* exp.(-2im * π * n * d ./ λ)) ./
        (1 .- rho^2 .* exp.(-4im * π * n * d ./ λ))
    return abs.(t).^2
end

function run_precision_test(precision::DataType; resolution=resolution,
                            d_slab=d_slab, n_slab=n_slab, nfreq=nfreq)

    Khronos.choose_backend(Khronos.CUDADevice(), precision)

    λ = range(1.0, 2.0, length=nfreq)
    freqs = 1.0 ./ λ

    dpml = 0.5
    sxy = 2*dpml + 0.5
    sz  = 2*dpml + 4*0.5

    function build_sim(; with_slab=false)
        sources = [
            Khronos.PlaneWaveSource(
                time_profile = Khronos.GaussianPulseSource(fcen=1/1.5, fwidth=0.4),
                center = [0.0, 0.0, -0.5],
                size   = [Inf, Inf, 0.0],
                polarization_angle = 0.0,
                k_vector = [0.0, 0.0, 1.0],
            ),
        ]

        geometry = with_slab ?
            [Khronos.Object(Cuboid([0,0,0], [sxy, sxy, d_slab]),
                            Khronos.Material(ε=n_slab^2))] : []

        monitors = [
            Khronos.DFTMonitor(
                component = Khronos.Ex(),
                center = [0.0, 0.0, 0.5],
                size   = [0.4, 0.4, 0.0],
                frequencies = collect(freqs),
            ),
        ]

        return Khronos.Simulation(
            cell_size   = [sxy, sxy, sz],
            cell_center = [0.0, 0.0, 0.0],
            resolution  = resolution,
            geometry    = geometry,
            sources     = sources,
            boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
            monitors    = monitors,
        )
    end

    compute_power(sim) =
        sum(abs.(sim.monitors[1].monitor_data.fields).^2, dims=[1,2])[1,1,1,:]

    # Normalization run
    sim_norm = build_sim(with_slab=false)
    Khronos.run(sim_norm,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-9, minimum_runtime=0.0, maximum_runtime=300.0))
    P_norm = compute_power(sim_norm)

    # Slab run
    sim_slab = build_sim(with_slab=true)
    Khronos.run(sim_slab,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-9, minimum_runtime=0.0, maximum_runtime=300.0))
    T_khronos = Array(compute_power(sim_slab) ./ P_norm)

    # Throughput benchmark
    sim_bench = build_sim(with_slab=true)
    rate = Khronos.run_benchmark(sim_bench, n_steps)

    return T_khronos, rate
end

function main()
    λ = collect(range(1.0, 2.0, length=nfreq))
    T_analytic = slab_analytical(d_slab, n_slab, λ)

    # ── Float64 ──────────────────────────────────────────────────────────────
    println("\n", "="^60, "\nFloat64 precision\n", "="^60)
    T_f64, rate_f64 = run_precision_test(Float64)
    err_f64 = maximum(abs.(T_f64 .- T_analytic))

    # ── Float32 ──────────────────────────────────────────────────────────────
    println("\n", "="^60, "\nFloat32 precision\n", "="^60)
    T_f32, rate_f32 = run_precision_test(Float32)
    err_f32 = maximum(abs.(T_f32 .- T_analytic))

    # ── Summary ──────────────────────────────────────────────────────────────
    println("\n", "="^60)
    println("Float32 vs Float64 Comparison")
    println("="^60)
    println("  Precision  MCells/s    Max |T error|    Speedup")
    println("  " * "-"^50)
    println("  Float64    $(rpad(round(rate_f64,digits=1), 12))$(rpad(round(err_f64,digits=6), 17))1.00×")
    println("  Float32    $(rpad(round(rate_f32,digits=1), 12))$(rpad(round(err_f32,digits=6), 17))$(round(rate_f32/rate_f64, digits=2))×")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 400))

    ax1 = Axis(f[1, 1], xlabel="Wavelength (μm)", ylabel="Transmission",
               title="Slab Transmission: Float32 vs Float64")
    lines!(ax1, λ, T_analytic, label="Analytic", color=:black, linewidth=2)
    scatterlines!(ax1, λ, T_f64, label="Float64", color=:blue, markersize=4)
    scatterlines!(ax1, λ, T_f32, label="Float32", color=:red, markersize=4)
    axislegend(ax1, position=:rb)

    ax2 = Axis(f[1, 2], xlabel="Precision", ylabel="Value",
               title="Performance Comparison",
               xticks=([1, 2], ["Float64", "Float32"]))
    barplot!(ax2, [1, 2], [rate_f64, rate_f32], color=[:blue, :red],
             bar_labels=:y, label_formatter=x -> "$(round(x, digits=0)) MC/s")

    save("precision_comparison.png", f)
    println("Saved: precision_comparison.png")

    return (rate_f64, err_f64), (rate_f32, err_f32)
end

main()
