# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E19: Anisotropic Material Slab
#
# Demonstrates polarization-dependent transmission through a slab with
# ε = diag(εx, εy, εz). An x-polarized wave sees εx while a y-polarized
# wave sees εy, leading to different effective wavelengths inside the slab
# and thus different Fabry-Perot fringes.
#
# Reference: fdtdx simulate_gaussian_source_anisotropic.py

import Khronos
using CairoMakie
using GeometryPrimitives

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution = 50         # pixels/μm
εx         = 4.0        # permittivity seen by x-polarized light
εy         = 1.5        # permittivity seen by y-polarized light
εz         = 4.0        # permittivity along propagation direction
d_slab     = 0.5        # slab thickness (μm)
dpml       = 0.5        # PML thickness
# ─────────────────────────────────────────────────────────────────────────────

function main(; resolution=resolution, εx=εx, εy=εy, εz=εz,
                d_slab=d_slab, dpml=dpml)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    # Use the same source parameters as the known-working periodic_slab.jl
    λ = range(1.0, 2.0, length=100)
    freqs = 1.0 ./ λ

    monitor_xy = 0.5
    monitor_height = 0.5
    sxy = 2*dpml + monitor_xy
    sz  = 2*dpml + 4*monitor_height

    # Analytic Fabry-Perot for a slab of index n
    function slab_T_analytic(d, n, λ_vec)
        rho = (n .- 1) ./ (n .+ 1)
        t = ((1 .+ rho) .* (1 .- rho) .* exp.(-2im * π * n * d ./ λ_vec)) ./
            (1 .- rho.^2 .* exp.(-4im * π * n * d ./ λ_vec))
        return abs.(t).^2
    end

    function build_sim(pol_angle; with_slab=false)
        sources = [
            Khronos.PlaneWaveSource(
                time_profile = Khronos.GaussianPulseSource(fcen=1/1.5, fwidth=0.4),
                center  = [0.0, 0.0, -monitor_height],
                size    = [Inf, Inf, 0.0],
                k_vector = [0.0, 0.0, 1.0],
                polarization_angle = pol_angle,
            ),
        ]

        component = pol_angle == 0.0 ? Khronos.Ex() : Khronos.Ey()

        geometry = []
        if with_slab
            geometry = [
                Khronos.Object(
                    Cuboid([0.0, 0.0, 0.0], [sxy, sxy, d_slab]),
                    Khronos.Material(εx=εx, εy=εy, εz=εz),
                ),
            ]
        end

        monitors = [
            Khronos.DFTMonitor(
                component = component,
                center = [0.0, 0.0, monitor_height],
                size   = [monitor_xy, monitor_xy, 0.0],
                frequencies = collect(freqs),
            ),
        ]

        sim = Khronos.Simulation(
            cell_size   = [sxy, sxy, sz],
            cell_center = [0.0, 0.0, 0.0],
            resolution  = resolution,
            geometry    = geometry,
            sources     = sources,
            boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
            monitors    = monitors,
        )
        return sim
    end

    compute_power(sim) =
        sum(abs.(sim.monitors[1].monitor_data.fields).^2, dims=[1,2])[1,1,1,:]

    run_dft!(sim) = Khronos.run(sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-11, minimum_runtime=0.0, maximum_runtime=300.0))

    # ── X-polarized runs ─────────────────────────────────────────────────────
    println("="^60, "\nX-polarized: normalization\n", "="^60)
    sim_x_norm = build_sim(0.0; with_slab=false)
    run_dft!(sim_x_norm)
    P_x_norm = compute_power(sim_x_norm)

    println("="^60, "\nX-polarized: with slab\n", "="^60)
    sim_x_slab = build_sim(0.0; with_slab=true)
    run_dft!(sim_x_slab)
    T_x = Array(compute_power(sim_x_slab) ./ P_x_norm)

    # ── Y-polarized runs ─────────────────────────────────────────────────────
    println("="^60, "\nY-polarized: normalization\n", "="^60)
    sim_y_norm = build_sim(π/2; with_slab=false)
    run_dft!(sim_y_norm)
    P_y_norm = compute_power(sim_y_norm)

    println("="^60, "\nY-polarized: with slab\n", "="^60)
    sim_y_slab = build_sim(π/2; with_slab=true)
    run_dft!(sim_y_slab)
    T_y = Array(compute_power(sim_y_slab) ./ P_y_norm)

    # ── Analytic comparison ──────────────────────────────────────────────────
    T_x_analytic = slab_T_analytic(d_slab, sqrt(εx), collect(λ))
    T_y_analytic = slab_T_analytic(d_slab, sqrt(εy), collect(λ))

    err_x = maximum(abs.(T_x .- T_x_analytic))
    err_y = maximum(abs.(T_y .- T_y_analytic))
    rms_x = sqrt(sum((T_x .- T_x_analytic).^2) / length(T_x))
    rms_y = sqrt(sum((T_y .- T_y_analytic).^2) / length(T_y))

    println("\n", "="^60)
    println("Anisotropic Slab Validation")
    println("="^60)
    println("  ε = diag($εx, $εy, $εz), slab thickness = $d_slab μm")
    println("  X-pol max |error|: $(round(err_x, digits=5))  RMS: $(round(rms_x, digits=5))")
    println("  Y-pol max |error|: $(round(err_y, digits=5))  RMS: $(round(rms_y, digits=5))")
    println("  Status: $(max(rms_x,rms_y) < 0.05 ? "PASS" : "FAIL") (RMS threshold=0.05)")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 500))

    ax1 = Axis(f[1, 1], xlabel="Wavelength (μm)", ylabel="Transmission",
               title="X-polarized (sees εx=$εx)")
    lines!(ax1, collect(λ), T_x_analytic, label="Analytic (n=$(round(sqrt(εx),digits=2)))",
           color=:red, linewidth=2)
    scatterlines!(ax1, collect(λ), T_x, label="Khronos", color=:blue, markersize=4)
    axislegend(ax1, position=:rb)

    ax2 = Axis(f[1, 2], xlabel="Wavelength (μm)", ylabel="Transmission",
               title="Y-polarized (sees εy=$εy)")
    lines!(ax2, collect(λ), T_y_analytic, label="Analytic (n=$(round(sqrt(εy),digits=2)))",
           color=:red, linewidth=2)
    scatterlines!(ax2, collect(λ), T_y, label="Khronos", color=:blue, markersize=4)
    axislegend(ax2, position=:rb)

    save("anisotropic_slab.png", f)
    println("Saved: anisotropic_slab.png")

    return T_x, T_y
end

main()
