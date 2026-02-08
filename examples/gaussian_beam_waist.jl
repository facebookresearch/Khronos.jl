# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E24: Gaussian Beam Waist Verification
#
# Injects a Gaussian beam and measures the beam waist w(z) at multiple planes
# along the propagation axis using DFT monitors. Compares the measured waist
# to the analytical Gaussian beam formula:
#   w(z) = w₀ √(1 + (z/z_R)²),   z_R = π w₀² / λ
#
# Reference: Tidy3D AdvancedGaussianSources, fdtdx simulate_gaussian_source.py

import Khronos
using CairoMakie
using GeometryPrimitives
using Statistics

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution   = 20       # pixels/μm
λ            = 1.0      # wavelength (μm)
waist_radius = 1.5      # beam waist w₀ (μm)
dpml         = 1.0      # PML thickness
n_monitors   = 7        # number of z-planes to measure waist
# ─────────────────────────────────────────────────────────────────────────────

function main(; resolution=resolution, λ=λ, waist_radius=waist_radius,
                dpml=dpml, n_monitors=n_monitors)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    fcen = 1.0 / λ
    z_R = π * waist_radius^2 / λ  # Rayleigh range

    # Domain size: enough transverse extent for the beam + PML
    sxy = 2*dpml + 4*waist_radius + 2.0
    sz  = 2*dpml + 2*z_R + 4.0

    # Source at z = -z_R (beam waist at z=0 by convention)
    src_z = -z_R - 1.0

    # Monitor z-positions: from before the waist to well past it
    z_monitor = range(-z_R, z_R, length=n_monitors)
    monitor_xy_size = sxy - 2*dpml - 0.5

    sources = [
        Khronos.GaussianBeamSource(
            time_profile = Khronos.GaussianPulseSource(fcen=fcen, fwidth=0.15*fcen),
            center     = [0.0, 0.0, src_z],
            size       = [sxy, sxy, 0.0],
            beam_center = [0.0, 0.0, 0.0],
            beam_waist = waist_radius,
            k_vector   = [0.0, 0.0, 1.0],
            polarization = [1.0, 0.0, 0.0],
        ),
    ]

    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0.0, 0.0, z],
            size   = [monitor_xy_size, 0.0, 0.0],  # 1D line along x
            frequencies = [fcen],
        )
        for z in z_monitor
    ]

    sim = Khronos.Simulation(
        cell_size   = [sxy, sxy, sz],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = resolution,
        sources     = sources,
        boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        monitors    = monitors,
    )

    println("="^60)
    println("Gaussian Beam Waist Verification")
    println("="^60)

    Khronos.run(sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-7, minimum_runtime=0.0, maximum_runtime=300.0))

    # ── Extract waist at each z-plane ────────────────────────────────────────
    measured_waists = Float64[]
    for (i, z) in enumerate(z_monitor)
        dft_data = Array(sim.monitors[i].monitor_data.fields[:, 1, 1, 1])
        intensity = abs.(dft_data).^2

        # Compute second moment (variance) of intensity profile → beam waist
        nx = length(intensity)
        xs = range(-monitor_xy_size/2, monitor_xy_size/2, length=nx)
        total = sum(intensity)
        if total > 0
            x_mean = sum(xs .* intensity) / total
            x_var = sum((xs .- x_mean).^2 .* intensity) / total
            w_measured = 2 * sqrt(x_var)  # 1/e² radius from second moment
        else
            w_measured = NaN
        end
        push!(measured_waists, w_measured)
    end

    # ── Analytic waist ───────────────────────────────────────────────────────
    w_analytic = [waist_radius * sqrt(1 + (z/z_R)^2) for z in z_monitor]

    # ── Correctness check ────────────────────────────────────────────────────
    valid = .!isnan.(measured_waists)
    rel_errors = abs.(measured_waists[valid] .- w_analytic[valid]) ./ w_analytic[valid]
    mean_rel_err = mean(rel_errors)

    println("\n  w₀ = $waist_radius μm, z_R = $(round(z_R, digits=2)) μm")
    println("  z (μm)     w_measured   w_analytic   rel_error")
    for (i, z) in enumerate(z_monitor)
        if valid[i]
            println("  $(round(z, digits=2))\t\t$(round(measured_waists[i], digits=3))\t\t$(round(w_analytic[i], digits=3))\t\t$(round(rel_errors[i], digits=3))")
        end
    end
    println("\n  Mean relative error: $(round(mean_rel_err, digits=4))")
    println("  Status: $(mean_rel_err < 0.25 ? "PASS" : "FAIL") (threshold=0.25)")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 400))

    ax1 = Axis(f[1, 1], xlabel="z (μm)", ylabel="Beam waist w(z) (μm)",
               title="Gaussian Beam Waist: w₀=$(waist_radius) μm")
    z_dense = range(-z_R, z_R, length=200)
    w_dense = waist_radius .* sqrt.(1 .+ (collect(z_dense) ./ z_R).^2)
    lines!(ax1, collect(z_dense), w_dense, label="Analytic w(z)=w₀√(1+(z/z_R)²)",
           color=:red, linewidth=2)
    scatter!(ax1, collect(z_monitor)[valid], measured_waists[valid],
             label="Khronos (second moment)", color=:blue, markersize=8)
    axislegend(ax1, position=:ct)

    # Plot one representative beam profile
    mid_idx = div(n_monitors, 2) + 1
    dft_mid = Array(sim.monitors[mid_idx].monitor_data.fields[:, 1, 1, 1])
    intensity_mid = abs.(dft_mid).^2
    nx = length(intensity_mid)
    xs = range(-monitor_xy_size/2, monitor_xy_size/2, length=nx)

    ax2 = Axis(f[1, 2], xlabel="x (μm)", ylabel="|Ex|² (a.u.)",
               title="Beam profile at z=$(round(collect(z_monitor)[mid_idx], digits=1)) μm")
    lines!(ax2, collect(xs), intensity_mid ./ maximum(intensity_mid), color=:blue,
           label="Khronos", linewidth=1.5)
    # Overlay Gaussian fit
    w_at_z = w_analytic[mid_idx]
    gauss_fit = exp.(-2 .* collect(xs).^2 ./ w_at_z^2)
    lines!(ax2, collect(xs), gauss_fit, color=:red, linestyle=:dash,
           label="Gaussian (w=$(round(w_at_z, digits=2)))", linewidth=1.5)
    axislegend(ax2, position=:rt)

    save("gaussian_beam_waist.png", f)
    println("Saved: gaussian_beam_waist.png")

    return measured_waists, w_analytic
end

main()
