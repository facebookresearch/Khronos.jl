# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E25: CW Source Steady-State Verification
#
# Verifies that a CW point source in vacuum reaches steady state and the
# field amplitude (measured via DFT monitors) decays as 1/r, consistent with
# the 3D Green's function for the free-space Helmholtz equation.
#
# Reference: Meep solve-cw.py (CW steady-state concept)

import Khronos
using CairoMakie

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution  = 15       # pixels/μm  (increase for accuracy)
cell_xyz    = 14.0     # domain size (μm)
dpml        = 1.0      # PML thickness (μm)
fcen        = 1.0      # source frequency (1/μm → λ = 1 μm)
t_end       = 25.0     # simulation time (enough CW cycles for steady state)
n_probes    = 8        # number of radial probe points
# ─────────────────────────────────────────────────────────────────────────────

function main(; resolution=resolution, cell_xyz=cell_xyz, dpml=dpml,
                fcen=fcen, t_end=t_end, n_probes=n_probes)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    # Probe distances from the source (along +x axis)
    r_min = 2.0
    r_max = cell_xyz/2 - dpml - 0.5
    r_probes = collect(range(r_min, r_max, length=n_probes))

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen=fcen),
            component = Khronos.Ez(),
            center = [0.0, 0.0, 0.0],
            size   = [0.0, 0.0, 0.0],
            amplitude = 1im,  # sine-like excitation
        ),
    ]

    # Place a DFT monitor at each probe distance along the x-axis.
    # Each monitor is a single small plane to capture the amplitude at that point.
    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [r, 0.0, 0.0],
            size   = [0.0, 0.0, 0.0],
            frequencies = [fcen],
        )
        for r in r_probes
    ]

    sim = Khronos.Simulation(
        cell_size   = [cell_xyz, cell_xyz, cell_xyz],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = resolution,
        sources     = sources,
        boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        monitors    = monitors,
    )

    Khronos.run(sim, until=t_end)

    # ── Extract DFT amplitudes at each probe ─────────────────────────────────
    amplitudes = Float64[]
    for i in 1:n_probes
        dft_val = sim.monitors[i].monitor_data.fields
        amp = abs(Array(dft_val)[1, 1, 1, 1])
        push!(amplitudes, amp)
    end

    # ── Fit 1/r decay ────────────────────────────────────────────────────────
    # |E| ∝ A/r in 3D. Fit A = mean(amplitude * r)
    A_fit = sum(amplitudes .* r_probes) / n_probes

    predicted = A_fit ./ r_probes
    rel_errors = abs.(amplitudes .- predicted) ./ predicted
    mean_rel_err = sum(rel_errors) / length(rel_errors)

    println("\n", "="^60)
    println("CW Steady-State Verification (3D)")
    println("="^60)
    println("  Frequency:           $fcen (λ = $(1/fcen) μm)")
    println("  Probes:              $n_probes points, r ∈ [$(r_probes[1]), $(round(r_probes[end],digits=1))] μm")
    println("  Fit coefficient A:   $(round(A_fit, digits=6))")
    println()
    println("  r (μm)     |Ez|_DFT     A/r          rel_error")
    for i in 1:n_probes
        println("  $(rpad(round(r_probes[i],digits=2), 11))$(rpad(round(amplitudes[i],digits=6), 13))$(rpad(round(predicted[i],digits=6), 13))$(round(rel_errors[i],digits=4))")
    end
    println()
    println("  Mean relative error: $(round(mean_rel_err, digits=4))")
    println("  Status:              $(mean_rel_err < 0.15 ? "PASS" : "FAIL") (threshold=0.15)")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 400))

    # Plot 1: field cross section
    Khronos.prepare_simulation!(sim)
    kc = div(sim.Nz, 2) + 1
    ez_slice = Array(collect(sim.fields.fEz[1:sim.Nx, 1:sim.Ny, kc]))
    xs = range(-cell_xyz/2, cell_xyz/2, length=size(ez_slice,1))
    ys = range(-cell_xyz/2, cell_xyz/2, length=size(ez_slice,2))
    vmax = maximum(abs.(ez_slice))

    ax1 = Axis(f[1, 1], xlabel="x (μm)", ylabel="y (μm)",
               title="CW Dipole Ez (XY plane)", aspect=DataAspect())
    heatmap!(ax1, collect(xs), collect(ys), ez_slice,
             colormap=:bluesreds, colorrange=(-vmax, vmax))

    # Plot 2: DFT amplitude vs 1/r
    ax2 = Axis(f[1, 2], xlabel="r (μm)", ylabel="|Ez| (DFT amplitude)",
               title="Amplitude Decay: |Ez| vs A/r", yscale=log10, xscale=log10)
    scatter!(ax2, r_probes, amplitudes, label="Khronos DFT", color=:blue, markersize=8)
    r_dense = range(r_probes[1], r_probes[end], length=100)
    lines!(ax2, collect(r_dense), A_fit ./ collect(r_dense),
           label="A/r fit", color=:red, linewidth=2)
    axislegend(ax2, position=:rt)

    save("cw_steady_state.png", f)
    println("Saved: cw_steady_state.png")

    return mean_rel_err
end

main()
