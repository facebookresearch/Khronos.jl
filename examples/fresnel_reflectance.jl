# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E2: Fresnel Reflectance (Normal Incidence)
#
# Computes R(λ) for a planar air → n=3.5 dielectric interface using a two-run
# normalization workflow with reflected-field subtraction. Compares to the
# analytic Fresnel formula:  R = |(n-1)/(n+1)|²
#
# Reference: Meep refl-quartz.py (simplified to non-dispersive dielectric)

import Khronos
using CairoMakie
using GeometryPrimitives

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution = 40         # pixels/μm  (increase for higher accuracy)
n_slab     = 3.5        # refractive index of dielectric half-space
wvl_min    = 0.9        # shortest wavelength (μm)
wvl_max    = 1.6        # longest wavelength (μm)
nfreq      = 50         # number of frequency points
dpml       = 1.0        # PML thickness (μm)
# ─────────────────────────────────────────────────────────────────────────────

function main(; resolution=resolution, n_slab=n_slab, wvl_min=wvl_min,
                wvl_max=wvl_max, nfreq=nfreq, dpml=dpml)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    λ = range(wvl_min, wvl_max, length=nfreq)
    freqs = 1.0 ./ λ
    fcen = 0.5 * (1/wvl_min + 1/wvl_max)
    fwidth = (1/wvl_min - 1/wvl_max)

    # Thin transverse extent (1D-like in 3D); propagation along z
    sxy = 2 * dpml + 0.5
    sz  = 2 * dpml + 8.0
    src_z = -sz/2 + dpml + 1.0           # source near -z boundary
    mon_refl_z = src_z + 1.0             # reflection monitor between source and slab
    slab_z_start = 0.0                   # slab fills z > 0

    function build_sim(; with_slab=false)
        sources = [
            Khronos.PlaneWaveSource(
                time_profile = Khronos.GaussianPulseSource(fcen=fcen, fwidth=fwidth),
                center = [0.0, 0.0, src_z],
                size   = [Inf, Inf, 0.0],
                k_vector = [0.0, 0.0, 1.0],
                polarization_angle = 0.0,
            ),
        ]

        geometry = []
        if with_slab
            # Semi-infinite slab (fills the +z half of the domain into PML)
            geometry = [
                Khronos.Object(
                    Cuboid([0.0, 0.0, slab_z_start + sz/4], [sxy, sxy, sz/2]),
                    Khronos.Material(ε = n_slab^2),
                ),
            ]
        end

        # Monitor to record DFT of Ex between source and slab (in vacuum)
        refl_monitor = Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0.0, 0.0, mon_refl_z],
            size   = [0.4, 0.4, 0.0],
            frequencies = collect(freqs),
        )

        sim = Khronos.Simulation(
            cell_size   = [sxy, sxy, sz],
            cell_center = [0.0, 0.0, 0.0],
            resolution  = resolution,
            geometry    = geometry,
            sources     = sources,
            boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
            monitors    = [refl_monitor],
        )
        return sim
    end

    # ── Run 1: normalization (no slab) ───────────────────────────────────────
    println("="^60)
    println("Run 1: Normalization (vacuum)")
    println("="^60)
    sim_norm = build_sim(with_slab=false)
    Khronos.run(sim_norm,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-11, minimum_runtime=20.0, maximum_runtime=500.0))
    # Complex DFT fields at reflection monitor (in vacuum for both runs)
    inc_fields = Array(sim_norm.monitors[1].monitor_data.fields)

    # ── Run 2: with dielectric slab ──────────────────────────────────────────
    println("="^60)
    println("Run 2: With dielectric slab (n=$n_slab)")
    println("="^60)
    sim_slab = build_sim(with_slab=true)
    Khronos.run(sim_slab,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance=1e-11, minimum_runtime=20.0, maximum_runtime=500.0))
    tot_fields = Array(sim_slab.monitors[1].monitor_data.fields)

    # ── Compute reflectance via field subtraction ────────────────────────────
    # E_refl = E_total - E_incident (complex subtraction preserves phase)
    refl_fields = tot_fields .- inc_fields
    P_inc  = sum(abs.(inc_fields).^2, dims=[1,2])[1,1,1,:]
    P_refl = sum(abs.(refl_fields).^2, dims=[1,2])[1,1,1,:]
    R_all  = P_refl ./ P_inc

    # Filter to frequencies with good SNR (spectral energy > 1% of peak)
    P_inc_max = maximum(P_inc)
    good = P_inc .> 0.01 * P_inc_max
    good_idx = findall(good)

    R_khronos = R_all[good_idx]
    λ_good = collect(λ)[good_idx]

    # ── Analytic Fresnel ─────────────────────────────────────────────────────
    R_analytic_val = ((n_slab - 1) / (n_slab + 1))^2
    R_analytic = R_analytic_val .* ones(length(R_khronos))

    # ── Correctness check ────────────────────────────────────────────────────
    max_err = maximum(abs.(R_khronos .- R_analytic))
    rms_err = sqrt(sum((R_khronos .- R_analytic).^2) / length(R_khronos))
    println("\n", "="^60)
    println("Fresnel Reflectance Validation")
    println("="^60)
    println("  n_slab:          $n_slab")
    println("  R_analytic:      $(round(R_analytic_val, digits=6))")
    println("  R_khronos mean:  $(round(sum(R_khronos)/length(R_khronos), digits=6))")
    println("  Good freqs:      $(length(R_khronos)) / $nfreq")
    println("  Max |error|:     $(round(max_err, digits=6))")
    println("  RMS error:       $(round(rms_err, digits=6))")
    println("  Status:          $(rms_err < 0.05 ? "PASS" : "FAIL") (RMS threshold=0.05)")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 400))

    ax1 = Axis(f[1, 1], xlabel="Wavelength (μm)", ylabel="Reflectance",
               title="Fresnel Reflectance: air → n=$n_slab")
    lines!(ax1, λ_good, R_analytic, label="Analytic R=|(n-1)/(n+1)|²",
           color=:red, linewidth=2)
    scatterlines!(ax1, λ_good, R_khronos, label="Khronos",
                  color=:blue, markersize=4)
    axislegend(ax1, position=:rt)

    ax2 = Axis(f[1, 2], xlabel="Wavelength (μm)", ylabel="|R_khronos - R_analytic|",
               title="Absolute Error", yscale=log10)
    scatterlines!(ax2, λ_good, abs.(R_khronos .- R_analytic) .+ 1e-16,
                  color=:black, markersize=3)

    save("fresnel_reflectance.png", f)
    println("Saved: fresnel_reflectance.png")

    return R_khronos, R_analytic
end

main()
