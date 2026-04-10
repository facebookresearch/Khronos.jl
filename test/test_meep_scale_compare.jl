#!/usr/bin/env julia
# Compare meep's adj_src_scale with Khronos's to identify discrepancies.
import Khronos
using Printf

function compare_scales()
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    fcen = 0.5; fwidth = 0.15; resolution = 20
    pml = 1.0; pad = 0.5; design_Lx = 1.0; cell_y = 4.0
    cell_x = design_Lx + 2 * pad + 2 * pml

    # Create a simulation to get dt
    geometry = [Khronos.Object(
        GeometryPrimitives.Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, 0.0]),
        Khronos.Material(ε = 1.0))]
    sources = [Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-1.0, 0.0, 0.0], size = [0.0, 0.0, 0.0], amplitude = 1.0)]
    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, 0.0], cell_center = [0.0, 0.0, 0.0],
        resolution = resolution, geometry = geometry, sources = sources,
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[])
    Khronos.prepare_simulation!(sim)

    dt = Float64(sim.Δt)
    runtime = 80.0
    Khronos.run(sim; until = runtime)
    T_sim = Float64(Khronos.round_time(sim))
    n_steps = floor(Int, T_sim / dt)
    t_vals = collect(0:n_steps-1) .* dt

    tp = Khronos.get_time_profile(sim.sources[1])

    println("="^60)
    println("MEEP vs KHRONOS adj_src_scale comparison")
    println("="^60)
    @printf("  fcen = %.2f, fwidth = %.2f, resolution = %d\n", fcen, fwidth, resolution)
    @printf("  dt = %.6e, T_sim = %.2f, n_steps = %d\n", dt, T_sim, n_steps)
    @printf("  ndims = %d, dV = %.6e\n", sim.ndims, 1.0/resolution^sim.ndims)

    # ── Evaluate source waveforms ──
    y_dipole = [Complex{Float64}(Khronos.eval_time_source(tp, t)) for t in t_vals]

    # meep's current() = (dipole(t+dt) - dipole(t))/dt
    y_current = zeros(ComplexF64, n_steps)
    for i in 1:n_steps-1
        y_current[i] = (y_dipole[i+1] - y_dipole[i]) / dt
    end

    println("\nSource waveform:")
    @printf("  Khronos eval_time_source(0) = %+.6e %+.6ei\n", real(y_dipole[1]), imag(y_dipole[1]))
    @printf("  Khronos eval_time_source(dt) = %+.6e %+.6ei\n", real(y_dipole[2]), imag(y_dipole[2]))
    @printf("  meep current(0) ≈ (dipole(dt)-dipole(0))/dt = %+.6e %+.6ei\n",
        real(y_current[1]), imag(y_current[1]))

    # ── DTFT comparisons ──
    # Khronos source: dipole(t)
    dtft_dipole = sum(dt / sqrt(2π) .* exp.(im .* 2π .* fcen .* t_vals) .* y_dipole)
    # meep source: current(t) = d(dipole)/dt
    dtft_current = sum(dt / sqrt(2π) .* exp.(im .* 2π .* fcen .* t_vals) .* y_current)

    println("\nDTFT at fcen=$fcen:")
    @printf("  DTFT(dipole):  %+.6e %+.6ei, |val| = %.6e\n",
        real(dtft_dipole), imag(dtft_dipole), abs(dtft_dipole))
    @printf("  DTFT(current): %+.6e %+.6ei, |val| = %.6e\n",
        real(dtft_current), imag(dtft_current), abs(dtft_current))
    @printf("  Ratio |current/dipole|: %.6f\n", abs(dtft_current) / abs(dtft_dipole))

    # iomega
    iomega = (1.0 - exp(-im * 2π * fcen * dt)) / dt
    @printf("\niomega: %+.6e %+.6ei, |val| = %.6e\n", real(iomega), imag(iomega), abs(iomega))

    # Theoretical: DTFT(current) = iomega * DTFT(dipole)
    # because current = d(dipole)/dt and discrete derivative in freq domain = iomega
    dtft_current_theory = iomega * dtft_dipole * dt  # ??? need to check normalization
    @printf("  iomega * dtft_dipole = %+.6e %+.6ei\n",
        real(iomega * dtft_dipole), imag(iomega * dtft_dipole))
    @printf("  Actual dtft_current  = %+.6e %+.6ei\n",
        real(dtft_current), imag(dtft_current))
    @printf("  Ratio: %.4f\n", abs(dtft_current) / abs(iomega * dtft_dipole))

    # ── adj_src_scale comparison ──
    dV = 1.0 / Float64(resolution)^sim.ndims

    # Phase correction
    cutoff = 5.0; fwidth_frac = 0.1
    fwidth_scale = exp(-2im * π * cutoff / fwidth_frac)
    adj_src_phase = exp(im * angle(dtft_dipole)) * fwidth_scale

    # meep formula (uses current DTFT + ×2 for real fields):
    # scale_meep = dV * iomega / dtft_current / adj_src_phase * 2
    # But meep's fwd_dtft IS dtft_current (meep evaluates current(), not dipole())
    scale_meep_style = dV * iomega / dtft_current / adj_src_phase * 2

    # Khronos formula as currently coded (uses dipole DTFT + ×2):
    # scale_khronos = dV * iomega / dtft_dipole / adj_src_phase * 2
    scale_khronos = dV * iomega / dtft_dipole / adj_src_phase * 2

    # Khronos's actual adj_src_scale function
    scale_actual = Khronos.adj_src_scale(sim, [fcen], tp; include_resolution=true)

    println("\nadj_src_scale values:")
    @printf("  meep-style (current DTFT):   %+.6e %+.6ei, |val| = %.6e\n",
        real(scale_meep_style), imag(scale_meep_style), abs(scale_meep_style))
    @printf("  Khronos (dipole DTFT + ×2):  %+.6e %+.6ei, |val| = %.6e\n",
        real(scale_khronos), imag(scale_khronos), abs(scale_khronos))
    @printf("  Khronos actual function:     %+.6e %+.6ei, |val| = %.6e\n",
        real(scale_actual[1]), imag(scale_actual[1]), abs(scale_actual[1]))
    @printf("  Ratio |meep/khronos|: %.4f\n", abs(scale_meep_style) / abs(scale_khronos))

    # Key insight: the difference is fwd_dtft
    # meep uses DTFT(current) = DTFT(d(dipole)/dt)
    # khronos uses DTFT(dipole)
    # The ratio is |DTFT(current)/DTFT(dipole)| = |iomega| (approximately)
    @printf("\n  Expected ratio |iomega|: %.4f\n", abs(iomega))
    @printf("  Actual ratio |current_dtft/dipole_dtft|: %.4f\n",
        abs(dtft_current) / abs(dtft_dipole))
    @printf("  These should match if current = d(dipole)/dt exactly.\n")

    # So the KEY DIFFERENCE: meep divides by DTFT(current), Khronos divides by DTFT(dipole)
    # The iomega factor is supposed to compensate for this:
    # meep: iomega / DTFT(current) = iomega / (iomega * DTFT(dipole)) = 1/DTFT(dipole)
    # khronos: iomega / DTFT(dipole) = iomega/DTFT(dipole)
    # These differ by a factor of iomega!
    #
    # meep's formula with current: scale ∝ iomega / dtft_current ≈ 1/dtft_dipole
    # khronos's formula with dipole: scale ∝ iomega / dtft_dipole
    # Ratio = iomega
    #
    # So Khronos has an EXTRA iomega factor that meep doesn't!
    println("\n" * "="^60)
    println("CONCLUSION:")
    println("  meep:    scale ∝ iomega / DTFT(current) ≈ 1 / DTFT(dipole)")
    println("  khronos: scale ∝ iomega / DTFT(dipole)")
    println("  The iomega factor cancels in meep but NOT in Khronos.")
    println("  Khronos should either:")
    println("  1. Remove iomega from scale (since it uses dipole not current), OR")
    println("  2. Use DTFT(current) instead of DTFT(dipole)")
    println("="^60)
end

using GeometryPrimitives
compare_scales()
