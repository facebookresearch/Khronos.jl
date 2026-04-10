# Derive the correct adj_src_scale by computing each factor independently
#
# Strategy: Run a simulation with a known source, measure the DFT response,
# and work backwards to determine the correct normalization.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random

function derive_scale()
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    fcen = 0.5; fwidth = 0.15
    resolution = 20
    pml = 1.0; pad = 0.5
    cell_x = 4.0 + 2*pml; cell_y = 4.0

    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, 0.0]),
            Khronos.Material(ε = 1.0)),
    ]
    fwd_src = Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-1.0, 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
        amplitude = 1.0,
    )

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, 0.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = [fwd_src],
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[],
    )
    Khronos.prepare_simulation!(sim)

    dt = Float64(sim.Δt)
    runtime = 80.0

    # ── Step 1: Forward source DFT calibration ──
    # Inject unit-amplitude forward source, measure DFT at a point
    Khronos.reset_fields!(sim)
    empty!(sim.monitor_data)
    mon = Khronos.DFTMonitor(component = Khronos.Ez(),
        center = [1.0, 0.0, 0.0], size = [0.0, 0.0, 0.0],
        frequencies = [fcen])
    push!(sim.monitor_data, Khronos.init_monitors(sim, mon))
    for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end

    Khronos.run(sim; until = runtime)
    fwd_dft_at_point = Array(mon.monitor_data.fields)[1]
    println("Forward source (amp=1) DFT at x=1: $(fwd_dft_at_point), |val|=$(abs(fwd_dft_at_point))")

    # ── Step 2: Compute adj_src_scale components ──
    fwd_tp = Khronos.get_time_profile(sim.sources[1])
    T_sim = Float64(Khronos.round_time(sim))
    n_steps = floor(Int, T_sim / dt)
    t_vals = collect(0:n_steps-1) .* dt
    y = [Complex{Float64}(Khronos.eval_time_source(fwd_tp, t)) for t in t_vals]

    # DTFT of the forward source
    fwd_dtft = sum(dt / sqrt(2π) .* exp.(im .* 2π .* fcen .* t_vals) .* y)
    println("\nForward source DTFT at fcen=$fcen:")
    @printf("  fwd_dtft = %+.6e %+.6ei, |fwd_dtft| = %.6e\n",
        real(fwd_dtft), imag(fwd_dtft), abs(fwd_dtft))

    # iomega
    iomega = (1.0 - exp(-im * 2π * fcen * dt)) / dt
    @printf("  iomega = %+.6e %+.6ei, |iomega| = %.6e\n",
        real(iomega), imag(iomega), abs(iomega))

    # adj_src_phase
    src_center_dtft = fwd_dtft  # at center frequency
    cutoff = 5.0; fwidth_frac = 0.1
    fwidth_scale = exp(-2im * π * cutoff / fwidth_frac)
    adj_src_phase = exp(im * angle(src_center_dtft)) * fwidth_scale
    @printf("  adj_src_phase = %+.6e %+.6ei, |phase| = %.6e\n",
        real(adj_src_phase), imag(adj_src_phase), abs(adj_src_phase))
    @printf("  fwidth_scale = %+.6e %+.6ei\n", real(fwidth_scale), imag(fwidth_scale))

    # Full scale (no dV)
    scale_noDV = iomega / fwd_dtft / adj_src_phase
    @printf("\n  scale_noDV = iomega/fwd_dtft/phase = %+.6e %+.6ei, |val| = %.6e\n",
        real(scale_noDV), imag(scale_noDV), abs(scale_noDV))

    # ── Step 3: Inject adjoint source with known amplitude, measure DFT ──
    # If we inject amplitude A with the forward time profile, the DFT at
    # the same point should be A * fwd_dft_at_point (by linearity/reciprocity
    # for the same source-monitor pair).
    # The ACTUAL DFT produced by the adjoint source with amplitude `dJ * scale`
    # should be `dJ * scale * fwd_dft_at_point`.
    # For the gradient to be correct, we need the DFT to equal some target value.

    # ── Step 4: Compute what the DTFT of the source waveform alone gives ──
    # The source kernel does: fSD += real(scalar_amp(t) * spatial_amp)
    # scalar_amp(t) = eval_time_source(fwd_tp, t) = Gaussian(t) * exp(-iωt) / (-2πfi)
    # The DFT accumulator records E at each timestep.
    # In free space (ε_inv=1): E = D + fSD (ignoring curl for the source point)
    # DFT of fSD = Σ_t fSD(t) * exp(-iωt)
    #            = Σ_t real(scalar_amp(t) * spatial_amp) * exp(-iωt)
    #            = spatial_amp * Σ_t real(Gaussian(t) * exp(-iωt) / (-2πfi)) * exp(-iωt)

    # Actually, let me just directly measure the ratio
    println("\n── Direct measurement ──")

    # Run with adjoint source at the monitoring location
    # and measure what DFT it produces at x=-1 (reciprocal to forward)
    Khronos.reset_fields!(sim)
    empty!(sim.monitor_data)
    mon2 = Khronos.DFTMonitor(component = Khronos.Ez(),
        center = [-1.0, 0.0, 0.0], size = [0.0, 0.0, 0.0],
        frequencies = [fcen])
    push!(sim.monitor_data, Khronos.init_monitors(sim, mon2))
    for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end

    # Use forward time profile with amplitude = 1
    adj_src = Khronos.UniformSource(
        time_profile = fwd_tp,
        component = Khronos.Ez(),
        center = [1.0, 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
        amplitude = 1.0,
    )
    saved = sim.sources
    sim.sources = [adj_src]
    Khronos.add_sources(sim, sim.sources)
    for chunk in sim.chunk_data; chunk.source_data = sim.source_data; end

    Khronos.run(sim; until = runtime)
    adj_dft_at_point = Array(mon2.monitor_data.fields)[1]

    sim.sources = saved
    Khronos.add_sources(sim, sim.sources)
    for chunk in sim.chunk_data; chunk.source_data = sim.source_data; end

    println("Adjoint source (amp=1) DFT at x=-1: $(adj_dft_at_point), |val|=$(abs(adj_dft_at_point))")
    println("Forward source (amp=1) DFT at x=1:  $(fwd_dft_at_point), |val|=$(abs(fwd_dft_at_point))")
    println("Reciprocity check: |fwd|/|adj| = $(abs(fwd_dft_at_point)/abs(adj_dft_at_point))")

    # ── Step 5: Derive the correct scale ──
    # For f = |E_obj|² at a single point, the adjoint gradient is:
    #   df/dρ = Re[ E_adj * E_fwd * dε_inv/dρ ]
    # where E_adj is the DFT field from the adjoint source with amplitude dJ*scale.
    #
    # We need E_adj at the design region to have the right relationship to dJ.
    # The correct relationship from the continuous adjoint is:
    #   E_adj(x,ω) = G(x,x_obj,ω) * dJ * scale_correct
    # where G is the Green's function.
    #
    # From the FD gradient: df/dρ_k = FD value
    # From the adjoint:     df/dρ_k = Re[ (dJ * scale * G(x_k, x_obj)) * E_fwd(x_k) * dε_inv/dρ_k ]
    #
    # The scale_correct satisfies: adjoint gradient = FD gradient
    # We measured that α_opt * adj_grad ≈ FD_grad, so scale_correct = α_opt * scale
    # And α_opt * |scale_noDV| ≈ 0.02 across resolutions.
    #
    # What should scale_correct be?
    # From the FDTD adjoint derivation for Khronos's source convention:
    #   E = ε_inv * (D + fSD)
    #   Ê(ω) = Σ_n E^n exp(-iωnΔt) = Σ_n ε_inv * (D^n + fSD^n) * exp(-iωnΔt)
    #
    # The source contribution to Ê at the source point:
    #   Ê_source(ω) = ε_inv * Σ_n fSD^n * exp(-iωnΔt)
    #                = ε_inv * spatial_amp * Σ_n real(eval_time_source(t_n)) * exp(-iωnΔt)
    #
    # The DTFT of real(f(t))*exp(-iωt) where f(t) = Gaussian * exp(-iω₀t) / (-2πf₀i):
    #   Σ_n real(f(t_n)) * exp(-iωnΔt)
    # = Σ_n (1/2)(f(t_n) + conj(f(t_n))) * exp(-iωnΔt)
    # = (1/2)[Σ f(t_n)exp(-iωnΔt) + Σ conj(f(t_n))exp(-iωnΔt)]
    # = (1/2)[DTFT{f}(ω) + DTFT{conj(f)}(ω)]
    # = (1/2)[DTFT{f}(ω) + conj(DTFT{f}(-ω))]

    # Let's compute this directly
    y_real = real.(y)
    dtft_real_at_fcen = sum(y_real .* exp.(-im .* 2π .* fcen .* t_vals))
    dtft_full_at_fcen = sum(y .* exp.(-im .* 2π .* fcen .* t_vals))
    println("\nDTFT analysis:")
    @printf("  DTFT of complex source at fcen: %+.6e %+.6ei\n", real(dtft_full_at_fcen), imag(dtft_full_at_fcen))
    @printf("  DTFT of real(source) at fcen:   %+.6e %+.6ei\n", real(dtft_real_at_fcen), imag(dtft_real_at_fcen))
    @printf("  Ratio |real/complex|: %.4f\n", abs(dtft_real_at_fcen) / abs(dtft_full_at_fcen))
    @printf("  fwd_dtft (with dt/√2π): %+.6e %+.6ei\n", real(fwd_dtft), imag(fwd_dtft))
    @printf("  Ratio |fwd_dtft / dtft_full| = dt/√2π = %.6e (dt=%.6e, dt/√2π=%.6e)\n",
        abs(fwd_dtft) / abs(dtft_full_at_fcen), dt, dt/sqrt(2π))

    # The `update_source!` kernel does `fSD += real(scalar_amp * spatial_amp)`
    # So the frequency-domain source is: DTFT{real(scalar_amp)} * spatial_amp
    # The adj_src_scale divides by fwd_dtft which includes dt/√2π weighting
    # But the actual DFT accumulation has no dt/√2π — it's just Σ E*exp(-iωt)
    #
    # So the mismatch factor is:
    # DTFT_with_normalization / DTFT_without_normalization = dt/√2π
    ratio_dtft = abs(fwd_dtft) / abs(dtft_real_at_fcen)
    @printf("\n  Key ratio: |fwd_dtft| / |dtft_real| = %.6e\n", ratio_dtft)
    @printf("  dt/√2π = %.6e\n", dt/sqrt(2π))

    # Expected correction to scale: divide by (dt/√2π)² because both
    # the source DTFT and the DFT monitor use different conventions?
    # Or maybe just 1/(dt/√2π) for one side?
    @printf("\n  If correction = 1/(dt/√2π) = %.4f → corrected |scale_noDV| = %.4f\n",
        sqrt(2π)/dt, abs(scale_noDV) * sqrt(2π)/dt)
    @printf("  If correction = (2π/dt) = %.4f → corrected |scale_noDV| = %.4f\n",
        2π/dt, abs(scale_noDV) * 2π/dt)
    @printf("  If correction = √2π/dt² = %.4f → corrected |scale_noDV| = %.4f\n",
        sqrt(2π)/dt^2, abs(scale_noDV) * sqrt(2π)/dt^2)

    # Target: α_opt * |scale_noDV| ≈ 0.02, and |scale_noDV| ≈ 1.49
    # So we need scale_correct ≈ 0.02, meaning we need to DIVIDE by ~75
    # Or equivalently, the current scale is 75x too large.
    println("\n  Target |scale_correct| ≈ 0.02")
    @printf("  Current |scale_noDV| = %.4f → need factor = %.4f\n",
        abs(scale_noDV), 0.02 / abs(scale_noDV))
end

derive_scale()
