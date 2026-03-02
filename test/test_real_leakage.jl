#!/usr/bin/env julia
# Compute F(ω₀) and G(ω₀) to quantify the real() leakage
# No simulation needed — just analyze the time-domain source waveform
using Printf

fcen = 0.5; fwidth = 0.15; resolution = 20
dt = 0.5 / resolution  # Courant factor
T = 80.0
n_steps = floor(Int, T/dt)
t_vals = collect(0:n_steps-1) .* dt

# Gaussian pulse dipole: Gaussian(t) * exp(-iω₀t) / (-2πif₀)
width = 1.0 / fwidth
peak_time = 5.0 * width  # cutoff=5.0
amp = 1.0 / (-2π * im * fcen)

y = zeros(ComplexF64, n_steps)
for (i, t) in enumerate(t_vals)
    tt = t - peak_time
    if abs(tt) < 5.0 * width
        y[i] = exp(-tt^2 / (2*width^2)) * exp(-2π*im*fcen*tt) * amp
    end
end

ω₀ = 2π * fcen
# F(ω₀) = Σ dipole(t_n) * exp(-iω₀t_n) — "intended" frequency component
F = sum(y .* exp.(-im .* ω₀ .* t_vals))
# G(ω₀) = Σ conj(dipole(t_n)) * exp(-iω₀t_n) — "conjugate leakage"
G = sum(conj.(y) .* exp.(-im .* ω₀ .* t_vals))

println("=== real() leakage analysis ===")
@printf("F(ω₀) = %+.6e %+.6ei,  |F| = %.6e\n", real(F), imag(F), abs(F))
@printf("G(ω₀) = %+.6e %+.6ei,  |G| = %.6e\n", real(G), imag(G), abs(G))
@printf("|G|/|F| = %.6f\n", abs(G)/abs(F))

# The DFT of real(dipole * A) at ω₀ is: (A*F + conj(A)*G) / 2
# For the adjoint to work correctly, we need:
#   (A*F + conj(A)*G) / 2 = target
# If G ≈ 0, then A = 2*target/F (the standard formula)
# If G ≠ 0, the solution is more complex

# Check: what fraction of the DFT response comes from G?
println("\nFor unit real A=1:")
eff_real = (F + G) / 2
@printf("  DFT = (F+G)/2 = %+.6e %+.6ei, |val| = %.6e\n", real(eff_real), imag(eff_real), abs(eff_real))

println("For unit imaginary A=i:")
eff_imag = im * (F - G) / 2
@printf("  DFT = i(F-G)/2 = %+.6e %+.6ei, |val| = %.6e\n", real(eff_imag), imag(eff_imag), abs(eff_imag))

ratio = abs(eff_real) / abs(eff_imag)
@printf("\n|DFT(real A)| / |DFT(imag A)| = %.4f\n", ratio)
@printf("If this ratio ≠ 1, then the adjoint response depends on the\n")
@printf("complex phase of dJ*scale, causing non-constant per-parameter ratios.\n")

# Also compute using dt/√2π normalization (matching Khronos's fwd_dtft)
fwd_dtft = sum(dt / sqrt(2π) .* exp.(im .* 2π .* fcen .* t_vals) .* y)
@printf("\nfwd_dtft (dipole, +iωt) = %+.6e %+.6ei, |val| = %.6e\n",
    real(fwd_dtft), imag(fwd_dtft), abs(fwd_dtft))

# The relationship:
# F = Σ y * exp(-iωt) is the -iωt convention
# fwd_dtft = Σ (dt/√2π) * y * exp(+iωt) is the +iωt convention with normalization
# F = N/dt * √2π * conj(fwd_dtft) approximately
@printf("F / (N*√2π*conj(fwd_dtft)/dt) ≈ 1? ratio = %.6f\n",
    abs(F) / (n_steps * sqrt(2π) * abs(fwd_dtft) / dt))
