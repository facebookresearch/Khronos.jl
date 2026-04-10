# Diagnostic: straight waveguide mode monitor test
#
# A straight waveguide with ModeSource + two ModeMonitors.
# The ratio |a+_through / a+_ref|² should be ≈ 1.0 if mode overlap is correct.

import Khronos
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

n_Si = 3.47
n_SiO2 = 1.44
wg_width = 0.5
wg_height = 0.22

λ_center = 1.55
freq_center = 1.0 / λ_center
fwidth = 0.05 * freq_center  # narrow bandwidth for quick convergence

# Few frequency points
n_freqs = 5
freq_min = freq_center - fwidth
freq_max = freq_center + fwidth
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# Small domain: just enough for waveguide + source + monitors
domain_x = 6.0   # μm
domain_y = 3.0    # 6 × wg_width
domain_z = 2.0    # ~9 × wg_height
pml_thickness = 0.5

geometry = [
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [domain_x + 4.0, wg_width, wg_height]),
        Khronos.Material(ε = n_Si^2),
    ),
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        Khronos.Material(ε = n_SiO2^2),
    ),
]

source_x = -2.0
sources = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(fcen = freq_center, fwidth = fwidth),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = 1,
        center = [source_x, 0.0, 0.0],
        size = [0.0, 6 * wg_width, 6 * wg_height],
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

ref_monitor = Khronos.ModeMonitor(
    center = [source_x + 0.5, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

through_monitor = Khronos.ModeMonitor(
    center = [2.0, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

monitors = Khronos.Monitor[ref_monitor, through_monitor]

sim = Khronos.Simulation(
    cell_size = [domain_x, domain_y, domain_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 25,
    geometry = geometry,
    sources = sources,
    monitors = monitors,
    boundaries = [[pml_thickness, pml_thickness], [pml_thickness, pml_thickness], [pml_thickness, pml_thickness]],
)

println("=" ^ 60)
println("Straight waveguide mode monitor diagnostic")
println("=" ^ 60)

Khronos.run(sim, until_after_sources = 50.0)

println("\nComputing mode amplitudes...")

ref_a_plus, ref_a_minus = Khronos.compute_mode_amplitudes(ref_monitor.monitor_data)
through_a_plus, through_a_minus = Khronos.compute_mode_amplitudes(through_monitor.monitor_data)

println("\nRaw mode amplitudes:")
for i in 1:n_freqs
    λ = 1.0 / monitor_freqs[i]
    println("  λ=$(round(λ, digits=4)) μm:")
    println("    |ref_a+|   = $(abs(ref_a_plus[i]))")
    println("    |thru_a+|  = $(abs(through_a_plus[i]))")
    println("    ratio      = $(abs(through_a_plus[i]) / abs(ref_a_plus[i]))")
    println("    |ratio|²   = $(abs2(through_a_plus[i] / ref_a_plus[i]))")
    println("    phase diff = $(angle(through_a_plus[i]) - angle(ref_a_plus[i])) rad")
end

println("\nExpected: |ratio|² ≈ 1.0 (straight waveguide, no coupling)")
println("=" ^ 60)
