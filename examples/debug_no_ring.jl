# Diagnostic: ring coupler geometry WITHOUT ring
# If |t|² ≈ 1, the issue is ring physics; if |t|² ≈ 0.1, it's normalization.

import Khronos
using GeometryPrimitives
using Printf

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

n_Si = 3.47
n_SiO2 = 1.44
wg_width = 0.5
wg_height = 0.22
ring_radius = 5.0
coupling_gap = 0.05
ring_center_y = wg_width + coupling_gap + ring_radius

λ_center = 1.55
freq_center = 1.0 / λ_center
freq_min = 1.0 / 1.6
freq_max = 1.0 / 1.5
fwidth = 2π * 0.5 * (freq_max - freq_min)

n_freqs = 11
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

domain_x = 2 * ring_radius + 2 * λ_center
domain_y = ring_radius / 2 + coupling_gap + 2 * wg_width + λ_center
domain_z = 9 * wg_height
pml_thickness = 1.0

# Straight waveguide ONLY — no ring
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

source_x = -(ring_radius + λ_center / 4)
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

through_x = ring_radius + λ_center / 4
ref_x = source_x + 1.0

ref_monitor = Khronos.ModeMonitor(
    center = [ref_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(num_modes = 1, geometry = geometry, mode_solver_resolution = 50),
)

through_monitor = Khronos.ModeMonitor(
    center = [through_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(num_modes = 1, geometry = geometry, mode_solver_resolution = 50),
)

monitors = Khronos.Monitor[ref_monitor, through_monitor]

sim = Khronos.Simulation(
    # PML is INSIDE cell_size in Khronos — add margins so waveguide
    # has adequate clearance from PML inner edge
    cell_size = [domain_x + 2 * pml_thickness, domain_y + 2 * pml_thickness, domain_z + 2 * pml_thickness],
    cell_center = [0.0, domain_y / 4, 0.0],
    resolution = 25,
    geometry = geometry,
    sources = sources,
    monitors = monitors,
    boundaries = [
        [pml_thickness, pml_thickness],
        [pml_thickness, pml_thickness],
        [pml_thickness, pml_thickness],
    ],
)

println("=" ^ 60)
println("No-ring diagnostic: straight waveguide only")
println("=" ^ 60)
println("Domain: $(domain_x) × $(domain_y) × $(domain_z)")
println("Source at x=$source_x, ref at x=$ref_x, through at x=$through_x")

Khronos.run(sim, until_after_sources = Khronos.stop_when_dft_decayed(
    tolerance=1e-6, minimum_runtime=100.0, maximum_runtime=300.0))

println("\nComputing mode amplitudes (ref monitor)...")
ref_a_plus, _ = Khronos.compute_mode_amplitudes(ref_monitor.monitor_data, verbose=true)
println("\nComputing mode amplitudes (through monitor)...")
through_a_plus, _ = Khronos.compute_mode_amplitudes(through_monitor.monitor_data, verbose=true)

ref_amp = abs.(ref_a_plus)
through_amp = abs.(through_a_plus)
T = abs2.(through_a_plus ./ ref_a_plus)
wavelengths = 1.0 ./ monitor_freqs

println("\n" * "-" ^ 60)
println("  freq_idx | λ (μm)  | |ref_a+|  | |thru_a+| | |t|²")
println("-" ^ 60)
for i in 1:n_freqs
    @printf("  %3d      | %.4f  | %.6f | %.6f  | %.4f\n",
        i, wavelengths[i], ref_amp[i], through_amp[i], T[i])
end
println("-" ^ 60)
println("\nExpected: |t|² ≈ 1.0 everywhere (straight waveguide, no ring)")
println("=" ^ 60)
