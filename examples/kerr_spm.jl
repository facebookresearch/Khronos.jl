# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# χ3 Kerr nonlinearity validation: self-phase modulation (SPM).
#
# A Gaussian pulse propagates through a slab with χ3 nonlinearity.
# The nonlinear phase shift causes spectral broadening. We verify
# that the output spectrum is broader than the input (linear case).
#
# This validates:
#   - chi3 material parameter
#   - χ3 correction kernel in the E-field update
#   - Periodic BC (xy) + PML (z)

t_script_start = time()

import Khronos
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Nonlinear slab
ε_slab = 2.25       # n = 1.5
chi3_val = 0.1       # moderate χ3 (in simulation units)
slab_thickness = 3.0 # μm

# Source
λ_center = 1.0      # μm
freq_center = 1.0 / λ_center
fwidth = 2π * 0.15  # narrow bandwidth for clear SPM

# Monitor frequencies (wider than source to see broadening)
n_freqs = 101
freq_min = 0.5 * freq_center
freq_max = 1.5 * freq_center
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# ------------------------------------------------------------------- #
# Domain
# ------------------------------------------------------------------- #

cell_xy = 0.1
pml_thickness = 1.0
buffer = 1.0
domain_z = slab_thickness + 2 * buffer
cell_z = domain_z + 2 * pml_thickness

resolution = 40

source_z = -slab_thickness / 2 - buffer / 2
monitor_z = slab_thickness / 2 + buffer / 2

# ------------------------------------------------------------------- #
# Geometry: slab with χ3
# ------------------------------------------------------------------- #

geometry_nonlinear = [
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_xy + 1.0, cell_xy + 1.0, slab_thickness]),
        Khronos.Material(ε = ε_slab, chi3 = chi3_val),
    ),
]

geometry_linear = [
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_xy + 1.0, cell_xy + 1.0, slab_thickness]),
        Khronos.Material(ε = ε_slab),
    ),
]

# ------------------------------------------------------------------- #
# Sources and monitors (shared between both runs)
# ------------------------------------------------------------------- #

function make_source()
    [Khronos.PlaneWaveSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        center = [0.0, 0.0, source_z],
        size = [cell_xy, cell_xy, 0.0],
        k_vector = [0.0, 0.0, 1.0],
        polarization_angle = 0.0,
        amplitude = 5.0,  # high amplitude to trigger nonlinear effects
    )]
end

function make_monitor()
    Khronos.Monitor[
        Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0.0, 0.0, monitor_z],
            size = [0.0, 0.0, 0.0],
            frequencies = monitor_freqs,
        ),
    ]
end

boundaries = [
    [0.0, 0.0],
    [0.0, 0.0],
    [pml_thickness, pml_thickness],
]

boundary_conditions = [
    [Khronos.Periodic(), Khronos.Periodic()],
    [Khronos.Periodic(), Khronos.Periodic()],
    [Khronos.PML(), Khronos.PML()],
]

stop_cond() = Khronos.stop_when_dft_decayed(
    tolerance = 1e-6,
    minimum_runtime = 30.0,
    maximum_runtime = 300.0,
)

# ------------------------------------------------------------------- #
# Run linear simulation
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("χ3 Kerr Nonlinearity Validation: Self-Phase Modulation")
println("Slab: ε=$(ε_slab), χ3=$(chi3_val), t=$(slab_thickness) μm")
println("=" ^ 60)

monitors_linear = make_monitor()
sim_linear = Khronos.Simulation(
    cell_size = [cell_xy, cell_xy, cell_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution = resolution,
    geometry = geometry_linear,
    sources = make_source(),
    monitors = monitors_linear,
    boundaries = boundaries,
    boundary_conditions = boundary_conditions,
)

println("\n--- Linear simulation ---")
Khronos.run(sim_linear, until_after_sources = stop_cond())

# ------------------------------------------------------------------- #
# Run nonlinear simulation
# ------------------------------------------------------------------- #

monitors_nonlinear = make_monitor()
sim_nonlinear = Khronos.Simulation(
    cell_size = [cell_xy, cell_xy, cell_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution = resolution,
    geometry = geometry_nonlinear,
    sources = make_source(),
    monitors = monitors_nonlinear,
    boundaries = boundaries,
    boundary_conditions = boundary_conditions,
)

println("\n--- Nonlinear simulation (χ3=$(chi3_val)) ---")
Khronos.run(sim_nonlinear, until_after_sources = stop_cond())

# ------------------------------------------------------------------- #
# Compare spectra
# ------------------------------------------------------------------- #

dft_linear = vec(abs.(Array(Khronos.get_dft_fields(monitors_linear[1]))))
dft_nonlinear = vec(abs.(Array(Khronos.get_dft_fields(monitors_nonlinear[1]))))

# Compute spectral width (RMS bandwidth)
function rms_bandwidth(spectrum, freqs)
    power = abs2.(spectrum)
    total = sum(power)
    total == 0 && return 0.0
    f_mean = sum(power .* freqs) / total
    f_rms = sqrt(sum(power .* (freqs .- f_mean).^2) / total)
    return f_rms
end

bw_linear = rms_bandwidth(dft_linear[1:min(end,n_freqs)], monitor_freqs[1:min(length(dft_linear),n_freqs)])
bw_nonlinear = rms_bandwidth(dft_nonlinear[1:min(end,n_freqs)], monitor_freqs[1:min(length(dft_nonlinear),n_freqs)])

println("\n" * "=" ^ 60)
println("RESULTS")
println("=" ^ 60)
println("  Linear RMS bandwidth:    $(round(bw_linear, digits=6))")
println("  Nonlinear RMS bandwidth: $(round(bw_nonlinear, digits=6))")
println("  Broadening ratio:        $(round(bw_nonlinear / bw_linear, digits=3))")

if bw_nonlinear > bw_linear * 1.01
    println("  ✓ PASS: Nonlinear spectrum is broader (SPM detected)")
else
    println("  ✗ FAIL: No spectral broadening detected")
end

t_script_total = time() - t_script_start
println("\nScript wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
