# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# 2×2 MMI Coupler for 90-Degree Optical Hybrid
#
# Adapted from the Tidy3D 90OpticalHybrid tutorial.
# The full optical hybrid device uses STL geometry files which Khronos
# does not currently support. This example focuses on the standalone
# 2×2 MMI coupler sub-simulation, built from native Cuboid shapes.
#
# Demonstrates:
#   - MMI coupler design using Cuboid geometry primitives
#   - ModeMonitor for through/cross port amplitude extraction
#   - Phase difference analysis between output ports
#
# Physics: A 2×2 multimode interferometer (MMI) splits input power
# equally between two output ports with a 90-degree phase difference.
# This property is the building block of 90-degree optical hybrids
# used in coherent optical receivers.
#
# Reference: Hang Guan et al., "Compact and low loss 90 degree optical
# hybrid on a silicon-on-insulator platform,"
# Opt. Express 25, 28957-28968 (2017).

t_script_start = time()

import Khronos
using GeometryPrimitives
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Materials (non-dispersive at C-band ~1.55 μm)
n_Si = 3.47
n_SiO2 = 1.45
ε_Si = n_Si^2
ε_SiO2 = n_SiO2^2
mat_si = Khronos.Material(ε = ε_Si)
mat_sio2 = Khronos.Material(ε = ε_SiO2)

# Waveguide dimensions
wg_width = 0.5    # μm — single-mode waveguide width
thickness = 0.22  # μm — SOI waveguide thickness

# MMI dimensions (typical for 2×2 MMI on SOI at 1.55 μm)
# The MMI width and length are designed for equal power splitting
# with a 90° phase difference between through and cross ports.
mmi_width = 2.8    # μm — multimode section width
mmi_length = 5.4   # μm — multimode section length
y_port = 0.45      # μm — port offset from MMI center

# Access waveguide lengths (input/output straight sections)
L_access = 4.0     # μm

# Wavelength range (C-band)
λ_min = 1.53    # μm
λ_max = 1.56    # μm
λ_center = 1.55 # μm
freq_center = 1.0 / λ_center
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
fwidth = 2π * 0.5 * (freq_max - freq_min)

# Monitor frequencies
n_freqs = 51
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))
wavelengths = 1.0 ./ monitor_freqs

# ------------------------------------------------------------------- #
# Geometry: 2×2 MMI from Cuboid primitives
# ------------------------------------------------------------------- #
#
# Layout (top view at z=0):
#
#   Input upper  ──────┐              ┌────── Output upper (through)
#                      │   MMI body   │
#   Input lower  ──────┘              └────── Output lower (cross)
#
# The MMI body is a wide rectangular waveguide. The input and output
# waveguides connect at y = ±y_port from the MMI center.

# Positions
mmi_center_x = 0.0
input_x = mmi_center_x - mmi_length / 2 - L_access / 2
output_x = mmi_center_x + mmi_length / 2 + L_access / 2

geometry = [
    # MMI multimode section (wide Si rectangle)
    Khronos.Object(
        Cuboid([mmi_center_x, 0.0, 0.0], [mmi_length, mmi_width, thickness]),
        mat_si,
    ),
    # Input waveguide — upper port (excited)
    Khronos.Object(
        Cuboid([input_x, y_port, 0.0], [L_access, wg_width, thickness]),
        mat_si,
    ),
    # Input waveguide — lower port (not excited in this simulation)
    Khronos.Object(
        Cuboid([input_x, -y_port, 0.0], [L_access, wg_width, thickness]),
        mat_si,
    ),
    # Output waveguide — upper port (through)
    Khronos.Object(
        Cuboid([output_x, y_port, 0.0], [L_access, wg_width, thickness]),
        mat_si,
    ),
    # Output waveguide — lower port (cross)
    Khronos.Object(
        Cuboid([output_x, -y_port, 0.0], [L_access, wg_width, thickness]),
        mat_si,
    ),
    # Background cladding (SiO2) — covers entire domain
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        mat_sio2,
    ),
]

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

# Domain encompassing the MMI + access waveguides with margin
domain_x = mmi_length + 2 * L_access + 2 * λ_center  # ~16.5 μm
domain_y = mmi_width + 2 * λ_center                   # ~5.9 μm
domain_z = 10 * thickness                              # 2.2 μm

pml_thickness = 1.0
resolution = 30  # higher resolution for accurate MMI phase measurement

cell_x = domain_x + 2 * pml_thickness
cell_y = domain_y + 2 * pml_thickness
cell_z = domain_z + 2 * pml_thickness

cell_center = [0.0, 0.0, 0.0]

# ------------------------------------------------------------------- #
# Source: TE0 mode at upper input port
# ------------------------------------------------------------------- #

source_x = -(mmi_length / 2 + L_access - λ_center / 2)
source_center = [source_x, y_port, 0.0]
source_size = [0.0, 2 * wg_width, 6 * thickness]

sources = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = 1,  # TE0 (fundamental mode)
        center = source_center,
        size = source_size,
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

# ------------------------------------------------------------------- #
# Monitors
# ------------------------------------------------------------------- #

monitor_x = mmi_length / 2 + L_access - λ_center / 2

# Through port monitor (upper output, same y as input)
through_monitor = Khronos.ModeMonitor(
    center = [monitor_x, y_port, 0.0],
    size = [0.0, 2 * wg_width, 6 * thickness],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Cross port monitor (lower output)
cross_monitor = Khronos.ModeMonitor(
    center = [monitor_x, -y_port, 0.0],
    size = [0.0, 2 * wg_width, 6 * thickness],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Reference monitor (on input waveguide, between source and MMI)
ref_monitor = Khronos.ModeMonitor(
    center = [source_x + 1.0, y_port, 0.0],
    size = source_size,
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Field monitor at z=0
field_monitor = Khronos.DFTMonitor(
    component = Khronos.Ey(),
    center = [0.0, 0.0, 0.0],
    size = [domain_x, domain_y, 0.0],
    frequencies = [freq_center],
)

monitors = Khronos.Monitor[through_monitor, cross_monitor, ref_monitor, field_monitor]

# ------------------------------------------------------------------- #
# Boundaries: PML on all sides
# ------------------------------------------------------------------- #

boundaries = [
    [pml_thickness, pml_thickness],  # x
    [pml_thickness, pml_thickness],  # y
    [pml_thickness, pml_thickness],  # z
]

# ------------------------------------------------------------------- #
# Simulation
# ------------------------------------------------------------------- #

sim = Khronos.Simulation(
    cell_size = [cell_x, cell_y, cell_z],
    cell_center = cell_center,
    resolution = resolution,
    geometry = geometry,
    sources = sources,
    monitors = monitors,
    boundaries = boundaries,
    symmetry = (0, 0, 1),  # TE z-symmetry
)

println("=" ^ 60)
println("2×2 MMI Coupler (Optical Hybrid Building Block)")
println("=" ^ 60)
println("MMI dimensions: $(mmi_length) × $(mmi_width) × $(thickness) μm")
println("Waveguide width: $(wg_width) μm")
println("Port offset: ±$(y_port) μm from center")
println("Wavelength range: $(λ_min)–$(λ_max) μm (C-band)")
println("Domain: $(domain_x) × $(domain_y) × $(domain_z) μm")
println("Cell (incl. PML): $(cell_x) × $(cell_y) × $(cell_z) μm")
println("Resolution: $(resolution) pts/μm")
println("=" ^ 60)

# ------------------------------------------------------------------- #
# Run simulation
# ------------------------------------------------------------------- #

t_prep_start = time()
Khronos.prepare_simulation!(sim)
t_prep = time() - t_prep_start

num_voxels = sim.Nx * sim.Ny * sim.Nz
println("\nGrid: $(sim.Nx) × $(sim.Ny) × $(sim.Nz) = $(num_voxels) voxels")
println("Preparation time: $(round(t_prep, digits=2))s")

t_run_start = time()
step_start = sim.timestep

Khronos.run(sim,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-6,
        minimum_runtime = 300.0,
        maximum_runtime = 800.0,
    ),
)

t_run = time() - t_run_start
total_steps = sim.timestep - step_start

println("\n", "=" ^ 60)
println("PERFORMANCE METRICS")
println("=" ^ 60)
println("  Grid size:       $(sim.Nx) × $(sim.Ny) × $(sim.Nz)")
println("  Total voxels:    $(num_voxels) ($(round(num_voxels / 1e6, digits=2))M)")
println("  Timesteps:       $(total_steps)")
println("  Prep time:       $(round(t_prep, digits=2))s")
println("  Run time:        $(round(t_run, digits=2))s")
if t_run > 0 && total_steps > 0
    mcells_per_s = num_voxels * total_steps / t_run / 1e6
    println("  Throughput:      $(round(mcells_per_s, digits=1)) MVoxels/s")
end
println("=" ^ 60)

# ------------------------------------------------------------------- #
# Post-processing: insertion loss and phase difference
# ------------------------------------------------------------------- #

println("\nPost-processing...")

through_data = through_monitor.monitor_data
cross_data = cross_monitor.monitor_data
ref_data = ref_monitor.monitor_data

through_a_plus, _ = Khronos.compute_mode_amplitudes(through_data)
cross_a_plus, _ = Khronos.compute_mode_amplitudes(cross_data)
ref_a_plus, _ = Khronos.compute_mode_amplitudes(ref_data)

# Normalize by reference
# compute_mode_amplitudes returns 1D vectors indexed by frequency
ref_amp = abs.(ref_a_plus)
ref_threshold = 0.05 * maximum(ref_amp)
reliable = ref_amp .> ref_threshold

# Complex transmission coefficients (normalized)
t_through = fill(NaN + 0im, n_freqs)
t_cross = fill(NaN + 0im, n_freqs)
t_through[reliable] .= through_a_plus[reliable] ./ ref_a_plus[reliable]
t_cross[reliable] .= cross_a_plus[reliable] ./ ref_a_plus[reliable]

# Insertion loss at each port
IL_through = fill(NaN, n_freqs)
IL_cross = fill(NaN, n_freqs)
IL_through[reliable] .= -10 .* log10.(abs2.(t_through[reliable]))
IL_cross[reliable] .= -10 .* log10.(abs2.(t_cross[reliable]))

# Phase difference (should be ~90°)
phase_through = angle.(t_through)
phase_cross = angle.(t_cross)
delta_phase = fill(NaN, n_freqs)
delta_phase[reliable] .= mod.(
    (phase_through[reliable] .- phase_cross[reliable]) .* (180 / π),
    360.0
)

println("\nResults at λ=$(λ_center) μm:")
center_idx = argmin(abs.(wavelengths .- λ_center))
println("  Through port IL: $(round(IL_through[center_idx], digits=3)) dB")
println("  Cross port IL:   $(round(IL_cross[center_idx], digits=3)) dB")
println("  Phase difference: $(round(delta_phase[center_idx], digits=2))°")
println("  Ideal: 3.0 dB insertion loss at each port, 90° phase difference")

# ------------------------------------------------------------------- #
# Plot 1: Insertion loss
# ------------------------------------------------------------------- #

fig_loss = Figure(size = (800, 400))
ax_loss = Axis(fig_loss[1, 1],
    xlabel = "Wavelength (μm)",
    ylabel = "Insertion loss (dB)",
    title = "2×2 MMI Coupler: Insertion Loss",
    xlabelsize = 14,
    ylabelsize = 14,
)

lines!(ax_loss, wavelengths, IL_through,
    color = :blue, linewidth = 2, label = "Through port")
lines!(ax_loss, wavelengths, IL_cross,
    color = :red, linewidth = 2, label = "Cross port")
hlines!(ax_loss, [3.01], color = :gray, linestyle = :dash,
    linewidth = 1, label = "Ideal (3 dB)")

axislegend(ax_loss, position = :rt)
ylims!(ax_loss, 2.5, 4.0)

save("mmi_insertion_loss.png", fig_loss)
println("Saved: mmi_insertion_loss.png")

# ------------------------------------------------------------------- #
# Plot 2: Phase difference
# ------------------------------------------------------------------- #

fig_phase = Figure(size = (800, 400))
ax_phase = Axis(fig_phase[1, 1],
    xlabel = "Wavelength (μm)",
    ylabel = "Phase difference (degrees)",
    title = "2×2 MMI Coupler: Through − Cross Phase Difference",
    xlabelsize = 14,
    ylabelsize = 14,
)

lines!(ax_phase, wavelengths, delta_phase,
    color = :blue, linewidth = 2)
hlines!(ax_phase, [90.0], color = :gray, linestyle = :dash, linewidth = 1)

ylims!(ax_phase, 85, 95)

save("mmi_phase_difference.png", fig_phase)
println("Saved: mmi_phase_difference.png")

# ------------------------------------------------------------------- #
# Plot 3: Field intensity at z=0
# ------------------------------------------------------------------- #

field_dft_raw = Khronos.get_dft_fields(field_monitor)
field_dft = Array(field_dft_raw[:, :, :, 1])

if size(field_dft, 3) == 1
    field_2d = field_dft[:, :, 1]
elseif size(field_dft, 1) == 1
    field_2d = field_dft[1, :, :]
else
    field_2d = field_dft[:, :, 1]
end

field_intensity = abs2.(field_2d)

mon_cx, mon_cy = field_monitor.center[1], field_monitor.center[2]
mon_sx, mon_sy = field_monitor.size[1], field_monitor.size[2]
nfx, nfy = size(field_intensity)
fx_range = range(mon_cx - mon_sx / 2, mon_cx + mon_sx / 2, length = nfx)
fy_range = range(mon_cy - mon_sy / 2, mon_cy + mon_sy / 2, length = nfy)

fig_field = Figure(size = (900, 400))
ax_field = Axis(fig_field[1, 1],
    title = "|Ey|² at z=0, λ=$(λ_center) μm",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    aspect = DataAspect(),
)

imax = maximum(field_intensity)
imax = imax > 0 ? imax : 1.0
heatmap!(ax_field, collect(fx_range), collect(fy_range), Float32.(field_intensity),
    colormap = :inferno,
    colorrange = (0.0, imax),
)

save("mmi_field_intensity.png", fig_field)
println("Saved: mmi_field_intensity.png")

# ------------------------------------------------------------------- #
# Summary
# ------------------------------------------------------------------- #

println("\n", "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Output files:")
println("  mmi_insertion_loss.png    — Through/cross port insertion loss")
println("  mmi_phase_difference.png  — Phase difference (target: 90°)")
println("  mmi_field_intensity.png   — |Ey|² field at z=0")
println("")
println("Note: This simulates the standalone 2×2 MMI coupler from the")
println("90-degree optical hybrid tutorial. The full optical hybrid device")
println("(Y-branch + 3 MMIs + 4 bends) requires STL geometry import,")
println("which is not yet supported in Khronos.")
t_script_total = time() - t_script_start
println("  Script wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
