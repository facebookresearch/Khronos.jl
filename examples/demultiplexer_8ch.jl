# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# 8-Channel Mode and Polarization De-multiplexer
#
# Ported from the Tidy3D 8ChannelDemultiplexer tutorial.
# Demonstrates:
#   - GDSII file import using read_gds / flatten_cell / gds_polygons_to_objects
#   - Large-scale photonic device simulation (~200 μm)
#   - ModeMonitor for higher-order mode extraction
#   - DFTMonitor for field visualization
#
# Physics: An 8-channel mode/polarization demultiplexer uses asymmetric
# directional couplers to selectively convert input TE0/TM0 modes from
# access waveguides into higher-order modes (TE1-TE3, TM1-TM3) in a
# central bus waveguide. Each coupler's bus width is designed to satisfy
# the phase-matching condition with the access waveguide.
#
# Reference: Wang, J., He, S. and Dai, D. (2014),
# "On-chip silicon 8-channel hybrid (de)multiplexer enabling simultaneous
# mode- and polarization-division-multiplexing,"
# Laser & Photonics Reviews, 8: L18-L22.

t_script_start = time()

import Khronos
using GeometryPrimitives
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Materials (constant refractive indices at ~1.55 μm)
n_Si = 3.48
n_SiO2 = 1.44
ε_Si = n_Si^2
ε_SiO2 = n_SiO2^2
mat_si = Khronos.Material(ε = ε_Si)
mat_sio2 = Khronos.Material(ε = ε_SiO2)

# Waveguide thickness
h = 0.22  # μm (standard 220 nm SOI)

# Wavelength range
λ_min = 1.50   # μm
λ_max = 1.60   # μm
λ_center = 1.55  # μm
freq_center = 1.0 / λ_center
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
fwidth = 2π * 0.5 * (freq_max - freq_min)

# Frequency monitor points
n_freqs = 51
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))
wavelengths = 1.0 ./ monitor_freqs

# ------------------------------------------------------------------- #
# Import device geometry from GDS
# ------------------------------------------------------------------- #

gds_path = joinpath(@__DIR__, "..", "..", "tidy3d-notebooks", "misc", "8ChannelDemultiplexer.gds")
if !isfile(gds_path)
    error("GDS file not found: $gds_path\n" *
          "Please ensure tidy3d-notebooks/misc/8ChannelDemultiplexer.gds is available.")
end

println("Reading GDS file: $gds_path")
lib = Khronos.read_gds(gds_path)
println("  Library: $(lib.name)")
println("  Cells: $(Khronos.get_cell_names(lib))")

cell_name = first(Khronos.get_cell_names(lib))
println("  Using cell: $cell_name")

# Flatten all polygons on layer 0
polys = Khronos.flatten_cell(lib, cell_name, 0, 0)
println("  Flattened polygons: $(length(polys))")

# Convert to Khronos Objects: extrude from z = -h/2 to z = +h/2
# (center the waveguide at z=0 for symmetry)
demux_objects = Khronos.gds_polygons_to_objects(polys, -h / 2, h / 2, mat_si; axis=3)
println("  Created $(length(demux_objects)) prism objects")

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

# Domain covering the full GDS device extent with margin
# GDS bounding box: x=[-195.2, 6.8], y=[-17.9, 17.4]
# Add ~1 wavelength margin on each side
domain_x = 204.0
domain_y = 38.0   # full device height (was 20.0)
domain_z = 10 * h  # 2.2 μm

pml_thickness = 1.0  # μm on each side
resolution = 20  # grid points per μm

# Cell includes PML
cell_x = domain_x + 2 * pml_thickness
cell_y = domain_y + 2 * pml_thickness
cell_z = domain_z + 2 * pml_thickness

cell_center = [-94.2, -0.2, 0.0]

# ------------------------------------------------------------------- #
# Geometry: device objects + background cladding
# ------------------------------------------------------------------- #

# Background cladding (SiO2) — large box covering entire domain
# Listed last so device objects take priority
background = Khronos.Object(
    Cuboid([cell_center[1], cell_center[2], 0.0],
           [cell_x + 10.0, cell_y + 10.0, cell_z + 10.0]),
    mat_sio2,
)

geometry = vcat(demux_objects, [background])

# ------------------------------------------------------------------- #
# Source: TM0 mode at I7 port (bottom-most access waveguide, Poly 2)
# ------------------------------------------------------------------- #
# The I7 port is the bottom TM input access waveguide.
# From GDS: Poly 2 straight section at y_center=-17.68, width=0.4 μm.
# TM0 excitation should convert to a higher-order TM mode in the bus.

source_center = [-180.0, -17.68, 0.0]
source_size = [0.0, 2.5, 8 * h]  # (0, 2.5, 1.76) — YZ cross-section

sources = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = 1,  # fundamental mode (TM0 with anti-symmetry)
        center = source_center,
        size = source_size,
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

# ------------------------------------------------------------------- #
# Monitors
# ------------------------------------------------------------------- #

# Bus waveguide mode monitors: one per mode index (1-4) for mode-resolved analysis
# Each monitor computes overlap with a single mode profile
bus_mode_monitors = [
    Khronos.ModeMonitor(
        center = [5.5, 0.0, 0.0],
        size = [0.0, 4.0, 8 * h],
        frequencies = monitor_freqs,
        mode_spec = Khronos.ModeSpec(
            num_modes = m,  # mode index
            target_neff = Float64(n_Si),
            geometry = geometry,
            mode_solver_resolution = 50,
        ),
    ) for m in 1:4
]

# Field monitor at z=0 for visualization
field_monitor = Khronos.DFTMonitor(
    component = Khronos.Ey(),
    center = [cell_center[1], cell_center[2], 0.0],
    size = [domain_x, domain_y, 0.0],
    frequencies = [freq_center],
)

monitors = Khronos.Monitor[bus_mode_monitors..., field_monitor]

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
    symmetry = (0, 0, 0),  # no symmetry (device not symmetric in z for coupled modes)
)

# ------------------------------------------------------------------- #
# Run simulation
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("8-Channel Mode/Polarization De-multiplexer")
println("=" ^ 60)
println("Device length: ~200 μm")
println("Waveguide thickness: $(h) μm (Si)")
println("Wavelength range: $(λ_min)–$(λ_max) μm")
println("Domain: $(domain_x) × $(domain_y) × $(domain_z) μm")
println("Cell (incl. PML): $(cell_x) × $(cell_y) × $(cell_z) μm")
println("Resolution: $(resolution) pts/μm")
println("Input: TM0 at I7 port → expected TM2 in bus")
println("=" ^ 60)

# Prepare
t_prep_start = time()
Khronos.prepare_simulation!(sim)
t_prep = time() - t_prep_start

num_voxels = sim.Nx * sim.Ny * sim.Nz
println("\nGrid: $(sim.Nx) × $(sim.Ny) × $(sim.Nz) = $(num_voxels) voxels")
println("Δx = $(round(sim.Δx, sigdigits=4)) μm")
println("Preparation time: $(round(t_prep, digits=2))s")

# Run with auto-shutoff
t_run_start = time()
step_start = sim.timestep

Khronos.run(sim,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-5,
        minimum_runtime = 1500.0,
        maximum_runtime = 4000.0,
    ),
)

t_run = time() - t_run_start
total_steps = sim.timestep - step_start

# ------------------------------------------------------------------- #
# Performance metrics
# ------------------------------------------------------------------- #

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
# Post-processing: mode amplitudes at bus waveguide
# ------------------------------------------------------------------- #

println("\n", "=" ^ 60)
println("Post-processing: mode composition at bus output")
println("=" ^ 60)

# Compute mode amplitudes for each mode index separately
# Each monitor returns a 1D vector indexed by frequency
mode_a_plus = [Khronos.compute_mode_amplitudes(bus_mode_monitors[m].monitor_data)[1] for m in 1:4]

# Compute transmission for each mode (|a+|²)
T_modes = [abs2.(mode_a_plus[m]) for m in 1:4]

println("\nMode transmission at λ=$(λ_center) μm:")
center_idx = argmin(abs.(wavelengths .- λ_center))
for m in 1:4
    println("  Mode $m: T = $(round(T_modes[m][center_idx], digits=4))")
end

# ------------------------------------------------------------------- #
# Plot: mode-resolved transmission spectra
# ------------------------------------------------------------------- #

println("\nPlotting mode transmission spectra...")

fig_modes = Figure(size = (800, 500))
ax_modes = Axis(fig_modes[1, 1],
    xlabel = "Wavelength (μm)",
    ylabel = "Transmission to bus waveguide",
    title = "8-Channel Demux: TM0 input at I7 port",
    xlabelsize = 14,
    ylabelsize = 14,
)

colors = [:blue, :red, :green, :orange]
labels = ["TM0", "TM1", "TM2", "TM3"]
for m in 1:4
    lines!(ax_modes, wavelengths, T_modes[m],
        color = colors[m], linewidth = 2, label = labels[m])
end

axislegend(ax_modes, position = :rt)
xlims!(ax_modes, λ_min, λ_max)
ylims!(ax_modes, -0.05, 1.05)

save("demux_8ch_mode_spectra.png", fig_modes)
println("Saved: demux_8ch_mode_spectra.png")

# ------------------------------------------------------------------- #
# Plot: field intensity at z=0
# ------------------------------------------------------------------- #

println("Plotting field intensity...")

field_dft_raw = Khronos.get_dft_fields(field_monitor)
field_dft = Array(field_dft_raw[:, :, :, 1])

if size(field_dft, 3) == 1
    field_2d = field_dft[:, :, 1]
elseif size(field_dft, 1) == 1
    field_2d = field_dft[1, :, :]
elseif size(field_dft, 2) == 1
    field_2d = field_dft[:, 1, :]
else
    field_2d = field_dft[:, :, 1]
end

field_intensity = abs2.(field_2d)

mon_cx, mon_cy = field_monitor.center[1], field_monitor.center[2]
mon_sx, mon_sy = field_monitor.size[1], field_monitor.size[2]
nfx, nfy = size(field_intensity)
fx_range = range(mon_cx - mon_sx / 2, mon_cx + mon_sx / 2, length = nfx)
fy_range = range(mon_cy - mon_sy / 2, mon_cy + mon_sy / 2, length = nfy)

fig_field = Figure(size = (1200, 400))
ax_field = Axis(fig_field[1, 1],
    title = "|Ey|² field intensity at z=0, λ=$(λ_center) μm",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
)

imax = maximum(field_intensity)
imax = imax > 0 ? imax : 1.0
heatmap!(ax_field, collect(fx_range), collect(fy_range), Float32.(field_intensity),
    colormap = :inferno,
    colorrange = (0.0, min(imax, 1000.0)),
)

save("demux_8ch_field.png", fig_field)
println("Saved: demux_8ch_field.png")

# ------------------------------------------------------------------- #
# Summary
# ------------------------------------------------------------------- #

println("\n", "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Output files:")
println("  demux_8ch_mode_spectra.png  — Mode-resolved transmission spectra")
println("  demux_8ch_field.png         — |Ey|² field at z=0")
t_script_total = time() - t_script_start
println("  Script wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
