# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# 90-Degree Bend Polarization Splitter-Rotator (PSR)
#
# Ported from the Tidy3D 90BendPolarizationSplitterRotator tutorial.
# Demonstrates:
#   - Programmatic curved waveguide geometry generation (arcs, S-bends)
#   - Using expand_path to convert centerlines to polygon outlines
#   - gds_polygons_to_objects for PolygonalPrism extrusion
#   - Mode conversion analysis (TM→TE at cross port)
#
# Physics: Two concentric 90-degree bend waveguides on SOI are designed
# so the TM mode of the inner bend phase-matches with the TE mode of
# the outer bend. This enables simultaneous polarization splitting and
# rotation: TM input on the inner bend couples to the outer bend as TE
# (cross port), while TE input remains on the inner bend (through port).
#
# Reference: Kang Tan et al., Opt. Express 24, 14506-14512 (2016).

t_script_start = time()

import Khronos
using GeometryPrimitives
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Materials (non-dispersive at O-band ~1.31 μm)
n_Si = 3.47
n_SiO2 = 1.44
ε_Si = n_Si^2
ε_SiO2 = n_SiO2^2
mat_si = Khronos.Material(ε = ε_Si)
mat_sio2 = Khronos.Material(ε = ε_SiO2)

# Geometric parameters
R1 = 10.0       # inner bend radius (μm)
W1 = 0.4        # inner waveguide width (μm) — fully etched
W2 = 0.21       # outer waveguide width (μm) — fully etched
W3 = 0.285      # outer waveguide width (μm) — partially etched
Wg = 0.2        # gap between inner and outer bends (μm)
H1 = 0.22       # full etch thickness (μm)
H2 = 0.11       # partial etch thickness (μm)
buffer = 5.0    # straight waveguide buffer length (μm)

# Outer bend radius (center of W2 waveguide)
R_outer = R1 + W1 / 2 + Wg + W2 / 2  # = 10.505 μm

# Wavelength range (O-band)
λ_min = 1.25    # μm
λ_max = 1.37    # μm
λ_center = 1.31 # μm
freq_center = 1.0 / λ_center
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
fwidth = 2π * 0.5 * (freq_max - freq_min)

# Monitor frequencies
n_freqs = 21
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))
wavelengths = 1.0 ./ monitor_freqs

# ------------------------------------------------------------------- #
# Helper: generate arc centerline points
# ------------------------------------------------------------------- #

function arc_centerline(cx, cy, radius, θ_start, θ_end; n_points=80)
    [(cx + radius * cos(θ), cy + radius * sin(θ))
     for θ in range(θ_start, θ_end, length=n_points)]
end

"""
Generate a waveguide polygon by offsetting a centerline by ±half_width.
"""
function centerline_to_polygon(
    centerline::Vector{Tuple{Float64,Float64}},
    half_width::Float64,
)::Vector{Tuple{Float64,Float64}}
    # Use Khronos's expand_path for robust polygon generation
    Khronos.expand_path(centerline, half_width, 0, 0.0, 0.0)
end

# ------------------------------------------------------------------- #
# Generate waveguide centerlines
# ------------------------------------------------------------------- #
#
# Both waveguides follow concentric 90-degree arcs centered at (0, R1).
# The inner bend goes from +x input to +y output; the outer bend
# follows at a larger radius separated by the coupling gap.
#
# Layout:
#   Inner input:  straight along +x, y=0
#   Inner output: straight along +y, x=R1
#   Outer input:  straight along +x, y=(R1-R_outer)≈-0.505
#   Outer output: straight along +y, x=R_outer≈10.505

arc_center = (0.0, R1)  # shared center of curvature

# --- Inner waveguide centerline ---
inner_straight_in = [(x, 0.0) for x in range(-buffer, 0.0, length=20)]
inner_arc = arc_centerline(arc_center[1], arc_center[2], R1, -π / 2, 0.0; n_points=80)
inner_straight_out = [(R1, y) for y in range(R1, R1 + buffer, length=20)]

inner_centerline = vcat(inner_straight_in, inner_arc[2:end], inner_straight_out[2:end])

# --- Outer waveguide centerline (fully etched) ---
y_outer_start = R1 - R_outer  # ≈ -0.505
outer_straight_in = [(x, y_outer_start) for x in range(-buffer, 0.0, length=20)]
outer_arc = arc_centerline(arc_center[1], arc_center[2], R_outer, -π / 2, 0.0; n_points=80)
outer_straight_out = [(R_outer, y) for y in range(R1, R1 + buffer, length=20)]

outer_centerline = vcat(outer_straight_in, outer_arc[2:end], outer_straight_out[2:end])

# --- Outer waveguide centerline (partially etched, same path but wider) ---
# The partially etched slab uses the same centerline as the outer waveguide

# ------------------------------------------------------------------- #
# Generate polygon outlines and convert to Khronos Objects
# ------------------------------------------------------------------- #

println("Generating waveguide polygons...")

# Inner waveguide polygon (W1 wide, fully etched)
inner_poly_verts = centerline_to_polygon(inner_centerline, W1 / 2)
inner_polygon = Khronos.GDSPolygon(inner_poly_verts, 1, 0)
inner_objects = Khronos.gds_polygons_to_objects(
    [inner_polygon], 0.0, H1, mat_si; axis=3,
)
println("  Inner waveguide: $(length(inner_poly_verts)) vertices → $(length(inner_objects)) objects")

# Outer waveguide polygon — fully etched (W2 wide)
outer_poly_verts = centerline_to_polygon(outer_centerline, W2 / 2)
outer_polygon = Khronos.GDSPolygon(outer_poly_verts, 1, 0)
outer_objects = Khronos.gds_polygons_to_objects(
    [outer_polygon], 0.0, H1, mat_si; axis=3,
)
println("  Outer waveguide (full etch): $(length(outer_poly_verts)) vertices → $(length(outer_objects)) objects")

# Outer waveguide polygon — partially etched (W3 wide, thinner)
partial_poly_verts = centerline_to_polygon(outer_centerline, W3 / 2)
partial_polygon = Khronos.GDSPolygon(partial_poly_verts, 2, 0)
partial_objects = Khronos.gds_polygons_to_objects(
    [partial_polygon], 0.0, H2, mat_si; axis=3,
)
println("  Outer waveguide (partial etch): $(length(partial_poly_verts)) vertices → $(length(partial_objects)) objects")

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

# Physical domain bounds (matching tidy3d: x∈[-3,12], y∈[-3,22], z∈[-1,1])
domain_x = 15.0
domain_y = 25.0
domain_z = 2.0

pml_thickness = 1.0
resolution = 25

cell_x = domain_x + 2 * pml_thickness
cell_y = domain_y + 2 * pml_thickness
cell_z = domain_z + 2 * pml_thickness

# Center the domain to encompass the bend region
cell_center = [4.5, 9.5, H1 / 2]

# ------------------------------------------------------------------- #
# Geometry: waveguide objects + background
# ------------------------------------------------------------------- #

# Objects listed first have higher priority (findfirst semantics).
# Inner waveguide takes priority over partially etched outer slab.
background = Khronos.Object(
    Cuboid([cell_center[1], cell_center[2], cell_center[3]],
           [cell_x + 10.0, cell_y + 10.0, cell_z + 10.0]),
    mat_sio2,
)

geometry = vcat(inner_objects, outer_objects, partial_objects, [background])

println("Total geometry objects: $(length(geometry))")

# ------------------------------------------------------------------- #
# Source: mode source on inner waveguide input
# ------------------------------------------------------------------- #

# Place source before the coupling region on the inner waveguide
# Inner waveguide input is along x-axis at y=0
source_x = -λ_center  # ~1 wavelength before bend start
source_center = [source_x, 0.0, H1 / 2]
source_size = [0.0, 4 * W1, 6 * H1]  # YZ cross-section

# TM simulation: mode_index=2 (second mode = TM0)
# TE simulation: mode_index=1 (first mode = TE0)
mode_index_tm = 2
mode_index_te = 1

sources_tm = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = mode_index_tm,
        center = source_center,
        size = source_size,
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

# ------------------------------------------------------------------- #
# Monitors
# ------------------------------------------------------------------- #

# Cross port monitor: on outer waveguide output (going +y at x=R_outer)
# The outer waveguide (W2=0.21 μm) is very narrow and may only support TE0.
# Monitor TE (mode 1) — the desired TM→TE conversion output.
# Place monitor within the output straight section (y ∈ [R1, R1+buffer])
cross_y = R1 + buffer - λ_center  # ~1 wavelength before end of waveguide
cross_center = [R_outer, cross_y, H1 / 2]
cross_size = [8 * W1, 0.0, 6 * H1]  # XZ cross-section (normal in y)

cross_monitor = Khronos.ModeMonitor(
    center = cross_center,
    size = cross_size,
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,  # TE0
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Through port monitors: on inner waveguide output (going +y at x=R1)
# The inner waveguide (W1=0.4 μm) supports both TE and TM.
# Need separate monitors for TE (mode 1) and TM (mode 2) overlap.
through_center = [R1, cross_y, H1 / 2]
through_size = [8 * W1, 0.0, 6 * H1]

through_monitor_te = Khronos.ModeMonitor(
    center = through_center,
    size = through_size,
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,  # TE0
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

through_monitor_tm = Khronos.ModeMonitor(
    center = through_center,
    size = through_size,
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 2,  # TM0
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Reference monitor: on inner waveguide input (before bend)
# Only need TM overlap since input is TM
ref_center = [source_x + 1.0, 0.0, H1 / 2]
ref_size = source_size

ref_monitor = Khronos.ModeMonitor(
    center = ref_center,
    size = ref_size,
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = mode_index_tm,  # TM0 (mode 2)
        target_neff = Float64(n_Si),
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Field monitor at z = H2/2 (partially etched slab mid-plane)
field_monitor = Khronos.DFTMonitor(
    component = Khronos.Ey(),
    center = [cell_center[1], cell_center[2], H2 / 2],
    size = [domain_x, domain_y, 0.0],
    frequencies = [freq_center],
)

monitors = Khronos.Monitor[cross_monitor, through_monitor_te, through_monitor_tm,
                          ref_monitor, field_monitor]

# ------------------------------------------------------------------- #
# Boundaries: PML on all sides
# ------------------------------------------------------------------- #

boundaries = [
    [pml_thickness, pml_thickness],  # x
    [pml_thickness, pml_thickness],  # y
    [pml_thickness, pml_thickness],  # z
]

# ------------------------------------------------------------------- #
# Simulation: TM input
# ------------------------------------------------------------------- #

sim_tm = Khronos.Simulation(
    cell_size = [cell_x, cell_y, cell_z],
    cell_center = cell_center,
    resolution = resolution,
    geometry = geometry,
    sources = sources_tm,
    monitors = monitors,
    boundaries = boundaries,
)

println("\n", "=" ^ 60)
println("90-Degree Bend Polarization Splitter-Rotator")
println("=" ^ 60)
println("Inner bend radius: $(R1) μm")
println("Coupling gap: $(Wg) μm")
println("Inner waveguide: W1=$(W1) μm, H1=$(H1) μm")
println("Outer waveguide: W2=$(W2) μm (full), W3=$(W3) μm (partial)")
println("Wavelength range: $(λ_min)–$(λ_max) μm (O-band)")
println("Domain: $(domain_x) × $(domain_y) × $(domain_z) μm")
println("Cell (incl. PML): $(cell_x) × $(cell_y) × $(cell_z) μm")
println("Resolution: $(resolution) pts/μm")
println("=" ^ 60)

# ================================================================== #
# RUN 1: TM input
# ================================================================== #

println("\n>>> TM SIMULATION <<<")

t_prep_start = time()
Khronos.prepare_simulation!(sim_tm)
t_prep = time() - t_prep_start

num_voxels = sim_tm.Nx * sim_tm.Ny * sim_tm.Nz
println("Grid: $(sim_tm.Nx) × $(sim_tm.Ny) × $(sim_tm.Nz) = $(num_voxels) voxels")
println("Preparation time: $(round(t_prep, digits=2))s")

t_run_start = time()
step_start = sim_tm.timestep

Khronos.run(sim_tm,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-5,
        minimum_runtime = 800.0,
        maximum_runtime = 2000.0,
    ),
)

t_run = time() - t_run_start
total_steps = sim_tm.timestep - step_start
println("TM simulation: $(total_steps) steps in $(round(t_run, digits=2))s")

# ------------------------------------------------------------------- #
# TM Post-processing: mode conversion efficiencies
# ------------------------------------------------------------------- #

println("\nTM post-processing...")

# compute_mode_amplitudes returns 1D vectors (one value per frequency)
# Each monitor tracks overlap with a single mode
cross_te_a_plus, _ = Khronos.compute_mode_amplitudes(cross_monitor.monitor_data)
through_te_a_plus, _ = Khronos.compute_mode_amplitudes(through_monitor_te.monitor_data)
through_tm_a_plus, _ = Khronos.compute_mode_amplitudes(through_monitor_tm.monitor_data)
ref_a_plus, _ = Khronos.compute_mode_amplitudes(ref_monitor.monitor_data)

# Normalize by reference (TM input)
ref_amp = abs.(ref_a_plus)
ref_threshold = 0.05 * maximum(ref_amp)
reliable = ref_amp .> ref_threshold

# Mode conversion efficiencies for TM input
T_cross_te = fill(NaN, n_freqs)
T_through_te = fill(NaN, n_freqs)
T_through_tm = fill(NaN, n_freqs)

T_cross_te[reliable] .= abs2.(cross_te_a_plus[reliable] ./ ref_a_plus[reliable])
T_through_te[reliable] .= abs2.(through_te_a_plus[reliable] ./ ref_a_plus[reliable])
T_through_tm[reliable] .= abs2.(through_tm_a_plus[reliable] ./ ref_a_plus[reliable])

# Convert to dB
dB(x) = 10 * log10(max(x, 1e-10))

T_cross_te_dB = dB.(T_cross_te)
T_through_te_dB = dB.(T_through_te)
T_through_tm_dB = dB.(T_through_tm)

println("TM→TE at cross port (desired): $(round(T_cross_te[argmin(abs.(wavelengths .- λ_center))], digits=4))")
println("TM→TM at through port: $(round(T_through_tm[argmin(abs.(wavelengths .- λ_center))], digits=4))")

# ------------------------------------------------------------------- #
# Plot TM results
# ------------------------------------------------------------------- #

fig_tm = Figure(size = (800, 500))
ax_tm = Axis(fig_tm[1, 1],
    xlabel = "Wavelength (μm)",
    ylabel = "Mode conversion efficiency (dB)",
    title = "PSR: TM input mode conversion",
    xlabelsize = 14,
    ylabelsize = 14,
)

lines!(ax_tm, wavelengths, T_cross_te_dB, color = :blue, linewidth = 2, label = "TM→TE (cross)")
lines!(ax_tm, wavelengths, T_through_te_dB, color = :blue, linewidth = 2,
    linestyle = :dash, label = "TM→TE (through)")
lines!(ax_tm, wavelengths, T_through_tm_dB, color = :red, linewidth = 2,
    linestyle = :dash, label = "TM→TM (through)")

axislegend(ax_tm, position = :lb)
ylims!(ax_tm, -30, 5)

save("psr_bend_tm_conversion.png", fig_tm)
println("Saved: psr_bend_tm_conversion.png")

# ------------------------------------------------------------------- #
# Plot field intensity
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

fig_field = Figure(size = (600, 700))
ax_f = Axis(fig_field[1, 1],
    title = "TM input: |Ey|² at z=$(H2/2) μm, λ=$(λ_center) μm",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    aspect = DataAspect(),
)

imax = maximum(field_intensity)
imax = imax > 0 ? imax : 1.0
heatmap!(ax_f, collect(fx_range), collect(fy_range), Float32.(field_intensity),
    colormap = :inferno,
    colorrange = (0.0, imax * 0.5),
)

save("psr_bend_tm_field.png", fig_field)
println("Saved: psr_bend_tm_field.png")

# ------------------------------------------------------------------- #
# Summary
# ------------------------------------------------------------------- #

println("\n", "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Output files:")
println("  psr_bend_tm_conversion.png  — TM mode conversion spectra")
println("  psr_bend_tm_field.png       — |Ey|² field intensity (TM input)")
println("")
println("To also run the TE simulation, modify mode_index to $(mode_index_te)")
println("in the ModeSource and re-run. The TE mode should remain on the")
println("inner waveguide (through port) with minimal coupling to the outer bend.")
t_script_total = time() - t_script_start
println("  Script wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
