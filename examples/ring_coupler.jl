# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Waveguide-to-ring resonator coupling simulation.
#
# Ported from the Tidy3D WaveguideToRingCoupling tutorial.
# Demonstrates:
#   - ModeMonitor for S-parameter extraction
#   - Absorber boundary for stable simulation with ring crossing boundary
#   - Ring resonator geometry via overlapping cylinders
#
# Physics: Light couples from a straight Si waveguide into a Si ring resonator.
# The coupling gap determines the coupling coefficient κ. At resonance,
# light circulates in the ring and the through-port transmission dips.

import Khronos
using GeometryPrimitives
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Materials (constant refractive indices at ~1.55 μm)
n_Si = 3.47
n_SiO2 = 1.44
ε_Si = n_Si^2      # 12.0409
ε_SiO2 = n_SiO2^2  # 2.0736

# Waveguide cross-section
wg_width = 0.5    # μm
wg_height = 0.22  # μm

# Ring resonator
ring_radius = 5.0  # μm (center radius)
coupling_gap = 0.05  # μm

# Ring center position: waveguide centered at y=0 spans ±wg_width/2,
# so its top edge is at y = wg_width/2. The gap is measured from this edge
# to the outer edge of the ring (at ring_center_y - ring_radius - wg_width/2).
# Solving: ring_center_y - (ring_radius + wg_width/2) = wg_width/2 + coupling_gap
# => ring_center_y = wg_width + coupling_gap + ring_radius
ring_center_y = wg_width + coupling_gap + ring_radius

# Wavelength range
λ_min = 1.5   # μm
λ_max = 1.6   # μm
λ_center = 1.55  # μm
freq_center = 1.0 / λ_center
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
fwidth = 2π * 0.5 * (freq_max - freq_min)  # Khronos uses fwidth = 1/temporal_width;
                                           # spectral 1/e half-width = fwidth/(2π).
                                           # To cover ±(freq_max-freq_min)/2 to 1/e,
                                           # we need fwidth = 2π × half-bandwidth.

# Number of frequency points for monitors
n_freqs = 101
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

# Physical domain (excluding PML/absorber margins). The ring extends above
# the domain boundary (absorbed by the adiabatic absorber on +y side).
# Matches Tidy3D domain: Lx = 2R + 2λ, Ly = R/2 + gap + 2w + λ, Lz = 9h
# NOTE: In Khronos, PML is INSIDE cell_size (unlike Tidy3D where PML is
# external). We add PML/absorber margins when constructing the Simulation.
domain_x = 2 * ring_radius + 2 * λ_center
domain_y = ring_radius / 2 + coupling_gap + 2 * wg_width + λ_center
domain_z = 9 * wg_height
pml_thickness = 1.0  # μm

resolution = 25  # grid points per μm (matching Tidy3D's min_steps_per_wvl=25)

# In Khronos, PML/absorber regions are INSIDE cell_size.
# Compute absorber thickness so we can size the cell correctly.
absorber_num_layers = 60
absorber_thickness = absorber_num_layers / resolution  # 2.4 μm

# ------------------------------------------------------------------- #
# Geometry
# ------------------------------------------------------------------- #
# Ring resonator: two overlapping cylinders (outer Si, inner SiO2)
# Objects listed first have higher priority in findfirst ordering.

geometry = [
    # Straight waveguide (Si) — extends across entire x-domain
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [domain_x + 4.0, wg_width, wg_height]),
        Khronos.Material(ε = ε_Si),
    ),
    # Ring: inner cylinder (SiO2 — carved out from outer)
    # Listed BEFORE outer cylinder so it takes priority (findfirst semantics)
    Khronos.Object(
        Cylinder([0.0, ring_center_y, 0.0], ring_radius - wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
        Khronos.Material(ε = ε_SiO2),
    ),
    # Ring: outer cylinder (Si)
    Khronos.Object(
        Cylinder([0.0, ring_center_y, 0.0], ring_radius + wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
        Khronos.Material(ε = ε_Si),
    ),
    # Background cladding (SiO2) — large box covering entire domain
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [domain_x + 10.0, domain_y + 10.0, domain_z + 10.0]),
        Khronos.Material(ε = ε_SiO2),
    ),
]

# ------------------------------------------------------------------- #
# Source: mode source launching TE fundamental mode
# ------------------------------------------------------------------- #

# Place source well before the coupling region
source_x = -(ring_radius + λ_center / 4)

sources = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = 1,
        center = [source_x, 0.0, 0.0],
        size = [0.0, 6 * wg_width, 6 * wg_height],
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

# ------------------------------------------------------------------- #
# Monitors
# ------------------------------------------------------------------- #

# Through port: mode monitor at output of straight waveguide
through_x = ring_radius + λ_center / 4
through_monitor = Khronos.ModeMonitor(
    center = [through_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Reference monitor: between source and coupling region (unperturbed waveguide)
# Used to normalize mode amplitudes and remove the source spectrum envelope.
ref_x = source_x + 1.0  # 1 μm after source
ref_monitor = Khronos.ModeMonitor(
    center = [ref_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Drop port: mode monitor on the ring at angle θ = π/4
# At this angle, the ring cross-section is approximately in the XZ plane
θ_drop = π / 4
drop_x = ring_radius * sin(θ_drop)               # R/√2
drop_y = ring_center_y - ring_radius * cos(θ_drop)  # ring_center_y - R/√2
drop_monitor = Khronos.ModeMonitor(
    center = [drop_x, drop_y, 0.0],
    size = [6 * wg_width, 0.0, 6 * wg_height],  # YZ plane → XZ-like cross section
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(
        num_modes = 1,
        geometry = geometry,
        mode_solver_resolution = 50,
    ),
)

# Field monitor at z=0 for visualization (covers full XY plane like Tidy3D)
field_monitor = Khronos.DFTMonitor(
    component = Khronos.Ey(),
    center = [0.0, domain_y / 4, 0.0],
    size = [domain_x, domain_y, 0.0],
    frequencies = [freq_center],
)

monitors = Khronos.Monitor[through_monitor, ref_monitor, drop_monitor, field_monitor]

# ------------------------------------------------------------------- #
# Boundaries
# ------------------------------------------------------------------- #
# PML on all sides EXCEPT +y where the ring intersects the boundary.
# An adiabatic absorber on +y prevents PML divergence from the
# non-translationally-invariant ring geometry.

boundaries = [
    [pml_thickness, pml_thickness],  # x: PML both sides
    [pml_thickness, 0.0],            # y: PML on -y, none on +y (absorber instead)
    [pml_thickness, pml_thickness],  # z: PML both sides
]

absorbers = [
    nothing,                                        # x: no absorber
    [nothing, Khronos.Absorber(num_layers = 60)],   # y: absorber on +y side only
    nothing,                                        # z: no absorber
]

# ------------------------------------------------------------------- #
# Simulation
# ------------------------------------------------------------------- #

# Cell size: physical domain + PML/absorber margins
# x: PML both sides; y: PML on -y, absorber on +y; z: PML both sides
cell_x = domain_x + 2 * pml_thickness
cell_y = domain_y + pml_thickness + absorber_thickness
cell_z = domain_z + 2 * pml_thickness

# Cell center: physical center shifted by asymmetric y margins
phys_center_y = domain_y / 4
cell_center_y = phys_center_y + (absorber_thickness - pml_thickness) / 2

sim = Khronos.Simulation(
    cell_size = [cell_x, cell_y, cell_z],
    cell_center = [0.0, cell_center_y, 0.0],
    resolution = resolution,
    geometry = geometry,
    sources = sources,
    monitors = monitors,
    boundaries = boundaries,
    absorbers = absorbers,
)

# ------------------------------------------------------------------- #
# Run simulation
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("Waveguide-to-Ring Resonator Coupling Simulation")
println("=" ^ 60)
println("Ring radius: $(ring_radius) μm")
println("Coupling gap: $(coupling_gap) μm")
println("Waveguide: $(wg_width) × $(wg_height) μm (Si)")
println("Wavelength range: $(λ_min)–$(λ_max) μm")
println("Physical domain: $(domain_x) × $(domain_y) × $(domain_z) μm")
println("Cell (incl. PML/absorber): $(cell_x) × $(round(cell_y, digits=1)) × $(cell_z) μm")
println("Resolution: $(resolution) pts/μm")
println("=" ^ 60)

# ================================================================== #
# 1. Pre-simulation geometry visualization
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 1: Geometry cross-sections")
println("=" ^ 60)

# XY cross-section at z=0 — shows waveguide and ring coupling region
fig_geom_xy = Khronos.plot_geometry_slice(
    geometry, :z, 0.0;
    x_range = (-domain_x / 2, domain_x / 2),
    y_range = (-1.0, ring_center_y + ring_radius + 1.0),
    resolution = 500,
    colormap = :viridis,
    title = "Ring coupler — XY cross-section (z=0)",
)
save("ring_coupler_geometry_xy.png", fig_geom_xy)
println("Saved: ring_coupler_geometry_xy.png")

# 3-panel cross-sections: XY, XZ, YZ through origin
fig_3panel = Khronos.plot_geometry_cross_sections(
    geometry;
    center = [0.0, ring_center_y / 2, 0.0],
    span = [domain_x, ring_center_y + ring_radius + 2.0, domain_z],
    resolution = 400,
    colormap = :viridis,
)
save("ring_coupler_geometry_crosssections.png", fig_3panel)
println("Saved: ring_coupler_geometry_crosssections.png")

# XY close-up of the coupling gap region
fig_gap = Khronos.plot_geometry_slice(
    geometry, :z, 0.0;
    x_range = (-2.0, 2.0),
    y_range = (-0.5, 1.5),
    resolution = 800,
    colormap = :viridis,
    title = "Coupling gap close-up (gap = $(coupling_gap) μm)",
)
save("ring_coupler_gap_closeup.png", fig_gap)
println("Saved: ring_coupler_gap_closeup.png")

# ================================================================== #
# 2. Run simulation with timing
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 2: Running FDTD simulation")
println("=" ^ 60)

t_total_start = time()

# Prepare and capture timing
t_prep_start = time()
Khronos.prepare_simulation!(sim)
t_prep = time() - t_prep_start

num_voxels = sim.Nx * sim.Ny * sim.Nz
println("\nGrid: $(sim.Nx) × $(sim.Ny) × $(sim.Nz) = $(num_voxels) voxels")
println("Δx = $(round(sim.Δx, sigdigits=4)) μm, Δt = $(round(sim.Δt, sigdigits=4))")
println("Preparation time: $(round(t_prep, digits=2))s")

# Run simulation
t_run_start = time()
step_start = sim.timestep

Khronos.run(sim,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-6,
        minimum_runtime = 600.0,  # several ring round-trips (2πR·n_eff ≈ 88 per trip)
        maximum_runtime = 1500.0,
    ),
)

t_run = time() - t_run_start
t_total = time() - t_total_start
total_steps = sim.timestep - step_start

# ================================================================== #
# 3. Performance metrics
# ================================================================== #

println("\n", "=" ^ 60)
println("PERFORMANCE METRICS")
println("=" ^ 60)
println("  Grid size:         $(sim.Nx) × $(sim.Ny) × $(sim.Nz)")
println("  Total voxels:      $(num_voxels) ($(round(num_voxels / 1e6, digits=2))M)")
println("  Timesteps:         $(total_steps)")
println("  Simulation time:   $(round(sim.timestep * sim.Δt, digits=2)) time units")
println("  Δx:                $(round(sim.Δx, sigdigits=4)) μm")
println("  Δt:                $(round(sim.Δt, sigdigits=4))")
println("  Courant number:    $(round(sim.Courant, sigdigits=4))")
println("  ────────────────────────────────")
println("  Prep time:         $(round(t_prep, digits=2))s")
println("  Run time:          $(round(t_run, digits=2))s")
println("  Total wall time:   $(round(t_total, digits=2))s")
if t_run > 0 && total_steps > 0
    mcells_per_s = num_voxels * total_steps / t_run / 1e6
    println("  Throughput:        $(round(mcells_per_s, digits=1)) MVoxels/s")
    println("  Time per step:     $(round(t_run / total_steps * 1e3, digits=2)) ms")
end
println("=" ^ 60)

# ================================================================== #
# 4. Post-processing: mode amplitudes
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 3: Post-processing — mode overlap integrals")
println("=" ^ 60)

t_post_start = time()

through_data = through_monitor.monitor_data
ref_data = ref_monitor.monitor_data
drop_data = drop_monitor.monitor_data

through_a_plus, through_a_minus = Khronos.compute_mode_amplitudes(through_data)
ref_a_plus, ref_a_minus = Khronos.compute_mode_amplitudes(ref_data)
drop_a_plus, drop_a_minus = Khronos.compute_mode_amplitudes(drop_data)

# Normalize by reference monitor to remove source spectrum envelope.
# T(f) = |a_port(f) / a_ref(f)|²
# This is the standard approach (MEEP, Tidy3D) for computing S-parameters.
#
# At band edges where the source has negligible energy, |ref_a_plus| → 0
# and the ratio becomes unreliable. Mask out frequencies where the reference
# amplitude is below a threshold fraction of the peak.
ref_amplitude = abs.(ref_a_plus)
ref_threshold = 0.05 * maximum(ref_amplitude)  # 5% of peak
reliable = ref_amplitude .> ref_threshold

T_through = zeros(n_freqs)
T_drop = zeros(n_freqs)
T_through[reliable] .= abs2.(through_a_plus[reliable] ./ ref_a_plus[reliable])
T_drop[reliable] .= abs2.(drop_a_plus[reliable] ./ ref_a_plus[reliable])

# Mark unreliable points as NaN for plotting (gaps in lines)
T_through[.!reliable] .= NaN
T_drop[.!reliable] .= NaN

n_reliable = count(reliable)
println("Reliable frequency points: $n_reliable / $n_freqs (threshold = $(round(ref_threshold, sigdigits=3)))")

# Convert frequencies to wavelengths
wavelengths = 1.0 ./ monitor_freqs

t_post = time() - t_post_start
println("Mode overlap computation: $(round(t_post, digits=2))s")

println("\nThrough port transmission |t|² (sampled):")
for idx in 1:20:n_freqs
    val = T_through[idx]
    valstr = isnan(val) ? "masked" : "$(round(val, digits=4))"
    println("  λ = $(round(wavelengths[idx], digits=4)) μm: |t|² = $valstr")
end

println("\nDrop port coupling |κ|² (sampled):")
for idx in 1:20:n_freqs
    val = T_drop[idx]
    valstr = isnan(val) ? "masked" : "$(round(val, digits=4))"
    println("  λ = $(round(wavelengths[idx], digits=4)) μm: |κ|² = $valstr")
end

# Report conservation (only for reliable points)
T_total = T_through .+ T_drop
println("\nEnergy conservation |t|² + |κ|² (sampled):")
for idx in 1:20:n_freqs
    val = T_total[idx]
    valstr = isnan(val) ? "masked" : "$(round(val, digits=4))"
    println("  λ = $(round(wavelengths[idx], digits=4)) μm: $valstr")
end

# Expected FSR
FSR_nm = λ_center^2 / (2.8 * 2π * ring_radius) * 1e3
println("\nExpected FSR ≈ $(round(FSR_nm, digits=1)) nm")

# ================================================================== #
# 5. Transmission spectra plot (main result — matches Tidy3D Fig. 3)
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 4: Plotting transmission spectra")
println("=" ^ 60)

fig_spectra = Figure(size = (900, 500))

ax_spectra = Axis(fig_spectra[1, 1],
    xlabel = "Wavelength (μm)",
    ylabel = "Transmission",
    title = "Waveguide-to-Ring Coupler: Through & Drop Port Spectra",
    xlabelsize = 14,
    ylabelsize = 14,
)

# NaN values are automatically skipped by CairoMakie lines!
lines!(ax_spectra, wavelengths, T_through,
    color = :blue, linewidth = 2, label = "|t|² (through)")
lines!(ax_spectra, wavelengths, T_drop,
    color = :red, linewidth = 2, label = "|κ|² (drop)")
lines!(ax_spectra, wavelengths, T_total,
    color = :gray, linewidth = 1, linestyle = :dash, label = "|t|² + |κ|²")

axislegend(ax_spectra, position = :rt)
ylims!(ax_spectra, -0.05, 1.1)

save("ring_coupler_spectra.png", fig_spectra)
println("Saved: ring_coupler_spectra.png")

# ================================================================== #
# 6. Field intensity at z=0 (matches Tidy3D Fig. 2)
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 5: Plotting DFT field intensity at z=0")
println("=" ^ 60)

field_dft_raw = Khronos.get_dft_fields(field_monitor)
println("  DFT field array size: ", size(field_dft_raw))
field_dft_max = maximum(abs.(Array(field_dft_raw)))
println("  DFT field max |value|: ", field_dft_max)

if isnan(field_dft_max) || isinf(field_dft_max)
    @error "Simulation produced NaN/Inf fields — check stability"
end

# Extract the first (only) frequency
field_dft = Array(field_dft_raw[:, :, :, 1])

# The monitor is an XY plane (z-size=0), so shape is (Nx, Ny, 1)
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

# Compute physical coordinate ranges for the field monitor
mon_cx, mon_cy = field_monitor.center[1], field_monitor.center[2]
mon_sx, mon_sy = field_monitor.size[1], field_monitor.size[2]
nfx, nfy = size(field_intensity)
fx_range = range(mon_cx - mon_sx / 2, mon_cx + mon_sx / 2, length = nfx)
fy_range = range(mon_cy - mon_sy / 2, mon_cy + mon_sy / 2, length = nfy)

# --- Fig A: Field intensity ---
fig_field = Figure(size = (900, 600))
ax_field = Axis(fig_field[1, 1],
    title = "|Ey|² field intensity at z=0, λ=$(λ_center) μm",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    aspect = DataAspect(),
)

imax = maximum(field_intensity)
imax = imax > 0 ? imax : 1.0
hm_i = heatmap!(ax_field, collect(fx_range), collect(fy_range), Float32.(field_intensity),
    colormap = :inferno,
    colorrange = (0.0, imax),
)
Colorbar(fig_field[1, 2], hm_i, label = "|Ey|²")

save("ring_coupler_field_intensity.png", fig_field)
println("Saved: ring_coupler_field_intensity.png")

# --- Fig B: Real part of Ey (shows wavefronts) ---
field_real = real.(field_2d)
vmax = maximum(abs.(field_real))
vmax = vmax > 0 ? vmax : 1.0

fig_field_re = Figure(size = (900, 600))
ax_re = Axis(fig_field_re[1, 1],
    title = "Re(Ey) at z=0, λ=$(λ_center) μm",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    aspect = DataAspect(),
)
hm_re = heatmap!(ax_re, collect(fx_range), collect(fy_range), Float32.(field_real),
    colormap = :bluesreds,
    colorrange = (-vmax, vmax),
)
Colorbar(fig_field_re[1, 2], hm_re, label = "Re(Ey)")

save("ring_coupler_field_real.png", fig_field_re)
println("Saved: ring_coupler_field_real.png")

# ================================================================== #
# 7. Combined geometry + field overlay (matches Tidy3D Fig. 2)
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 6: Geometry + field overlay")
println("=" ^ 60)

# Sample geometry on the same region as the field monitor
eps_data, xs_eps, ys_eps = Khronos.sample_geometry_slice(
    geometry, :z, 0.0;
    x_range = (mon_cx - mon_sx / 2, mon_cx + mon_sx / 2),
    y_range = (mon_cy - mon_sy / 2, mon_cy + mon_sy / 2),
    resolution = 400,
)

fig_overlay = Figure(size = (900, 600))
ax_ov = Axis(fig_overlay[1, 1],
    title = "Ring coupler: geometry + |Ey|² overlay (z=0, λ=$(λ_center) μm)",
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    aspect = DataAspect(),
)

# Plot geometry as background
heatmap!(ax_ov, xs_eps, ys_eps, Float32.(eps_data),
    colormap = :binary,
    colorrange = (1.0, maximum(eps_data)),
)

# Overlay field intensity with transparency
heatmap!(ax_ov, collect(fx_range), collect(fy_range), Float32.(field_intensity),
    colormap = Makie.cgrad(:inferno, alpha = 0.6),
    colorrange = (0.0, imax),
    transparency = true,
)

save("ring_coupler_overlay.png", fig_overlay)
println("Saved: ring_coupler_overlay.png")

# ================================================================== #
# 8. Summary
# ================================================================== #

println("\n", "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Output files:")
println("  ring_coupler_geometry_xy.png           — XY geometry cross-section")
println("  ring_coupler_geometry_crosssections.png — 3-panel geometry (XY, XZ, YZ)")
println("  ring_coupler_gap_closeup.png           — Coupling gap close-up")
println("  ring_coupler_spectra.png               — Through/drop transmission spectra")
println("  ring_coupler_field_intensity.png        — |Ey|² field at z=0")
println("  ring_coupler_field_real.png            — Re(Ey) field at z=0")
println("  ring_coupler_overlay.png               — Geometry + field overlay")
println("=" ^ 60)
