# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Blue micro-LED simulation — ported from the Tidy3D BlueMicroLED tutorial.
#
# Simulates a blue InGaN micro-LED with:
#   - Tapered mesa with 45-degree sidewalls (TruncatedCone)
#   - Silver back reflector (dispersive Drude metal)
#   - SiO2/Al2O3 passivation layers
#   - nGaN/MQW/pGaN/ITO epitaxial stack
#   - Near-to-far field projection for LEE computation
#   - Batch execution over 5 dipole positions x 2 polarizations
#
# Reference: Vogl et al., Opt. Lett. 49, 5095-5098 (2024)

import Khronos
using GeometryPrimitives
using LinearAlgebra
using CairoMakie
using Statistics

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

# ================================================================== #
# Physical parameters (all in micrometers)
# ================================================================== #

const nm = 1e-3  # nm → μm

# Wavelength & frequency
const lda0 = 450nm        # 450 nm blue emission
const freq0 = 1.0 / lda0  # frequency in c/μm units (Khronos uses c=1)
const fwidth = freq0 / 10

# Layer thicknesses
const t_mirror = 120nm
const t_SiO2   = 100nm
const t_Al2O3  = 40nm
const t_ITO    = 50nm
const t_pGaN   = 150nm
const t_MQW    = 50nm
const t_nGaN   = 1000nm  # 1 μm

# Refractive indices at 450 nm
const n_nGaN = 2.45
const n_MQW  = 2.60
const n_pGaN = 2.46
const n_ITO  = 2.00
const n_Al2O3 = 1.76
const n_SiO2  = 1.46
# Silver: n=0.028, k=2.88 at 450 nm → ε ≈ (0.028 + 2.88i)² ≈ -8.29 + 0.16i

# Mesa geometry
const mesa_radius = 500nm  # 0.5 μm
const sidewall_angle_deg = -45.0  # Tidy3D convention: negative = widens toward top

# Vertical coordinate system (z=0 at bottom of nGaN slab)
# nGaN slab: z ∈ [0, t_nGaN], air above z = t_nGaN
# nGaN cylinder extends below: z ∈ [-t_1, 0]
const t_1 = 500nm  # mesa extension below nGaN slab

# Dipole position (center of MQW layer)
const z_dipole = -t_1 - t_MQW / 2  # ≈ -0.525 μm

# Simulation domain
const dpml = 0.5   # PML thickness in μm
const n2f_margin = 0.5  # margin between n2f surface edge and PML (μm)
const sim_xy = 2 * dpml + 2 * n2f_margin + 3.0  # lateral: dpml + margin + 3μm physical + margin + dpml
const sim_z_min = -1.7   # cell bottom (margin below structures for PML)
const sim_z_max = 1.5    # cell top; ends inside nGaN (no GaN/air interface, matching Tidy3D)
const sim_z = sim_z_max - sim_z_min
const sim_center_z = (sim_z_min + sim_z_max) / 2
const n2f_xy = sim_xy - 2 * dpml - 2 * n2f_margin  # n2f surface lateral size (3.0 μm)

# Resolution — Tidy3D uses 15 pts/wavelength
# lda_in_material = lda0/n_max ≈ 450nm/2.6 ≈ 173 nm
# 15 pts/wvl → Δx ≈ 11.5 nm → resolution ≈ 87/μm
# We use a somewhat coarser resolution for speed; increase for accuracy
const resolution = 40  # pixels/μm (adjust for speed vs accuracy)

# ================================================================== #
# Build geometry
# ================================================================== #

function build_geometry()
    objects = Khronos.Object[]

    # Painter's algorithm: LATER objects override EARLIER ones.
    # Background layers first (silver, SiO2, Al2O3), then nGaN, then mesa
    # internals last. This matches Tidy3D's structure ordering.

    # Silver material for reflector and contact
    eps_ag = (0.028 + 2.88im)^2
    eps_inf = 1.0  # high-frequency permittivity
    chi_target = eps_ag - eps_inf  # susceptibility to fit
    ω0 = 2π * freq0  # angular frequency
    # From the Drude formula: Γ = -ω * Im(χ)/Re(χ), σ = -Re(χ)*(ω² + Γ²)/Γ
    chi_r = real(chi_target)
    chi_i = imag(chi_target)
    Gamma = -ω0 * chi_i / chi_r  # damping rate (angular)
    gamma_meep = Gamma / (2π)      # meep convention frequency
    sigma = -chi_r * (ω0^2 + Gamma^2) / Gamma  # oscillator strength
    ag_mat = Khronos.Material(
        ε = eps_inf,
        susceptibilities = [Khronos.DrudeSusceptibility(gamma_meep, sigma)],
    )

    # --- 1. Silver reflector (background, painted first — lowest priority) ---
    t_2 = 1.5
    z_ag_top = -(t_Al2O3 + t_SiO2)
    z_ag_bot = -(t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3 + t_SiO2 + t_2)
    ag_center_z = (z_ag_top + z_ag_bot) / 2
    ag_height = z_ag_top - z_ag_bot
    ag_lateral = sim_xy - 2 * dpml
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, ag_center_z], [ag_lateral, ag_lateral, ag_height]),
        material = ag_mat,
    ))

    # --- 2. SiO2 layers (slab + tapered cone around mesa) ---
    z_sio2_top = -t_Al2O3
    z_sio2_bot = z_sio2_top - t_SiO2
    sio2_center_z = (z_sio2_top + z_sio2_bot) / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, sio2_center_z], [sim_xy, sim_xy, t_SiO2]),
        material = Khronos.Material(ε = n_SiO2^2),
    ))

    mesa_stack_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    sio2_cone_center_z = -t_SiO2 - mesa_stack_height / 2
    radius_center = 1.2
    half_h = mesa_stack_height / 2
    r_bot_sio2 = radius_center + half_h * tan(deg2rad(sidewall_angle_deg))
    r_top_sio2 = radius_center - half_h * tan(deg2rad(sidewall_angle_deg))
    push!(objects, Khronos.Object(
        shape = TruncatedCone(
            [0.0, 0.0, sio2_cone_center_z],
            r_bot_sio2, r_top_sio2,
            mesa_stack_height, [0.0, 0.0, 1.0],
        ),
        material = Khronos.Material(ε = n_SiO2^2),
    ))

    # --- 3. Al2O3 layers (slab + coating cylinder around mesa) ---
    z_al2o3_top = 0.0
    z_al2o3_bot = -t_Al2O3
    al2o3_center_z = (z_al2o3_top + z_al2o3_bot) / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, al2o3_center_z], [sim_xy, sim_xy, t_Al2O3]),
        material = Khronos.Material(ε = n_Al2O3^2),
    ))

    al2o3_coat_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    al2o3_coat_center_z = -al2o3_coat_height / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, al2o3_coat_center_z],
                         mesa_radius + t_Al2O3, al2o3_coat_height),
        material = Khronos.Material(ε = n_Al2O3^2),
    ))

    # --- 4. nGaN (slab + mesa cylinder — overrides silver/SiO2/Al2O3 inside mesa) ---
    # Extend nGaN slab to fill entire upper domain including PML,
    # so there's no GaN/air interface (matching Tidy3D setup).
    nGaN_height = sim_z_max  # extends from z=0 to z=sim_z_max (fills upper PML)
    nGaN_slab_center_z = nGaN_height / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, nGaN_slab_center_z], [sim_xy, sim_xy, nGaN_height]),
        material = Khronos.Material(ε = n_nGaN^2),
    ))

    nGaN_cyl_center_z = -t_1 / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, nGaN_cyl_center_z],
                         mesa_radius, t_1),
        material = Khronos.Material(ε = n_nGaN^2),
    ))

    # --- 5. MQW active layer (inside mesa — overrides nGaN) ---
    mqw_center_z = -t_1 - t_MQW / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, mqw_center_z],
                         mesa_radius, t_MQW),
        material = Khronos.Material(ε = n_MQW^2),
    ))

    # --- 6. pGaN layer (inside mesa — overrides nGaN) ---
    pGaN_center_z = -t_1 - t_MQW - t_pGaN / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, pGaN_center_z],
                         mesa_radius, t_pGaN),
        material = Khronos.Material(ε = n_pGaN^2),
    ))

    # --- 7. ITO contact layer (inside mesa — overrides nGaN) ---
    ito_center_z = -t_1 - t_MQW - t_pGaN - t_ITO / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, ito_center_z],
                         mesa_radius, t_ITO),
        material = Khronos.Material(ε = n_ITO^2),
    ))

    # --- 8. Metal contact via (small cylinder — painted last, highest priority) ---
    contact_height = t_SiO2 + t_Al2O3
    contact_center_z = -t_1 - t_MQW - t_pGaN - t_ITO - contact_height / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, contact_center_z],
                         0.2, contact_height),
        material = ag_mat,
    ))

    return objects
end

# ================================================================== #
# Build simulation for a given dipole position and polarization
# ================================================================== #

function make_sim(dipole_x::Float64, pol::Symbol; geometry=nothing)
    if isnothing(geometry)
        geometry = build_geometry()
    end

    component = pol == :Ex ? Khronos.Ex() : Khronos.Ey()

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = freq0, fwidth = fwidth),
            component = component,
            center = [dipole_x, 0.0, z_dipole],
            size = [0.0, 0.0, 0.0],
            amplitude = 1.0,
        ),
    ]

    # Near2Far monitor inside nGaN slab (matching Tidy3D placement).
    # Tidy3D ends the domain inside nGaN and projects through the GaN medium,
    # so the monitor sees fields before the GaN/air interface.
    n2f_z = lda0 / 8  # λ₀/8 above nGaN base (inside nGaN, matching Tidy3D)
    n2f_monitor = Khronos.Near2FarMonitor(
        center = [0.0, 0.0, n2f_z],
        size = [n2f_xy, n2f_xy, 0.0],
        frequencies = [freq0],
        theta = collect(range(0.0, π / 2, length = 91)),  # 0-90 degrees, 1-degree steps
        phi = collect(range(0.0, 2π - 2π/72, length = 72)),  # 72 azimuthal points
        normal_dir = :+,
        medium_eps = n_nGaN^2,  # project through nGaN medium (matching Tidy3D)
    )

    # Field DFT monitor in xz-plane (for visualization)
    field_monitor = Khronos.DFTMonitor(
        component = component,
        center = [0.0, 0.0, sim_center_z],
        size = [n2f_xy, 0.0, sim_z - 2 * dpml],
        frequencies = [freq0],
    )

    sim = Khronos.Simulation(
        cell_size = [sim_xy, sim_xy, sim_z],
        cell_center = [0.0, 0.0, sim_center_z],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        monitors = [n2f_monitor, field_monitor],
    )

    return sim
end

# ================================================================== #
# 1. Geometry cross-section visualization (meep-style, from primitives)
# ================================================================== #

println("=" ^ 60)
println("STEP 1: Geometry cross-section (from shape primitives)")
println("=" ^ 60)

geom = build_geometry()

# XZ cross-section at y=0 — shows the full layer stack and tapered mesa
fig_xz = Khronos.plot_geometry_slice(
    geom, :y, 0.0;
    x_range = (-sim_xy / 2 + dpml, sim_xy / 2 - dpml),
    y_range = (sim_z_min + dpml, sim_z_max - dpml),
    resolution = 400,
    colormap = :viridis,
    title = "Blue micro-LED — XZ cross-section (y=0)",
)
save("uled_geometry_xz.png", fig_xz)
println("Saved: uled_geometry_xz.png")

# 3-panel cross-sections: XY, XZ, YZ
fig_3panel = Khronos.plot_geometry_cross_sections(
    geom;
    center = [0.0, 0.0, z_dipole],
    span = [sim_xy - 2 * dpml, sim_xy - 2 * dpml, sim_z - 2 * dpml],
    resolution = 400,
    colormap = :viridis,
)
save("uled_geometry_crosssections.png", fig_3panel)
println("Saved: uled_geometry_crosssections.png")

# Additional XY slice through the MQW layer (where dipoles are)
fig_xy_mqw = Khronos.plot_geometry_slice(
    geom, :z, z_dipole;
    x_range = (-1.5, 1.5),
    y_range = (-1.5, 1.5),
    resolution = 400,
    colormap = :viridis,
    title = "XY cross-section at MQW layer (z=$(round(z_dipole, digits=3)) μm)",
)
save("uled_geometry_xy_mqw.png", fig_xy_mqw)
println("Saved: uled_geometry_xy_mqw.png")

# ================================================================== #
# 2. Run a single simulation and show the field pattern
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 2: Single dipole simulation — field pattern")
println("=" ^ 60)

sim_single = make_sim(3 * mesa_radius / 4, :Ey; geometry=geom)

t_single_start = time()
Khronos.run(sim_single,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-6, minimum_runtime = 20.0, maximum_runtime = 200.0))
t_single = time() - t_single_start
println("Single simulation completed in $(round(t_single, digits=1))s")

# Extract field monitor data (the second monitor is the xz-plane DFT)
field_dft_raw = Khronos.get_dft_fields(sim_single.monitors[2])
println("  DFT field array size: ", size(field_dft_raw))
field_dft_max = maximum(abs.(Array(field_dft_raw)))
println("  DFT field max |value|: ", field_dft_max)

if isnan(field_dft_max) || isinf(field_dft_max)
    @error "Simulation produced NaN/Inf fields — check material/PML overlap and stability"
end

field_dft = Array(field_dft_raw[:, :, :, 1])
# The monitor is an xz-plane (y-size=0), so it should be (Nx, 1, Nz) or similar
if size(field_dft, 2) == 1
    field_2d = real(field_dft[:, 1, :])
elseif size(field_dft, 1) == 1
    field_2d = real(field_dft[1, :, :])
elseif size(field_dft, 3) == 1
    field_2d = real(field_dft[:, :, 1])
else
    field_2d = real(field_dft[:, 1, :])
end
println("  2D field size: ", size(field_2d), ", max abs: ", maximum(abs.(field_2d)))

fig_field = Figure(size = (800, 600))
ax_field = Axis(fig_field[1, 1],
    title = "Ey field pattern — dipole at x=$(round(3*mesa_radius/4, digits=3)) μm",
    xlabel = "x (μm)",
    ylabel = "z (μm)",
    aspect = DataAspect(),
)

nfx, nfz = size(field_2d)
fx_range = range(-sim_xy / 2 + dpml, sim_xy / 2 - dpml, length = nfx)
fz_range = range(sim_z_min + dpml, sim_z_max - dpml, length = nfz)

vmax = maximum(abs.(field_2d))
vmax = vmax > 0 ? vmax : 1.0  # guard against degenerate range
hm_f = heatmap!(ax_field, collect(fx_range), collect(fz_range), Float32.(field_2d),
    colormap = :bluesreds,
    colorrange = (-vmax, vmax),
)
try
    Colorbar(fig_field[1, 2], hm_f, label = "Re(Ey)")
catch e
    @warn "Colorbar creation failed ($(typeof(e))); skipping"
end

save("uled_field_pattern_single.png", fig_field)
println("Saved: uled_field_pattern_single.png")

# Also compute and show single-dipole far-field
md_single = sim_single.monitors[1].monitor_data
if isnothing(md_single)
    println("WARNING: Near2FarMonitorData is nothing — skipping far-field plots for single sim")
else
    far_field_single = Khronos.compute_far_field(md_single)
    theta_arr = Float64.(md_single.theta)
    phi_arr = Float64.(md_single.phi)
    power_single = Khronos.compute_far_field_power(far_field_single, theta_arr, phi_arr)
    println("  Single far-field max power: ", maximum(power_single))

    # Single-dipole far-field polar plot
    fig_ff_single = Figure(size = (600, 600))
    ax_ff = PolarAxis(fig_ff_single[1, 1],
        title = "Far-field power — single dipole (Ey, x=0.375 μm)",
        rlimits = (0, 90),
        rticks = [0, 15, 30, 45, 60, 75, 90],
    )
    theta_deg = rad2deg.(theta_arr)
    pmax = maximum(power_single)
    if pmax > 0
        surface!(ax_ff, phi_arr, theta_deg, power_single',
            colormap = :hot,
        )
    end

    save("uled_farfield_single.png", fig_ff_single)
    println("Saved: uled_farfield_single.png")
end

# ================================================================== #
# 3. Batch sweep: 5 dipole positions × 2 polarizations
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 3: Batch sweep — 5 positions × 2 polarizations = 10 sims")
println("=" ^ 60)

dipole_x_list = collect(range(0.0, mesa_radius, length = 5))
pol_list = [:Ex, :Ey]

# Build batch configurations
configs = NamedTuple[]
for dipole_x in dipole_x_list
    for pol in pol_list
        component = pol == :Ex ? Khronos.Ex() : Khronos.Ey()

        n2f_z = lda0 / 8  # inside nGaN (matching Tidy3D)
        config = (
            sources = [
                Khronos.UniformSource(
                    time_profile = Khronos.GaussianPulseSource(fcen = freq0, fwidth = fwidth),
                    component = component,
                    center = [dipole_x, 0.0, z_dipole],
                    size = [0.0, 0.0, 0.0],
                    amplitude = 1.0,
                ),
            ],
            monitors = [
                Khronos.Near2FarMonitor(
                    center = [0.0, 0.0, n2f_z],
                    size = [n2f_xy, n2f_xy, 0.0],
                    frequencies = [freq0],
                    theta = collect(range(0.0, π / 2, length = 91)),
                    phi = collect(range(0.0, 2π - 2π / 72, length = 72)),
                    normal_dir = :+,
                    medium_eps = n_nGaN^2,  # project through nGaN medium
                ),
                Khronos.DFTMonitor(
                    component = component,
                    center = [0.0, 0.0, sim_center_z],
                    size = [sim_xy - 2 * dpml, 0.0, sim_z - 2 * dpml],
                    frequencies = [freq0],
                ),
            ],
        )
        push!(configs, config)
    end
end

template_kwargs = (
    cell_size = [sim_xy, sim_xy, sim_z],
    cell_center = [0.0, 0.0, sim_center_z],
    resolution = resolution,
    geometry = geom,
    boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
)

t_batch_start = time()
results = Khronos.run_batch_concurrent(configs;
    template_kwargs = template_kwargs,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-6, minimum_runtime = 20.0, maximum_runtime = 200.0))
t_batch = time() - t_batch_start
println("Concurrent batch sweep completed in $(round(t_batch, digits=1))s")

# ================================================================== #
# 4. Show field patterns for a few selected dipoles
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 4: Field patterns from batch")
println("=" ^ 60)

# Show field patterns for 3 representative dipoles
selected_indices = [1, 5, 9]  # x=0/Ex, x=0.25/Ex, x=0.5/Ex
selected_labels = [
    "x=0, pol=Ex",
    "x=0.25μm, pol=Ex",
    "x=0.5μm, pol=Ex",
]

fig_fields = Figure(size = (1200, 400))
for (col, (idx, label)) in enumerate(zip(selected_indices, selected_labels))
    r = results[idx]
    # Field monitor is the second monitor in each config
    mon2 = r.monitors[2]
    if isnothing(mon2.monitor_data)
        println("  Result $idx: no monitor_data for field monitor")
        continue
    end
    field_data = Array(Khronos.get_dft_fields(mon2)[:, :, :, 1])
    if size(field_data, 2) == 1
        field_slice = real(field_data[:, 1, :])
    elseif size(field_data, 1) == 1
        field_slice = real(field_data[1, :, :])
    else
        field_slice = real(field_data[:, 1, :])
    end

    local nfx, nfz = size(field_slice)
    local fx_range = range(-sim_xy / 2 + dpml, sim_xy / 2 - dpml, length = nfx)
    local fz_range = range(sim_z_min + dpml, sim_z_max - dpml, length = nfz)

    ax = Axis(fig_fields[1, col],
        title = label,
        xlabel = col == 2 ? "x (μm)" : "",
        ylabel = col == 1 ? "z (μm)" : "",
        aspect = DataAspect(),
    )
    local vmax = maximum(abs.(field_slice))
    vmax = vmax > 0 ? vmax : 1.0
    heatmap!(ax, collect(fx_range), collect(fz_range), field_slice,
        colormap = :bluesreds, colorrange = (-vmax, vmax))
end

save("uled_field_patterns_batch.png", fig_fields)
println("Saved: uled_field_patterns_batch.png")

# ================================================================== #
# 5. Incoherent power summation and LEE
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 5: Incoherent power sum and LEE")
println("=" ^ 60)

theta_arr = collect(range(0.0, π / 2, length = 91))
phi_arr = collect(range(0.0, 2π - 2π / 72, length = 72))

total_power = zeros(length(theta_arr), length(phi_arr))
n_valid = 0

for (i, r) in enumerate(results)
    # Find the Near2FarMonitor (first monitor in each config)
    mon = r.monitors[1]
    if !(mon isa Khronos.Near2FarMonitor)
        continue
    end
    md = mon.monitor_data
    if isnothing(md)
        @warn "Result $i: no Near2FarMonitorData"
        continue
    end

    # Check if tangential E fields are non-zero and non-NaN
    e_fields = Array(Khronos.get_dft_fields(md.tangential_E_monitors[1]))
    e_max = maximum(abs.(e_fields))
    if isnan(e_max) || isinf(e_max)
        @warn "Result $i: NaN/Inf in tangential E fields — skipping"
        continue
    end
    if e_max == 0
        @warn "Result $i: zero tangential E fields"
        continue
    end

    far_field = Khronos.compute_far_field(md)
    power = Khronos.compute_far_field_power(far_field, theta_arr, phi_arr)
    total_power .+= power
    global n_valid += 1
    println("  Sim $i/$(length(results)): max power = $(maximum(power))")
end

println("Used $n_valid / $(length(results)) valid simulations")

# LEE at various cone half-angles (angles measured in GaN medium)
# Since medium_eps = n_nGaN², the far-field angles θ are in-medium angles.
# Snell’s law: n_nGaN*sin(θ_GaN) = sin(θ_air)
# Critical angle in GaN: arcsin(1/n_nGaN) ≈ 24.1° — light beyond this is TIR.
lee_15 = Khronos.compute_LEE(total_power, theta_arr, phi_arr;
    cone_half_angle = deg2rad(15.0))
lee_30 = Khronos.compute_LEE(total_power, theta_arr, phi_arr;
    cone_half_angle = deg2rad(30.0))
# Critical angle: all light within this cone escapes into air
theta_crit = asin(1.0 / n_nGaN)  # ≈ 24.1°
lee_crit = Khronos.compute_LEE(total_power, theta_arr, phi_arr;
    cone_half_angle = theta_crit)
# Full hemisphere
lee_full = Khronos.compute_LEE(total_power, theta_arr, phi_arr;
    cone_half_angle = π / 2)

println("\n", "=" ^ 60)
println("Light Extraction Efficiency Results")
println("=" ^ 60)
println("  Note: angles are in GaN medium (n=$n_nGaN), not air")
println("  Critical angle (TIR): $(round(rad2deg(theta_crit), digits=1))°")
println()
println("  LEE (±15° GaN): $(round(100 * lee_15, digits=2))%")
println("  LEE (±24° GaN / escape cone): $(round(100 * lee_crit, digits=2))%")
println("  LEE (±30° GaN): $(round(100 * lee_30, digits=2))%")
println("  LEE (full hemisphere):   $(round(100 * lee_full, digits=2))%")
println("=" ^ 60)

# ================================================================== #
# 6. Far-field polar plots
# ================================================================== #

println("\n", "=" ^ 60)
println("STEP 6: Far-field visualizations")
println("=" ^ 60)

# --- Incoherent sum far-field polar plot ---
fig_ff = Figure(size = (600, 600))
ax_polar = PolarAxis(fig_ff[1, 1],
    title = "Far-field power — incoherent sum (10 dipoles)",
    rlimits = (0, 90),
    rticks = [0, 15, 30, 45, 60, 75, 90],
)

theta_deg = rad2deg.(theta_arr)
pmax_total = maximum(total_power)
if pmax_total > 0
    surface!(ax_polar, phi_arr, theta_deg, total_power',
        colormap = :hot,
    )
end

save("uled_farfield_incoherent.png", fig_ff)
println("Saved: uled_farfield_incoherent.png")

# --- Azimuthally-averaged far-field (rotationally symmetric) ---
power_avg = mean(total_power, dims=2)  # average over phi

# --- 1D line plot: power vs theta (averaged over phi) ---
fig_line = Figure(size = (700, 400))
ax_line = Axis(fig_line[1, 1],
    title = "Radiated power vs polar angle in GaN (azimuthally averaged)",
    xlabel = "Polar angle θ in GaN (degrees)",
    ylabel = "Power (a.u.)",
)

lines!(ax_line, theta_deg, vec(power_avg), color = :red, linewidth = 2)
vlines!(ax_line, [15.0], color = :blue, linestyle = :dash, label = "15° cone")
vlines!(ax_line, [rad2deg(theta_crit)], color = :orange, linestyle = :dashdot, label = "critical angle ($(round(rad2deg(theta_crit),digits=1))°)")
vlines!(ax_line, [30.0], color = :green, linestyle = :dash, label = "30° cone")
axislegend(ax_line, position = :rt)

save("uled_power_vs_angle.png", fig_line)
println("Saved: uled_power_vs_angle.png")

println("\n", "=" ^ 60)
println("ALL OUTPUTS COMPLETE")
println("=" ^ 60)
println("\nGenerated files:")
println("  uled_geometry_xz.png            — Detailed XZ permittivity cross-section")
println("  uled_geometry_crosssections.png — XY/XZ/YZ cross-sections at dipole plane")
println("  uled_geometry_xy_mqw.png        — XY slice through MQW layer")
println("  uled_field_pattern_single.png   — Ey field for single dipole")
println("  uled_farfield_single.png        — Polar far-field for single dipole")
println("  uled_field_patterns_batch.png   — Field patterns for 3 dipole positions")
println("  uled_farfield_incoherent.png    — Polar far-field (incoherent sum)")
println("  uled_power_vs_angle.png         — Power vs angle line plot")
