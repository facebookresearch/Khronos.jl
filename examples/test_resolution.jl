# Quick resolution study: single dipole LEE at low vs high resolution
# Tests whether grid resolution is the primary cause of LEE discrepancy

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Statistics

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

const nm = 1e-3
const lda0 = 450nm
const freq0 = 1.0 / lda0
const fwidth = freq0 / 10
const t_mirror = 120nm; const t_SiO2 = 100nm; const t_Al2O3 = 40nm
const t_ITO = 50nm; const t_pGaN = 150nm; const t_MQW = 50nm; const t_nGaN = 1000nm
const n_nGaN = 2.45; const n_MQW = 2.60; const n_pGaN = 2.46
const n_ITO = 2.00; const n_Al2O3 = 1.76; const n_SiO2 = 1.46
const mesa_radius = 500nm; const sidewall_angle_deg = -45.0
const t_1 = 500nm; const z_dipole = -t_1 - t_MQW / 2
const dpml = 0.5; const sim_xy = 2 * dpml + 3.0
const sim_z_min = -1.7; const sim_z_max = 3.0
const sim_z = sim_z_max - sim_z_min
const sim_center_z = (sim_z_min + sim_z_max) / 2

function run_at_resolution(res; geometry=nothing)
    if isnothing(geometry)
        geometry = build_geometry_full()
    end

    n2f_z = t_nGaN + 2 * lda0  # 2 wavelengths above GaN/air interface (eliminates evanescent contamination)

    sim = Khronos.Simulation(
        cell_size = [sim_xy, sim_xy, sim_z],
        cell_center = [0.0, 0.0, sim_center_z],
        resolution = res,
        geometry = geometry,
        sources = [
            Khronos.UniformSource(
                time_profile = Khronos.GaussianPulseSource(fcen = freq0, fwidth = fwidth),
                component = Khronos.Ey(),
                center = [3 * mesa_radius / 4, 0.0, z_dipole],
                size = [0.0, 0.0, 0.0],
                amplitude = 1.0,
            ),
        ],
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        monitors = [
            Khronos.Near2FarMonitor(
                center = [0.0, 0.0, n2f_z],
                size = [sim_xy - 2 * dpml, sim_xy - 2 * dpml, 0.0],
                frequencies = [freq0],
                theta = collect(range(0.0, π / 2, length = 91)),
                phi = collect(range(0.0, 2π - 2π / 72, length = 72)),
                normal_dir = :+,
                medium_eps = 1.0,
            ),
        ],
    )

    Nx = round(Int, sim_xy * res)
    Ny = Nx
    Nz = round(Int, sim_z * res)
    nvoxels = Nx * Ny * Nz
    lda_min = lda0 / n_nGaN
    pts_per_wvl = lda_min * res
    println("  Grid: $(Nx)×$(Ny)×$(Nz) = $(nvoxels) voxels, $(round(pts_per_wvl, digits=1)) pts/wvl in nGaN")

    t0 = time()
    Khronos.run(sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance = 1e-6, minimum_runtime = 20.0, maximum_runtime = 200.0))
    t_sim = time() - t0

    md = sim.monitors[1].monitor_data
    theta_arr = Float64.(md.theta)
    phi_arr = Float64.(md.phi)
    ff = Khronos.compute_far_field(md)
    power = Khronos.compute_far_field_power(ff, theta_arr, phi_arr)

    lee_15 = Khronos.compute_LEE(power, theta_arr, phi_arr; cone_half_angle = deg2rad(15.0))
    lee_30 = Khronos.compute_LEE(power, theta_arr, phi_arr; cone_half_angle = deg2rad(30.0))

    println("  Simulation: $(round(t_sim, digits=1))s")
    println("  LEE (±15°): $(round(100 * lee_15, digits=2))%")
    println("  LEE (±30°): $(round(100 * lee_30, digits=2))%")
    println("  Max power: $(maximum(power))")

    return lee_15, lee_30
end

# Extract geometry builder to a shared function
function build_geometry_full()
    objects = Khronos.Object[]

    # Silver material (Drude model for Ag at 450nm)
    eps_ag = (0.028 + 2.88im)^2
    eps_inf = 1.0
    chi_target = eps_ag - eps_inf
    ω0 = 2π * freq0
    chi_r = real(chi_target); chi_i = imag(chi_target)
    Gamma = -ω0 * chi_i / chi_r
    gamma_meep = Gamma / (2π)
    sigma = -chi_r * (ω0^2 + Gamma^2) / Gamma
    ag_mat = Khronos.Material(ε = eps_inf, susceptibilities = [Khronos.DrudeSusceptibility(gamma_meep, sigma)])

    # ================================================================
    # Painter's algorithm: LATER objects override EARLIER ones.
    # Must match Tidy3D ordering: background layers first, mesa last.
    # ================================================================

    # --- 1. Silver reflector (background, painted first) ---
    # Thick slab covering bottom of domain (matching Tidy3D t_2=1.5)
    t_2 = 1.5
    z_ag_top = -(t_Al2O3 + t_SiO2)
    z_ag_bot = -(t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3 + t_SiO2 + t_2)
    ag_center_z = (z_ag_top + z_ag_bot) / 2; ag_height = z_ag_top - z_ag_bot
    ag_lateral = sim_xy - 2 * dpml  # stay within PML boundaries laterally
    push!(objects, Khronos.Object(shape = Cuboid([0.0, 0.0, ag_center_z], [ag_lateral, ag_lateral, ag_height]), material = ag_mat))

    # --- 2. SiO2 layers (slab + tapered cone around mesa) ---
    z_sio2_top = -t_Al2O3; z_sio2_bot = z_sio2_top - t_SiO2
    sio2_center_z = (z_sio2_top + z_sio2_bot) / 2
    push!(objects, Khronos.Object(shape = Cuboid([0.0, 0.0, sio2_center_z], [sim_xy, sim_xy, t_SiO2]), material = Khronos.Material(ε = n_SiO2^2)))

    mesa_stack_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    sio2_cone_center_z = -t_SiO2 - mesa_stack_height / 2
    radius_center = 1.2; half_h = mesa_stack_height / 2
    r_bot_sio2 = radius_center + half_h * tan(deg2rad(sidewall_angle_deg))
    r_top_sio2 = radius_center - half_h * tan(deg2rad(sidewall_angle_deg))
    push!(objects, Khronos.Object(shape = TruncatedCone([0.0, 0.0, sio2_cone_center_z], r_bot_sio2, r_top_sio2, mesa_stack_height, [0.0, 0.0, 1.0]), material = Khronos.Material(ε = n_SiO2^2)))

    # --- 3. Al2O3 layers (slab + coating cylinder around mesa) ---
    z_al2o3_top = 0.0; z_al2o3_bot = -t_Al2O3
    al2o3_center_z = (z_al2o3_top + z_al2o3_bot) / 2
    push!(objects, Khronos.Object(shape = Cuboid([0.0, 0.0, al2o3_center_z], [sim_xy, sim_xy, t_Al2O3]), material = Khronos.Material(ε = n_Al2O3^2)))

    al2o3_coat_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    al2o3_coat_center_z = -al2o3_coat_height / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, al2o3_coat_center_z], mesa_radius + t_Al2O3, al2o3_coat_height), material = Khronos.Material(ε = n_Al2O3^2)))

    # --- 4. nGaN (slab + mesa cylinder — overrides silver/SiO2/Al2O3 inside mesa) ---
    nGaN_slab_center_z = t_nGaN / 2
    push!(objects, Khronos.Object(shape = Cuboid([0.0, 0.0, nGaN_slab_center_z], [sim_xy, sim_xy, t_nGaN]), material = Khronos.Material(ε = n_nGaN^2)))

    nGaN_cyl_center_z = -t_1 / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, nGaN_cyl_center_z], mesa_radius, t_1), material = Khronos.Material(ε = n_nGaN^2)))

    # --- 5. MQW (inside mesa — overrides nGaN) ---
    mqw_center_z = -t_1 - t_MQW / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, mqw_center_z], mesa_radius, t_MQW), material = Khronos.Material(ε = n_MQW^2)))

    # --- 6. pGaN (inside mesa — overrides nGaN) ---
    pGaN_center_z = -t_1 - t_MQW - t_pGaN / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, pGaN_center_z], mesa_radius, t_pGaN), material = Khronos.Material(ε = n_pGaN^2)))

    # --- 7. ITO (inside mesa — overrides nGaN) ---
    ito_center_z = -t_1 - t_MQW - t_pGaN - t_ITO / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, ito_center_z], mesa_radius, t_ITO), material = Khronos.Material(ε = n_ITO^2)))

    # --- 8. Metal contact (small cylinder through SiO2/Al2O3 — painted last) ---
    contact_height = t_SiO2 + t_Al2O3
    contact_center_z = -t_1 - t_MQW - t_pGaN - t_ITO - contact_height / 2
    push!(objects, Khronos.Object(shape = Cylinder([0.0, 0.0, contact_center_z], 0.2, contact_height), material = ag_mat))

    return objects
end

geom = build_geometry_full()

println("=== Resolution study: single Ey dipole at x=0.375μm ===")
println("  Geometry ordering: background layers first, mesa layers last (Tidy3D style)")
println()

for res in [40]
    println("--- Resolution: $res pts/μm ---")
    run_at_resolution(res; geometry=geom)
    println()
end
