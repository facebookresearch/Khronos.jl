# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Blue micro-LED with dispersive metal and near-to-far field.
#
# Exercises: Drude dispersive materials (silver reflector — ADE update kernels),
# TruncatedCone geometry, Cylinder geometry, multiple material layers,
# Near2FarMonitor with layered projection, point dipole source.
#
# Based on examples/uled.jl (ported from Tidy3D BlueMicroLED tutorial).
# Reference: Vogl et al., Opt. Lett. 49, 5095-5098 (2024).

import YAML
import Khronos

using KernelAbstractions
using Logging
using Test
using GeometryPrimitives
if !@isdefined(BenchmarkUtils)
    include("benchmark_utils.jl")
end
using .BenchmarkUtils
if !@isdefined(BenchmarkMetrics)
    include("benchmark_metrics.jl")
end
using .BenchmarkMetrics

debuglogger = ConsoleLogger(stderr, Logging.Warn)
global_logger(debuglogger)

# contains all the relevant profiling metrics
YAML_FILENAME = joinpath(@__DIR__, "uled.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_uled_sim(resolution, domain_scale)
    nm = 1e-3  # nm → μm

    # Wavelength & frequency
    lda0 = 450nm
    freq0 = 1.0 / lda0
    fwidth = freq0 / 10

    # Layer thicknesses
    t_mirror = 120nm
    t_SiO2   = 100nm
    t_Al2O3  = 40nm
    t_ITO    = 50nm
    t_pGaN   = 150nm
    t_MQW    = 50nm
    t_nGaN   = 1000nm

    # Refractive indices at 450 nm
    n_nGaN = 2.45
    n_MQW  = 2.60
    n_pGaN = 2.46
    n_ITO  = 2.00
    n_Al2O3_val = 1.76
    n_SiO2_val  = 1.46

    # Mesa geometry
    mesa_radius = 500nm * domain_scale
    t_1 = 500nm

    z_dipole = -t_1 - t_MQW / 2

    # Simulation domain
    dpml = 0.5
    sim_xy = 2 * dpml + 4.0 * domain_scale
    sim_z_min = -1.7
    sim_z_max = 0.8
    sim_z = sim_z_max - sim_z_min
    sim_center_z = (sim_z_min + sim_z_max) / 2

    # Silver Drude model
    eps_ag = (0.028 + 2.88im)^2
    eps_inf = 1.0
    chi_target = eps_ag - eps_inf
    ω0 = 2π * freq0
    chi_r = real(chi_target)
    chi_i = imag(chi_target)
    Gamma = -ω0 * chi_i / chi_r
    gamma_meep = Gamma / (2π)
    sigma = -chi_r * (ω0^2 + Gamma^2) / Gamma
    ag_mat = Khronos.Material(
        ε = eps_inf,
        susceptibilities = [Khronos.DrudeSusceptibility(gamma_meep, sigma)],
    )

    objects = Khronos.Object[]

    # Silver reflector
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

    # SiO2 layers
    z_sio2_top = -t_Al2O3
    z_sio2_bot = z_sio2_top - t_SiO2
    sio2_center_z = (z_sio2_top + z_sio2_bot) / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, sio2_center_z], [sim_xy, sim_xy, t_SiO2]),
        material = Khronos.Material(ε = n_SiO2_val^2),
    ))

    mesa_stack_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    sio2_cone_center_z = -t_SiO2 - mesa_stack_height / 2
    radius_center = 1.2 * domain_scale
    half_h = mesa_stack_height / 2
    sidewall_angle_deg = -45.0
    r_bot_sio2 = radius_center + half_h * tan(deg2rad(sidewall_angle_deg))
    r_top_sio2 = radius_center - half_h * tan(deg2rad(sidewall_angle_deg))
    push!(objects, Khronos.Object(
        shape = TruncatedCone(
            [0.0, 0.0, sio2_cone_center_z],
            r_bot_sio2, r_top_sio2,
            mesa_stack_height, [0.0, 0.0, 1.0],
        ),
        material = Khronos.Material(ε = n_SiO2_val^2),
    ))

    # Al2O3 layers
    al2o3_center_z = -t_Al2O3 / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, al2o3_center_z], [sim_xy, sim_xy, t_Al2O3]),
        material = Khronos.Material(ε = n_Al2O3_val^2),
    ))

    al2o3_coat_height = t_1 + t_MQW + t_pGaN + t_ITO + t_Al2O3
    al2o3_coat_center_z = -al2o3_coat_height / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, al2o3_coat_center_z],
                         mesa_radius + t_Al2O3, al2o3_coat_height),
        material = Khronos.Material(ε = n_Al2O3_val^2),
    ))

    # nGaN slab + mesa cylinder
    nGaN_slab_center_z = t_nGaN / 2
    push!(objects, Khronos.Object(
        shape = Cuboid([0.0, 0.0, nGaN_slab_center_z], [sim_xy, sim_xy, t_nGaN]),
        material = Khronos.Material(ε = n_nGaN^2),
    ))

    nGaN_cyl_center_z = -t_1 / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, nGaN_cyl_center_z], mesa_radius, t_1),
        material = Khronos.Material(ε = n_nGaN^2),
    ))

    # MQW active layer
    mqw_center_z = -t_1 - t_MQW / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, mqw_center_z], mesa_radius, t_MQW),
        material = Khronos.Material(ε = n_MQW^2),
    ))

    # pGaN layer
    pGaN_center_z = -t_1 - t_MQW - t_pGaN / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, pGaN_center_z], mesa_radius, t_pGaN),
        material = Khronos.Material(ε = n_pGaN^2),
    ))

    # ITO contact
    ito_center_z = -t_1 - t_MQW - t_pGaN - t_ITO / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, ito_center_z], mesa_radius, t_ITO),
        material = Khronos.Material(ε = n_ITO^2),
    ))

    # Metal contact via
    contact_height = t_SiO2 + t_Al2O3
    contact_center_z = -t_1 - t_MQW - t_pGaN - t_ITO - contact_height / 2
    push!(objects, Khronos.Object(
        shape = Cylinder([0.0, 0.0, contact_center_z], 0.2, contact_height),
        material = ag_mat,
    ))

    # Reverse for findfirst priority
    geometry = reverse(objects)

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = freq0),
            component = Khronos.Ey(),
            center = [mesa_radius * 0.75, 0.0, z_dipole],
            size = [0.0, 0.0, 0.0],
            amplitude = 1.0,
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = [sim_xy, sim_xy, sim_z],
        cell_center = [0.0, 0.0, sim_center_z],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
    )

    return sim
end

@testset "Benchmark: micro-LED with dispersive metal" begin
    TESTNAME = "uled_dispersive"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        domain_scale = benchmark["domain_scale"]

        @testset "resolution: $resolution | domain_scale: $domain_scale" begin

            sim = build_uled_sim(resolution, domain_scale)
            timstep_rate = Khronos.run_benchmark(sim, 110)
            benchmark_result(
                timstep_rate,
                benchmark_rate,
                tolerance,
                profile_run,
                benchmark,
            )

            if metrics_run
                collect_and_store_metrics(sim, precision_type, benchmark;
                    label="uled (res=$resolution, scale=$domain_scale)")
            end
        end
    end

    # Store kernel metrics once (not per-config — registers don't change with grid size)
    if metrics_run
        km = collect_kernel_metrics(precision_type)
        if !haskey(profiling_results[TESTNAME], "kernel_metrics")
            profiling_results[TESTNAME]["kernel_metrics"] = Dict{String,Any}()
        end
        profiling_results[TESTNAME]["kernel_metrics"][precision] = kernel_metrics_to_dict(km)
    end
end

if profile_run || metrics_run
    YAML.write_file(YAML_FILENAME, profiling_results)
end
