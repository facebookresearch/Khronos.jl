# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: 2×2 MMI coupler for optical hybrid.
#
# Exercises: Multiple Cuboid geometry objects, ModeSource, ModeMonitor,
# DFTMonitor, TE z-symmetry, SOI platform.
#
# Based on examples/optical_hybrid_mmi.jl (ported from Tidy3D 90OpticalHybrid).

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
YAML_FILENAME = joinpath(@__DIR__, "mmi_coupler.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_mmi_coupler_sim(resolution, mmi_length_scale)
    # Standard SOI parameters
    n_Si = 3.47
    n_SiO2 = 1.45
    ε_Si = n_Si^2
    ε_SiO2 = n_SiO2^2
    mat_si = Khronos.Material(ε = ε_Si)
    mat_sio2 = Khronos.Material(ε = ε_SiO2)

    wg_width = 0.5    # μm
    thickness = 0.22  # μm

    # MMI dimensions scaled by mmi_length_scale
    mmi_width = 2.8 * mmi_length_scale
    mmi_length = 5.4 * mmi_length_scale
    y_port = 0.45 * mmi_length_scale
    L_access = 4.0

    λ_center = 1.55 # μm

    mmi_center_x = 0.0
    input_x = mmi_center_x - mmi_length / 2 - L_access / 2
    output_x = mmi_center_x + mmi_length / 2 + L_access / 2

    geometry = [
        # MMI multimode section
        Khronos.Object(
            Cuboid([mmi_center_x, 0.0, 0.0], [mmi_length, mmi_width, thickness]),
            mat_si,
        ),
        # Input waveguide — upper port
        Khronos.Object(
            Cuboid([input_x, y_port, 0.0], [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Input waveguide — lower port
        Khronos.Object(
            Cuboid([input_x, -y_port, 0.0], [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Output waveguide — upper port
        Khronos.Object(
            Cuboid([output_x, y_port, 0.0], [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Output waveguide — lower port
        Khronos.Object(
            Cuboid([output_x, -y_port, 0.0], [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Background cladding (SiO2)
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            mat_sio2,
        ),
    ]

    freq_center = 1.0 / λ_center

    source_x = -(mmi_length / 2 + L_access - λ_center / 2)

    sources = [
        Khronos.ModeSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = freq_center),
            frequency = freq_center,
            mode_solver_resolution = 50,
            mode_index = 1,
            center = [source_x, y_port, 0.0],
            size = [0.0, 2 * wg_width, 6 * thickness],
            solver_tolerance = 1e-6,
            geometry = geometry,
        ),
    ]

    domain_x = mmi_length + 2 * L_access + 2 * λ_center
    domain_y = mmi_width + 2 * λ_center
    domain_z = 10 * thickness
    pml_thickness = 1.0

    cell_x = domain_x + 2 * pml_thickness
    cell_y = domain_y + 2 * pml_thickness
    cell_z = domain_z + 2 * pml_thickness

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, cell_z],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[pml_thickness, pml_thickness], [pml_thickness, pml_thickness], [pml_thickness, pml_thickness]],
        symmetry = (0, 0, 1),
    )

    return sim
end

@testset "Benchmark: MMI coupler" begin
    TESTNAME = "mmi_coupler"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        mmi_length_scale = benchmark["mmi_length_scale"]

        @testset "resolution: $resolution | mmi_length_scale: $mmi_length_scale" begin

            sim = build_mmi_coupler_sim(resolution, mmi_length_scale)
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
                    label="mmi_coupler (res=$resolution, scale=$mmi_length_scale)")
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
