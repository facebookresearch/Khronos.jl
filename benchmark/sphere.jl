# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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
YAML_FILENAME = joinpath(@__DIR__, "sphere.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_sphere_sim(resolution, radius; include_loss = false)

    s_xyz = 2.0 + 1.0 + 2 * radius

    src_z = -s_xyz / 2.0 + 1.0
    #TODO swap out for actual planewave source once ready
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ex(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Hy(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
    ]

    if include_loss
        mat = Khronos.Material(ε = 3, σD = 5)
    else
        mat = Khronos.Material(ε = 3)
    end

    geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], radius), mat)]

    sim = Khronos.Simulation(
        cell_size = [s_xyz, s_xyz, s_xyz],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        geometry = geometry,
    )

    return sim
end

@testset "Benchmark: scattering off sphere" begin
    TESTNAME = "scattering_off_sphere"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        radius = benchmark["radius"]

        @testset "resolution: $resolution | radius: $radius" begin

            sim = build_sphere_sim(resolution, radius)
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
                    label="sphere (res=$resolution, r=$radius)")
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
