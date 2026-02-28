# Copyright (c) Meta Platforms, Inc. and affiliates.
#
import YAML
import Khronos

using KernelAbstractions
using Logging
using Test
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
YAML_FILENAME = joinpath(@__DIR__, "dipole.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_dipole_simulation(resolution, sim_xyz)
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ez(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ]

    sim = Khronos.Simulation(
        cell_size = sim_xyz * [1.0, 1.0, 1.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )

    return sim
end

@testset "Benchmark: dipole in vacuum" begin
    TESTNAME = "simple_dipole"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        size_xyz = benchmark["size_xyz"]

        @testset "resolution: $resolution | size_xyz: $size_xyz" begin

            sim = build_dipole_simulation(resolution, size_xyz)
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
                    label="dipole (res=$resolution, size=$size_xyz)")
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
