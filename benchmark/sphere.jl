# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
import YAML
import fdtd

using KernelAbstractions
using Logging
using Test
using GeometryPrimitives
include("benchmark_utils.jl")
using .BenchmarkUtils

debuglogger = ConsoleLogger(stderr, Logging.Warn)
global_logger(debuglogger)

# contains all the relevant profiling metrics
YAML_FILENAME = joinpath(@__DIR__, "sphere.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run = detect_and_set_backend()

# current hardware
hardware_key = get_hardware_key()

function build_sphere_sim(resolution, radius; include_loss = false)

    s_xyz = 2.0 + 1.0 + 2 * radius

    src_z = -s_xyz / 2.0 + 1.0
    #TODO swap out for actual planewave source once ready
    sources = [
        fdtd.UniformSource(
            time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
            component = fdtd.Ex(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
        fdtd.UniformSource(
            time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
            component = fdtd.Hy(),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
        ),
    ]

    if include_loss
        mat = fdtd.Material(ε = 3, σD = 5)
    else
        mat = fdtd.Material(ε = 3)
    end

    geometry = [fdtd.Object(Ball([0.0, 0.0, 0.0], radius), mat)]

    sim = fdtd.Simulation(
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
            timstep_rate = fdtd.run_benchmark(sim, 110)
            benchmark_result(
                timstep_rate,
                benchmark_rate,
                tolerance,
                profile_run,
                benchmark,
            )
        end
    end
end

if profile_run
    YAML.write_file(YAML_FILENAME, profiling_results)
end
