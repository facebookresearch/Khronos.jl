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
YAML_FILENAME = joinpath(@__DIR__, "periodic_stack.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run = detect_and_set_backend()

# current hardware
hardware_key = get_hardware_key()

function build_periodic_stack(resolution::Real, z_scaling::Real)
    z_thickness = 10 * z_scaling

    #TODO swap out for actual planewave source once ready
    src_z = (-z_thickness / 2.0) + 1.0
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

    mat_low = fdtd.Material(ε = 1.5)
    mat_mid = fdtd.Material(ε = 2.5)
    mat_high = fdtd.Material(ε = 3.5)

    materials = [mat_low, mat_mid, mat_low, mat_high, mat_mid, mat_high]
    thicknesses = [0.5, 1.0, 0.75, 1.0, 0.25, 0.5] * z_scaling

    z_cur = -sum(thicknesses) / 2
    geometry = []
    for (current_mat, current_thick) in zip(materials, thicknesses)
        z_cur += current_thick / 2.0
        append!(
            geometry,
            [
                fdtd.Object(
                    Cuboid([0.0, 0.0, z_cur], [4.0, 4.0, current_thick]),
                    current_mat,
                ),
            ],
        )
        z_cur += current_thick / 2.0
    end

    sim = fdtd.Simulation(
        cell_size = [4.0, 4.0, z_thickness],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )

    return sim
end

@testset "Benchmark: dielectric periodic stack" begin
    TESTNAME = "dielectric_periodic_stack"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        z_scaling = benchmark["z_scaling"]

        @testset "resolution: $resolution | z_scaling: $z_scaling" begin

            sim = build_periodic_stack(resolution, z_scaling)
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
