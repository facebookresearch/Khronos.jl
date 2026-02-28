# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Silicon waveguide with mode source excitation.
#
# Exercises: ModeSource, ModeMonitor, DFTMonitor, SOI geometry (Si/SiO2),
# subpixel smoothing at waveguide boundaries.
#
# Based on examples/waveguide.jl and examples/ring_coupler.jl.

import YAML
import Khronos

using KernelAbstractions
using Logging
using Test
using GeometryPrimitives
include("benchmark_utils.jl")
using .BenchmarkUtils

debuglogger = ConsoleLogger(stderr, Logging.Warn)
global_logger(debuglogger)

# contains all the relevant profiling metrics
YAML_FILENAME = joinpath(@__DIR__, "waveguide_mode.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run = detect_and_set_backend()

# current hardware
hardware_key = get_hardware_key()

function build_waveguide_mode_sim(resolution, wg_length)
    # Standard SOI parameters
    n_Si = 3.47
    n_SiO2 = 1.44
    ε_Si = n_Si^2
    ε_SiO2 = n_SiO2^2

    wg_width = 0.5    # μm
    wg_height = 0.22  # μm
    λ = 1.55          # μm

    geometry = [
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [wg_length + 4.0, wg_width, wg_height]),
            Khronos.Material(ε = ε_Si),
        ),
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [wg_length + 10.0, wg_length + 10.0, wg_length + 10.0]),
            Khronos.Material(ε = ε_SiO2),
        ),
    ]

    sources = [
        Khronos.ModeSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / λ),
            frequency = 1.0 / λ,
            mode_solver_resolution = 50,
            mode_index = 1,
            center = [-wg_length / 2 + 1.0, 0.0, 0.0],
            size = [0.0, 2.0, 2.0],
            solver_tolerance = 1e-6,
            geometry = geometry,
        ),
    ]

    pml_thickness = 1.0
    cell_xy = 4.0 + 2 * pml_thickness
    cell_x = wg_length + 2 * pml_thickness
    cell_z = 6 * wg_height + 2 * pml_thickness

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_xy, cell_z],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[pml_thickness, pml_thickness], [pml_thickness, pml_thickness], [pml_thickness, pml_thickness]],
    )

    return sim
end

@testset "Benchmark: waveguide with mode source" begin
    TESTNAME = "waveguide_mode_source"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        wg_length = benchmark["wg_length"]

        @testset "resolution: $resolution | wg_length: $wg_length" begin

            sim = build_waveguide_mode_sim(resolution, wg_length)
            timstep_rate = Khronos.run_benchmark(sim, 110)
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
