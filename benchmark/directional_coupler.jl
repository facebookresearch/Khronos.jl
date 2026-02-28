# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Silicon photonics directional coupler.
#
# Exercises: Closely spaced parallel waveguides (fine geometry features),
# ModeSource, evanescent coupling region, SOI platform.
# Tests subpixel smoothing accuracy at narrow gaps.
#
# Inspired by SiEPIC directional coupler designs and examples/psr_bend_90.jl.

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
YAML_FILENAME = joinpath(@__DIR__, "directional_coupler.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run = detect_and_set_backend()

# current hardware
hardware_key = get_hardware_key()

function build_directional_coupler_sim(resolution, coupling_length)
    # Standard SOI parameters
    n_Si = 3.47
    n_SiO2 = 1.44
    ε_Si = n_Si^2
    ε_SiO2 = n_SiO2^2
    mat_si = Khronos.Material(ε = ε_Si)
    mat_sio2 = Khronos.Material(ε = ε_SiO2)

    wg_width = 0.5    # μm
    thickness = 0.22  # μm
    gap = 0.2         # μm — coupling gap between waveguides

    λ_center = 1.55   # μm
    freq_center = 1.0 / λ_center

    # Waveguide center y-positions
    y_upper = (gap + wg_width) / 2
    y_lower = -(gap + wg_width) / 2

    # Total device length: input taper + coupling region + output taper
    L_access = 3.0   # μm — straight access waveguide length
    L_taper = 2.0    # μm — S-bend taper length (simplified as straight here)
    L_total = 2 * L_access + 2 * L_taper + coupling_length

    geometry = [
        # Upper waveguide — coupling region
        Khronos.Object(
            Cuboid([0.0, y_upper, 0.0], [coupling_length + 2 * L_taper, wg_width, thickness]),
            mat_si,
        ),
        # Lower waveguide — coupling region
        Khronos.Object(
            Cuboid([0.0, y_lower, 0.0], [coupling_length + 2 * L_taper, wg_width, thickness]),
            mat_si,
        ),
        # Upper input access waveguide (wider separation)
        Khronos.Object(
            Cuboid([-(coupling_length / 2 + L_taper + L_access / 2), y_upper + 0.5, 0.0],
                   [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Lower input access waveguide
        Khronos.Object(
            Cuboid([-(coupling_length / 2 + L_taper + L_access / 2), y_lower - 0.5, 0.0],
                   [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Upper output access waveguide
        Khronos.Object(
            Cuboid([(coupling_length / 2 + L_taper + L_access / 2), y_upper + 0.5, 0.0],
                   [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Lower output access waveguide
        Khronos.Object(
            Cuboid([(coupling_length / 2 + L_taper + L_access / 2), y_lower - 0.5, 0.0],
                   [L_access, wg_width, thickness]),
            mat_si,
        ),
        # Background cladding (SiO2)
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [L_total + 10.0, 20.0, 20.0]),
            mat_sio2,
        ),
    ]

    # Mode source on upper input waveguide
    source_x = -(coupling_length / 2 + L_taper + L_access) + 1.0

    sources = [
        Khronos.ModeSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = freq_center),
            frequency = freq_center,
            mode_solver_resolution = 50,
            mode_index = 1,
            center = [source_x, y_upper + 0.5, 0.0],
            size = [0.0, 3 * wg_width, 6 * thickness],
            solver_tolerance = 1e-6,
            geometry = geometry,
        ),
    ]

    pml_thickness = 1.0
    domain_x = L_total + 2 * λ_center
    domain_y = 2 * (y_upper + 0.5) + wg_width + 2 * λ_center
    domain_z = 10 * thickness

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
    )

    return sim
end

@testset "Benchmark: directional coupler" begin
    TESTNAME = "directional_coupler"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        coupling_length = benchmark["coupling_length"]

        @testset "resolution: $resolution | coupling_length: $coupling_length" begin

            sim = build_directional_coupler_sim(resolution, coupling_length)
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
