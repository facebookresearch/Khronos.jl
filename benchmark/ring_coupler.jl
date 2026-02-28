# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Waveguide-to-ring resonator coupler.
#
# Exercises: Cylinder geometry, ModeSource, ModeMonitor, DFTMonitor,
# absorber boundaries, overlapping geometry objects with priority ordering,
# SOI waveguide platform.
#
# Based on examples/ring_coupler.jl (ported from Tidy3D WaveguideToRingCoupling).

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
YAML_FILENAME = joinpath(@__DIR__, "ring_coupler.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run = detect_and_set_backend()

# current hardware
hardware_key = get_hardware_key()

function build_ring_coupler_sim(resolution, ring_radius)
    # Standard SOI parameters
    n_Si = 3.47
    n_SiO2 = 1.44
    ε_Si = n_Si^2
    ε_SiO2 = n_SiO2^2

    wg_width = 0.5    # μm
    wg_height = 0.22  # μm
    coupling_gap = 0.05  # μm
    λ_center = 1.55   # μm

    ring_center_y = wg_width + coupling_gap + ring_radius

    # Domain sizing (matches Tidy3D tutorial)
    domain_x = 2 * ring_radius + 2 * λ_center
    domain_y = ring_radius / 2 + coupling_gap + 2 * wg_width + λ_center
    domain_z = 9 * wg_height
    pml_thickness = 1.0

    absorber_num_layers = 60
    absorber_thickness = absorber_num_layers / resolution

    geometry = [
        # Straight waveguide (Si)
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [domain_x + 4.0, wg_width, wg_height]),
            Khronos.Material(ε = ε_Si),
        ),
        # Ring: inner cylinder (SiO2 — carved out)
        Khronos.Object(
            Cylinder([0.0, ring_center_y, 0.0], ring_radius - wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
            Khronos.Material(ε = ε_SiO2),
        ),
        # Ring: outer cylinder (Si)
        Khronos.Object(
            Cylinder([0.0, ring_center_y, 0.0], ring_radius + wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
            Khronos.Material(ε = ε_Si),
        ),
        # Background cladding (SiO2)
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [domain_x + 10.0, domain_y + 10.0, domain_z + 10.0]),
            Khronos.Material(ε = ε_SiO2),
        ),
    ]

    freq_center = 1.0 / λ_center
    source_x = -(ring_radius + λ_center / 4)

    sources = [
        Khronos.ModeSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = freq_center),
            frequency = freq_center,
            mode_solver_resolution = 50,
            mode_index = 1,
            center = [source_x, 0.0, 0.0],
            size = [0.0, 6 * wg_width, 6 * wg_height],
            solver_tolerance = 1e-6,
            geometry = geometry,
        ),
    ]

    boundaries = [
        [pml_thickness, pml_thickness],
        [pml_thickness, 0.0],
        [pml_thickness, pml_thickness],
    ]

    absorbers = [
        nothing,
        [nothing, Khronos.Absorber(num_layers = 60)],
        nothing,
    ]

    cell_x = domain_x + 2 * pml_thickness
    cell_y = domain_y + pml_thickness + absorber_thickness
    cell_z = domain_z + 2 * pml_thickness

    phys_center_y = domain_y / 4
    cell_center_y = phys_center_y + (absorber_thickness - pml_thickness) / 2

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, cell_z],
        cell_center = [0.0, cell_center_y, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = boundaries,
        absorbers = absorbers,
    )

    return sim
end

@testset "Benchmark: ring coupler" begin
    TESTNAME = "ring_coupler"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        ring_radius = benchmark["ring_radius"]

        @testset "resolution: $resolution | ring_radius: $ring_radius" begin

            sim = build_ring_coupler_sim(resolution, ring_radius)
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
