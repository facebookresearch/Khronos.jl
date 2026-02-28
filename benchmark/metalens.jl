# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Pancharatnam-Berry phase metalens.
#
# Exercises: Large number of rotated Cuboid objects (geometry rasterization stress test),
# PlaneWaveSource, large simulation domain, high voxel count.
# This is the primary throughput benchmark for large-scale simulations.
#
# Based on examples/metalens.jl (from Khorasaninejad et al., Science 2016).

import YAML
import Khronos

using KernelAbstractions
using Logging
using Test
using GeometryPrimitives
using StaticArrays
using LinearAlgebra
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
YAML_FILENAME = joinpath(@__DIR__, "metalens.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_metalens_sim(resolution, n_cells_side)
    nm = 1e-3  # Khronos uses μm

    wavelength   = 660 * nm
    NA           = 0.8
    n_TiO2       = 2.40
    n_SiO2       = 1.46
    ε_TiO2       = n_TiO2^2
    ε_SiO2       = n_SiO2^2

    rect_width   = 85  * nm
    rect_length  = 410 * nm
    lens_thick   = 600 * nm
    cell_length  = 430 * nm
    spacing      = 1.0 * wavelength

    length_xy = n_cells_side * cell_length
    focal_length = length_xy / (2 * NA) * sqrt(1 - NA^2)
    length_z = spacing + lens_thick + 1.1 * focal_length + spacing

    dl = wavelength / 18
    sim_resolution = 1.0 / dl

    # Use provided resolution to override if specified
    if resolution > 0
        sim_resolution = Float64(resolution)
    end

    TiO2_mat = Khronos.Material(ε = ε_TiO2)
    SiO2_mat = Khronos.Material(ε = ε_SiO2)

    center_z = -length_z / 2 + spacing + lens_thick / 2

    substrate = Khronos.Object(
        Cuboid(
            [0.0, 0.0, -length_z + spacing],
            [2 * length_xy, 2 * length_xy, length_z],
        ),
        SiO2_mat,
    )

    function pb_theta(x, y)
        return π / wavelength * (focal_length - sqrt(x^2 + y^2 + focal_length^2))
    end

    function make_pillar(x, y, theta)
        axes = SMatrix{3,3}(
            cos(theta), sin(theta), 0.0,
           -sin(theta), cos(theta), 0.0,
            0.0,        0.0,        1.0,
        )
        Cuboid(
            SVector(x, y, center_z),
            SVector(rect_width, rect_length, lens_thick),
            axes,
        )
    end

    centers = cell_length .* (0:n_cells_side-1) .- length_xy / 2 .+ cell_length / 2

    geometry = Khronos.Object[substrate]
    sizehint!(geometry, n_cells_side * n_cells_side + 1)

    for cx in centers, cy in centers
        push!(geometry, Khronos.Object(make_pillar(cx, cy, pb_theta(cx, cy)), TiO2_mat))
    end

    # Source
    fcen = 1.0 / wavelength
    fwidth = fcen / 10.0
    src_z = -length_z / 2 + 2 * dl

    sources = [
        Khronos.PlaneWaveSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = fcen),
            center = [0.0, 0.0, src_z],
            size = [Inf, Inf, 0.0],
            polarization_angle = 0.0,
            k_vector = [0.0, 0.0, 1.0],
            amplitude = 1.0,
        ),
    ]

    dpml = 15 * dl
    boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

    sim = Khronos.Simulation(
        cell_size   = [length_xy, length_xy, length_z],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = sim_resolution,
        Courant     = 0.55,
        geometry    = geometry,
        sources     = sources,
        boundaries  = boundaries,
    )

    return sim
end

@testset "Benchmark: metalens" begin
    TESTNAME = "metalens"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        n_cells_side = benchmark["n_cells_side"]

        @testset "resolution: $resolution | n_cells_side: $n_cells_side" begin

            sim = build_metalens_sim(resolution, n_cells_side)
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
                    label="metalens (res=$resolution, cells=$n_cells_side)")
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
