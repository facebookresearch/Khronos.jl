# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark: Photonic crystal slab with Bloch-periodic boundaries.
#
# Exercises: Bloch boundary conditions (complex fields, phase-shifted halos),
# Cylinder geometry (air holes), DFTMonitor, point dipole source.
# Tests a fundamentally different kernel path from PML-based benchmarks.
#
# Based on examples/bandstructure.jl.

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
YAML_FILENAME = joinpath(@__DIR__, "periodic_bloch.yml")

profiling_results = YAML.load_file(YAML_FILENAME)

# set the appropriate backend and determine if this is a profile run
backend, precision, profile_run, metrics_run = detect_and_set_backend()
precision_type = precision == "Float32" ? Float32 : Float64

# current hardware
hardware_key = get_hardware_key()

function build_bloch_sim(resolution, n_cells)
    a = 1.0           # lattice constant (μm)
    r_hole = 0.2 * a  # hole radius
    t_slab = 0.5 * a  # slab thickness
    ε_slab = 12.0     # silicon-like

    pml_z = 1.0
    buffer_z = 1.0
    cell_z = t_slab + 2 * buffer_z + 2 * pml_z
    cell_xy = n_cells * a

    # Build geometry: n_cells × n_cells supercell
    geometry = Khronos.Object[]

    for ix in 0:n_cells-1
        for iy in 0:n_cells-1
            cx = (ix + 0.5) * a - cell_xy / 2
            cy = (iy + 0.5) * a - cell_xy / 2
            push!(geometry, Khronos.Object(
                Cylinder([cx, cy, 0.0], r_hole, t_slab, [0.0, 0.0, 1.0]),
                Khronos.Material(ε = 1.0),
            ))
        end
    end

    # Dielectric slab (lower priority)
    push!(geometry, Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_xy + 1.0, cell_xy + 1.0, t_slab]),
        Khronos.Material(ε = ε_slab),
    ))

    # Bloch wavevector at X-point
    kx = 0.5 * 2π / a
    ky = 0.0

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 0.3),
            component = Khronos.Hz(),
            center = [0.13 * a - cell_xy / 2, 0.27 * a - cell_xy / 2, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ]

    boundaries = [
        [0.0, 0.0],
        [0.0, 0.0],
        [pml_z, pml_z],
    ]

    boundary_conditions = [
        [Khronos.Bloch(k = kx), Khronos.Bloch(k = kx)],
        [Khronos.Bloch(k = ky), Khronos.Bloch(k = ky)],
        [Khronos.PML(), Khronos.PML()],
    ]

    sim = Khronos.Simulation(
        cell_size = [cell_xy, cell_xy, cell_z],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = boundaries,
        boundary_conditions = boundary_conditions,
    )

    return sim
end

@testset "Benchmark: Bloch-periodic photonic crystal" begin
    TESTNAME = "bloch_photonic_crystal"

    current_testset = profiling_results[TESTNAME][hardware_key][backend][precision]

    for benchmark in current_testset
        resolution = benchmark["resolution"]
        tolerance = benchmark["tolerance"]
        benchmark_rate = benchmark["timestep_rate"]
        n_cells = benchmark["n_cells"]

        @testset "resolution: $resolution | n_cells: $n_cells" begin

            sim = build_bloch_sim(resolution, n_cells)
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
                    label="periodic_bloch (res=$resolution, cells=$n_cells)")
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
