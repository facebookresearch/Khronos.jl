# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Metalens Benchmark and Profiling Script
#
# Measures single-GPU throughput for the metalens simulation at various
# scales, and provides nsys/ncu profiling hooks.
#
# Usage:
#   # Throughput sweep across lens sizes:
#   julia --project=. examples/metalens_benchmark.jl
#
#   # nsys profiling (profile just the stepping loop):
#   nsys profile --trace=cuda --capture-range=cudaProfilerApi \
#       --output=metalens_nsys \
#       julia --project=. examples/metalens_benchmark.jl --profile
#
#   # ncu kernel analysis:
#   ncu --set full --kernel-name step_curl --launch-count 5 \
#       --output=metalens_ncu \
#       julia --project=. examples/metalens_benchmark.jl --profile
#
#   # Float32 mode:
#   julia --project=. examples/metalens_benchmark.jl --float32
#
#   # Multi-GPU scaling test:
#   for N in 1 2 4 8; do
#       mpirun -np $N julia --project=. examples/metalens_benchmark.jl --scale
#   done

import Khronos
using GeometryPrimitives
using StaticArrays
using LinearAlgebra
using Printf
using CUDA

const PROFILE_MODE = "--profile" in ARGS
const USE_FLOAT32 = "--float32" in ARGS
const SCALING_TEST = "--scale" in ARGS

# ── Backend selection ─────────────────────────────────────────────────────────

precision = USE_FLOAT32 ? Float32 : Float64
Khronos.choose_backend(Khronos.CUDADevice(), precision)

if haskey(ENV, "OMPI_COMM_WORLD_SIZE") || haskey(ENV, "PMI_RANK") || haskey(ENV, "SLURM_PROCID")
    Khronos.init_mpi!()
    Khronos.select_device_for_rank!()
end

# ── Helper: build metalens simulation at given scale ──────────────────────────

function build_metalens_sim(; n_cells_side::Int, resolution_factor::Float64=1.0)
    nm = 1e-3
    wavelength   = 660 * nm
    NA           = 0.8
    rect_width   = 85  * nm
    rect_length  = 410 * nm
    lens_thick   = 600 * nm
    cell_length  = 430 * nm
    spacing      = 1.0 * wavelength

    ε_TiO2 = 2.40^2
    ε_SiO2 = 1.46^2
    TiO2_mat = Khronos.Material(ε = ε_TiO2)
    SiO2_mat = Khronos.Material(ε = ε_SiO2)

    N_cells = n_cells_side
    length_xy = N_cells * cell_length
    focal_length = length_xy / (2 * NA) * sqrt(1 - NA^2)
    length_z = spacing + lens_thick + 1.1 * focal_length + spacing

    grids_per_wavelength = 18
    dl = wavelength / grids_per_wavelength
    resolution = (1.0 / dl) * resolution_factor

    center_z = -length_z / 2 + spacing + lens_thick / 2

    # Substrate
    substrate = Khronos.Object(
        Cuboid([0.0, 0.0, -length_z + spacing], [2*length_xy, 2*length_xy, length_z]),
        SiO2_mat,
    )

    # Phase formula
    pb_theta(x, y) = π / wavelength * (focal_length - sqrt(x^2 + y^2 + focal_length^2))

    function make_pillar(x, y, theta)
        axes = SMatrix{3,3}(
            cos(theta), sin(theta), 0.0,
           -sin(theta), cos(theta), 0.0,
            0.0,        0.0,        1.0,
        )
        Cuboid(SVector(x, y, center_z), SVector(rect_width, rect_length, lens_thick), axes)
    end

    centers = cell_length .* (0:N_cells-1) .- length_xy / 2 .+ cell_length / 2
    geometry = Khronos.Object[substrate]
    sizehint!(geometry, N_cells * N_cells + 1)
    for cx in centers, cy in centers
        push!(geometry, Khronos.Object(make_pillar(cx, cy, pb_theta(cx, cy)), TiO2_mat))
    end

    # Sources (no monitors for benchmarking)
    fcen = 1.0 / wavelength
    fwidth = fcen / 10.0
    src_z = -length_z / 2 + 2 * dl

    sources = [
        Khronos.PlaneWaveSource(
            time_profile = Khronos.GaussianPulseSource(fcen=fcen, fwidth=fwidth),
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
        resolution  = resolution,
        geometry    = geometry,
        sources     = sources,
        boundaries  = boundaries,
        monitors    = Khronos.Monitor[],
    )

    return sim
end

# ── Profile mode: warmup then CUPTI-profiled steps ────────────────────────────

if PROFILE_MODE
    println("Building metalens simulation for profiling (10×10 cells)...")
    sim = build_metalens_sim(n_cells_side=10)

    println("Preparing and warming up...")
    Khronos.prepare_simulation!(sim)
    for _ in 1:20
        Khronos.step!(sim)
    end
    CUDA.synchronize()

    println("Warmup complete. Starting profiled region (5 timesteps)...")
    CUDA.@profile external=true begin
        for _ in 1:5
            Khronos.step!(sim)
        end
        CUDA.synchronize()
    end
    println("Profiling complete.")

# ── Scaling test: single size, report throughput ──────────────────────────────

elseif SCALING_TEST
    n_cells = 50  # Medium lens for scaling tests
    n_steps = 110

    println("Building metalens simulation ($(n_cells)×$(n_cells) cells)...")
    sim = build_metalens_sim(n_cells_side=n_cells)

    rate = Khronos.run_benchmark(sim, n_steps)
    num_voxels = sim.Nx * sim.Ny * sim.Nz

    rank = Khronos.is_distributed() ? Khronos.mpi_rank() : 0
    nranks = Khronos.is_distributed() ? Khronos.mpi_size() : 1

    if rank == 0
        println("\n" * "="^60)
        println("Metalens Scaling Test Results")
        println("="^60)
        println("  Lens:       $(n_cells)×$(n_cells) cells")
        println("  Grid:       $(sim.Nx)×$(sim.Ny)×$(sim.Nz) = $(num_voxels) voxels")
        println("  Precision:  $(precision)")
        println("  GPUs:       $(nranks)")
        println("  Steps:      $(n_steps)")
        println("  Rate:       $(round(rate, digits=1)) MCells/s")
        println("              $(round(rate / 1000, digits=3)) GCells/s")
        println("  Per-GPU:    $(round(rate / nranks, digits=1)) MCells/s")
        println("="^60)
    end

# ── Default: throughput sweep across lens sizes ───────────────────────────────

else
    # Lens sizes to benchmark (number of cells per side)
    lens_sizes = [5, 10, 20, 30, 50]
    n_steps = 110

    println("="^60)
    println("Metalens Throughput Benchmark")
    println("Precision: $(precision)")
    println("="^60)

    results = Dict{Int, NamedTuple}()

    for n_cells in lens_sizes
        println("\n--- $(n_cells)×$(n_cells) cells ---")

        try
            sim = build_metalens_sim(n_cells_side=n_cells)
            num_voxels = sim.Nx * sim.Ny * sim.Nz
            num_objects = n_cells * n_cells + 1

            println("  Grid:    $(sim.Nx)×$(sim.Ny)×$(sim.Nz) = $(num_voxels) voxels")
            println("  Objects: $(num_objects)")

            rate = Khronos.run_benchmark(sim, n_steps)
            results[n_cells] = (
                voxels = num_voxels,
                rate_mcells = rate,
                rate_gcells = rate / 1000,
                Nx = sim.Nx, Ny = sim.Ny, Nz = sim.Nz,
            )
            println("  Rate:    $(round(rate, digits=1)) MCells/s ($(round(rate/1000, digits=3)) GCells/s)")
        catch e
            println("  SKIPPED: $(typeof(e)): $(e)")
        end
    end

    # Summary table
    println("\n" * "="^60)
    println("Summary")
    println("="^60)
    println("  Cells    Grid                    Voxels         MCells/s    GCells/s")
    println("  " * "-"^70)
    for n_cells in sort(collect(keys(results)))
        r = results[n_cells]
        grid_str = "$(r.Nx)×$(r.Ny)×$(r.Nz)"
        @printf("  %-8d %-23s %-14d %-11.1f %.3f\n",
                n_cells, grid_str, r.voxels, r.rate_mcells, r.rate_gcells)
    end
    println("="^60)

    # Target comparison
    println("\nTarget: 60 GCells/s (multi-GPU)")
    if !isempty(results)
        peak = maximum(r.rate_gcells for r in values(results))
        println("Peak single-GPU: $(round(peak, digits=3)) GCells/s")
        println("GPUs needed (linear scaling): $(ceil(Int, 60.0 / peak))")
    end
end
