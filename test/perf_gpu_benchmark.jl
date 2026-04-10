# GPU Performance Benchmark: PML grid splitting on A100
#
# Compares single-chunk vs PML-grid (:auto) for several 3D configurations.
# Each config is run with run_benchmark (warmup + timed stepping).
# Pin to GPU 0 to avoid multi-GPU variance.

import Khronos
using GeometryPrimitives
using Logging
using CUDA

# Suppress info logging during benchmarks for clean output
debuglogger = ConsoleLogger(stderr, Logging.Warn)
global_logger(debuglogger)

# Pin to GPU 0
CUDA.device!(0)
Khronos.choose_backend(Khronos.CUDADevice(), Float64)

# ── Helpers ──────────────────────────────────────────────────────────────────

function count_nothing_aux(chunk)
    n = 0
    for name in (:fCBx,:fCBy,:fCBz,:fUBx,:fUBy,:fUBz,:fWBx,:fWBy,:fWBz,
                  :fCDx,:fCDy,:fCDz,:fUDx,:fUDy,:fUDz,:fWDx,:fWDy,:fWDz)
        isnothing(getfield(chunk.fields, name)) && (n += 1)
    end
    return n
end

function chunk_summary(sim)
    n = length(sim.chunk_data)
    n_int = count(c -> !Khronos.has_any_pml(c.spec.physics), sim.chunk_data)
    n1 = count(c -> count([c.spec.physics.has_pml_x, c.spec.physics.has_pml_y, c.spec.physics.has_pml_z]) == 1, sim.chunk_data)
    n2 = count(c -> count([c.spec.physics.has_pml_x, c.spec.physics.has_pml_y, c.spec.physics.has_pml_z]) == 2, sim.chunk_data)
    n3 = count(c -> count([c.spec.physics.has_pml_x, c.spec.physics.has_pml_y, c.spec.physics.has_pml_z]) == 3, sim.chunk_data)
    return "$n chunks (int=$n_int face=$n1 edge=$n2 corner=$n3)"
end

function aux_mem_mb(sim)
    total = 0
    for chunk in sim.chunk_data
        for name in (:fCBx,:fCBy,:fCBz,:fUBx,:fUBy,:fUBz,:fWBx,:fWBy,:fWBz,
                      :fCDx,:fCDy,:fCDz,:fUDx,:fUDy,:fUDz,:fWDx,:fWDy,:fWDz)
            arr = getfield(chunk.fields, name)
            !isnothing(arr) && (total += sizeof(parent(arr)))
        end
    end
    return total / 1024^2
end

function bench_config(label, sim; n_steps=200)
    println("  Preparing...")
    t_prep = @elapsed Khronos.prepare_simulation!(sim)

    nvox = sim.Nx * sim.Ny * max(sim.Nz, 1)
    cs = chunk_summary(sim)
    mem = aux_mem_mb(sim)

    println("  $cs")
    println("  Prep: $(round(t_prep, digits=2))s  |  Aux mem: $(round(mem, digits=2)) MB  |  Voxels: $nvox")

    # Run benchmark (10 warmup steps built-in, then n_steps - 10 timed)
    rate = Khronos.run_benchmark(sim, n_steps)

    println("  Throughput: $(round(rate, digits=1)) MCells/s")
    return (label=label, rate=rate, prep=t_prep, mem=mem, nvox=nvox, nchunks=length(sim.chunk_data))
end

# ── Benchmark configurations ─────────────────────────────────────────────────

function run_all_benchmarks()
    all_results = []

    # ── Config 1: Dipole (small, vacuum+PML) ──────────────────────────────
    println("\n" * "=" ^ 70)
    println("CONFIG 1: Dipole (40^3, vacuum + PML)")
    println("=" ^ 70)

    for (tag, nc) in [("1-chunk", nothing), ("PML-grid", :auto)]
        println("\n--- $tag ---")
        sim = Khronos.Simulation(
            cell_size = [4.0, 4.0, 4.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ex(),
                center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0],
                amplitude = 1im,
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = nc,
        )
        r = bench_config("Dipole/$tag", sim, n_steps=200)
        push!(all_results, r)
        CUDA.reclaim()
    end

    # ── Config 2: Sphere (medium, geometry+PML+loss) ──────────────────────
    println("\n" * "=" ^ 70)
    println("CONFIG 2: Lossy sphere (100^3, ε=3 σD=5 + PML)")
    println("=" ^ 70)

    s_xyz = 2.0 + 1.0 + 2 * 1.0  # = 5.0
    for (tag, nc) in [("1-chunk", nothing), ("PML-grid", :auto)]
        println("\n--- $tag ---")
        sim = Khronos.Simulation(
            cell_size = [s_xyz, s_xyz, s_xyz],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 20,
            sources = [Khronos.PlaneWaveSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                center = [0.0, 0.0, -s_xyz/2 + 1.0],
                size = [Inf, Inf, 0.0],
                k_vector = [0.0, 0.0, 1.0],
                polarization_angle = 0.0, amplitude = 1im,
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 1.0),
                        Khronos.Material(ε = 3, σD = 5))],
            num_chunks = nc,
        )
        r = bench_config("Sphere/$tag", sim, n_steps=200)
        push!(all_results, r)
        CUDA.reclaim()
    end

    # ── Config 3: Large vacuum (throughput ceiling) ───────────────────────
    println("\n" * "=" ^ 70)
    println("CONFIG 3: Large vacuum + PML (128^3)")
    println("=" ^ 70)

    cell_val = 128.0 / 10
    for (tag, nc) in [("1-chunk", nothing), ("PML-grid", :auto)]
        println("\n--- $tag ---")
        sim = Khronos.Simulation(
            cell_size = [cell_val, cell_val, cell_val],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = nc,
        )
        r = bench_config("Vacuum128/$tag", sim, n_steps=200)
        push!(all_results, r)
        CUDA.reclaim()
    end

    # ── Config 4: Large vacuum + PML (256^3) ──────────────────────────────
    println("\n" * "=" ^ 70)
    println("CONFIG 4: Large vacuum + PML (256^3)")
    println("=" ^ 70)

    cell_val = 256.0 / 10
    for (tag, nc) in [("1-chunk", nothing), ("PML-grid", :auto)]
        println("\n--- $tag ---")
        sim = Khronos.Simulation(
            cell_size = [cell_val, cell_val, cell_val],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 10,
            sources = [Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0],
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            num_chunks = nc,
        )
        r = bench_config("Vacuum256/$tag", sim, n_steps=110)
        push!(all_results, r)
        CUDA.reclaim()
    end

    # ── Config 5: Large sphere (256^3, production-like) ───────────────────
    println("\n" * "=" ^ 70)
    println("CONFIG 5: Large sphere (256^3, ε=3.5 + PML)")
    println("=" ^ 70)

    cell_val = 256.0 / 20   # 12.8 μm at res=20
    for (tag, nc) in [("1-chunk", nothing), ("PML-grid", :auto)]
        println("\n--- $tag ---")
        sim = Khronos.Simulation(
            cell_size = [cell_val, cell_val, cell_val],
            cell_center = [0.0, 0.0, 0.0],
            resolution = 20,
            sources = [Khronos.PlaneWaveSource(
                time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
                center = [0.0, 0.0, -cell_val/2 + 1.0],
                size = [Inf, Inf, 0.0],
                k_vector = [0.0, 0.0, 1.0],
                polarization_angle = 0.0, amplitude = 1im,
            )],
            boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], cell_val/4),
                        Khronos.Material(ε = 3.5))],
            num_chunks = nc,
        )
        r = bench_config("Sphere256/$tag", sim, n_steps=110)
        push!(all_results, r)
        CUDA.reclaim()
    end

    # ── Summary ───────────────────────────────────────────────────────────
    println("\n\n")
    println("=" ^ 78)
    println("GPU BENCHMARK SUMMARY (A100)")
    println("=" ^ 78)
    println()
    println("  $(rpad("Config", 22)) $(rpad("Voxels", 12)) $(rpad("Chunks", 8)) $(rpad("Aux MB", 10)) $(rpad("MCells/s", 12)) $(rpad("vs 1-chunk", 12))")
    println("  " * "-" ^ 72)

    # Print in pairs
    for i in 1:2:length(all_results)
        r1 = all_results[i]
        r2 = all_results[i+1]
        speedup = r2.rate / r1.rate

        # Extract config name
        config = split(r1.label, "/")[1]

        println("  $(rpad(r1.label, 22)) $(rpad(r1.nvox, 12)) $(rpad(r1.nchunks, 8)) $(rpad(round(r1.mem, digits=1), 10)) $(rpad(round(r1.rate, digits=1), 12))")
        println("  $(rpad(r2.label, 22)) $(rpad(r2.nvox, 12)) $(rpad(r2.nchunks, 8)) $(rpad(round(r2.mem, digits=1), 10)) $(rpad(round(r2.rate, digits=1), 12)) $(round(speedup, digits=2))x")
        println()
    end
    println("=" ^ 78)

    return all_results
end

results = run_all_benchmarks()
