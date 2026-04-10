# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Performance test: PML grid chunk splitting vs single-chunk baseline.
#
# Part 1: Memory & allocation analysis (fast, no stepping needed)
# Part 2: Stepping comparison at small grid size (to keep JIT time manageable)
#
# The key benefits of PML grid splitting are:
#   1. Memory savings: interior chunks eliminate all 18 C/U/W auxiliary arrays
#   2. Per-direction sigma: face chunks only allocate sigma for their PML direction
#   3. Nothing-dispatch: simpler kernel variants for non-PML regions

import Khronos
using GeometryPrimitives
using Statistics
using Logging

debuglogger = ConsoleLogger(stderr, Logging.Warn)
global_logger(debuglogger)

# ── Helpers ──────────────────────────────────────────────────────────────────

function count_nothing_aux_fields(chunk)
    f = chunk.fields
    n = 0
    for name in (:fCBx, :fCBy, :fCBz, :fUBx, :fUBy, :fUBz, :fWBx, :fWBy, :fWBz,
                  :fCDx, :fCDy, :fCDz, :fUDx, :fUDy, :fUDz, :fWDx, :fWDy, :fWDz)
        isnothing(getfield(f, name)) && (n += 1)
    end
    return n
end

function count_nothing_sigma(chunk)
    bd = chunk.boundary_data
    n = 0
    for name in (:σBx, :σBy, :σBz, :σDx, :σDy, :σDz)
        isnothing(getfield(bd, name)) && (n += 1)
    end
    return n
end

function estimate_aux_memory_bytes(sim)
    total = 0
    for chunk in sim.chunk_data
        for name in (:fCBx, :fCBy, :fCBz, :fUBx, :fUBy, :fUBz, :fWBx, :fWBy, :fWBz,
                      :fCDx, :fCDy, :fCDz, :fUDx, :fUDy, :fUDz, :fWDx, :fWDy, :fWDz)
            arr = getfield(chunk.fields, name)
            !isnothing(arr) && (total += sizeof(parent(arr)))
        end
        for name in (:σBx, :σBy, :σBz, :σDx, :σDy, :σDz)
            arr = getfield(chunk.boundary_data, name)
            !isnothing(arr) && (total += sizeof(arr))
        end
    end
    return total
end

function build_sim(; N, res, pml, geometry=nothing, num_chunks=nothing)
    cell_val = Float64(N) / res
    src = Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
    )
    kwargs = Dict{Symbol,Any}(
        :cell_size => [cell_val, cell_val, cell_val],
        :cell_center => [0.0, 0.0, 0.0],
        :resolution => res,
        :sources => [src],
    )
    pml && (kwargs[:boundaries] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    !isnothing(geometry) && (kwargs[:geometry] = geometry)
    !isnothing(num_chunks) && (kwargs[:num_chunks] = num_chunks)
    return Khronos.Simulation(; kwargs...)
end

function classify_chunk(chunk)
    pf = chunk.spec.physics
    n_pml = count([pf.has_pml_x, pf.has_pml_y, pf.has_pml_z])
    if n_pml == 0
        return "interior"
    elseif n_pml == 1
        return "face"
    elseif n_pml == 2
        return "edge"
    else
        return "corner"
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Memory & Allocation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

function run_memory_test(; N=64, res=10)
    cell_val = Float64(N) / res
    geom = [Khronos.Object(Ball([0.0, 0.0, 0.0], cell_val / 4), Khronos.Material(ε = 3.5))]

    println("=" ^ 78)
    println("PART 1: MEMORY & ALLOCATION ANALYSIS ($(N)^3 = $(N^3) voxels)")
    println("=" ^ 78)
    println("  PML: 1.0 μm on all sides ($(ceil(Int, res * 1.0)) cells/side)")
    pml_frac = (2 * ceil(Int, res * 1.0))^3 / N^3
    interior_frac = (N - 2*ceil(Int, res * 1.0))^3 / N^3
    println("  PML volume fraction:      $(round(100 * (1 - interior_frac), digits=1))%")
    println("  Interior volume fraction: $(round(100 * interior_frac, digits=1))%")
    println()

    configs = [
        ("1-chunk (baseline)", nothing),
        ("PML-grid (:auto)",   :auto),
        ("BSP (num_chunks=8)", 8),
    ]

    for (label, nc) in configs
        println("-" ^ 78)
        println("Config: $label")

        sim = build_sim(N=N, res=res, pml=true, geometry=geom, num_chunks=nc)

        t0 = time()
        Khronos.prepare_simulation!(sim)
        t_prep = time() - t0

        n_chunks = length(sim.chunk_data)

        # Classify chunks
        types = [classify_chunk(c) for c in sim.chunk_data]
        n_int = count(==("interior"), types)
        n_face = count(==("face"), types)
        n_edge = count(==("edge"), types)
        n_corner = count(==("corner"), types)

        # Memory
        aux_mem = estimate_aux_memory_bytes(sim)

        # Nothing-dispatch analysis per chunk type
        println("  Chunks:       $n_chunks  (interior=$n_int  face=$n_face  edge=$n_edge  corner=$n_corner)")
        println("  Prep time:    $(round(t_prep, digits=2))s")
        println("  Aux memory:   $(round(aux_mem / 1024^2, digits=2)) MB")
        println()

        # Per-type analysis
        for ctype in ["interior", "face", "edge", "corner"]
            matching = filter(c -> classify_chunk(c) == ctype, sim.chunk_data)
            isempty(matching) && continue
            c = first(matching)
            n_nil_aux = count_nothing_aux_fields(c)
            n_nil_sig = count_nothing_sigma(c)
            gv = c.spec.grid_volume
            nvox = gv.Nx * gv.Ny * max(1, gv.Nz)
            pml_dirs = String[]
            c.spec.physics.has_pml_x && push!(pml_dirs, "x")
            c.spec.physics.has_pml_y && push!(pml_dirs, "y")
            c.spec.physics.has_pml_z && push!(pml_dirs, "z")
            dir_str = isempty(pml_dirs) ? "none" : join(pml_dirs, ",")
            println("    $ctype (×$(length(matching))): $(nvox) vox, PML=[$dir_str], nothing_aux=$(n_nil_aux)/18, nothing_σ=$(n_nil_sig)/6")
        end
        println()

        sim = nothing
        GC.gc()
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Stepping Performance
# ═══════════════════════════════════════════════════════════════════════════════

function run_step_test(; N=32, res=10, n_warmup=3, n_measure=10)
    cell_val = Float64(N) / res
    geom = [Khronos.Object(Ball([0.0, 0.0, 0.0], cell_val / 4), Khronos.Material(ε = 3.5))]

    println("=" ^ 78)
    println("PART 2: STEPPING PERFORMANCE ($(N)^3 = $(N^3) voxels)")
    println("=" ^ 78)
    println("  Warmup: $n_warmup steps, Measure: $n_measure steps")
    println()

    configs = [
        ("1-chunk (baseline)", nothing),
        ("PML-grid (:auto)",   :auto),
    ]

    results = []

    for (label, nc) in configs
        println("-" ^ 78)
        println("Config: $label")

        sim = build_sim(N=N, res=res, pml=true, geometry=geom, num_chunks=nc)
        Khronos.prepare_simulation!(sim)

        n_chunks = length(sim.chunk_data)
        num_voxels = sim.Nx * sim.Ny * sim.Nz

        # Warmup (includes JIT)
        print("  Warming up ($n_warmup steps + JIT)...")
        t_warmup_start = time()
        for _ in 1:n_warmup
            Khronos.step!(sim)
        end
        println(" $(round(time() - t_warmup_start, digits=1))s")

        # Measure
        times = Float64[]
        for _ in 1:n_measure
            t0 = time_ns()
            Khronos.step!(sim)
            push!(times, (time_ns() - t0) / 1e6)
        end

        med = median(times)
        mn = mean(times)
        mi = minimum(times)
        throughput = num_voxels / (med / 1000) / 1e6

        println("  Chunks:     $n_chunks")
        println("  Step time:  median=$(round(med, digits=2))ms  mean=$(round(mn, digits=2))ms  min=$(round(mi, digits=2))ms")
        println("  Throughput: $(round(throughput, digits=1)) MCells/s")

        push!(results, (label=label, med=med, throughput=throughput, n_chunks=n_chunks))
        sim = nothing
        GC.gc()
    end

    if length(results) >= 2
        speedup = results[2].throughput / results[1].throughput
        println()
        println("  PML-grid step speedup: $(round(speedup, digits=2))x")
    end

    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Scaling analysis (memory only — fast)
# ═══════════════════════════════════════════════════════════════════════════════

function run_scaling_memory()
    println()
    println("=" ^ 78)
    println("PART 3: MEMORY SAVINGS SCALING")
    println("=" ^ 78)
    println()
    println("  $(rpad("Grid", 10)) $(rpad("1-chunk MB", 14)) $(rpad("PML-grid MB", 14)) $(rpad("Savings", 10)) $(rpad("Interior%", 12))")
    println("  " * "-" ^ 56)

    for N in [32, 48, 64, 80]
        cell_val = Float64(N) / 10
        geom = [Khronos.Object(Ball([0.0, 0.0, 0.0], cell_val / 4), Khronos.Material(ε = 3.5))]

        sim1 = build_sim(N=N, res=10, pml=true, geometry=geom, num_chunks=nothing)
        Khronos.prepare_simulation!(sim1)
        mem1 = estimate_aux_memory_bytes(sim1)

        sim2 = build_sim(N=N, res=10, pml=true, geometry=geom, num_chunks=:auto)
        Khronos.prepare_simulation!(sim2)
        mem2 = estimate_aux_memory_bytes(sim2)

        savings = round(100 * (1 - mem2 / mem1), digits=1)

        # Interior volume fraction
        pml_cells = ceil(Int, 10 * 1.0)
        interior_frac = round(100 * ((N - 2*pml_cells) / N)^3, digits=1)

        println("  $(rpad("$(N)^3", 10)) $(rpad(round(mem1/1024^2, digits=2), 14)) $(rpad(round(mem2/1024^2, digits=2), 14)) $(rpad("$(savings)%", 10)) $(rpad("$(interior_frac)%", 12))")

        sim1 = nothing; sim2 = nothing; GC.gc()
    end
    println("=" ^ 78)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

println()
println("╔" * "═" ^ 76 * "╗")
println("║" * lpad("PML GRID CHUNK SPLITTING — PERFORMANCE ANALYSIS", 62) * " " ^ 14 * "║")
println("╚" * "═" ^ 76 * "╝")
println()

# Part 1: Memory analysis at reasonable size
run_memory_test(N=64, res=10)

# Part 2: Stepping (small grid to keep JIT time manageable on CPU)
println("\n")
run_step_test(N=32, res=10, n_warmup=3, n_measure=10)

# Part 3: Memory scaling
println("\n")
run_scaling_memory()

println("\nDone.")
