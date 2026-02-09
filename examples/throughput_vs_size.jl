# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E28: Single-GPU Throughput vs. Problem Size
#
# Measures cells/second as a function of grid size N³, from small
# (GPU-underutilized) to large (memory-bandwidth-limited). This reveals the
# GPU crossover point and peak achievable throughput.
#
# Reference: Extends existing benchmark/dipole.jl

import Khronos
using CairoMakie

# ── Scalable parameters ──────────────────────────────────────────────────────
grid_sizes = [32, 64, 96, 128, 192, 256]  # N values for N³ grids
n_steps    = 100                           # timesteps per measurement
# ─────────────────────────────────────────────────────────────────────────────
# For larger GPUs, extend: push!(grid_sizes, 384, 512, 768, 1024)

function main(; grid_sizes=grid_sizes, n_steps=n_steps)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    results = Dict{Int, Float64}()

    for N in grid_sizes
        # Compute cell_size and resolution so that we get exactly N voxels
        cell_size_val = Float64(N) / 10.0  # resolution = 10 pixels/μm
        dpml = 1.0

        println("\n", "="^60)
        println("Grid size: $(N)^3 = $(N^3) voxels")
        println("="^60)

        sources = [
            Khronos.UniformSource(
                time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
                component = Khronos.Ez(),
                center = [0.0, 0.0, 0.0],
                size   = [0.0, 0.0, 0.0],
            ),
        ]

        try
            sim = Khronos.Simulation(
                cell_size   = [cell_size_val, cell_size_val, cell_size_val],
                cell_center = [0.0, 0.0, 0.0],
                resolution  = 10,
                sources     = sources,
                boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
            )

            rate = Khronos.run_benchmark(sim, n_steps)
            results[N] = rate
            println("  $(N)^3 → $(round(rate, digits=1)) MCells/s")
        catch e
            println("  $(N)^3 → SKIPPED ($(typeof(e)))")
        end
    end

    # ── Results summary ──────────────────────────────────────────────────────
    println("\n", "="^60)
    println("Throughput vs. Problem Size Summary")
    println("="^60)
    println("  N       Voxels       MCells/s")
    println("  " * "-"^40)
    sorted_keys = sort(collect(keys(results)))
    for N in sorted_keys
        println("  $N\t$(N^3)\t\t$(round(results[N], digits=1))")
    end
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    if length(results) >= 2
        Ns = sorted_keys
        voxels = [N^3 for N in Ns]
        rates = [results[N] for N in Ns]

        f = Figure(size=(800, 500))
        ax = Axis(f[1, 1],
            xlabel = "Total voxels (N³)",
            ylabel = "Throughput (MCells/s)",
            title  = "Single-GPU FDTD Throughput vs. Problem Size",
            xscale = log10,
        )
        scatterlines!(ax, Float64.(voxels), rates,
                      color=:blue, markersize=10, linewidth=2,
                      label="Khronos (CUDA, Float64)")
        axislegend(ax, position=:rb)

        save("throughput_vs_size.png", f)
        println("Saved: throughput_vs_size.png")
    end

    return results
end

main()
