# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E29: Throughput vs. Physics Complexity
#
# Measures how adding physics features (PML, materials, monitors) affects
# throughput at a fixed grid size. Demonstrates the value of Khronos's
# Nothing-dispatch kernel specialization: unused features have zero overhead.
#
# Configs:
#   A: Vacuum, no PML, no monitors       (baseline — fastest)
#   B: Vacuum + PML                       (+σ field updates)
#   C: Dielectric geometry + PML          (+ε lookup)
#   D: Multi-material (6 layers) + PML    (same kernel)
#   E: Config C + DFT monitor (10 freqs)  (+DFT accumulation)

import Khronos
using CairoMakie
using GeometryPrimitives
using OrderedCollections

# ── Scalable parameters ──────────────────────────────────────────────────────
N        = 128         # grid dimension (N³ voxels) — increase for larger GPUs
res      = 10          # resolution (pixels/μm)
n_steps  = 100         # timesteps per measurement
dpml     = 1.0         # PML thickness
# ─────────────────────────────────────────────────────────────────────────────

function main(; N=N, res=res, n_steps=n_steps, dpml=dpml)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    cell_val = Float64(N) / res

    # Common source
    base_source = Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0],
        size   = [0.0, 0.0, 0.0],
    )

    configs = OrderedDict{String, NamedTuple}()

    # Config A: Vacuum, no PML
    configs["A: Vacuum (no PML)"] = (
        sources    = [base_source],
        boundaries = nothing,
        geometry   = nothing,
        monitors   = nothing,
    )

    # Config B: Vacuum + PML
    configs["B: Vacuum + PML"] = (
        sources    = [base_source],
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        geometry   = nothing,
        monitors   = nothing,
    )

    # Config C: Single dielectric + PML
    configs["C: Dielectric + PML"] = (
        sources    = [base_source],
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        geometry   = [
            Khronos.Object(
                Cuboid([0.0, 0.0, 0.0], [cell_val/2, cell_val/2, cell_val/2]),
                Khronos.Material(ε=3.5),
            ),
        ],
        monitors   = nothing,
    )

    # Config D: 6-layer stack + PML
    mats = [Khronos.Material(ε=e) for e in [1.5, 2.5, 3.5, 2.0, 1.8, 3.0]]
    layer_thick = cell_val / 6
    layers = [
        Khronos.Object(
            Cuboid([0.0, 0.0, -cell_val/2 + (i-0.5)*layer_thick],
                   [cell_val, cell_val, layer_thick]),
            mats[i],
        )
        for i in 1:6
    ]
    configs["D: 6-layer stack + PML"] = (
        sources    = [base_source],
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        geometry   = layers,
        monitors   = nothing,
    )

    # Config E: Dielectric + PML + DFT monitor (10 freqs)
    configs["E: Dielectric + PML + DFT"] = (
        sources    = [base_source],
        boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
        geometry   = [
            Khronos.Object(
                Cuboid([0.0, 0.0, 0.0], [cell_val/2, cell_val/2, cell_val/2]),
                Khronos.Material(ε=3.5),
            ),
        ],
        monitors   = [
            Khronos.DFTMonitor(
                component = Khronos.Ez(),
                center = [0.0, 0.0, cell_val/4],
                size   = [cell_val/2, cell_val/2, 0.0],
                frequencies = collect(range(0.5, 1.5, length=10)),
            ),
        ],
    )

    # ── Run benchmarks ───────────────────────────────────────────────────────
    results = OrderedDict{String, Float64}()

    for (name, cfg) in configs
        println("\n", "="^60)
        println("Config: $name")
        println("="^60)

        kwargs = Dict{Symbol, Any}(
            :cell_size   => [cell_val, cell_val, cell_val],
            :cell_center => [0.0, 0.0, 0.0],
            :resolution  => res,
            :sources     => cfg.sources,
        )
        if !isnothing(cfg.boundaries)
            kwargs[:boundaries] = cfg.boundaries
        end
        if !isnothing(cfg.geometry)
            kwargs[:geometry] = cfg.geometry
        end
        if !isnothing(cfg.monitors)
            kwargs[:monitors] = cfg.monitors
        end

        sim = Khronos.Simulation(; kwargs...)
        rate = Khronos.run_benchmark(sim, n_steps)
        results[name] = rate
        println("  → $(round(rate, digits=1)) MCells/s")
    end

    # ── Summary ──────────────────────────────────────────────────────────────
    baseline = first(values(results))
    println("\n", "="^60)
    println("Physics Complexity Benchmark ($(N)^3 grid, $n_steps steps)")
    println("="^60)
    println("  Config                        MCells/s    Relative")
    println("  " * "-"^55)
    for (name, rate) in results
        rel = round(rate / baseline, digits=2)
        println("  $(rpad(name, 30))$(rpad(round(rate,digits=1), 12))$(rel)×")
    end
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(800, 500))
    ax = Axis(f[1, 1],
        xlabel = "Configuration",
        ylabel = "Throughput (MCells/s)",
        title  = "FDTD Throughput vs. Physics Complexity ($(N)^3 grid)",
        xticks = (1:length(results), [split(k, ":")[1] for k in keys(results)]),
    )
    barplot!(ax, 1:length(results), collect(values(results)),
             color=:steelblue, bar_labels=:y,
             label_formatter=x -> "$(round(x, digits=0))")

    save("throughput_vs_complexity.png", f)
    println("Saved: throughput_vs_complexity.png")

    return results
end

main()
