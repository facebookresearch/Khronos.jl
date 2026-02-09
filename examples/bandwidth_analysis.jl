# Corrected throughput vs size benchmark with proper GPU synchronization
# The existing run_benchmark() lacks CUDA.synchronize(), so its reported
# MCells/s values are inflated (measuring CPU launch time, not GPU execution).

import Khronos
using CUDA
using CairoMakie

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

grid_sizes = [32, 64, 96, 128, 192, 256]
n_warmup = 20
n_measure = 50

println("="^80)
println("Throughput vs Size (with GPU synchronization)")
println("="^80)
println("GPU: $(CUDA.name(CUDA.device()))")
println()

results = Dict{Int, NamedTuple}()

for N in grid_sizes
    cell_val = Float64(N) / 10.0
    dpml = 1.0

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
            component = Khronos.Ez(),
            center = [0.0, 0.0, 0.0],
            size   = [0.0, 0.0, 0.0],
        ),
    ]

    sim = Khronos.Simulation(
        cell_size   = [cell_val, cell_val, cell_val],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = 10,
        sources     = sources,
        boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
    )

    Khronos.prepare_simulation!(sim)

    # Count field arrays
    n_arrays = 0
    for fname in fieldnames(typeof(sim.fields))
        f = getfield(sim.fields, fname)
        if f !== nothing && isa(f, AbstractArray)
            n_arrays += 1
        end
    end
    array_bytes = (N+2)^3 * 8
    working_set_mb = n_arrays * array_bytes / 1024^2

    # Warmup
    for _ in 1:n_warmup
        Khronos.step!(sim)
    end
    CUDA.synchronize()

    # Measure with CUDA events (precise GPU-side timing)
    start_event = CUDA.CuEvent()
    stop_event = CUDA.CuEvent()

    CUDA.record(start_event)
    for _ in 1:n_measure
        Khronos.step!(sim)
    end
    CUDA.record(stop_event)
    CUDA.synchronize()

    elapsed_s = CUDA.elapsed(start_event, stop_event)
    voxels = N^3
    mcells_s = voxels * n_measure / elapsed_s / 1e6
    ms_per_step = elapsed_s / n_measure * 1000

    # Also measure with run_benchmark for comparison (has the sync bug)
    sim2 = Khronos.Simulation(
        cell_size   = [cell_val, cell_val, cell_val],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = 10,
        sources     = sources,
        boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
    )
    reported_rate = Khronos.run_benchmark(sim2, 100)

    results[N] = (mcells_s=mcells_s, reported=reported_rate,
                  working_set_mb=working_set_mb, ms_per_step=ms_per_step)

    println("$(N)^3:  $(round(mcells_s, digits=0)) MCells/s (actual)  vs  $(round(reported_rate, digits=0)) MCells/s (run_benchmark)  |  working set: $(round(working_set_mb, digits=0)) MB  |  $(round(ms_per_step, digits=2)) ms/step")
end

println()
println("="^80)
println("Summary")
println("="^80)
println("  N      Actual MCells/s   Reported MCells/s   Inflation    Working Set")
println("  " * "-"^70)
sorted = sort(collect(keys(results)))
for N in sorted
    r = results[N]
    inflation = round(r.reported / r.mcells_s, digits=1)
    println("  $(rpad(N, 6)) $(rpad(round(r.mcells_s, digits=0), 18)) $(rpad(round(r.reported, digits=0), 20)) $(rpad(string(inflation)*"x", 13)) $(round(r.working_set_mb, digits=0)) MB")
end

# Visualization
f = Figure(size=(900, 500))
ax = Axis(f[1, 1],
    xlabel = "Grid size N (N^3 voxels)",
    ylabel = "Throughput (MCells/s)",
    title  = "FDTD Throughput: Actual (synced) vs Reported (no sync)",
)
Ns = sorted
actual = [results[N].mcells_s for N in Ns]
reported = [results[N].reported for N in Ns]
scatterlines!(ax, Float64.(Ns), actual, label="Actual (CUDA events)", color=:blue, markersize=10, linewidth=2)
scatterlines!(ax, Float64.(Ns), reported, label="Reported (run_benchmark)", color=:red, markersize=10, linewidth=2, linestyle=:dash)
axislegend(ax, position=:lt)

save("throughput_sync_comparison.png", f)
println("\nSaved: throughput_sync_comparison.png")
