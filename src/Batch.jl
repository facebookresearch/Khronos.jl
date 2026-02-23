# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Batch simulation execution for incoherent source averaging.
# Runs multiple independent FDTD simulations that share geometry but differ
# in source configuration, then sums far-field power incoherently.

export run_batch, run_batch_concurrent, plan_batch, compute_incoherent_LEE

# ------------------------------------------------------------------- #
# Memory planning
# ------------------------------------------------------------------- #

"""
    plan_batch(sim_template::SimulationData, n_sims::Int;
               target_utilization=0.85) -> NamedTuple

Estimate how many simulations can run concurrently on a single GPU.
Returns planning information including max batch size and number of waves.

Shared arrays (geometry) are allocated once; field arrays are per-simulation.
"""
function plan_batch(sim_template::SimulationData, n_sims::Int;
                    target_utilization::Float64=0.85)
    Nx, Ny, Nz = sim_template.Nx, sim_template.Ny, max(1, sim_template.Nz)
    elem = sizeof(backend_number)
    voxels = (Nx + 2) * (Ny + 2) * (Nz + 2)  # with ghost cells

    # Count field arrays per simulation
    # Primary: E, H, B, D = 12 arrays
    # PML auxiliary: depends on boundaries, estimate conservatively
    n_primary = 12
    n_pml_est = 0
    if !isnothing(sim_template.boundaries)
        n_pml_est = 18  # worst case: C, U, W for B and D, all 3 directions
    end
    n_source_est = 6  # worst case: all 6 source components
    n_field_arrays = n_primary + n_pml_est + n_source_est

    per_sim_bytes = n_field_arrays * voxels * elem

    # Shared geometry (not batched)
    geom_voxels = Nx * Ny * Nz
    n_geom_arrays = 6  # eps_inv_x/y/z, potentially sigma_Dx/y/z
    shared_bytes = n_geom_arrays * geom_voxels * elem

    # Query GPU memory
    gpu_mem = 0
    try
        if backend_engine isa CUDABackend
            dev = CUDA.device()
            gpu_mem = CUDA.totalmem(dev)
        end
    catch
        # No GPU
    end

    available = gpu_mem > 0 ? target_utilization * gpu_mem - shared_bytes : typemax(Int)
    max_batch = max(1, floor(Int, available / per_sim_bytes))
    n_waves = ceil(Int, n_sims / max_batch)

    return (
        max_batch = min(max_batch, n_sims),
        n_waves = n_waves,
        shared_bytes = shared_bytes,
        per_sim_bytes = per_sim_bytes,
        total_bytes = shared_bytes + min(max_batch, n_sims) * per_sim_bytes,
        gpu_memory = gpu_mem,
    )
end

# ------------------------------------------------------------------- #
# Batch execution
# ------------------------------------------------------------------- #

"""
    run_batch(sim_configs::Vector{<:NamedTuple};
              until=nothing, until_after_sources=nothing,
              stop_condition=nothing) -> Vector

Run multiple simulations sequentially. Each config is a NamedTuple with fields
that override the template simulation's source configuration.

Each simulation shares the same geometry but has different sources.

Returns a vector of simulation results (monitor data).

# Arguments
- `sim_configs`: Vector of NamedTuples, each with at minimum:
  - `sources`: Vector of source objects for this simulation
  - Optionally other overrides (monitors, etc.)
- `until`, `until_after_sources`: time limits (passed to `run`)
- `stop_condition`: optional convergence criterion function

# Example
```julia
configs = [
    (sources = [UniformSource(...)],),
    (sources = [UniformSource(...)],),
]
results = run_batch(configs; until_after_sources=100.0)
```
"""
function run_batch(sim_configs::Vector;
                   template_kwargs::NamedTuple = NamedTuple(),
                   until=nothing,
                   until_after_sources=nothing,
                   stop_condition=nothing)
    results = []
    n_sims = length(sim_configs)

    for (i, config) in enumerate(sim_configs)
        if !is_distributed() || is_root()
            @info("Batch simulation $i / $n_sims")
        end

        # Create simulation from template + config overrides
        sim_kwargs = merge(template_kwargs, config)
        sim = Simulation(; sim_kwargs...)

        # Prepare and run
        prepare_simulation!(sim)

        if !isnothing(stop_condition)
            run(sim; until=until, until_after_sources=until_after_sources)
        elseif !isnothing(until)
            run(sim; until=until)
        elseif !isnothing(until_after_sources)
            run(sim; until_after_sources=until_after_sources)
        end

        # Collect results (monitor data)
        push!(results, (
            monitor_data = deepcopy(sim.monitor_data),
            monitors = sim.monitors,
        ))

        # Free GPU memory for next simulation
        sim.fields = nothing
        sim.chunk_data = nothing
        sim.geometry_data = nothing
        GC.gc(false)
    end

    return results
end

# ------------------------------------------------------------------- #
# Concurrent batch execution (multi-stream GPU)
# ------------------------------------------------------------------- #

"""
    run_batch_concurrent(sim_configs; template_kwargs, until_after_sources,
                         max_concurrent=0) -> Vector

Run multiple simulations **concurrently** on a single GPU using CUDA
multi-stream execution. Each simulation runs in its own Julia task,
which maps to a separate CUDA stream. The GPU scheduler overlaps
kernel execution across streams, maximizing SM utilization.

Geometry arrays are shared (read-only, allocated once). Field arrays
are allocated per-simulation. Simulations are processed in waves if
total memory exceeds GPU capacity.

# Arguments
- `sim_configs`: Vector of NamedTuples with per-sim overrides (sources, monitors)
- `template_kwargs`: Shared simulation parameters (cell_size, geometry, etc.)
- `until_after_sources`: Convergence criterion or time limit
- `max_concurrent`: Max sims per wave (0 = auto from GPU memory)
"""
function run_batch_concurrent(sim_configs::Vector;
                               template_kwargs::NamedTuple = NamedTuple(),
                               until=nothing,
                               until_after_sources=nothing,
                               max_concurrent::Int=0)
    n_sims = length(sim_configs)

    # Determine concurrency level from GPU memory
    if max_concurrent <= 0
        # Estimate per-sim memory from template
        est_sim = Simulation(; template_kwargs..., sim_configs[1]...)
        plan = plan_batch(est_sim, n_sims)
        max_concurrent = plan.max_batch
        if !is_distributed() || is_root()
            @info("Concurrent batch: $(n_sims) sims, max_concurrent=$(max_concurrent) " *
                  "(per_sim=$(round(plan.per_sim_bytes / 1e9, digits=2)) GB, " *
                  "GPU=$(round(plan.gpu_memory / 1e9, digits=1)) GB)")
        end
    end

    # Disable CUDA Graph capture for concurrent mode (graphs are per-stream
    # and would conflict; stream-level concurrency provides the parallelism)
    old_cuda_graphs = get(ENV, "KHRONOS_CUDA_GRAPHS", "1")
    ENV["KHRONOS_CUDA_GRAPHS"] = "0"

    all_results = Vector{Any}(undef, n_sims)

    try
        for wave_start in 1:max_concurrent:n_sims
            wave_end = min(wave_start + max_concurrent - 1, n_sims)
            wave_size = wave_end - wave_start + 1

            if !is_distributed() || is_root()
                @info("Wave $(cld(wave_start, max_concurrent)): sims $(wave_start)-$(wave_end) ($(wave_size) concurrent)")
            end

            # Prepare all simulations in this wave
            sims = Vector{SimulationData}(undef, wave_size)
            for (j, i) in enumerate(wave_start:wave_end)
                sim_kwargs = merge(template_kwargs, sim_configs[i])
                sims[j] = Simulation(; sim_kwargs...)
                prepare_simulation!(sims[j])
            end

            # Launch all concurrently via Julia tasks
            # Each task gets its own CUDA stream (CUDA.jl per-task stream binding)
            t_wave_start = time()
            tasks = map(sims) do sim
                Threads.@spawn begin
                    if !isnothing(until)
                        run(sim; until=until)
                    elseif !isnothing(until_after_sources)
                        run(sim; until_after_sources=until_after_sources)
                    end
                end
            end

            # Wait for all tasks in this wave
            for t in tasks
                fetch(t)
            end

            t_wave = time() - t_wave_start
            if !is_distributed() || is_root()
                total_steps = sum(s.timestep for s in sims)
                total_voxels = sims[1].Nx * sims[1].Ny * sims[1].Nz
                agg_rate = total_voxels * total_steps / t_wave / 1e6
                @info("  Wave complete: $(round(t_wave, digits=1))s, " *
                      "aggregate $(round(agg_rate, digits=0)) MVoxels/s")
            end

            # Collect results
            for (j, i) in enumerate(wave_start:wave_end)
                all_results[i] = (
                    monitor_data = deepcopy(sims[j].monitor_data),
                    monitors = sims[j].monitors,
                )
            end

            # Free GPU memory for next wave
            for sim in sims
                sim.fields = nothing
                sim.chunk_data = nothing
                sim.geometry_data = nothing
            end
            GC.gc(false)
        end
    finally
        # Restore CUDA Graph setting
        ENV["KHRONOS_CUDA_GRAPHS"] = old_cuda_graphs
    end

    return all_results
end

# ------------------------------------------------------------------- #
# Incoherent power summation
# ------------------------------------------------------------------- #

"""
    compute_incoherent_LEE(batch_results, theta, phi;
                           cone_half_angle=deg2rad(90),
                           monitor_selector=nothing)

Compute Light Extraction Efficiency by incoherently summing far-field power
from multiple batch simulation results.

Each batch element's near2far data produces a far-field power pattern.
Power (not fields) is summed across all batch elements to model spatially
incoherent emission.

# Arguments
- `batch_results`: Vector of batch result NamedTuples (from `run_batch`)
- `theta`, `phi`: angular grids for far-field evaluation
- `cone_half_angle`: half-angle of extraction cone (radians)
- `monitor_selector`: optional function to select the Near2FarMonitor from each result
"""
function compute_incoherent_LEE(batch_results::Vector, theta::Vector{Float64},
                                phi::Vector{Float64};
                                cone_half_angle::Float64=deg2rad(90.0),
                                monitor_selector=nothing)
    n_theta = length(theta)
    n_phi = length(phi)
    total_power = zeros(n_theta, n_phi)

    for (i, result) in enumerate(batch_results)
        # Find the Near2FarMonitorData in this result
        n2f_data = nothing
        if !isnothing(monitor_selector)
            n2f_data = monitor_selector(result)
        else
            # Default: find first Near2FarMonitor in the monitors list
            if haskey(result, :monitors) && !isnothing(result.monitors)
                for mon in result.monitors
                    if mon isa Near2FarMonitor && !isnothing(mon.monitor_data)
                        n2f_data = mon.monitor_data
                        break
                    end
                end
            end
        end

        if isnothing(n2f_data)
            @warn("Batch result $i: no Near2FarMonitorData found, skipping")
            continue
        end

        # Compute far field and power for this simulation
        far_field = compute_far_field(n2f_data)
        power = compute_far_field_power(far_field, theta, phi;
                                         eps=n2f_data.medium_eps,
                                         mu=n2f_data.medium_mu)

        # Incoherent sum: add POWER not fields
        total_power .+= power
    end

    return compute_LEE(total_power, theta, phi; cone_half_angle=cone_half_angle)
end

# ------------------------------------------------------------------- #
# Multi-GPU batch execution
# ------------------------------------------------------------------- #

"""
    run_batch_multi_gpu(sim_configs; kwargs...)

Run batch simulations across multiple GPUs on a single node.
Each GPU processes a subset of the simulations sequentially.

Requires CUDA.jl with multiple GPU devices available.
"""
function run_batch_multi_gpu(sim_configs::Vector;
                              template_kwargs::NamedTuple = NamedTuple(),
                              n_gpus::Int = 1,
                              kwargs...)
    if !(backend_engine isa CUDABackend)
        @warn("Multi-GPU batch requires CUDA backend, falling back to single device")
        return run_batch(sim_configs; template_kwargs=template_kwargs, kwargs...)
    end

    n_sims = length(sim_configs)
    actual_gpus = min(n_gpus, n_sims)

    # Partition simulations across GPUs
    batch_per_gpu = ceil(Int, n_sims / actual_gpus)

    tasks = map(1:actual_gpus) do gpu_id
        start_idx = (gpu_id - 1) * batch_per_gpu + 1
        end_idx = min(gpu_id * batch_per_gpu, n_sims)
        gpu_configs = sim_configs[start_idx:end_idx]

        Threads.@spawn begin
            CUDA.device!(gpu_id - 1)
            run_batch(gpu_configs; template_kwargs=template_kwargs, kwargs...)
        end
    end

    # Collect and concatenate results
    results = vcat(fetch.(tasks)...)
    return results
end
