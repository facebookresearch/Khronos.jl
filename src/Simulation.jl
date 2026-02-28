# Copyright (c) Meta Platforms, Inc. and affiliates.


export prepare_simulation!, run, run_until, stop_when_dft_decayed, run_benchmark

"""
    increment_timestep!(sim::SimulationData)

Increment the timestep by one (integer) time unit.
"""
function increment_timestep!(sim::SimulationData)
    sim.timestep += 1
    return
end

"""
    round_time(sim::SimulationData)::Float

When comparing times e.g. for source cutoffs, it is useful to round to float to
avoid sensitivities in roundoff.
"""
round_time(sim::SimulationData)::Float64 = Float64(sim.timestep * sim.Δt)

"""
    time_to_steps(sim::SimulationData, time::Real)::Int

Determine how many timesteps are need to step a certain time period.
"""
time_to_steps(sim::SimulationData, time::Real)::Int = floor(Int, time / sim.Δt)

"""
    get_sim_dims(sim)

Determine the simulation dimensionality, coordinate system, and polarization.

Since we have abstract types that can't be initialized in Julia (weird) we
return the type itself.

For now, since we don't have off-diagonal materials, we don't need to check
those when determining the 2D polarization.
"""
function get_sim_dims(sim::SimulationData)
    if sim.ndims == 3
        sim.dimensionality = ThreeD
    elseif (contains_TE_sources(sim) && !contains_TM_sources(sim))
        sim.dimensionality = TwoD_TE
    elseif (!contains_TE_sources(sim) && contains_TM_sources(sim))
        sim.dimensionality = TwoD_TM
    else
        sim.dimensionality = TwoD
    end
    #TODO add support for cylindrical coordinates
end

"""
    prepare_simulation!(sim::SimulationData)

Initialize a Simulation object (`sim`) such that it can be properly times.

Specifically, allocate all of the relevant field components, build the
permittivity and permeability arrays, etc.
"""
function prepare_simulation!(sim::SimulationData)
    if sim.is_prepared
        return
    end

    num_voxels = sim.Nx * sim.Ny * sim.Nz
    if !is_distributed() || is_root()
        @info("Preparing simulation object ($(sim.Nx)×$(sim.Ny)×$(sim.Nz) = $(num_voxels) voxels)...")
    end

    # Log non-uniform grid info (before GPU transfer)
    if sim.Δx isa AbstractVector && (!is_distributed() || is_root())
        @info("  Non-uniform grid: Δx has $(length(sim.Δx)) cells, " *
              "min=$(round(minimum(sim.Δx), sigdigits=4)), max=$(round(maximum(sim.Δx), sigdigits=4))")
    end

    t_start = time()

    # prepare geometry
    # Rasterize full-domain geometry on CPU, then transfer to GPU.
    # In distributed mode, global arrays are used temporarily for slicing
    # into per-chunk geometry (in create_local_chunks), then freed.
    t0 = time()
    init_geometry(sim, sim.geometry)
    t1 = time()
    if !is_distributed() || is_root()
        @info("  init_geometry:   $(round(t1-t0, digits=3))s")
    end

    # prepare boundaries
    init_boundaries(sim, sim.boundaries)
    t2 = time()
    if !is_distributed() || is_root()
        @info("  init_boundaries: $(round(t2-t1, digits=3))s")
    end

    # plan domain decomposition into chunks
    sim.chunk_plan = plan_chunks(sim)
    t2b = time()
    if !is_distributed() || is_root()
        @info("  plan_chunks:     $(round(t2b-t2, digits=3))s ($(sim.chunk_plan.total_chunks) chunk(s))")
    end

    # MPI chunk-to-rank assignment
    if is_distributed()
        sim.chunk_rank_assignment = assign_chunks_to_ranks(sim.chunk_plan, mpi_size())
    end

    # prepare sources
    add_sources(sim, sim.sources)
    t3 = time()
    if !is_distributed() || is_root()
        @info("  add_sources:     $(round(t3-t2b, digits=3))s")
    end

    # determine dimensionality
    get_sim_dims(sim)

    # estimate memory requirements before allocation
    mem_est = estimate_memory(sim)
    t3b = time()
    if !is_distributed() || is_root()
        @info("  estimate_mem:    $(round(t3b-t3, digits=3))s")
    end

    # prepare fields
    # In distributed mode or multi-chunk, skip allocating full-domain arrays —
    # stepping operates on per-chunk fields allocated by create_all_chunks.
    if !is_distributed() && sim.chunk_plan.total_chunks == 1
        init_fields(sim, sim.dimensionality)
    end
    t4 = time()
    if !is_distributed() || is_root()
        @info("  init_fields:     $(round(t4-t3b, digits=3))s")
    end

    # Auto-decimate monitors based on source bandwidth (Nyquist criterion)
    auto_decimate!(sim)

    # prepare time and dft monitors
    init_monitors(sim, sim.monitors)
    t5 = time()
    if !is_distributed() || is_root()
        @info("  init_monitors:   $(round(t5-t4, digits=3))s")
    end

    # create per-chunk runtime data
    if is_distributed()
        sim.chunk_data = create_local_chunks(sim)
    else
        sim.chunk_data = create_all_chunks(sim)
    end
    connect_chunks!(sim)

    # Precompute halo copy operations for fast runtime exchange
    # (only for CUDA backend — HaloCopyOp uses CuPtr)
    if !is_distributed() && !(backend_engine isa CPU)
        precompute_halo_ops!(sim)
    end

    # Multi-stream setup: create per-chunk CUDA streams for concurrent kernel dispatch
    if backend_engine isa CUDABackend && length(sim.chunk_data) > 1 &&
       get(ENV, "KHRONOS_MULTI_STREAM", "1") == "1"
        sim._chunk_streams = [CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING) for _ in 1:length(sim.chunk_data)]
        sim._use_multi_stream = true
    end

    # Allocate MPI staging buffers for cross-rank halo exchange
    if is_distributed()
        allocate_mpi_halo_buffers!(sim)
        init_nccl!()  # GPU-direct halo exchange via NCCL (falls back to MPI if unavailable)
        if nccl_initialized()
            precompute_nccl_exchange!(sim)
        end
    end

    t6 = time()
    if !is_distributed() || is_root()
        @info("  create_chunks:   $(round(t6-t5, digits=3))s")
    end

    # Initialize dispersive polarization (ADE) if any materials have susceptibilities
    init_polarization!(sim)
    t6b = time()
    if any_material_has_susceptibilities(sim.geometry)
        if !is_distributed() || is_root()
            @info("  init_polariz.:   $(round(t6b-t6, digits=3))s")
        end
    end

    if !is_distributed() || is_root()
        @info("  Total prepare:   $(round(t6b-t_start, digits=3))s")
    end

    sim.is_prepared = true

    # Transfer non-uniform grid spacing vectors to GPU for kernel use
    # (must happen after all CPU-side geometry/boundary setup is complete)
    if sim.Δx isa AbstractVector && !(sim.Δx isa backend_array)
        sim.Δx = backend_array(backend_number.(sim.Δx))
    end
    if sim.Δy isa AbstractVector && !(sim.Δy isa backend_array)
        sim.Δy = backend_array(backend_number.(sim.Δy))
    end
    if sim.Δz isa AbstractVector && sim.ndims > 2 && !(sim.Δz isa backend_array)
        sim.Δz = backend_array(backend_number.(sim.Δz))
    end

    # P.2: Cache kernel objects to avoid repeated construction from non-const global
    wg = parse(Int, get(ENV, "KHRONOS_WORKGROUP_SIZE", "64"))
    sim._cached_curl_kernel = step_curl!(backend_engine, (wg,))
    sim._cached_update_kernel = update_field!(backend_engine, (wg,))
    sim._cached_curl_comp_kernel = step_curl_comp!(backend_engine, (wg,))
    sim._cached_update_comp_kernel = update_field_comp!(backend_engine, (wg,))
    sim._cached_fused_kernel = step_curl_and_update!(backend_engine, (wg,))
    sim._cached_fused_pml_kernel = step_curl_and_update_pml!(backend_engine, (wg,))
    sim._cached_source_kernel = update_source!(backend_engine, (wg,))
    sim._cached_dft_kernel = update_dft_monitor!(backend_engine, (wg,))
    sim._cached_dft_chunk_kernel = update_dft_monitor_chunk!(backend_engine, (wg,))

    # P.3: Cache CUDA dispatch constants (avoid per-timestep ENV parsing and recomputation)
    sim._cached_grid_is_uniform = _grid_is_uniform(sim)
    if sim._cached_grid_is_uniform
        sim._cached_dt_dx = backend_number(sim.Δt / sim.Δx)
        sim._cached_dt_dy = backend_number(sim.Δt / sim.Δy)
        sim._cached_dt_dz = backend_number(sim.Δt / sim.Δz)
    else
        sim._cached_dt_dx = zero(backend_number)
        sim._cached_dt_dy = zero(backend_number)
        sim._cached_dt_dz = zero(backend_number)
    end
    cuda_wg_total = parse(Int, get(ENV, "KHRONOS_CUDA_WORKGROUP_SIZE", "256"))
    sim._cached_cuda_wg_x = Int32(32)
    sim._cached_cuda_wg_y = Int32(cld(cuda_wg_total, 32))

    return
end


"""
    run(
    sim::SimulationData;
    until::Union{Real,Function,Nothing} = nothing,
    until_after_sources::Union{Real,Function,Nothing} = nothing,
)

TBW
"""
function run(
    sim::SimulationData;
    until::Union{Real,Function,Nothing} = nothing,
    until_after_sources::Union{Real,Function,Nothing} = nothing,
)

    # santize the input
    if isnothing(until) && isnothing(until_after_sources)
        error("Must specify a terminating conditon in the run functions.")
    end
    if !isnothing(until) && !isnothing(until_after_sources)
        error("Must specify a single terminating conditon in the run functions.")
    end

    # initialize fields and structure
    if (!sim.is_prepared)
        prepare_simulation!(sim)
    end

    num_voxels = sim.Nx * sim.Ny * sim.Nz
    step_start = sim.timestep
    if !is_distributed() || is_root()
        @info("Starting simulation...")
    end
    t_run_start = time()
    warmup_steps = 50
    t_after_warmup = nothing
    steps_at_warmup = 0
    # prepare the step functions
    wait_for_sources = false
    if until isa Function
        _run_func = until
    elseif until isa Real
        _run_func = run_until(sim, until)
    elseif until_after_sources isa Function
        _run_func = until_after_sources
        wait_for_sources = true
    elseif until_after_sources isa Real
        _run_func = run_until(sim, until_after_sources)
        wait_for_sources = true
    else
        error("Must specify either `until` or `until_after_sources`.")
    end

    # Timestep until sources are done.
    if wait_for_sources
        time_for_sources = last_source_time(sim)
        while round_time(sim) <= time_for_sources
            step!(sim)
            if isnothing(t_after_warmup) && (sim.timestep - step_start) == warmup_steps
                t_after_warmup = time()
                steps_at_warmup = sim.timestep - step_start
            end
        end
        if !is_distributed() || is_root()
            @info("All sources have terminated...")
        end
    end

    # Execute step function. The function should return true if it's time to
    # break out of the loop.
    while !(_run_func(sim))
        step!(sim)
        if isnothing(t_after_warmup) && (sim.timestep - step_start) == warmup_steps
            t_after_warmup = time()
            steps_at_warmup = sim.timestep - step_start
        end
    end

    t_run_end = time()
    total_steps = sim.timestep - step_start
    elapsed = t_run_end - t_run_start
    if total_steps > 0 && elapsed > 0 && (!is_distributed() || is_root())
        mcells_per_s = num_voxels * total_steps / elapsed / 1e6
        if !isnothing(t_after_warmup) && total_steps > steps_at_warmup
            steady_steps = total_steps - steps_at_warmup
            steady_elapsed = t_run_end - t_after_warmup
            steady_rate = num_voxels * steady_steps / steady_elapsed / 1e6
            @info("Simulation complete: $(total_steps) steps in $(round(elapsed, digits=3))s " *
                  "($(round(mcells_per_s, digits=1)) MVoxels/s overall, " *
                  "$(round(steady_rate, digits=1)) MVoxels/s after warmup)")
        else
            @info("Simulation complete: $(total_steps) steps in $(round(elapsed, digits=3))s ($(round(mcells_per_s, digits=1)) MVoxels/s)")
        end
    end

    return
end

function run_until(sim::SimulationData, until::Real)
    _stop(sim) = round_time(sim) > until
    return _stop
end

"""
    stop_when_dft_decayed(;
    tolerance::Real = 1e-6,
    minimum_runtime::Real = 0.0,
    maximum_runtime::Real = Inf,
)::Function

Return a callback for use with `run(sim, until_after_sources = ...)` that stops
the simulation when all DFT monitors have converged.

Convergence is evaluated **per DFT monitor**: each monitor independently tracks
its step-to-step norm change relative to the largest change it has ever seen.
The simulation stops only when every monitor that has received significant signal
reports `rel_change <= tolerance`.

# Arguments
- `tolerance`: relative change threshold (default `1e-6`).
- `minimum_runtime`: simulation time (in natural units) before convergence
  checks begin.  Set this to at least the light travel time from the source
  to the farthest monitor.
- `maximum_runtime`: hard upper bound on simulation time.
"""
function stop_when_dft_decayed(;
    tolerance::Real = 1e-6,
    minimum_runtime::Real = 0.0,
    maximum_runtime::Real = Inf,
)::Function

    if minimum_runtime > maximum_runtime
        error(
            "Minimum runtime ($minimum_runtime) cannot be greater than maximum runtime ($maximum_runtime).",
        )
    end

    # Per-monitor convergence state.
    # Each entry stores (prev_norm, maxchange) for one DFT monitor.
    monitor_state = Dict{Int, Tuple{Float64, Float64}}()

    function _stop(sim::SimulationData)::Bool
        if round_time(sim) < minimum_runtime
            return false
        end

        if round_time(sim) > maximum_runtime
            @warn("Maximum runtime reached.")
            return true
        end

        all_converged = true
        n_active = 0  # monitors that have seen real signal

        for (i, md) in enumerate(sim.monitor_data)
            # Only DFT monitors contribute (ModeMonitorData, FluxMonitorData etc.
            # return 0.0 from the fallback dispatch — their internal DFT sub-monitors
            # are pushed into sim.monitor_data separately).
            current_norm = dft_fields_norm(md)
            current_norm == 0.0 && continue

            if !haskey(monitor_state, i)
                # First time seeing this monitor — record initial norm,
                # don't allow convergence yet.
                monitor_state[i] = (current_norm, 0.0)
                all_converged = false
                continue
            end

            prev_norm, maxchange = monitor_state[i]
            change = abs(current_norm - prev_norm)
            maxchange = max(maxchange, change)
            monitor_state[i] = (current_norm, maxchange)

            if maxchange == 0.0
                # Haven't seen any change yet — not converged.
                all_converged = false
                continue
            end

            n_active += 1
            rel_change = change / maxchange

            if rel_change > tolerance
                all_converged = false
            end
        end

        # Don't stop if no monitors have seen signal yet.
        if n_active == 0
            return false
        end

        @verbose("DFT fields decay: $n_active active monitors, all_converged=$all_converged")

        return all_converged
    end

    return _stop
end

"""
    run_benchmark(sim::SimulationData, until::Int)

TBW
"""
function run_benchmark(sim::SimulationData, until::Int)
    if (!sim.is_prepared)
        prepare_simulation!(sim)
    end

    if !is_distributed() || is_root()
        @info("Running simulation...")
    end

    t_tic = time()
    for it = 1:until
        if (it == 10)
            KernelAbstractions.synchronize(backend_engine)
            t_tic = time()
        end
        step!(sim)
    end
    KernelAbstractions.synchronize(backend_engine)

    time_s = time() - t_tic
    if !is_distributed() || is_root()
        @info("Run complete.")
        @info("===========================================")
        @info("Total number of iterations: $until.")
        @info("Total simulation time: $time_s seconds.")
        num_voxels = sim.Nx * sim.Ny * sim.Nz
        steprate = num_voxels * (until - 10) / time_s / 1e6
        @info("Simulation speed:  $steprate MCells/S.")
        @info("===========================================")
    else
        num_voxels = sim.Nx * sim.Ny * sim.Nz
        steprate = num_voxels * (until - 10) / time_s / 1e6
    end

    return steprate
end

"""
dft_fields_norm(fields_array::AbstractArray)

TBW
"""
dft_fields_norm(fields_array::AbstractArray) = sqrt(sum(abs.(fields_array) .^ 2))
dft_fields_norm(monitor::DFTMonitorData) = dft_fields_norm(monitor.fields)
dft_fields_norm(monitor::MonitorData) = 0.0 # default case
dft_fields_norm(monitor_list::Vector{<:MonitorData}) =
    sum([dft_fields_norm(m) for m in monitor_list])
dft_fields_norm(sim::SimulationData) = dft_fields_norm(sim.monitor_data)

# ---------------------------------------------------------- #
# Simulation reset for adjoint runs
# ---------------------------------------------------------- #

"""
    _zero_field!(arr)

Zero a field array on the GPU (or CPU). No-op if the array is nothing.
"""
function _zero_field!(arr::AbstractArray)
    fill!(arr, zero(eltype(arr)))
end
_zero_field!(::Nothing) = nothing

"""
    reset_fields!(sim::SimulationData)

Reset all field arrays to zero for a new simulation run (e.g., the adjoint run).
This zeros:
- Primary fields (E, H, D, B)
- PML auxiliary fields (C, U, W)
- Source arrays (SD, SB)
- DFT monitor accumulators
- Per-chunk field arrays
- Polarization P/P_prev arrays
- Timestep counter

Does NOT re-run `prepare_simulation!` -- geometry and boundaries stay intact.
Sources and monitors should be reconfigured by the caller after reset.
"""
function reset_fields!(sim::SimulationData)
    if !sim.is_prepared
        return  # nothing to reset if not yet prepared
    end

    # Reset timestep
    sim.timestep = 0
    sim.sources_active = true

    # Zero per-chunk field arrays (this is where the actual field data lives)
    if !isnothing(sim.chunk_data)
        for chunk in sim.chunk_data
            f = chunk.fields
            # Primary fields
            for fn in (:fEx, :fEy, :fEz, :fHx, :fHy, :fHz,
                        :fBx, :fBy, :fBz, :fDx, :fDy, :fDz)
                arr = getfield(f, fn)
                _zero_field!(arr)
            end

            # PML auxiliary fields
            for fn in (:fCBx, :fCBy, :fCBz, :fUBx, :fUBy, :fUBz,
                        :fWBx, :fWBy, :fWBz, :fSBx, :fSBy, :fSBz,
                        :fPBx, :fPBy, :fPBz,
                        :fCDx, :fCDy, :fCDz, :fUDx, :fUDy, :fUDz,
                        :fWDx, :fWDy, :fWDz, :fSDx, :fSDy, :fSDz,
                        :fPDx, :fPDy, :fPDz)
                arr = getfield(f, fn)
                _zero_field!(arr)
            end

            # Polarization data (dispersive materials)
            if !isnothing(chunk.polarization_data)
                for pole in chunk.polarization_data.poles
                    _zero_field!(pole.Px)
                    _zero_field!(pole.Py)
                    _zero_field!(pole.Pz)
                    _zero_field!(pole.Px_prev)
                    _zero_field!(pole.Py_prev)
                    _zero_field!(pole.Pz_prev)
                end
            end
        end
    end

    # Zero global field arrays (single-chunk mode uses these)
    if !isnothing(sim.fields)
        f = sim.fields
        for fn in fieldnames(Fields)
            arr = getfield(f, fn)
            _zero_field!(arr)
        end
    end

    # Clear DFT monitor accumulators
    for md in sim.monitor_data
        if md isa DFTMonitorData
            _zero_field!(md.fields)
        end
    end

    # Invalidate CUDA graph captures (field state changed)
    sim._cuda_graph_exec_H = nothing
    sim._cuda_graph_exec_E = nothing

    return nothing
end

export reset_fields!

export Simulation
