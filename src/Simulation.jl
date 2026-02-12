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

    t_start = time()

    # prepare geometry
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

    # prepare fields
    init_fields(sim, sim.dimensionality)
    t4 = time()
    if !is_distributed() || is_root()
        @info("  init_fields:     $(round(t4-t3, digits=3))s")
    end

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

    # Allocate MPI staging buffers for cross-rank halo exchange
    if is_distributed()
        allocate_mpi_halo_buffers!(sim)
    end

    t6 = time()
    if !is_distributed() || is_root()
        @info("  create_chunks:   $(round(t6-t5, digits=3))s")
        @info("  Total prepare:   $(round(t6-t_start, digits=3))s")
    end

    sim.is_prepared = true

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
        _run_func = run_until(sim, wait_for_sources)
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

TBW
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

    closure = Dict("previous_fields" => 0.0, "t0" => 0.0, "dt" => 0.0, "maxchange" => 0.0)

    function _stop(sim::SimulationData)::Bool
        if round_time(sim) < minimum_runtime
            return false
        end

        if round_time(sim) > maximum_runtime
            @warn("Maximum runtime reached.")
            return true
        end

        # TODO
        # if round_time(sim) <= closure["dt"] + closure["t0"]
        #     return false
        # end

        previous_fields = closure["previous_fields"]
        current_fields = dft_fields_norm(sim)
        change = abs(previous_fields - current_fields)
        closure["maxchange"] = max(closure["maxchange"], change)

        if previous_fields == 0.0
            closure["previous_fields"] = current_fields
            return false
        end

        closure["previous_fields"] = current_fields
        closure["t0"] = sim.timestep

        rel_change = (change / closure["maxchange"])

        @verbose("DFT fields decay()")

        return rel_change <= tolerance
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

export Simulation
