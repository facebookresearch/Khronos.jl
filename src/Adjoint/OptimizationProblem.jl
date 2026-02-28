# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    OptimizationProblem.jl

Top-level orchestrator for adjoint-based topology optimization.
Manages the forward run, adjoint run, and gradient computation
in a state-machine pattern: INIT → FWD → ADJ → INIT.

Ported from meep/python/adjoint/optimization_problem.py
"""

"""
    OptimizationProblem

Top-level struct for adjoint-based topology optimization.

Holds the simulation, design regions, objective functions/arguments,
and manages the forward/adjoint/gradient computation cycle.
"""
@with_kw mutable struct OptimizationProblem
    sim::SimulationData
    objective_functions::Vector{Function}
    objective_arguments::Vector{<:ObjectiveQuantity}
    design_regions::Vector{DesignRegion}
    frequencies::Vector{Float64}

    # Convergence parameters
    decay_by::Float64 = 1e-6
    minimum_run_time::Float64 = 0.0
    maximum_run_time::Float64 = Inf

    # State machine
    current_state::Symbol = :INIT  # :INIT, :FWD, :ADJ

    # Cached results
    f0::Union{Nothing, Any} = nothing
    gradient::Union{Nothing, AbstractArray} = nothing
    results_list::Union{Nothing, Vector} = nothing
    forward_sources::Union{Nothing, Any} = nothing

    # Forward monitors
    forward_design_monitors::Union{Nothing, Vector{DFTMonitor}} = nothing
    adjoint_design_monitors::Union{Nothing, Vector{DFTMonitor}} = nothing
    forward_obj_monitors::Union{Nothing, Vector} = nothing

    # History
    f_bank::Vector{Any} = Any[]
end

"""
    (opt::OptimizationProblem)(rho; need_gradient=true)

Callable interface: given design parameters `rho`, compute the objective
function value and (optionally) its gradient.

Returns (f0, gradient).
"""
function (opt::OptimizationProblem)(
    rho::Vector{Float64};
    need_gradient::Bool = true,
)
    # Update design
    update_design_all!(opt, rho)

    # Forward run
    if opt.current_state == :INIT
        forward_run!(opt)
    end

    # Adjoint + gradient
    if need_gradient
        if opt.current_state == :FWD
            adjoint_run!(opt)
            calculate_gradient!(opt)
        end
    end

    return opt.f0, opt.gradient
end

"""
    update_design_all!(opt, rho)

Update all design regions with new parameters and reset the simulation.
"""
function update_design_all!(opt::OptimizationProblem, rho::Vector{Float64})
    for dr in opt.design_regions
        update_design!(opt.sim, dr, rho)
    end
    reset_fields!(opt.sim)
    opt.current_state = :INIT
end

"""
    forward_run!(opt::OptimizationProblem)

Run the forward simulation:
1. Reset fields
2. Install design region DFT monitors
3. Run until DFT fields converge
4. Evaluate objective functions
"""
function forward_run!(opt::OptimizationProblem)
    sim = opt.sim

    # Save forward sources reference
    opt.forward_sources = sim.sources

    # Reset fields for fresh run
    reset_fields!(sim)

    # Clear monitor data to avoid duplication on repeated calls
    empty!(sim.monitor_data)

    # Register objective quantity monitors
    opt.forward_obj_monitors = []
    for oq in opt.objective_arguments
        mon = register_monitors!(oq, sim, opt.frequencies)
        push!(opt.forward_obj_monitors, mon)
        if !isnothing(mon) && mon isa DFTMonitor
            # Initialize the DFT monitor and add to sim.monitor_data
            push!(sim.monitor_data, init_monitors(sim, mon))
        end
    end

    # Install design region DFT monitors
    opt.forward_design_monitors = DFTMonitor[]
    for dr in opt.design_regions
        mons = install_design_region_monitors!(sim, dr, opt.frequencies)
        append!(opt.forward_design_monitors, mons)
    end

    # Distribute monitor data to chunks
    if !isnothing(sim.chunk_data)
        for chunk in sim.chunk_data
            chunk.monitor_data = sim.monitor_data
        end
    end

    # Prepare if needed (first call only)
    if !sim.is_prepared
        prepare_simulation!(sim)
    end

    run(sim; until_after_sources = stop_when_dft_decayed(
        tolerance = opt.decay_by,
        minimum_runtime = opt.minimum_run_time,
        maximum_runtime = opt.maximum_run_time,
    ))

    # Evaluate objectives
    opt.results_list = [evaluate(oq, sim) for oq in opt.objective_arguments]
    opt.f0 = [f(opt.results_list...) for f in opt.objective_functions]

    if length(opt.f0) == 1
        opt.f0 = opt.f0[1]
    end

    push!(opt.f_bank, opt.f0)
    opt.current_state = :FWD

    return opt.f0
end

"""
    adjoint_run!(opt::OptimizationProblem)

Run the adjoint simulation:
1. Compute dJ/d(monitor) via ForwardDiff
2. Place adjoint sources
3. Negate k-point (if applicable)
4. Reset fields, swap sources, re-init DFT monitors
5. Run until convergence
6. Restore k-point and original sources
"""
function adjoint_run!(opt::OptimizationProblem)
    sim = opt.sim

    # Compute adjoint sources for each objective function
    adjoint_sources = Source[]
    for (ar, obj_func) in enumerate(opt.objective_functions)
        for (mi, oq) in enumerate(opt.objective_arguments)
            # Compute Jacobian of objective w.r.t. this monitor's output
            dJ = _compute_jacobian(obj_func, opt.results_list, mi)

            if any(x -> abs(x) > 0, dJ)
                srcs = place_adjoint_source(oq, sim, dJ, opt.frequencies)
                append!(adjoint_sources, srcs)
            end
        end
    end

    if isempty(adjoint_sources)
        @warn "No adjoint sources placed -- gradient is zero."
        opt.gradient = zeros(length(opt.design_regions[1].design_parameters), length(opt.frequencies))
        opt.current_state = :ADJ
        return
    end

    # Negate Bloch k-point for adjoint
    if !isnothing(sim.boundary_conditions)
        for (axis_idx, axis_bcs) in enumerate(sim.boundary_conditions)
            for (side_idx, bc) in enumerate(axis_bcs)
                if bc isa Bloch
                    sim.boundary_conditions[axis_idx][side_idx] = Bloch(k = -bc.k)
                end
            end
        end
    end

    # Save the forward design region DFT data before resetting
    # (reset_fields! will zero all DFT accumulators including forward ones)
    saved_fwd_dft = []
    for m in opt.forward_design_monitors
        push!(saved_fwd_dft, Array(m.monitor_data.fields))
    end

    # Reset fields (zeros all field/DFT arrays, resets timestep)
    reset_fields!(sim)

    # Restore saved forward DFT data into the forward monitors
    for (i, m) in enumerate(opt.forward_design_monitors)
        copyto!(m.monitor_data.fields, saved_fwd_dft[i])
    end

    # Clear existing monitor data so DFT accumulators start fresh
    # We keep the design region DFT monitors from the forward run structure
    # but zero their accumulators (already done by reset_fields!)

    # Replace sources: swap the forward sources with adjoint sources
    saved_sources = sim.sources
    sim.sources = adjoint_sources

    # Re-assemble source data for the new adjoint sources
    # This computes spatial amplitude profiles and uploads to GPU
    add_sources(sim, sim.sources)

    # Distribute source data to chunks
    for chunk in sim.chunk_data
        chunk.source_data = sim.source_data
    end

    # Reinstall design region DFT monitors (zeros accumulators, installs into sim.monitor_data)
    # First, clear existing monitor data to avoid duplication
    empty!(sim.monitor_data)
    opt.adjoint_design_monitors = DFTMonitor[]
    for dr in opt.design_regions
        mons = install_design_region_monitors!(sim, dr, opt.frequencies)
        append!(opt.adjoint_design_monitors, mons)
    end

    # Distribute monitor data to chunks
    for chunk in sim.chunk_data
        chunk.monitor_data = sim.monitor_data
    end

    # Run adjoint simulation
    run(sim; until_after_sources = stop_when_dft_decayed(
        tolerance = opt.decay_by,
        minimum_runtime = opt.minimum_run_time,
        maximum_runtime = opt.maximum_run_time,
    ))

    # Restore forward sources and negate k-point back
    sim.sources = saved_sources
    add_sources(sim, sim.sources)
    for chunk in sim.chunk_data
        chunk.source_data = sim.source_data
    end

    if !isnothing(sim.boundary_conditions)
        for (axis_idx, axis_bcs) in enumerate(sim.boundary_conditions)
            for (side_idx, bc) in enumerate(axis_bcs)
                if bc isa Bloch
                    sim.boundary_conditions[axis_idx][side_idx] = Bloch(k = -bc.k)
                end
            end
        end
    end

    opt.current_state = :ADJ

    return nothing
end

"""
    _compute_jacobian(func, args, arg_idx)

Compute the Jacobian of `func` with respect to argument `arg_idx`
using ForwardDiff.

For a scalar-valued function of complex inputs, this extracts the
derivative as a complex number.
"""
function _compute_jacobian(func::Function, args::Vector, arg_idx::Int)
    # The objective function takes all results_list entries as arguments
    # We need dJ/d(args[arg_idx])
    x = args[arg_idx]

    if x isa Vector{<:Number}
        # Use ForwardDiff for the Jacobian
        function f_real(x_re)
            new_args = copy(args)
            new_args[arg_idx] = complex.(x_re, imag.(args[arg_idx]))
            result = func(new_args...)
            return isa(result, Number) ? [real(result)] : real.(result)
        end

        function f_imag(x_im)
            new_args = copy(args)
            new_args[arg_idx] = complex.(real.(args[arg_idx]), x_im)
            result = func(new_args...)
            return isa(result, Number) ? [real(result)] : real.(result)
        end

        jac_re = ForwardDiff.jacobian(f_real, real.(x))
        jac_im = ForwardDiff.jacobian(f_imag, imag.(x))

        # Combine: dJ/dz = dJ/dx - i*dJ/dy (Wirtinger derivative)
        return vec(jac_re .- im .* jac_im)
    else
        # Scalar case
        return [1.0 + 0.0im]
    end
end
