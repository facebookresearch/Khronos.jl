# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    ChainRulesIntegration.jl

Defines a ChainRulesCore.rrule for the simulation function, allowing
the adjoint method to compose with any ChainRules-compatible AD framework
(Zygote, Enzyme, Diffractor, etc.).

This is analogous to JAX's custom_vjp decorator used in meep's adjoint module.
"""

"""
    simulate(opt::OptimizationProblem, designs::Vector{Float64})

Run the forward simulation and return monitor values.
This function has a ChainRulesCore.rrule defined so that reverse-mode AD
automatically invokes the adjoint solver for gradient computation.
"""
function simulate(opt::OptimizationProblem, designs::Vector{Float64})
    # Update design parameters
    update_design_all!(opt, designs)

    # Run forward simulation
    f0 = forward_run!(opt)

    return f0
end

"""
    ChainRulesCore.rrule(::typeof(simulate), opt, designs)

Custom reverse-mode rule for the simulate function.
The pullback runs the adjoint simulation and computes gradients
via the overlap integral, avoiding the need to differentiate through
the entire FDTD kernel.
"""
function ChainRulesCore.rrule(
    ::typeof(simulate),
    opt::OptimizationProblem,
    designs::Vector{Float64},
)
    # Forward pass
    monitor_values = simulate(opt, designs)

    # Define the pullback (VJP)
    function simulate_pullback(Δ)
        # Δ is the cotangent (upstream gradient) w.r.t. monitor_values
        # Run adjoint simulation
        adjoint_run!(opt)

        # Compute gradient via overlap integral
        grads = calculate_gradient!(opt)

        # Return tangents: NoTangent for function and opt, actual gradient for designs
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), grads
    end

    return monitor_values, simulate_pullback
end
