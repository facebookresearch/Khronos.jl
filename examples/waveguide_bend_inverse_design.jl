# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end inverse design example using the ADJOINT METHOD
#
# Optimizes a 2D waveguide bend design region to maximize |Ez|^2
# at an output monitor. Uses the adjoint method:
#   1 forward simulation + 1 adjoint simulation = gradient w.r.t. all params
#
# This replaces the O(N) finite-difference approach with O(1) adjoint.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf

function run_adjoint_optimization(;
    resolution = 20,
    n_iters = 10,
    design_resolution = 10,
)
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    # ── Problem parameters ──────────────────────────────────────────────
    n_si = 3.4
    n_sio2 = 1.44
    ε_si = n_si^2
    ε_sio2 = n_sio2^2
    wg_width = 0.5
    design_size = 1.5
    pml = 0.5
    pad = 0.5
    fcen = 0.625
    fwidth = 0.1
    dz = 1.0 / resolution

    cell_x = design_size + 2 * pad + 2 * pml + 2 * wg_width
    cell_y = design_size + 2 * pad + 2 * pml + 2 * wg_width

    # ── Geometry ────────────────────────────────────────────────────────
    cladding = Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, dz + 1]),
        Khronos.Material(ε = ε_sio2),
    )
    wg_in_cx = -(design_size / 2 + pad + wg_width / 2)
    wg_in = Khronos.Object(
        Cuboid([wg_in_cx, 0.0, 0.0], [pad + wg_width + pml, wg_width, dz]),
        Khronos.Material(ε = ε_si),
    )
    wg_out_cy = design_size / 2 + pad + wg_width / 2
    wg_out = Khronos.Object(
        Cuboid([0.0, wg_out_cy, 0.0], [wg_width, pad + wg_width + pml, dz]),
        Khronos.Material(ε = ε_si),
    )
    geometry = [wg_in, wg_out, cladding]

    # ── Source ──────────────────────────────────────────────────────────
    src_x = -(design_size / 2 + pad)
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
            component = Khronos.Ez(),
            center = [src_x, 0.0, 0.0],
            size = [0.0, wg_width, 0.0],
            amplitude = 1.0,
        ),
    ]

    # ── Design Region ───────────────────────────────────────────────────
    Nx_d = round(Int, design_size * design_resolution) + 1
    Ny_d = round(Int, design_size * design_resolution) + 1
    n_params = Nx_d * Ny_d

    dr = Khronos.DesignRegion(
        volume = Khronos.Volume(center = [0.0, 0.0, 0.0], size = [design_size, design_size, 0.0]),
        design_parameters = fill(0.5, n_params),
        grid_size = (Nx_d, Ny_d),
        ε_min = ε_sio2,
        ε_max = ε_si,
    )

    # ── Objective: maximize |Ez|^2 at the output waveguide ─────────────
    output_mon_y = design_size / 2 + pad
    output_obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [0.0, output_mon_y, 0.0], size = [wg_width * 2, 0.0, 0.0]),
        component = Khronos.Ez(),
    )

    # Objective function: sum |Ez|^2 over the output monitor
    function objective(ez_fields)
        return sum(abs2.(ez_fields))
    end

    # ── Create simulation ───────────────────────────────────────────────
    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, dz],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[],
    )

    # Prepare once
    Khronos.prepare_simulation!(sim)
    Khronos.init_design_region!(sim, dr)

    # ── Build OptimizationProblem ───────────────────────────────────────
    opt = Khronos.OptimizationProblem(
        sim = sim,
        objective_functions = [objective],
        objective_arguments = Khronos.ObjectiveQuantity[output_obj],
        design_regions = [dr],
        frequencies = [fcen],
        decay_by = 1e-4,
        maximum_run_time = 200.0,
    )

    # ── Optimization loop ───────────────────────────────────────────────
    rho = fill(0.5, n_params)
    step_size = 1.0
    beta = 1.0

    println("Design region: $(Nx_d)×$(Ny_d) = $n_params parameters")
    println("\n" * "="^60)
    println("Starting adjoint-based waveguide bend optimization")
    println("="^60)

    for iter in 1:n_iters
        # Apply filter/projection
        rho_projected = Khronos.tanh_projection(rho, beta, 0.5)

        # Update design
        Khronos.update_design!(sim, dr, rho_projected)

        # Forward run (1 simulation)
        opt.current_state = :INIT
        f0 = Khronos.forward_run!(opt)

        @printf("Iter %d/%d: f0=%.6e", iter, n_iters, real(f0))

        # Adjoint run (1 simulation → gradient for ALL parameters)
        Khronos.adjoint_run!(opt)

        # Compute gradient via overlap integral
        Khronos.calculate_gradient!(opt)

        grad = vec(opt.gradient)
        grad_norm = norm(grad)

        # Steepest ascent update (we maximize f0)
        if grad_norm > 0
            rho .+= step_size .* grad ./ grad_norm
            rho .= clamp.(rho, 0.0, 1.0)
        end

        @printf(", |∇f|=%.4e, ρ∈[%.3f,%.3f]\n", grad_norm, minimum(rho), maximum(rho))
    end

    println("\n" * "="^60)
    println("Optimization complete!")
    println("="^60)

    return sim, dr, rho, opt
end

sim, dr, rho, opt = run_adjoint_optimization(
    resolution = 15,
    n_iters = 5,
    design_resolution = 5,
)
