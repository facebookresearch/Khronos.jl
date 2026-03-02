# Quick sign/scaling scan for adjoint gradient
# Tests 4 combinations of Jacobian sign × include_resolution
# with the SourceData per-voxel injection approach

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf

function test_one_config(; jac_sign, include_res, negate_electric)
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    ε_hi = 4.0
    ε_lo = 1.0
    design_Lx = 2.0
    pml = 1.0
    pad = 0.5
    fcen = 0.5
    fwidth = 0.15
    cell_y = 3.0
    cell_x = design_Lx + 2 * pad + 2 * pml

    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, 0.0]),
            Khronos.Material(ε = ε_lo)),
    ]

    sources = [Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-(design_Lx / 2 + pad), 0.0, 0.0],
        size = [0.0, cell_y - 2 * pml, 0.0],
        amplitude = 1.0,
    )]

    Nx_d = round(Int, design_Lx * 3) + 1
    Ny_d = round(Int, cell_y * 3) + 1
    n_params = Nx_d * Ny_d

    dr = Khronos.DesignRegion(
        volume = Khronos.Volume(center = [0.0, 0.0, 0.0],
            size = [design_Lx, cell_y - 2 * pml, 0.0]),
        design_parameters = fill(0.5, n_params),
        grid_size = (Nx_d, Ny_d),
        ε_min = ε_lo,
        ε_max = ε_hi,
    )

    output_mon_x = design_Lx / 2 + pad / 2
    output_obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [output_mon_x, 0.0, 0.0],
            size = [0.0, cell_y - 2 * pml, 0.0]),
        component = Khronos.Ez(),
    )

    objective = ez_fields -> real(sum(abs2.(ez_fields)))

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, 0.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 20,
        geometry = geometry,
        sources = sources,
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[],
    )

    Khronos.prepare_simulation!(sim)
    Khronos.init_design_region!(sim, dr)

    fixed_runtime = 50.0

    opt = Khronos.OptimizationProblem(
        sim = sim,
        objective_functions = [objective],
        objective_arguments = Khronos.ObjectiveQuantity[output_obj],
        design_regions = [dr],
        frequencies = [fcen],
        decay_by = 1e-4,
        maximum_run_time = fixed_runtime,
    )

    # Seed for reproducibility
    rho0 = 0.3 .+ 0.4 .* [0.5 + 0.3*sin(i*0.7) for i in 1:n_params]

    # ── Adjoint gradient ──
    Khronos.update_design!(sim, dr, rho0)
    opt.current_state = :INIT

    # Override _compute_jacobian sign
    original_jacobian = Khronos._compute_jacobian

    # Monkey-patch for this test by using the opt callable directly
    # The sign/config will be applied inside the functions we modified

    f0_adj = Khronos.forward_run!(opt)
    Khronos.adjoint_run!(opt)
    Khronos.calculate_gradient!(opt)
    adj_grad = vec(opt.gradient)

    # ── FD gradient (1 direction) ──
    d = randn(n_params)
    d ./= norm(d)
    adj_dd = dot(adj_grad, d)

    function eval_objective(rho_vec)
        Khronos.update_design!(sim, dr, rho_vec)
        Khronos.reset_fields!(sim)
        empty!(sim.monitor_data)

        mon = Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [output_mon_x, 0.0, 0.0],
            size = [0.0, cell_y - 2 * pml, 0.0],
            frequencies = [fcen],
        )
        push!(sim.monitor_data, Khronos.init_monitors(sim, mon))
        for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end

        Khronos.run(sim; until = fixed_runtime)

        return real(sum(abs2.(Array(mon.monitor_data.fields))))
    end

    h = 1e-2
    rho_p = clamp.(rho0 .+ h .* d, 0.0, 1.0)
    rho_m = clamp.(rho0 .- h .* d, 0.0, 1.0)
    f_p = eval_objective(rho_p)
    f_m = eval_objective(rho_m)
    fd_dd = (f_p - f_m) / (2 * h)

    ref = max(abs(adj_dd), abs(fd_dd), 1e-30)
    rel_err = abs(adj_dd - fd_dd) / ref

    return adj_dd, fd_dd, rel_err
end

# We'll test different configurations by modifying the source files
# For now, just run with the current config
println("Testing current configuration...")
adj, fd, err = test_one_config(jac_sign=-1, include_res=true, negate_electric=true)
@printf("  Adjoint DD: %+.6e\n", adj)
@printf("  FD DD:      %+.6e\n", fd)
@printf("  Rel error:  %.1f%%\n", err * 100)
@printf("  Ratio adj/fd: %.4f\n", adj/fd)
