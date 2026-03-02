# Resolution sweep to determine the adj_src_scale correction factor
import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random

function run_at_resolution(resolution)
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    ε_hi = 4.0; ε_lo = 1.0
    design_Lx = 1.0; pml = 1.0; pad = 0.5
    fcen = 0.5; fwidth = 0.15
    cell_y = 4.0
    cell_x = design_Lx + 2 * pad + 2 * pml

    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, 0.0]),
            Khronos.Material(ε = ε_lo)),
    ]
    sources = [Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-(design_Lx / 2 + pad), 0.0, 0.0],
        size = [0.0, 0.0, 0.0],
        amplitude = 1.0,
    )]

    Nx_d = 2; Ny_d = 2; n_params = 4
    dr = Khronos.DesignRegion(
        volume = Khronos.Volume(center = [0.0, 0.0, 0.0],
            size = [design_Lx, cell_y - 2 * pml, 0.0]),
        design_parameters = fill(0.5, n_params),
        grid_size = (Nx_d, Ny_d),
        ε_min = ε_lo, ε_max = ε_hi,
    )

    output_mon_x = design_Lx / 2 + pad / 2
    output_obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [output_mon_x, 0.0, 0.0],
            size = [0.0, cell_y - 2 * pml, 0.0]),  # LINE monitor
        component = Khronos.Ez(),
    )
    objective = ez_fields -> real(sum(abs2.(ez_fields)))

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, 0.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry, sources = sources,
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[],
    )

    Khronos.prepare_simulation!(sim)
    Khronos.init_design_region!(sim, dr)
    fixed_runtime = 80.0

    opt = Khronos.OptimizationProblem(
        sim = sim,
        objective_functions = [objective],
        objective_arguments = Khronos.ObjectiveQuantity[output_obj],
        design_regions = [dr], frequencies = [fcen],
        decay_by = 1e-4, maximum_run_time = fixed_runtime,
    )

    Random.seed!(123)
    rho0 = 0.3 .+ 0.4 .* rand(n_params)

    # Adjoint gradient
    Khronos.update_design!(sim, dr, rho0)
    opt.current_state = :INIT
    Khronos.forward_run!(opt)
    Khronos.adjoint_run!(opt)
    Khronos.calculate_gradient!(opt)
    adj_grad = vec(opt.gradient)

    # FD gradient (just param 4 for speed — it's the most sensitive)
    function eval_objective(rho_vec)
        Khronos.update_design!(sim, dr, rho_vec)
        Khronos.reset_fields!(sim)
        empty!(sim.monitor_data)
        mon = Khronos.DFTMonitor(component = Khronos.Ez(),
            center = [output_mon_x, 0.0, 0.0], size = [0.0, cell_y - 2*pml, 0.0],
            frequencies = [fcen])
        push!(sim.monitor_data, Khronos.init_monitors(sim, mon))
        for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end
        Khronos.run(sim; until = fixed_runtime)
        return real(sum(abs2.(Array(mon.monitor_data.fields))))
    end

    h = 1e-3
    fd_grad = zeros(n_params)
    for k in 1:n_params
        rho_p = copy(rho0); rho_p[k] = min(rho0[k] + h, 1.0)
        rho_m = copy(rho0); rho_m[k] = max(rho0[k] - h, 0.0)
        fd_grad[k] = (eval_objective(rho_p) - eval_objective(rho_m)) / (rho_p[k] - rho_m[k])
    end

    # Optimal scale factor
    alpha_opt = dot(adj_grad, fd_grad) / dot(adj_grad, adj_grad)

    dt = Float64(sim.Δt)
    dV = 1.0 / Float64(resolution)^sim.ndims
    n_steps = floor(Int, fixed_runtime / dt)

    # Also compute key intermediate values
    scale = Khronos.adj_src_scale(sim, [fcen], Khronos.get_time_profile(sim.sources[1]);
        include_resolution=true)
    scale_noDV = Khronos.adj_src_scale(sim, [fcen], Khronos.get_time_profile(sim.sources[1]);
        include_resolution=false)

    return (resolution=resolution, dt=dt, dV=dV, n_steps=n_steps,
            alpha=alpha_opt, scale=scale[1], scale_noDV=scale_noDV[1],
            adj_norm=norm(adj_grad), fd_norm=norm(fd_grad))
end

println("Resolution sweep for adj_src_scale calibration")
println("="^80)
@printf("%-6s  %-10s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
    "Res", "dt", "dV", "Nsteps", "α_opt", "|scale|", "|scaleNoV|", "α*|s|")
println("-"^80)

for res in [10, 15, 20, 30]
    r = run_at_resolution(res)
    @printf("%-6d  %-10.4e  %-10.4e  %-8d  %-10.4f  %-10.4e  %-10.4e  %-10.4e\n",
        r.resolution, r.dt, r.dV, r.n_steps, r.alpha,
        abs(r.scale), abs(r.scale_noDV), r.alpha * abs(r.scale))
end
println("="^80)
