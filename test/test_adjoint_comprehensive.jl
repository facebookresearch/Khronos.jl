#!/usr/bin/env julia
#
# Comprehensive adjoint gradient test suite.
# Tests across resolutions, design grid sizes, bandwidths, and FOM types.
# Compares adjoint gradient vs central finite differences.
#

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random

Khronos.choose_backend(Khronos.CPUDevice(), Float64)

function test_gradient(;
    resolution = 20,
    design_nx = 5,
    design_ny = 1,
    fwidth = 0.15,
    fcen = 0.5,
    pml = 1.0,
    pad = 0.5,
    design_Lx = 1.0,
    cell_y = 3.0,
    ε_min = 1.0,
    ε_max = 4.0,
    rho_val = 0.5,  # uniform design parameter value
    fd_step = 1e-3,
    max_runtime = 100.0,
    tolerance = 1e-6,
)
    cell_x = design_Lx + 2pad + 2pml
    out_x = design_Lx / 2 + pad / 2
    n_params = design_nx * design_ny

    geom = [Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, 0.0]),
        Khronos.Material(ε = ε_min))]
    fwd_src = Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-(design_Lx / 2 + pad), 0.0, 0.0],
        size = [0.0, cell_y - 2pml, 0.0],
        amplitude = 1.0)

    # Use uniform rho for reproducibility
    rho0 = fill(rho_val, n_params)

    dr = Khronos.DesignRegion(
        volume = Khronos.Volume(center = [0.0, 0.0, 0.0],
            size = [design_Lx, cell_y - 2pml, 0.0]),
        design_parameters = fill(rho_val, n_params),
        grid_size = (design_nx, design_ny),
        ε_min = ε_min, ε_max = ε_max)

    function mksim()
        s = Khronos.Simulation(
            cell_size = [cell_x, cell_y, 0.0],
            cell_center = [0.0, 0.0, 0.0],
            resolution = resolution,
            geometry = geom, sources = [fwd_src],
            boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
            monitors = Khronos.Monitor[])
        Khronos.prepare_simulation!(s)
        Khronos.init_design_region!(s, dr)
        Khronos.update_design!(s, dr, rho0)
        return s
    end

    function eval_obj(rv)
        s = mksim()
        Khronos.update_design!(s, dr, rv)
        Khronos.reset_fields!(s)
        empty!(s.monitor_data)
        m = Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [out_x, 0.0, 0.0],
            size = [0.0, cell_y - 2pml, 0.0],
            frequencies = [fcen])
        push!(s.monitor_data, Khronos.init_monitors(s, m))
        for c in s.chunk_data; c.monitor_data = s.monitor_data; end
        Khronos.run(s; until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance = tolerance, maximum_runtime = max_runtime))
        return real(sum(abs2.(Array(m.monitor_data.fields))))
    end

    # Adjoint gradient
    sim = mksim()
    obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [out_x, 0.0, 0.0],
            size = [0.0, cell_y - 2pml, 0.0]),
        component = Khronos.Ez())
    opt = Khronos.OptimizationProblem(
        sim = sim,
        objective_functions = [ez -> real(sum(abs2.(ez)))],
        objective_arguments = Khronos.ObjectiveQuantity[obj],
        design_regions = [dr],
        frequencies = [fcen],
        decay_by = tolerance,
        maximum_run_time = max_runtime)
    f0 = Khronos.forward_run!(opt)
    Khronos.adjoint_run!(opt)
    Khronos.calculate_gradient!(opt)
    adj_grad = vec(opt.gradient)

    # FD gradient (sample a few parameters)
    n_fd = min(n_params, 5)
    Random.seed!(42)
    fd_indices = sort(randperm(n_params)[1:n_fd])
    fd_grad = zeros(n_fd)
    for (i, k) in enumerate(fd_indices)
        rp = copy(rho0); rp[k] = min(rho0[k] + fd_step, 1.0)
        rm = copy(rho0); rm[k] = max(rho0[k] - fd_step, 0.0)
        fd_grad[i] = (eval_obj(rp) - eval_obj(rm)) / (rp[k] - rm[k])
    end

    adj_sampled = adj_grad[fd_indices]

    # Compute metrics
    corr = dot(adj_sampled, fd_grad) / (norm(adj_sampled) * norm(fd_grad) + 1e-30)
    rel_errors = [abs(a - f) / max(abs(a), abs(f), 1e-30) for (a, f) in zip(adj_sampled, fd_grad)]
    median_err = sort(rel_errors)[max(1, div(length(rel_errors) + 1, 2))]
    alpha = dot(fd_grad, adj_sampled) / (dot(adj_sampled, adj_sampled) + 1e-30)

    return (;
        corr, median_err, alpha,
        adj = adj_sampled, fd = fd_grad, indices = fd_indices,
        n_params, f0,
    )
end

# ═══════════════════════════════════════════════════════════════
println("="^80)
println("COMPREHENSIVE ADJOINT GRADIENT TEST SUITE")
println("="^80)

# Test 1: Resolution sweep
println("\n--- Resolution sweep (design_nx=3, design_ny=1, fwidth=0.15) ---")
@printf("%-6s  %-8s  %-8s  %-8s  %-8s\n", "Res", "Corr", "MedErr", "Alpha", "|∇f|")
for res in [10, 20, 30]
    r = test_gradient(resolution=res, design_nx=3, max_runtime=80.0)
    @printf("%-6d  %+.4f  %.1f%%    %+.4f  %.2e\n",
        res, r.corr, r.median_err*100, r.alpha, norm(r.adj))
end

# Test 2: Design grid size sweep
println("\n--- Design grid size sweep (res=20, fwidth=0.15) ---")
@printf("%-8s  %-8s  %-8s  %-8s\n", "Grid", "Corr", "MedErr", "Alpha")
for (nx, ny) in [(1,1), (3,1), (5,1), (10,1), (3,3)]
    r = test_gradient(resolution=20, design_nx=nx, design_ny=ny, max_runtime=80.0)
    @printf("(%d,%d)    %+.4f  %.1f%%    %+.4f\n", nx, ny, r.corr, r.median_err*100, r.alpha)
end

# Test 3: Bandwidth sweep
println("\n--- Bandwidth sweep (res=20, design_nx=3) ---")
@printf("%-8s  %-8s  %-8s  %-8s\n", "fwidth", "Corr", "MedErr", "Alpha")
for fw in [0.15, 0.10, 0.05, 0.02]
    r = test_gradient(resolution=20, design_nx=3, fwidth=fw, max_runtime=200.0, tolerance=1e-8)
    @printf("%.3f     %+.4f  %.1f%%    %+.4f\n", fw, r.corr, r.median_err*100, r.alpha)
end

# Test 4: Material contrast sweep
println("\n--- Material contrast sweep (res=20, design_nx=3) ---")
@printf("%-12s  %-8s  %-8s  %-8s\n", "eps_range", "Corr", "MedErr", "Alpha")
for (emin, emax) in [(1.0, 2.0), (1.0, 4.0), (1.0, 12.0)]
    r = test_gradient(resolution=20, design_nx=3, ε_min=emin, ε_max=emax, max_runtime=80.0)
    @printf("[%.0f,%.0f]      %+.4f  %.1f%%    %+.4f\n", emin, emax, r.corr, r.median_err*100, r.alpha)
end
