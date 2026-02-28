# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adjoint gradient validation: directional derivative test
#
# Tests the adjoint gradient against central finite differences along
# random direction vectors in design-parameter space.
#
# Uses a 1D-like geometry (uniform slab) where light propagates through
# a design region. This gives a well-conditioned objective with large
# signal-to-noise ratio for clean gradient validation.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf

function validate_adjoint_gradient(;
    resolution = 20,
    design_resolution = 5,
    n_directions = 5,
    step_sizes = [1e-1, 5e-2, 1e-2],
)
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    # ── 1D-like slab geometry ───────────────────────────────────────────
    # Light propagates in +x through a design region that fills the
    # entire transverse cross-section. The design region modulates the
    # dielectric, affecting transmission.
    ε_hi = 4.0     # high-ε material (ρ=1)
    ε_lo = 1.0     # low-ε material (ρ=0)
    design_Lx = 2.0
    pml = 1.0
    pad = 0.5
    fcen = 0.5
    fwidth = 0.15
    cell_y = 3.0   # wide enough so the slab fills it
    dz = 1.0 / resolution
    cell_x = design_Lx + 2 * pad + 2 * pml

    # Background dielectric required so ε_inv is allocated as per-voxel arrays
    # (without geometry, Khronos uses a scalar ε_inv=1 which can't be modified)
    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [cell_x + 1, cell_y + 1, dz + 1]),
            Khronos.Material(ε = ε_lo)),
    ]

    sources = [Khronos.UniformSource(
        time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
        component = Khronos.Ez(),
        center = [-(design_Lx / 2 + pad), 0.0, 0.0],
        size = [0.0, cell_y - 2 * pml, 0.0],
        amplitude = 1.0,
    )]

    Nx_d = round(Int, design_Lx * design_resolution) + 1
    Ny_d = round(Int, cell_y * design_resolution) + 1
    n_params = Nx_d * Ny_d

    dr = Khronos.DesignRegion(
        volume = Khronos.Volume(center = [0.0, 0.0, 0.0],
            size = [design_Lx, cell_y - 2 * pml, 0.0]),
        design_parameters = fill(0.5, n_params),
        grid_size = (Nx_d, Ny_d),
        ε_min = ε_lo,
        ε_max = ε_hi,
    )

    # Output monitor: measure Ez after the design region
    output_mon_x = design_Lx / 2 + pad / 2
    output_obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [output_mon_x, 0.0, 0.0],
            size = [0.0, cell_y - 2 * pml, 0.0]),
        component = Khronos.Ez(),
    )

    objective = ez_fields -> real(sum(abs2.(ez_fields)))

    sim = Khronos.Simulation(
        cell_size = [cell_x, cell_y, dz],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
        monitors = Khronos.Monitor[],
    )

    Khronos.prepare_simulation!(sim)
    Khronos.init_design_region!(sim, dr)

    # Fixed runtime for reproducible FD results (no DFT convergence variability)
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

    # ── Helper: evaluate f(ρ) ───────────────────────────────────────────
    # Use a fixed runtime (not DFT convergence) to ensure reproducible
    # results across FD perturbations.
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

    # ── Base point ──────────────────────────────────────────────────────
    rho0 = 0.3 .+ 0.4 .* rand(n_params)

    println("="^70)
    println("Adjoint Gradient Validation: Directional Finite Differences")
    println("="^70)
    @printf("  Design grid:     %d × %d = %d parameters\n", Nx_d, Ny_d, n_params)
    @printf("  ε range:         [%.1f, %.1f]\n", ε_lo, ε_hi)
    @printf("  Resolution:      %d pixels/μm\n", resolution)
    @printf("  Directions:      %d random\n", n_directions)
    @printf("  FD step sizes:   %s\n", join([@sprintf("%.0e", s) for s in step_sizes], ", "))
    println()

    # ── Adjoint gradient ────────────────────────────────────────────────
    println("Computing adjoint gradient (1 fwd + 1 adj simulation)...")
    Khronos.update_design!(sim, dr, rho0)
    opt.current_state = :INIT
    f0_adj = Khronos.forward_run!(opt)
    Khronos.adjoint_run!(opt)
    Khronos.calculate_gradient!(opt)
    adj_grad = vec(opt.gradient)

    @printf("  f(ρ₀) = %.8e\n", real(f0_adj))
    @printf("  |∇f|  = %.8e\n", norm(adj_grad))
    println()

    # ── Directional derivative comparison ───────────────────────────────
    println("-"^70)
    @printf("%-5s  %-12s  ", "Dir", "adjoint")
    for h in step_sizes
        @printf("FD(h=%.0e)    ", h)
    end
    println("  rel_err (best FD)")
    println("-"^70)

    all_rel_errors = Float64[]

    for di in 1:n_directions
        d = randn(n_params)
        d ./= norm(d)

        adj_dd = dot(adj_grad, d)

        fd_results = Float64[]
        for h in step_sizes
            rho_p = clamp.(rho0 .+ h .* d, 0.0, 1.0)
            rho_m = clamp.(rho0 .- h .* d, 0.0, 1.0)
            f_p = eval_objective(rho_p)
            f_m = eval_objective(rho_m)
            fd_dd = (f_p - f_m) / (2 * h)
            push!(fd_results, fd_dd)
        end

        # Pick the FD estimate with the smallest step as "best"
        # But also check which one is closest to adjoint (convergence check)
        best_fd = fd_results[end]
        ref = max(abs(adj_dd), abs(best_fd), 1e-30)
        rel_err = abs(adj_dd - best_fd) / ref

        push!(all_rel_errors, rel_err)

        @printf("%-5d  %+.5e  ", di, adj_dd)
        for fd in fd_results
            @printf("%+.5e  ", fd)
        end
        @printf("  %.1f%%\n", rel_err * 100)
    end
    println("-"^70)

    median_err = sort(all_rel_errors)[max(1, div(length(all_rel_errors) + 1, 2))]
    max_err = maximum(all_rel_errors)
    @printf("Median relative error: %.1f%%\n", median_err * 100)
    @printf("Max relative error:    %.1f%%\n", max_err * 100)

    # ── FD convergence study ────────────────────────────────────────────
    println("\n" * "="^70)
    println("FD Convergence Study (Direction 1)")
    println("="^70)
    d1 = randn(n_params); d1 ./= norm(d1)
    adj_dd1 = dot(adj_grad, d1)
    @printf("Adjoint dF/dα = %+.10e\n\n", adj_dd1)

    conv_steps = [2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
    prev_err = nothing
    @printf("  %-10s  %-16s  %-12s  %-8s\n", "h", "FD dF/dα", "|error|", "ratio")
    println("  " * "-"^52)

    for h in conv_steps
        rho_p = clamp.(rho0 .+ h .* d1, 0.0, 1.0)
        rho_m = clamp.(rho0 .- h .* d1, 0.0, 1.0)
        f_p = eval_objective(rho_p)
        f_m = eval_objective(rho_m)
        fd_dd = (f_p - f_m) / (2 * h)
        err = abs(fd_dd - adj_dd1)
        ratio_str = isnothing(prev_err) || prev_err < 1e-30 ? "  -" :
                    @sprintf("  %.2f", prev_err / err)
        @printf("  %-10.0e  %+.10e  %.4e  %s\n", h, fd_dd, err, ratio_str)
        prev_err = err
    end
    println("\n  Expected: ratio → 4 for 2nd-order central FD convergence")

    # ── Verdict ─────────────────────────────────────────────────────────
    println("\n" * "="^70)
    if median_err < 0.10
        println("PASS: Median relative error $(round(median_err*100,digits=1))% < 10%")
    elseif median_err < 0.30
        println("PASS (MARGINAL): Median relative error $(round(median_err*100,digits=1))% < 30%")
    else
        println("NEEDS INVESTIGATION: Median relative error $(round(median_err*100,digits=1))%")
    end
    println("="^70)

    return adj_grad, all_rel_errors
end

adj_grad, errors = validate_adjoint_gradient(
    resolution = 20,
    design_resolution = 3,
    n_directions = 5,
    step_sizes = [1e-1, 5e-2, 1e-2],
)
