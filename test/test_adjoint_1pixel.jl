# Minimal 1-pixel adjoint diagnostic
#
# Uses a SINGLE spatial point for the objective monitor, so dJ is a scalar
# and every intermediate value can be checked analytically.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random

function diagnostic_1pixel()
    Khronos.choose_backend(Khronos.CPUDevice(), Float64)

    ε_hi = 4.0
    ε_lo = 1.0
    design_Lx = 1.0
    pml = 1.0
    pad = 0.5
    fcen = 0.5
    fwidth = 0.15
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
        size = [0.0, 0.0, 0.0],  # POINT source
        amplitude = 1.0,
    )]

    # Small design region: just 2×2 design parameters
    Nx_d = 2
    Ny_d = 2
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

    # SINGLE POINT objective monitor
    output_obj = Khronos.FourierFieldsObjective(
        volume = Khronos.Volume(center = [output_mon_x, 0.0, 0.0],
            size = [0.0, 0.0, 0.0]),  # POINT monitor
        component = Khronos.Ez(),
    )

    # Simplest objective: |E|² at one point
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

    fixed_runtime = 30.0

    opt = Khronos.OptimizationProblem(
        sim = sim,
        objective_functions = [objective],
        objective_arguments = Khronos.ObjectiveQuantity[output_obj],
        design_regions = [dr],
        frequencies = [fcen],
        decay_by = 1e-4,
        maximum_run_time = fixed_runtime,
    )

    Random.seed!(123)
    rho0 = 0.3 .+ 0.4 .* rand(n_params)

    println("="^60)
    println("1-PIXEL ADJOINT DIAGNOSTIC")
    println("="^60)

    # ── Forward run ──
    Khronos.update_design!(sim, dr, rho0)
    opt.current_state = :INIT
    f0 = Khronos.forward_run!(opt)

    # Extract the single DFT value at the monitor
    E_obj = opt.results_list[1]
    println("\nForward results:")
    @printf("  f(ρ₀) = %.8e\n", real(f0))
    println("  E_obj shape: $(size(E_obj))")
    println("  E_obj value: $(E_obj[1])")
    @printf("  |E_obj|² = %.8e (should equal f0)\n", abs2(E_obj[1]))

    # ── Jacobian ──
    dJ = Khronos._compute_jacobian(objective, opt.results_list, 1)
    println("\nJacobian:")
    println("  dJ length: $(length(dJ))")
    println("  dJ[1] = $(dJ[1])")
    println("  Expected for negated Wirtinger of |z|²:")
    println("    -(2Re(z) - 2i·Im(z)) = -2·conj(z) = $(-2*conj(E_obj[1]))")

    # ── Adjoint source scale ──
    time_profile = Khronos.create_adjoint_time_profile([fcen])
    scale_with_dV = Khronos.adj_src_scale(sim, [fcen], Khronos.get_time_profile(sim.sources[1]); include_resolution=true)
    scale_no_dV = Khronos.adj_src_scale(sim, [fcen], Khronos.get_time_profile(sim.sources[1]); include_resolution=false)
    println("\nAdj source scale:")
    @printf("  scale (with dV):  %+.6e %+.6ei\n", real(scale_with_dV[1]), imag(scale_with_dV[1]))
    @printf("  scale (no dV):    %+.6e %+.6ei\n", real(scale_no_dV[1]), imag(scale_no_dV[1]))
    @printf("  dV = 1/res^ndims = %.6e\n", 1.0 / 20.0^sim.ndims)

    # ── Source amplitude ──
    amp_with_dV = dJ[1] * scale_with_dV[1]
    amp_no_dV = dJ[1] * scale_no_dV[1]
    println("\nAdjoint source amplitude:")
    @printf("  dJ * scale(dV):   %+.6e %+.6ei\n", real(amp_with_dV), imag(amp_with_dV))
    @printf("  dJ * scale(noDV): %+.6e %+.6ei\n", real(amp_no_dV), imag(amp_no_dV))

    # ── Adjoint run ──
    # Save forward DFT for verification
    fwd_ez_before = copy(Array(opt.forward_design_monitors[3].monitor_data.fields))

    # Debug: print the objective monitor's GridVolume
    obj_gv = output_obj.monitor.monitor_data.gv
    println("\nObjective monitor GridVolume:")
    println("  start_idx: $(obj_gv.start_idx)")
    println("  end_idx: $(obj_gv.end_idx)")
    println("  Nx=$(obj_gv.Nx), Ny=$(obj_gv.Ny), Nz=$(obj_gv.Nz)")
    println("  component: $(obj_gv.component)")

    Khronos.adjoint_run!(opt)

    # Verify forward DFT is preserved
    fwd_ez_after = Array(opt.forward_design_monitors[3].monitor_data.fields)
    println("\nForward DFT preservation check:")
    @printf("  Before adjoint: max|Ez_fwd| = %.6e\n", maximum(abs.(fwd_ez_before)))
    @printf("  After adjoint:  max|Ez_fwd| = %.6e\n", maximum(abs.(fwd_ez_after)))
    @printf("  Match: %s\n", fwd_ez_before ≈ fwd_ez_after)

    # Check adjoint DFT fields
    println("\nAdjoint DFT fields at design region:")
    for (i, m) in enumerate(opt.adjoint_design_monitors)
        f = Array(m.monitor_data.fields)
        maxval = maximum(abs.(f))
        if maxval > 0
            println("  Component $i: shape=$(size(f)), max|val|=$(@sprintf("%.6e", maxval))")
        end
    end

    # ── Gradient ──
    Khronos.calculate_gradient!(opt)
    adj_grad = vec(opt.gradient)
    println("\nAdjoint gradient: $(adj_grad)")

    # ── Debug: manually compute gradient to check ──
    println("\nManual gradient check:")
    fwd_mons = opt.forward_design_monitors
    adj_mons = opt.adjoint_design_monitors
    e_fwd = [Array(fwd_mons[i].monitor_data.fields) for i in 1:3]
    e_adj = [Array(adj_mons[i].monitor_data.fields) for i in 1:3]
    dε_range = ε_hi - ε_lo
    comp_names = ["Ex", "Ey", "Ez"]
    for (ci, name) in enumerate(comp_names)
        maxfwd = maximum(abs.(e_fwd[ci]))
        maxadj = maximum(abs.(e_adj[ci]))
        println("  $name: fwd max=$(maxfwd), adj max=$(maxadj), size=$(size(e_fwd[ci]))")
    end
    # Check interpolation weights
    println("\n  Per-design-parameter contribution from Ez:")
    weights_Ez = dr.interp_weights_Ez
    rho = dr.design_parameters
    n_Ex = dr.gv_Ex.Nx * dr.gv_Ex.Ny
    n_Ey = dr.gv_Ey.Nx * dr.gv_Ey.Ny
    n_Ez = dr.gv_Ez.Nx * dr.gv_Ez.Ny
    rho_yee_Ez = Khronos._interpolate_design_to_yee(rho, weights_Ez, n_Ez)

    manual_grad = zeros(n_params)
    nx_ez = size(e_fwd[3], 1)
    ny_ez = size(e_fwd[3], 2)
    nz_ez = size(e_fwd[3], 3)
    total_overlap = 0.0
    n_nonzero = 0
    for (yee_idx, design_idx, w) in weights_Ez
        iy = div(yee_idx - 1, nx_ez) + 1
        ix = mod(yee_idx - 1, nx_ez) + 1
        ix > nx_ez && continue
        iy > ny_ez && continue

        ρ_local = clamp(rho_yee_Ez[yee_idx], 0.0, 1.0)
        ε_local = ε_lo + ρ_local * dε_range
        d_eps_inv = -dε_range / (ε_local^2)

        overlap = zero(ComplexF64)
        for iz in 1:nz_ez
            overlap += e_adj[3][ix, iy, iz, 1] * e_fwd[3][ix, iy, iz, 1]
        end
        if nz_ez > 0 && dr.volume.size[3] == 0.0
            overlap /= max(1, nz_ez)
        end

        manual_grad[design_idx] += real(d_eps_inv * w * overlap)
        if abs(overlap) > 1e-15
            n_nonzero += 1
            total_overlap += abs(overlap)
        end
    end
    println("  Manual grad: $(manual_grad)")
    println("  Kernel grad: $(adj_grad)")
    println("  Match: $(manual_grad ≈ adj_grad)")
    println("  Total overlaps > 0: $n_nonzero / $(length(weights_Ez))")
    println("  Avg overlap: $(total_overlap / max(1, n_nonzero))")

    # ── FD gradient (all 4 parameters) ──
    function eval_objective(rho_vec)
        Khronos.update_design!(sim, dr, rho_vec)
        Khronos.reset_fields!(sim)
        empty!(sim.monitor_data)
        mon = Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [output_mon_x, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
            frequencies = [fcen],
        )
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
        f_p = eval_objective(rho_p)
        f_m = eval_objective(rho_m)
        fd_grad[k] = (f_p - f_m) / (rho_p[k] - rho_m[k])
    end
    println("\nFD gradient:      $(fd_grad)")

    # ── Comparison ──
    println("\n" * "-"^60)
    println("Per-parameter comparison:")
    @printf("%-5s  %12s  %12s  %8s\n", "Param", "Adjoint", "FD", "Ratio")
    for k in 1:n_params
        ratio = abs(fd_grad[k]) > 1e-30 ? adj_grad[k] / fd_grad[k] : NaN
        @printf("%-5d  %+12.5e  %+12.5e  %8.4f\n", k, adj_grad[k], fd_grad[k], ratio)
    end
    println("-"^60)

    # ── Empirical correction factor analysis ──
    # The adj_grad = Σ_i w_ki * Re[E_adj[i] * E_fwd[i] * dε_inv/dρ]
    # The fd_grad is ground truth. The ratio adj/fd should be constant
    # if the adjoint field shape is correct but the magnitude is off.
    # A non-constant ratio means the adjoint field spatial profile is wrong.
    #
    # Compute what global scale factor would minimize the error:
    if any(abs.(adj_grad) .> 1e-30)
        # Least squares: find α such that α*adj_grad ≈ fd_grad
        alpha_opt = dot(adj_grad, fd_grad) / dot(adj_grad, adj_grad)
        corrected = alpha_opt .* adj_grad
        println("\nEmpirical analysis:")
        @printf("  Optimal global scale factor α = %.4f\n", alpha_opt)
        @printf("  Corrected gradient error:\n")
        for k in 1:n_params
            ratio_c = abs(fd_grad[k]) > 1e-30 ? corrected[k] / fd_grad[k] : NaN
            @printf("    Param %d: corrected=%.5e, FD=%.5e, ratio=%.4f\n",
                k, corrected[k], fd_grad[k], ratio_c)
        end
        corr_err = norm(corrected - fd_grad) / norm(fd_grad)
        @printf("  Relative error after optimal scaling: %.1f%%\n", corr_err * 100)
    end

    return adj_grad, fd_grad
end

diagnostic_1pixel()
