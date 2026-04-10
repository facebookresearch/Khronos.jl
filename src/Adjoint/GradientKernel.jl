# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    GradientKernel.jl

Computes the adjoint gradient via the overlap integral between forward and
adjoint DFT D-fields, plus the restriction step that contracts the per-voxel
sensitivity onto the design grid.

Matches Meep's convention: design region monitors record D-fields (Dx, Dy, Dz),
and the gradient uses ∂(ε_inv)/∂ρ:

    ∂f/∂ρ = Re[ D̂_adj · (∂ε_inv/∂ρ) · D̂_fwd ]

Reference: Hammond et al., Optics Express (2022), Appendix A;
           meep/src/meepgeom.cpp: material_grids_addgradient()
"""

"""
    calculate_gradient!(opt::OptimizationProblem)

Compute the gradient ∂f/∂ρ by:
1. Computing the overlap integral D_adj · D_fwd at each voxel and frequency
   in the design region
2. Multiplying by ∂(ε_inv)/∂ρ at each voxel
3. Contracting the per-voxel sensitivity onto the design grid using
   the transpose of the interpolation weights (restriction step)

The design region monitors record D-fields (matching Meep's convention).
For linear interpolation ε(ρ) = ε_min + ρ(ε_max - ε_min):
    ∂(ε_inv)/∂ρ = -(ε_max - ε_min) / ε(ρ)²

Returns a matrix of shape (n_design_params, n_freqs).
"""
function calculate_gradient!(opt)
    dr = opt.design_regions[1]  # TODO: support multiple design regions
    sim = opt.sim
    freqs = opt.frequencies

    n_freqs = length(freqs)
    n_design = length(dr.design_parameters)

    # D-field gradient formula (matching Meep):
    #   df/dρ_k = Σ_voxels w_ki · Re[ D_adj_i · (∂ε_inv/∂ρ) · D_fwd_i ]
    # where ∂(ε_inv)/∂ρ = -(ε_max - ε_min) / ε² depends on local ε.
    dε_range = dr.ε_max - dr.ε_min

    # Pre-compute interpolated ρ at each Yee grid point for each component
    rho = dr.design_parameters
    n_Dx = dr.gv_Ex.Nx * dr.gv_Ex.Ny  # D and E share same Yee positions
    n_Dy = dr.gv_Ey.Nx * dr.gv_Ey.Ny
    n_Dz = dr.gv_Ez.Nx * dr.gv_Ez.Ny
    rho_yee_Dx = Khronos._interpolate_design_to_yee(rho, dr.interp_weights_Ex, n_Dx)
    rho_yee_Dy = Khronos._interpolate_design_to_yee(rho, dr.interp_weights_Ey, n_Dy)
    rho_yee_Dz = Khronos._interpolate_design_to_yee(rho, dr.interp_weights_Ez, n_Dz)

    # Get the DFT D-fields from the forward and adjoint design region monitors
    fwd_monitors = opt.forward_design_monitors
    adj_monitors = opt.adjoint_design_monitors

    # Extract DFT field arrays and transfer to CPU
    # Extract DFT D-field arrays and transfer to CPU.
    d_fwd = [Array(fwd_monitors[i].monitor_data.fields) for i in 1:3]
    d_adj = [Array(adj_monitors[i].monitor_data.fields) for i in 1:3]

    gradient = zeros(Float64, n_design, n_freqs)

    # Interp weights are shared between D and E (same Yee positions)
    for (ci, (fwd_f, adj_f, weights, rho_yee)) in enumerate(zip(
        d_fwd, d_adj,
        [dr.interp_weights_Ex, dr.interp_weights_Ey, dr.interp_weights_Ez],
        [rho_yee_Dx, rho_yee_Dy, rho_yee_Dz],
    ))
        nx = size(fwd_f, 1)
        ny = size(fwd_f, 2)
        nz = size(fwd_f, 3)

        for (yee_idx, design_idx, w) in weights
            iy = div(yee_idx - 1, nx) + 1
            ix = mod(yee_idx - 1, nx) + 1

            ix > nx && continue
            iy > ny && continue

            # ∂(ε_inv)/∂ρ at this Yee voxel
            ρ_local = clamp(rho_yee[yee_idx], 0.0, 1.0)
            ε_local = dr.ε_min + ρ_local * dε_range
            d_eps_inv = -dε_range / (ε_local^2)

            for fi in 1:n_freqs
                overlap = zero(ComplexF64)
                for iz in 1:nz
                    if iz <= size(adj_f, 3) && fi <= size(adj_f, 4) &&
                       iz <= size(fwd_f, 3) && fi <= size(fwd_f, 4)
                        overlap += adj_f[ix, iy, iz, fi] * fwd_f[ix, iy, iz, fi]
                    end
                end
                # For 2D designs (z-size=0), average over z-planes
                if dr.volume.size[3] == 0.0
                    overlap /= max(1, nz)
                end
                gradient[design_idx, fi] += real(d_eps_inv * w * overlap)
            end
        end
    end

    opt.gradient = gradient
    return opt.gradient
end

"""
    calculate_fd_gradient(opt, rho; db=1e-4, n_samples=10)

Estimate the gradient using central finite differences for validation.
Randomly samples `n_samples` design parameters.
"""
function calculate_fd_gradient(
    opt,
    rho::Vector{Float64};
    db::Float64 = 1e-4,
    n_samples::Int = 10,
)
    n_design = length(rho)
    n_samples = min(n_samples, n_design)

    # Randomly choose parameter indices (use shuffle via sortperm of random values)
    sample_idx = sort(sortperm(rand(n_design))[1:n_samples])

    fd_gradient = zeros(n_samples, length(opt.frequencies))

    for (si, k) in enumerate(sample_idx)
        # Perturb left
        rho_left = copy(rho)
        rho_left[k] = max(rho[k] - db, 0.0)

        f_left, _ = opt(rho_left; need_gradient=false)

        # Perturb right
        rho_right = copy(rho)
        rho_right[k] = min(rho[k] + db, 1.0)

        f_right, _ = opt(rho_right; need_gradient=false)

        # Central difference
        step = rho_right[k] - rho_left[k]
        if step > 0
            fd_gradient[si, :] .= real.((f_right .- f_left) ./ step)
        end
    end

    return fd_gradient, sample_idx
end
