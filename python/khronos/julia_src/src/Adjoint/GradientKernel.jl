# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    GradientKernel.jl

GPU kernel for computing the adjoint gradient via the overlap integral
between forward and adjoint DFT fields, plus the restriction step that
contracts the per-voxel sensitivity onto the design grid.

The gradient formula (from the adjoint method) is:
    ∂f/∂ρ = -Ê_adj^T · (∂ε/∂ρ) · Ê_fwd

where Ê_adj and Ê_fwd are the DFT electric fields from the adjoint and
forward simulations, and ∂ε/∂ρ encodes the material interpolation derivative.
"""

"""
    compute_voxel_sensitivity(
        e_adj_x, e_adj_y, e_adj_z,
        e_fwd_x, e_fwd_y, e_fwd_z,
        dε_dρ,
    )

Compute the per-voxel sensitivity (overlap integral) on the GPU.
For linear material interpolation ε(ρ) = ε_min + ρ*(ε_max - ε_min),
the derivative dε/dρ = (ε_max - ε_min), and we need:

    sensitivity[x,y,freq] = -(ε_max - ε_min) * sum_c( E_adj_c * E_fwd_c )

This operates on the DFT field arrays which have shape (Nx, Ny, Nz, Nfreq).
"""
@kernel function compute_voxel_sensitivity_kernel!(
    sensitivity::AbstractArray,    # output: (Nx, Ny, Nfreq)
    e_adj_x::AbstractArray,        # (Nx_ex, Ny_ex, Nfreq)
    e_adj_y::AbstractArray,        # (Nx_ey, Ny_ey, Nfreq)
    e_adj_z::AbstractArray,        # (Nx_ez, Ny_ez, Nfreq)
    e_fwd_x::AbstractArray,        # (Nx_ex, Ny_ex, Nfreq)
    e_fwd_y::AbstractArray,        # (Nx_ey, Ny_ey, Nfreq)
    e_fwd_z::AbstractArray,        # (Nx_ez, Ny_ez, Nfreq)
    dε_factor::Number,             # (ε_max - ε_min)
)
    ix, iy = @index(Global, NTuple)
    n_freq = size(sensitivity, 3)

    for k in 1:n_freq
        # Sum overlap from each E-field component
        # Note: different components may have slightly different grid sizes
        # due to Yee staggering. We use the minimum overlapping region.
        overlap = zero(eltype(sensitivity))

        # Ex contribution
        if ix <= size(e_adj_x, 1) && iy <= size(e_adj_x, 2)
            overlap += e_adj_x[ix, iy, k] * e_fwd_x[ix, iy, k]
        end
        # Ey contribution
        if ix <= size(e_adj_y, 1) && iy <= size(e_adj_y, 2)
            overlap += e_adj_y[ix, iy, k] * e_fwd_y[ix, iy, k]
        end
        # Ez contribution
        if ix <= size(e_adj_z, 1) && iy <= size(e_adj_z, 2)
            overlap += e_adj_z[ix, iy, k] * e_fwd_z[ix, iy, k]
        end

        sensitivity[ix, iy, k] = -dε_factor * overlap
    end
end

"""
    calculate_gradient!(
        opt::OptimizationProblem,
    )

Compute the gradient ∂f/∂ρ by:
1. Computing the overlap integral E_adj · E_fwd at each voxel and frequency
   in the design region
2. Contracting the per-voxel sensitivity onto the design grid using
   the transpose of the interpolation weights (restriction step)

The gradient formula (from the adjoint method, Hammond et al. Opt. Express 2022):
    ∂f/∂ρ = -Re[ Ê_adj^T · (∂ε/∂ρ) · Ê_fwd ]

Since the DFT monitors record E-fields (not D-fields), we use ∂ε/∂ρ directly.
For linear interpolation ε(ρ) = ε_min + ρ(ε_max - ε_min), ∂ε/∂ρ = ε_max - ε_min
is a constant (no ρ-dependence).

Note: The overlap uses E_adj (transpose, not Hermitian conjugate) per the
frequency-domain adjoint formulation. No conjugation on E_adj.

Returns a matrix of shape (n_design_params, n_freqs).
"""
function calculate_gradient!(opt)
    dr = opt.design_regions[1]  # TODO: support multiple design regions
    sim = opt.sim
    freqs = opt.frequencies

    n_freqs = length(freqs)
    n_design = length(dr.design_parameters)

    # For E-field DFT monitors, the gradient formula is:
    #   df/dρ_k = -Σ_voxels w_ki · Re[ E_adj_i · (∂ε/∂ρ) · E_fwd_i ]
    # where ∂ε/∂ρ = (ε_max - ε_min) is a constant for linear interpolation.
    #
    # Note: if D-field monitors were used instead, the formula would be:
    #   df/dρ_k = +Σ_voxels w_ki · Re[ D_adj_i · (∂ε_inv/∂ρ) · D_fwd_i ]
    # where ∂(ε_inv)/∂ρ = -(ε_max - ε_min)/ε² depends on local ε.
    dε_dρ = dr.ε_max - dr.ε_min  # constant ∂ε/∂ρ

    # Get the DFT fields from the forward and adjoint design region monitors
    fwd_monitors = opt.forward_design_monitors
    adj_monitors = opt.adjoint_design_monitors

    # Extract DFT field arrays and transfer to CPU
    e_fwd = [Array(fwd_monitors[i].monitor_data.fields) for i in 1:3]
    e_adj = [Array(adj_monitors[i].monitor_data.fields) for i in 1:3]

    gradient = zeros(Float64, n_design, n_freqs)

    for (ci, (fwd_f, adj_f, weights)) in enumerate(zip(
        e_fwd, e_adj,
        [dr.interp_weights_Ex, dr.interp_weights_Ey, dr.interp_weights_Ez],
    ))
        nx = size(fwd_f, 1)
        ny = size(fwd_f, 2)
        nz = size(fwd_f, 3)

        for (yee_idx, design_idx, w) in weights
            iy = div(yee_idx - 1, nx) + 1
            ix = mod(yee_idx - 1, nx) + 1

            ix > nx && continue
            iy > ny && continue

            for fi in 1:n_freqs
                overlap = zero(ComplexF64)
                for iz in 1:nz
                    if iz <= size(adj_f, 3) && fi <= size(adj_f, 4) &&
                       iz <= size(fwd_f, 3) && fi <= size(fwd_f, 4)
                        # Transpose (not Hermitian): no conj() on adjoint field
                        overlap += adj_f[ix, iy, iz, fi] * fwd_f[ix, iy, iz, fi]
                    end
                end
                # For 2D designs (z-size=0), the DFT is at 1-2 z-planes and
                # we average to get the per-z sensitivity.
                if dr.volume.size[3] == 0.0
                    overlap /= max(1, nz)
                end
                # ∂f/∂ρ = +Re[ E_adj · (∂ε/∂ρ) · E_fwd ]
                # The sign is + (not - as in the paper) because Khronos's
                # source convention (D += S) is opposite to Meep's (D -= dt*J),
                # which flips the sign of the adjoint E-fields.
                gradient[design_idx, fi] += real(dε_dρ * w * overlap)
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
