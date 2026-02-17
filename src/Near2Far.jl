# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Near-to-far field transformation via surface equivalence principle.
# Implements full 3D dyadic Green's functions (near + far field),
# GPU-accelerated surface integration, and power/LEE computation.
#
# Reference: meep/src/near2far.cpp (adapted from Homer Reid's SCUFF-EM)

export compute_far_field, compute_far_field_power, compute_LEE


# ------------------------------------------------------------------- #
# Full 3D dyadic Green's function
# ------------------------------------------------------------------- #

"""
    green3d!(EH, x, freq, eps, mu, x0, c0, f0)

Full 3D dyadic Green's function for a point current source.

Computes the 6-component (Ex,Ey,Ez,Hx,Hy,Hz) field at observation point `x`
due to a unit current source of component `c0` at position `x0` with
complex amplitude `f0`, in a homogeneous medium with permittivity `eps`
and permeability `mu`.

Includes near-field (1/r³), intermediate (1/r²), and far-field (1/r) terms.

Arguments:
- `EH`: MVector{6,ComplexF64} accumulator (modified in-place)
- `x`: SVector{3} observation point
- `freq`: frequency (Hz)
- `eps`, `mu`: medium permittivity and permeability
- `x0`: SVector{3} source point
- `c0`: source component (1=Jx, 2=Jy, 3=Jz for electric; 4=Mx, 5=My, 6=Mz for magnetic)
- `f0`: ComplexF64 source amplitude

Reference: meep/src/near2far.cpp:133-187
"""
@inline function green3d!(EH::MVector{6,ComplexF64}, x::SVector{3,Float64},
                          freq::Float64, eps::Float64, mu::Float64,
                          x0::SVector{3,Float64}, c0::Int, f0::ComplexF64)
    r_vec = x - x0
    r = norm(r_vec)

    # Guard against self-interaction (r ≈ 0); wrap entire body to avoid
    # `return` which is forbidden inside KernelAbstractions @kernel functions
    if r >= 1e-20
        r_hat = r_vec / r

        n = sqrt(eps * mu)
        k = 2π * freq * n
        Z = sqrt(mu / eps)

        ikr = im * k * r
        ikr2 = -(k * r)^2

        # expfac = f0 * (k*n / (4π*r)) * exp(i*(k*r + π/2))
        expfac = f0 * (k * n / (4π * r)) * exp(im * (k * r + π / 2))

        # Unit source direction vector
        pc = mod1(c0, 3)  # component index 1-3
        p = SVector(pc == 1 ? 1.0 : 0.0, pc == 2 ? 1.0 : 0.0, pc == 3 ? 1.0 : 0.0)
        pdotrhat = dot(p, r_hat)
        rhatcrossp = cross(r_hat, p)

        # Three scalar terms (meep near2far.cpp:158-160)
        term1 = 1.0 - 1.0 / ikr + 1.0 / ikr2
        term2 = (-1.0 + 3.0 / ikr - 3.0 / ikr2) * pdotrhat
        term3 = 1.0 - 1.0 / ikr

        if c0 <= 3  # Electric current source
            ef = expfac / eps
            EH[1] += ef * (term1 * p[1] + term2 * r_hat[1])
            EH[2] += ef * (term1 * p[2] + term2 * r_hat[2])
            EH[3] += ef * (term1 * p[3] + term2 * r_hat[3])
            EH[4] += ef * term3 * rhatcrossp[1] / Z
            EH[5] += ef * term3 * rhatcrossp[2] / Z
            EH[6] += ef * term3 * rhatcrossp[3] / Z
        else  # Magnetic current source (c0 = 4, 5, or 6)
            ef = expfac / mu
            EH[1] += -ef * term3 * rhatcrossp[1] * Z
            EH[2] += -ef * term3 * rhatcrossp[2] * Z
            EH[3] += -ef * term3 * rhatcrossp[3] * Z
            EH[4] += ef * (term1 * p[1] + term2 * r_hat[1])
            EH[5] += ef * (term1 * p[2] + term2 * r_hat[2])
            EH[6] += ef * (term1 * p[3] + term2 * r_hat[3])
        end
    end
end

# ------------------------------------------------------------------- #
# GPU-accelerated near-to-far field evaluation kernel
# ------------------------------------------------------------------- #

"""
    near2far_kernel_z!(EH_out, dft fields, base positions, obs coords, params)

KernelAbstractions kernel for near-to-far field computation on a z-normal surface.
Each component's source position is computed from its stored base position
(the physical position of dft[1,1,1]) plus grid index offsets.
"""
@kernel function near2far_kernel_z!(
    EH_out,  # (N_obs, 6, N_freq) ComplexF64
    @Const(dft_Ex), @Const(dft_Ey),
    @Const(dft_Hx), @Const(dft_Hy),
    # Per-component base positions: physical (x,y,z) of dft[1,1,1]
    ex_x0, ex_y0, ex_z0,
    ey_x0, ey_y0, ey_z0,
    hx_x0, hx_y0, hx_z0,
    hy_x0, hy_y0, hy_z0,
    dx_val, dy_val,  # grid spacing
    @Const(obs_x), @Const(obs_y), @Const(obs_z),
    freq_val, eps_val, mu_val, normal_sign_val, dA_val,
    n_surf_x::Int32, n_surf_y::Int32,
    i_freq::Int32,
)
    i_obs = @index(Global)

    x_obs = SVector(Float64(obs_x[i_obs]), Float64(obs_y[i_obs]), Float64(obs_z[i_obs]))
    EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)

    freq = Float64(freq_val)
    eps = Float64(eps_val)
    mu = Float64(mu_val)
    ns = Float64(normal_sign_val)
    dA = Float64(dA_val)
    dx = Float64(dx_val)
    dy = Float64(dy_val)

    for iy_src in 1:Int(n_surf_y)
        iy = iy_src - 1
        for ix_src in 1:Int(n_surf_x)
            ix = ix_src - 1

            ex = ComplexF64(dft_Ex[ix_src, iy_src, i_freq])
            ey = ComplexF64(dft_Ey[ix_src, iy_src, i_freq])
            hx = ComplexF64(dft_Hx[ix_src, iy_src, i_freq])
            hy = ComplexF64(dft_Hy[ix_src, iy_src, i_freq])

            # Exact physical positions from per-component GridVolume origins
            x0_Ex = SVector(Float64(ex_x0) + ix*dx, Float64(ex_y0) + iy*dy, Float64(ex_z0))
            x0_Ey = SVector(Float64(ey_x0) + ix*dx, Float64(ey_y0) + iy*dy, Float64(ey_z0))
            x0_Hx = SVector(Float64(hx_x0) + ix*dx, Float64(hx_y0) + iy*dy, Float64(hx_z0))
            x0_Hy = SVector(Float64(hy_x0) + ix*dx, Float64(hy_y0) + iy*dy, Float64(hy_z0))

            # z-normal: J = n̂×H → Jx=ns*Hy, Jy=-ns*Hx; M=-n̂×E → Mx=-ns*Ey, My=ns*Ex
            green3d!(EH, x_obs, freq, eps, mu, x0_Hy, 1, ns * hy * dA)   # Jx from Hy
            green3d!(EH, x_obs, freq, eps, mu, x0_Hx, 2, -ns * hx * dA)  # Jy from Hx
            green3d!(EH, x_obs, freq, eps, mu, x0_Ey, 4, -ns * ey * dA)  # Mx from Ey
            green3d!(EH, x_obs, freq, eps, mu, x0_Ex, 5, ns * ex * dA)   # My from Ex
        end
    end

    for j in 1:6
        EH_out[i_obs, j, Int(i_freq)] = EH[j]
    end
end

@kernel function near2far_kernel_x!(
    EH_out,
    @Const(dft_Ey), @Const(dft_Ez),
    @Const(dft_Hy), @Const(dft_Hz),
    ey_x0, ey_y0, ey_z0,
    ez_x0, ez_y0, ez_z0,
    hy_x0, hy_y0, hy_z0,
    hz_x0, hz_y0, hz_z0,
    dy_val, dz_val,
    @Const(obs_x), @Const(obs_y), @Const(obs_z),
    freq_val, eps_val, mu_val, normal_sign_val, dA_val,
    n_surf_1::Int32, n_surf_2::Int32,
    i_freq::Int32,
)
    i_obs = @index(Global)
    x_obs = SVector(Float64(obs_x[i_obs]), Float64(obs_y[i_obs]), Float64(obs_z[i_obs]))
    EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    freq = Float64(freq_val); eps = Float64(eps_val); mu = Float64(mu_val)
    ns = Float64(normal_sign_val); dA = Float64(dA_val)
    dy = Float64(dy_val); dz = Float64(dz_val)

    for i2 in 1:Int(n_surf_2), i1 in 1:Int(n_surf_1)
        iy = i1 - 1; iz = i2 - 1
        ey = ComplexF64(dft_Ey[i1, i2, i_freq])
        ez = ComplexF64(dft_Ez[i1, i2, i_freq])
        hy = ComplexF64(dft_Hy[i1, i2, i_freq])
        hz = ComplexF64(dft_Hz[i1, i2, i_freq])

        x0_Ey = SVector(Float64(ey_x0), Float64(ey_y0) + iy*dy, Float64(ey_z0) + iz*dz)
        x0_Ez = SVector(Float64(ez_x0), Float64(ez_y0) + iy*dy, Float64(ez_z0) + iz*dz)
        x0_Hy = SVector(Float64(hy_x0), Float64(hy_y0) + iy*dy, Float64(hy_z0) + iz*dz)
        x0_Hz = SVector(Float64(hz_x0), Float64(hz_y0) + iy*dy, Float64(hz_z0) + iz*dz)

        # x-normal: J = n̂×H → Jy=ns*Hz, Jz=-ns*Hy; M=-n̂×E → My=-ns*Ez, Mz=ns*Ey
        green3d!(EH, x_obs, freq, eps, mu, x0_Hz, 2, ns * hz * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Hy, 3, -ns * hy * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Ez, 5, -ns * ez * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Ey, 6, ns * ey * dA)
    end
    for j in 1:6; EH_out[i_obs, j, Int(i_freq)] = EH[j]; end
end

@kernel function near2far_kernel_y!(
    EH_out,
    @Const(dft_Ex), @Const(dft_Ez),
    @Const(dft_Hx), @Const(dft_Hz),
    ex_x0, ex_y0, ex_z0,
    ez_x0, ez_y0, ez_z0,
    hx_x0, hx_y0, hx_z0,
    hz_x0, hz_y0, hz_z0,
    dx_val, dz_val,
    @Const(obs_x), @Const(obs_y), @Const(obs_z),
    freq_val, eps_val, mu_val, normal_sign_val, dA_val,
    n_surf_1::Int32, n_surf_2::Int32,
    i_freq::Int32,
)
    i_obs = @index(Global)
    x_obs = SVector(Float64(obs_x[i_obs]), Float64(obs_y[i_obs]), Float64(obs_z[i_obs]))
    EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)
    freq = Float64(freq_val); eps = Float64(eps_val); mu = Float64(mu_val)
    ns = Float64(normal_sign_val); dA = Float64(dA_val)
    dx = Float64(dx_val); dz = Float64(dz_val)

    for i2 in 1:Int(n_surf_2), i1 in 1:Int(n_surf_1)
        ix = i1 - 1; iz = i2 - 1
        ex = ComplexF64(dft_Ex[i1, i2, i_freq])
        ez = ComplexF64(dft_Ez[i1, i2, i_freq])
        hx = ComplexF64(dft_Hx[i1, i2, i_freq])
        hz = ComplexF64(dft_Hz[i1, i2, i_freq])

        x0_Ex = SVector(Float64(ex_x0) + ix*dx, Float64(ex_y0), Float64(ex_z0) + iz*dz)
        x0_Ez = SVector(Float64(ez_x0) + ix*dx, Float64(ez_y0), Float64(ez_z0) + iz*dz)
        x0_Hx = SVector(Float64(hx_x0) + ix*dx, Float64(hx_y0), Float64(hx_z0) + iz*dz)
        x0_Hz = SVector(Float64(hz_x0) + ix*dx, Float64(hz_y0), Float64(hz_z0) + iz*dz)

        # y-normal: J = n̂×H → Jz=ns*Hx, Jx=-ns*Hz; M=-n̂×E → Mz=-ns*Ex, Mx=ns*Ez
        green3d!(EH, x_obs, freq, eps, mu, x0_Hx, 3, ns * hx * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Hz, 1, -ns * hz * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Ex, 6, -ns * ex * dA)
        green3d!(EH, x_obs, freq, eps, mu, x0_Ez, 4, ns * ez * dA)
    end
    for j in 1:6; EH_out[i_obs, j, Int(i_freq)] = EH[j]; end
end

# ------------------------------------------------------------------- #
# CPU fallback for near2far (plain Julia loop)
# ------------------------------------------------------------------- #

"""
    _compute_far_field_cpu(monitor_data::Near2FarMonitorData, obs_points::Matrix{Float64})
    -> Array{ComplexF64, 3}  # (N_obs, 6, N_freq)

CPU implementation of the near-to-far field evaluation.
"""
function _compute_far_field_cpu(md::Near2FarMonitorData, obs_points::Matrix{Float64})
    n_obs = size(obs_points, 1)
    n_freq = length(md.frequencies)
    EH_out = zeros(ComplexF64, n_obs, 6, n_freq)

    normal_axis = md.normal_axis
    ns = md.normal_sign
    eps = md.medium_eps
    mu = md.medium_mu
    dx = md.dx
    dy = md.dy
    dz = md.dz

    # Get DFT monitor data — transfer from GPU to CPU
    e_mons = md.tangential_E_monitors
    h_mons = md.tangential_H_monitors
    e1_fields = Array(e_mons[1].monitor_data.fields)
    e2_fields = Array(e_mons[2].monitor_data.fields)
    h1_fields = Array(h_mons[1].monitor_data.fields)
    h2_fields = Array(h_mons[2].monitor_data.fields)

    # Average fields across normal dimension when size > 1 (interpolate to monitor plane)
    _avg_dim(f, d) = size(f, d) >= 2 ?
        (selectdim(f, d, 1:1) .+ selectdim(f, d, 2:2)) ./ 2 : f
    e1_fields = _avg_dim(e1_fields, normal_axis)
    e2_fields = _avg_dim(e2_fields, normal_axis)
    h1_fields = _avg_dim(h1_fields, normal_axis)
    h2_fields = _avg_dim(h2_fields, normal_axis)

    # Grid dimensions (min across all 4 components for safety)
    e1_gv = e_mons[1].monitor_data.gv
    e2_gv = e_mons[2].monitor_data.gv
    h1_gv = h_mons[1].monitor_data.gv
    h2_gv = h_mons[2].monitor_data.gv

    # Per-component base positions
    e1b = md.e1_base
    e2b = md.e2_base
    h1b = md.h1_base
    h2b = md.h2_base

    if normal_axis == 3
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        dA = dx * dy
    elseif normal_axis == 1
        n1 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = dy * dz
    else  # normal_axis == 2
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = dx * dz
    end

    # Grid spacings for the two tangential directions
    d1, d2 = if normal_axis == 3; (dx, dy)
    elseif normal_axis == 1; (dy, dz)
    else; (dx, dz)
    end

    for i_freq in 1:n_freq
        freq = md.frequencies[i_freq]
        for i_obs in 1:n_obs
            x_obs = SVector{3,Float64}(obs_points[i_obs, 1], obs_points[i_obs, 2], obs_points[i_obs, 3])
            EH = MVector{6,ComplexF64}(0, 0, 0, 0, 0, 0)

            for i2 in 1:n2, i1 in 1:n1
                i1m = i1 - 1; i2m = i2 - 1

                et1 = ComplexF64(e1_fields[i1, i2, 1, i_freq])
                et2 = ComplexF64(e2_fields[i1, i2, 1, i_freq])
                ht1 = ComplexF64(h1_fields[i1, i2, 1, i_freq])
                ht2 = ComplexF64(h2_fields[i1, i2, 1, i_freq])

                if normal_axis == 3
                    # e1=Ex, e2=Ey, h1=Hx, h2=Hy
                    x0_e1 = SVector{3,Float64}(e1b[1]+i1m*d1, e1b[2]+i2m*d2, e1b[3])
                    x0_e2 = SVector{3,Float64}(e2b[1]+i1m*d1, e2b[2]+i2m*d2, e2b[3])
                    x0_h1 = SVector{3,Float64}(h1b[1]+i1m*d1, h1b[2]+i2m*d2, h1b[3])
                    x0_h2 = SVector{3,Float64}(h2b[1]+i1m*d1, h2b[2]+i2m*d2, h2b[3])
                    green3d!(EH, x_obs, freq, eps, mu, x0_h2, 1, ns * ht2 * dA)   # Jx from Hy
                    green3d!(EH, x_obs, freq, eps, mu, x0_h1, 2, -ns * ht1 * dA)  # Jy from Hx
                    green3d!(EH, x_obs, freq, eps, mu, x0_e2, 4, -ns * et2 * dA)  # Mx from Ey
                    green3d!(EH, x_obs, freq, eps, mu, x0_e1, 5, ns * et1 * dA)   # My from Ex
                elseif normal_axis == 1
                    # e1=Ey, e2=Ez, h1=Hy, h2=Hz
                    x0_e1 = SVector{3,Float64}(e1b[1], e1b[2]+i1m*d1, e1b[3]+i2m*d2)
                    x0_e2 = SVector{3,Float64}(e2b[1], e2b[2]+i1m*d1, e2b[3]+i2m*d2)
                    x0_h1 = SVector{3,Float64}(h1b[1], h1b[2]+i1m*d1, h1b[3]+i2m*d2)
                    x0_h2 = SVector{3,Float64}(h2b[1], h2b[2]+i1m*d1, h2b[3]+i2m*d2)
                    green3d!(EH, x_obs, freq, eps, mu, x0_h2, 2, ns * ht2 * dA)    # Jy from Hz
                    green3d!(EH, x_obs, freq, eps, mu, x0_h1, 3, -ns * ht1 * dA)   # Jz from Hy
                    green3d!(EH, x_obs, freq, eps, mu, x0_e2, 5, -ns * et2 * dA)   # My from Ez
                    green3d!(EH, x_obs, freq, eps, mu, x0_e1, 6, ns * et1 * dA)    # Mz from Ey
                else  # normal_axis == 2
                    # e1=Ex, e2=Ez, h1=Hx, h2=Hz
                    x0_e1 = SVector{3,Float64}(e1b[1]+i1m*d1, e1b[2], e1b[3]+i2m*d2)
                    x0_e2 = SVector{3,Float64}(e2b[1]+i1m*d1, e2b[2], e2b[3]+i2m*d2)
                    x0_h1 = SVector{3,Float64}(h1b[1]+i1m*d1, h1b[2], h1b[3]+i2m*d2)
                    x0_h2 = SVector{3,Float64}(h2b[1]+i1m*d1, h2b[2], h2b[3]+i2m*d2)
                    green3d!(EH, x_obs, freq, eps, mu, x0_h1, 3, ns * ht1 * dA)    # Jz from Hx
                    green3d!(EH, x_obs, freq, eps, mu, x0_h2, 1, -ns * ht2 * dA)   # Jx from Hz
                    green3d!(EH, x_obs, freq, eps, mu, x0_e1, 6, -ns * et1 * dA)   # Mz from Ex
                    green3d!(EH, x_obs, freq, eps, mu, x0_e2, 4, ns * et2 * dA)    # Mx from Ez
                end
            end

            EH_out[i_obs, :, i_freq] .= EH
        end
    end

    return EH_out
end

# ------------------------------------------------------------------- #
# GPU-accelerated near-to-far field evaluation
# ------------------------------------------------------------------- #

"""
    _compute_far_field_gpu(md::Near2FarMonitorData, obs_points::Matrix{Float64})
    -> Array{ComplexF64, 3}  # (N_obs, 6, N_freq)

GPU implementation of the near-to-far field evaluation. Uses per-component base
positions from GridVolume for exact Yee-grid-aware source coordinates.
"""
function _compute_far_field_gpu(md::Near2FarMonitorData, obs_points::Matrix{Float64})
    n_obs = size(obs_points, 1)
    n_freq = length(md.frequencies)

    normal_axis = md.normal_axis
    ns = md.normal_sign
    eps = md.medium_eps
    mu = md.medium_mu
    dx_val = md.dx
    dy_val = md.dy
    dz_val = md.dz

    # Per-component base positions (physical position of dft[1,1,1])
    e1b = md.e1_base
    e2b = md.e2_base
    h1b = md.h1_base
    h2b = md.h2_base

    # Get DFT monitor data — keep on GPU
    e_mons = md.tangential_E_monitors
    h_mons = md.tangential_H_monitors
    e1_raw = e_mons[1].monitor_data.fields
    e2_raw = e_mons[2].monitor_data.fields
    h1_raw = h_mons[1].monitor_data.fields
    h2_raw = h_mons[2].monitor_data.fields

    # Get grid dimensions
    e1_gv = e_mons[1].monitor_data.gv
    e2_gv = e_mons[2].monitor_data.gv
    h1_gv = h_mons[1].monitor_data.gv
    h2_gv = h_mons[2].monitor_data.gv

    if normal_axis == 3
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        dA = dx_val * dy_val
    elseif normal_axis == 1
        n1 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = dy_val * dz_val
    else  # normal_axis == 2
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = dx_val * dz_val
    end

    # Upload observation points to GPU
    obs_x_gpu = backend_array(Float64.(obs_points[:, 1]))
    obs_y_gpu = backend_array(Float64.(obs_points[:, 2]))
    obs_z_gpu = backend_array(Float64.(obs_points[:, 3]))

    # Allocate output on GPU: (N_obs, 6, N_freq)
    EH_gpu = complex_backend_array(zeros(ComplexF64, n_obs, 6, n_freq))

    # Slice DFT fields to common tangential dimensions and collapse normal axis.
    # Average when normal dimension > 1 (monitor between two Yee planes).
    if normal_axis == 3
        e1_gpu = size(e1_raw, 3) >= 2 ?
            (e1_raw[1:n1, 1:n2, 1, 1:n_freq] .+ e1_raw[1:n1, 1:n2, 2, 1:n_freq]) ./ 2 :
            e1_raw[1:n1, 1:n2, 1, 1:n_freq]
        e2_gpu = size(e2_raw, 3) >= 2 ?
            (e2_raw[1:n1, 1:n2, 1, 1:n_freq] .+ e2_raw[1:n1, 1:n2, 2, 1:n_freq]) ./ 2 :
            e2_raw[1:n1, 1:n2, 1, 1:n_freq]
        h1_gpu = size(h1_raw, 3) >= 2 ?
            (h1_raw[1:n1, 1:n2, 1, 1:n_freq] .+ h1_raw[1:n1, 1:n2, 2, 1:n_freq]) ./ 2 :
            h1_raw[1:n1, 1:n2, 1, 1:n_freq]
        h2_gpu = size(h2_raw, 3) >= 2 ?
            (h2_raw[1:n1, 1:n2, 1, 1:n_freq] .+ h2_raw[1:n1, 1:n2, 2, 1:n_freq]) ./ 2 :
            h2_raw[1:n1, 1:n2, 1, 1:n_freq]
    elseif normal_axis == 1
        e1_gpu = size(e1_raw, 1) >= 2 ?
            (e1_raw[1, 1:n1, 1:n2, 1:n_freq] .+ e1_raw[2, 1:n1, 1:n2, 1:n_freq]) ./ 2 :
            e1_raw[1, 1:n1, 1:n2, 1:n_freq]
        e2_gpu = size(e2_raw, 1) >= 2 ?
            (e2_raw[1, 1:n1, 1:n2, 1:n_freq] .+ e2_raw[2, 1:n1, 1:n2, 1:n_freq]) ./ 2 :
            e2_raw[1, 1:n1, 1:n2, 1:n_freq]
        h1_gpu = size(h1_raw, 1) >= 2 ?
            (h1_raw[1, 1:n1, 1:n2, 1:n_freq] .+ h1_raw[2, 1:n1, 1:n2, 1:n_freq]) ./ 2 :
            h1_raw[1, 1:n1, 1:n2, 1:n_freq]
        h2_gpu = size(h2_raw, 1) >= 2 ?
            (h2_raw[1, 1:n1, 1:n2, 1:n_freq] .+ h2_raw[2, 1:n1, 1:n2, 1:n_freq]) ./ 2 :
            h2_raw[1, 1:n1, 1:n2, 1:n_freq]
    else  # normal_axis == 2
        e1_gpu = size(e1_raw, 2) >= 2 ?
            (e1_raw[1:n1, 1, 1:n2, 1:n_freq] .+ e1_raw[1:n1, 2, 1:n2, 1:n_freq]) ./ 2 :
            e1_raw[1:n1, 1, 1:n2, 1:n_freq]
        e2_gpu = size(e2_raw, 2) >= 2 ?
            (e2_raw[1:n1, 1, 1:n2, 1:n_freq] .+ e2_raw[1:n1, 2, 1:n2, 1:n_freq]) ./ 2 :
            e2_raw[1:n1, 1, 1:n2, 1:n_freq]
        h1_gpu = size(h1_raw, 2) >= 2 ?
            (h1_raw[1:n1, 1, 1:n2, 1:n_freq] .+ h1_raw[1:n1, 2, 1:n2, 1:n_freq]) ./ 2 :
            h1_raw[1:n1, 1, 1:n2, 1:n_freq]
        h2_gpu = size(h2_raw, 2) >= 2 ?
            (h2_raw[1:n1, 1, 1:n2, 1:n_freq] .+ h2_raw[1:n1, 2, 1:n2, 1:n_freq]) ./ 2 :
            h2_raw[1:n1, 1, 1:n2, 1:n_freq]
    end

    # Grid spacings for the two tangential directions
    d1, d2 = if normal_axis == 3; (dx_val, dy_val)
    elseif normal_axis == 1; (dy_val, dz_val)
    else; (dx_val, dz_val)
    end

    # Launch kernel for each frequency
    wg = 256
    for i_freq in 1:n_freq
        freq = md.frequencies[i_freq]

        if normal_axis == 3
            kernel! = near2far_kernel_z!(backend_engine, (wg,))
            kernel!(
                EH_gpu,
                e1_gpu, e2_gpu, h1_gpu, h2_gpu,
                Float64(e1b[1]), Float64(e1b[2]), Float64(e1b[3]),
                Float64(e2b[1]), Float64(e2b[2]), Float64(e2b[3]),
                Float64(h1b[1]), Float64(h1b[2]), Float64(h1b[3]),
                Float64(h2b[1]), Float64(h2b[2]), Float64(h2b[3]),
                Float64(d1), Float64(d2),
                obs_x_gpu, obs_y_gpu, obs_z_gpu,
                Float64(freq), Float64(eps), Float64(mu),
                Float64(ns), Float64(dA),
                Int32(n1), Int32(n2), Int32(i_freq),
                ndrange=n_obs
            )
        elseif normal_axis == 1
            kernel! = near2far_kernel_x!(backend_engine, (wg,))
            kernel!(
                EH_gpu,
                e1_gpu, e2_gpu, h1_gpu, h2_gpu,
                Float64(e1b[1]), Float64(e1b[2]), Float64(e1b[3]),
                Float64(e2b[1]), Float64(e2b[2]), Float64(e2b[3]),
                Float64(h1b[1]), Float64(h1b[2]), Float64(h1b[3]),
                Float64(h2b[1]), Float64(h2b[2]), Float64(h2b[3]),
                Float64(d1), Float64(d2),
                obs_x_gpu, obs_y_gpu, obs_z_gpu,
                Float64(freq), Float64(eps), Float64(mu),
                Float64(ns), Float64(dA),
                Int32(n1), Int32(n2), Int32(i_freq),
                ndrange=n_obs
            )
        else  # normal_axis == 2
            kernel! = near2far_kernel_y!(backend_engine, (wg,))
            kernel!(
                EH_gpu,
                e1_gpu, e2_gpu, h1_gpu, h2_gpu,
                Float64(e1b[1]), Float64(e1b[2]), Float64(e1b[3]),
                Float64(e2b[1]), Float64(e2b[2]), Float64(e2b[3]),
                Float64(h1b[1]), Float64(h1b[2]), Float64(h1b[3]),
                Float64(h2b[1]), Float64(h2b[2]), Float64(h2b[3]),
                Float64(d1), Float64(d2),
                obs_x_gpu, obs_y_gpu, obs_z_gpu,
                Float64(freq), Float64(eps), Float64(mu),
                Float64(ns), Float64(dA),
                Int32(n1), Int32(n2), Int32(i_freq),
                ndrange=n_obs
            )
        end
    end

    # Synchronize and download results
    KernelAbstractions.synchronize(backend_engine)
    EH_out = Array(EH_gpu)

    @info "Near2Far GPU: $(n_obs) obs × $(n1*n2) src × $(n_freq) freq = $(n_obs*n1*n2*n_freq) Green's function evaluations"

    return EH_out
end

# ------------------------------------------------------------------- #
# High-level interface
# ------------------------------------------------------------------- #

"""
    compute_far_field(md::Near2FarMonitorData) -> Array{ComplexF64, 3}

Compute the far-field E,H at observation points from the near-field DFT data.
Returns array of shape (N_obs, 6, N_freq) where columns are (Ex,Ey,Ez,Hx,Hy,Hz).

Automatically uses GPU acceleration when a CUDA backend is active.
"""
function compute_far_field(md::Near2FarMonitorData)
    # Compute observation points from theta/phi/r or use provided points
    if !isnothing(md.theta) && !isnothing(md.phi)
        n_obs = length(md.theta) * length(md.phi)
        obs = zeros(Float64, n_obs, 3)
        idx = 1
        for phi in md.phi
            for theta in md.theta
                obs[idx, 1] = md.r * sin(theta) * cos(phi)
                obs[idx, 2] = md.r * sin(theta) * sin(phi)
                obs[idx, 3] = md.r * cos(theta)
                idx += 1
            end
        end
    elseif !isnothing(md.observation_points)
        obs = Float64.(md.observation_points)
    else
        error("Near2FarMonitorData must have either theta/phi or observation_points")
    end

    # Dispatch to GPU or CPU
    if backend_engine isa CUDABackend
        return _compute_far_field_gpu(md, obs)
    else
        return _compute_far_field_cpu(md, obs)
    end
end

# ------------------------------------------------------------------- #
# Power and LEE computation
# ------------------------------------------------------------------- #

"""
    compute_far_field_power(EH, theta, phi, eps, mu) -> Matrix{Float64}

Compute Poynting flux (radiated power per solid angle) from far-field E,H data.
Returns power(theta, phi) matrix.

For far field: S_r = (|E_θ|² + |E_φ|²) / (2Z)
"""
function compute_far_field_power(EH::Array{ComplexF64,3}, theta::Vector{Float64},
                                  phi::Vector{Float64}; eps::Float64=1.0, mu::Float64=1.0)
    n_theta = length(theta)
    n_phi = length(phi)
    Z = sqrt(mu / eps)

    power = zeros(n_theta, n_phi)
    idx = 1
    for (ip, φ) in enumerate(phi)
        for (it, θ) in enumerate(theta)
            # E field at this observation point
            Ex = EH[idx, 1, 1]
            Ey = EH[idx, 2, 1]
            Ez = EH[idx, 3, 1]

            # Decompose E into θ̂ and φ̂ components
            θ_hat = SVector(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
            φ_hat = SVector(-sin(φ), cos(φ), 0.0)

            E_θ = Ex * θ_hat[1] + Ey * θ_hat[2] + Ez * θ_hat[3]
            E_φ = Ex * φ_hat[1] + Ey * φ_hat[2] + Ez * φ_hat[3]

            # Poynting flux per solid angle
            power[it, ip] = 0.5 * (abs2(E_θ) + abs2(E_φ)) / Z
            idx += 1
        end
    end

    return power
end

"""
    compute_LEE(power, theta, phi; cone_half_angle=deg2rad(90))

Compute Light Extraction Efficiency: ratio of power within an angular cone
to total hemispherical power. Uses trapezoidal integration with sin(θ) Jacobian.

LEE = ∫₀²π ∫₀^α P(θ,φ) sin(θ) dθ dφ / ∫₀²π ∫₀^(π/2) P(θ,φ) sin(θ) dθ dφ
"""
function compute_LEE(power::Matrix{Float64}, theta::Vector{Float64},
                     phi::Vector{Float64}; cone_half_angle::Float64=deg2rad(90.0))
    n_theta = length(theta)
    n_phi = length(phi)

    # Trapezoidal integration weights
    dtheta = length(theta) > 1 ? diff(theta) : [0.0]
    dphi = length(phi) > 1 ? diff(phi) : [0.0]

    P_total = 0.0
    P_cone = 0.0

    for ip in 1:n_phi
        dphi_w = if ip == 1
            length(dphi) > 0 ? dphi[1] / 2 : 2π
        elseif ip == n_phi
            dphi[end] / 2
        else
            (dphi[ip-1] + dphi[ip]) / 2
        end

        for it in 1:n_theta
            dtheta_w = if it == 1
                length(dtheta) > 0 ? dtheta[1] / 2 : π / 2
            elseif it == n_theta
                dtheta[end] / 2
            else
                (dtheta[it-1] + dtheta[it]) / 2
            end

            integrand = power[it, ip] * sin(theta[it]) * dtheta_w * dphi_w

            if theta[it] ≤ π / 2  # upper hemisphere
                P_total += integrand
            end

            if theta[it] ≤ cone_half_angle
                P_cone += integrand
            end
        end
    end

    return P_total > 0 ? P_cone / P_total : 0.0
end

# ------------------------------------------------------------------- #
# Monitor initialization (called from Monitors.jl)
# ------------------------------------------------------------------- #

"""
    init_near2far_monitor(sim, monitor::Near2FarMonitor) -> Near2FarMonitorData

Initialize a Near2FarMonitor by creating 4 internal DFT monitors for the
tangential E and H components on the specified planar surface.
"""
function init_near2far_monitor(sim, monitor::Near2FarMonitor)
    # Determine normal axis (the axis where size == 0)
    normal_axis = findfirst(x -> x == 0.0, monitor.size)
    isnothing(normal_axis) && error("Near2FarMonitor size must have exactly one zero dimension")

    normal_sign = monitor.normal_dir == :+ ? 1.0 : -1.0

    # Determine tangential E and H components based on normal axis
    tangential_E_comps, tangential_H_comps = if normal_axis == 1  # x-normal
        ([Ey(), Ez()], [Hy(), Hz()])
    elseif normal_axis == 2  # y-normal
        ([Ex(), Ez()], [Hx(), Hz()])
    else  # z-normal
        ([Ex(), Ey()], [Hx(), Hy()])
    end

    # Create 4 DFT monitors for tangential fields
    tangential_E_monitors = DFTMonitor[]
    tangential_H_monitors = DFTMonitor[]

    for comp in tangential_E_comps
        dft_mon = DFTMonitor(
            component = comp,
            center = copy(monitor.center),
            size = copy(monitor.size),
            frequencies = copy(monitor.frequencies),
            decimation = monitor.decimation,
        )
        push!(tangential_E_monitors, dft_mon)
    end

    for comp in tangential_H_comps
        dft_mon = DFTMonitor(
            component = comp,
            center = copy(monitor.center),
            size = copy(monitor.size),
            frequencies = copy(monitor.frequencies),
            decimation = monitor.decimation,
        )
        push!(tangential_H_monitors, dft_mon)
    end

    md = Near2FarMonitorData(
        normal_axis = normal_axis,
        normal_sign = normal_sign,
        tangential_E_monitors = tangential_E_monitors,
        tangential_H_monitors = tangential_H_monitors,
        frequencies = Float64.(monitor.frequencies),
        medium_eps = monitor.medium_eps,
        medium_mu = monitor.medium_mu,
        observation_points = isnothing(monitor.observation_points) ?
            nothing : Float64.(monitor.observation_points),
        theta = isnothing(monitor.theta) ? nothing : Float64.(monitor.theta),
        phi = isnothing(monitor.phi) ? nothing : Float64.(monitor.phi),
        r = monitor.r,
        dx = Float64(sim.Δx),
        dy = Float64(sim.Δy),
        dz = Float64(sim.Δz),
    )

    monitor.monitor_data = md
    return md
end
