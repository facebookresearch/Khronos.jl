# Copyright (c) Meta Platforms, Inc. and affiliates.

# ------------------------------------------------------------------- #
# Raw CUDA.jl fused kernels for interior (no-PML) chunks
#
# These bypass KernelAbstractions entirely for the hot path, avoiding
# overhead from CartesianIndex construction, multiple dispatch in inner
# loops, and KA's index management. Achieves ~2× higher throughput
# than the KA equivalent by using simple integer indexing.
#
# B and D fields are eliminated for the interior. Since H = μ⁻¹·B and
# E = ε⁻¹·D with constant material properties (no PML W-accumulation),
# we can reformulate as:
#   H_new = H_old + μ⁻¹·Δt·curl(E)   (instead of B_new = B_old + Δt·curl(E); H = μ⁻¹·B)
#   E_new = E_old + ε⁻¹·Δt·curl(H)   (instead of D_new = D_old + Δt·curl(H); E = ε⁻¹·D)
# This saves 6 array reads + 6 array writes per voxel per step (24 bytes
# total), and reduces the number of unique arrays from 15 to 9, improving
# L2 cache and TLB utilization.
# ------------------------------------------------------------------- #

function _cuda_fused_BH_kernel!(
    Ex, Ey, Ez,     # source E fields (read-only, stencil)
    Hx, Hy, Hz,     # H fields (read-write): H_new = H_old + μ⁻¹·Δt·curl(E)
    m_dt_dx, m_dt_dy, m_dt_dz,   # μ⁻¹ · Δt/Δ{x,y,z}
    iNx, iNy,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing

    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)

    @inbounds begin
        # Forward curl: uses [ix+1], [iy+1], [iz+1]
        Hx[ix,iy,iz] = Hx[ix,iy,iz] + m_dt_dz * (Ey[ix,iy,iz+Int32(1)] - Ey[ix,iy,iz]) -
                                          m_dt_dy * (Ez[ix,iy+Int32(1),iz] - Ez[ix,iy,iz])
        Hy[ix,iy,iz] = Hy[ix,iy,iz] + m_dt_dx * (Ez[ix+Int32(1),iy,iz] - Ez[ix,iy,iz]) -
                                          m_dt_dz * (Ex[ix,iy,iz+Int32(1)] - Ex[ix,iy,iz])
        Hz[ix,iy,iz] = Hz[ix,iy,iz] + m_dt_dy * (Ex[ix,iy+Int32(1),iz] - Ex[ix,iy,iz]) -
                                          m_dt_dx * (Ey[ix+Int32(1),iy,iz] - Ey[ix,iy,iz])
    end
    return nothing
end

function _cuda_fused_DE_kernel!(
    Hx, Hy, Hz,     # source H fields (read-only, stencil)
    Ex, Ey, Ez,     # E fields (read-write): E_new = E_old + ε⁻¹·Δt·curl(H)
    eps_inv_x, eps_inv_y, eps_inv_z,  # per-voxel ε⁻¹ (read-only)
    dt_dx, dt_dy, dt_dz,
    iNx, iNy,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing

    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)

    @inbounds begin
        # Backward curl: uses [ix-1], [iy-1], [iz-1]
        Kx = dt_dz * (Hy[ix,iy,iz-Int32(1)] - Hy[ix,iy,iz]) -
             dt_dy * (Hz[ix,iy-Int32(1),iz] - Hz[ix,iy,iz])
        Ky = dt_dx * (Hz[ix-Int32(1),iy,iz] - Hz[ix,iy,iz]) -
             dt_dz * (Hx[ix,iy,iz-Int32(1)] - Hx[ix,iy,iz])
        Kz = dt_dy * (Hx[ix,iy-Int32(1),iz] - Hx[ix,iy,iz]) -
             dt_dx * (Hy[ix-Int32(1),iy,iz] - Hy[ix,iy,iz])

        Ex[ix,iy,iz] = Ex[ix,iy,iz] + eps_inv_x[ix_0,iy_0,iz_0] * Kx
        Ey[ix,iy,iz] = Ey[ix,iy,iz] + eps_inv_y[ix_0,iy_0,iz_0] * Ky
        Ez[ix,iy,iz] = Ez[ix,iy,iz] + eps_inv_z[ix_0,iy_0,iz_0] * Kz
    end
    return nothing
end

function _cuda_fused_DE_scalar_kernel!(
    Hx, Hy, Hz,
    Ex, Ey, Ez,
    eps_inv,         # scalar ε⁻¹
    dt_dx, dt_dy, dt_dz,
    iNx, iNy,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing

    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)

    @inbounds begin
        Kx = dt_dz * (Hy[ix,iy,iz-Int32(1)] - Hy[ix,iy,iz]) -
             dt_dy * (Hz[ix,iy-Int32(1),iz] - Hz[ix,iy,iz])
        Ky = dt_dx * (Hz[ix-Int32(1),iy,iz] - Hz[ix,iy,iz]) -
             dt_dz * (Hx[ix,iy,iz-Int32(1)] - Hx[ix,iy,iz])
        Kz = dt_dy * (Hx[ix,iy-Int32(1),iz] - Hx[ix,iy,iz]) -
             dt_dx * (Hy[ix-Int32(1),iy,iz] - Hy[ix,iy,iz])

        Ex[ix,iy,iz] = Ex[ix,iy,iz] + eps_inv * Kx
        Ey[ix,iy,iz] = Ey[ix,iy,iz] + eps_inv * Ky
        Ez[ix,iy,iz] = Ez[ix,iy,iz] + eps_inv * Kz
    end
    return nothing
end

# ------------------------------------------------------------------- #
# Raw CUDA PML kernels (fused curl + material update)
#
# These handle PML boundary chunks by fusing the step_curl! and
# update_field! stages into a single kernel, eliminating the
# intermediate B/D re-read and bypassing KernelAbstractions overhead.
#
# Each kernel processes all 3 field components (x, y, z) to maximize
# data reuse of shared stencil source values.
#
# PML direction flags (pml_x, pml_y, pml_z) control which cascade
# stages are active per component. All threads evaluate the same
# branch (uniform condition), so there is no warp divergence.
#
# Inactive U/W auxiliary fields and σ arrays are replaced with dummy
# arrays at the launch site; the flags guarantee they are never accessed.
#
# Limitations: no material conductivity (σ_D/σ_B), no polarizability,
# no active sources. Chunks with these features fall back to KA.
# ------------------------------------------------------------------- #

function _cuda_pml_BH_kernel!(
    Ex, Ey, Ez,          # source E fields (read)
    Bx, By, Bz,          # target B fields (read/write)
    Hx, Hy, Hz,          # output H fields (read/write)
    UBx, UBy, UBz,       # U auxiliary (read/write, may be dummy)
    WBx, WBy, WBz,       # W auxiliary (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,  # PML σ 1D (read, may be dummy)
    m_inv,               # scalar μ⁻¹
    dt_dx, dt_dy, dt_dz,
    iNx, iNy,                 # x-dimension (Int32)
    pml_x::Int32, pml_y::Int32, pml_z::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(m_inv)

    @inbounds begin
        # Load shared source field values (reused across components)
        ey_cur = Ey[ix,iy,iz]; ez_cur = Ez[ix,iy,iz]; ex_cur = Ex[ix,iy,iz]

        # ==================== X component ====================
        # curl_x(E) = dEy/dz - dEz/dy (forward diff)
        Kx = dt_dz * (Ey[ix,iy,iz+Int32(1)] - ey_cur) - dt_dy * (Ez[ix,iy+Int32(1),iz] - ez_cur)

        # U stage: X uses σ_next = σ_pml_y (pml_y)
        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            u_old = UBx[ix,iy,iz]
            UBx[ix,iy,iz] = ((_one - σ_val) * u_old + Kx) / (_one + σ_val)
            in_t = UBx[ix,iy,iz] - u_old
        else
            in_t = Kx
        end

        # T stage: X uses σ_prev = σ_pml_z (pml_z)
        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            Bx[ix,iy,iz] = ((_one - σ_val) * Bx[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Bx[ix,iy,iz] = Bx[ix,iy,iz] + in_t
        end

        # H update: X uses σ_own = σ_pml_x (pml_x)
        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            w_old = WBx[ix,iy,iz]
            WBx[ix,iy,iz] = m_inv * Bx[ix,iy,iz]
            Hx[ix,iy,iz] = Hx[ix,iy,iz] + (_one + σ_val) * WBx[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Hx[ix,iy,iz] = m_inv * Bx[ix,iy,iz]
        end

        # ==================== Y component ====================
        # curl_y(E) = dEz/dx - dEx/dz (forward diff)
        Ky = dt_dx * (Ez[ix+Int32(1),iy,iz] - ez_cur) - dt_dz * (Ex[ix,iy,iz+Int32(1)] - ex_cur)

        # U stage: Y uses σ_next = σ_pml_z (pml_z)
        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            u_old = UBy[ix,iy,iz]
            UBy[ix,iy,iz] = ((_one - σ_val) * u_old + Ky) / (_one + σ_val)
            in_t = UBy[ix,iy,iz] - u_old
        else
            in_t = Ky
        end

        # T stage: Y uses σ_prev = σ_pml_x (pml_x)
        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            By[ix,iy,iz] = ((_one - σ_val) * By[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            By[ix,iy,iz] = By[ix,iy,iz] + in_t
        end

        # H update: Y uses σ_own = σ_pml_y (pml_y)
        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            w_old = WBy[ix,iy,iz]
            WBy[ix,iy,iz] = m_inv * By[ix,iy,iz]
            Hy[ix,iy,iz] = Hy[ix,iy,iz] + (_one + σ_val) * WBy[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Hy[ix,iy,iz] = m_inv * By[ix,iy,iz]
        end

        # ==================== Z component ====================
        # curl_z(E) = dEx/dy - dEy/dx (forward diff)
        Kz = dt_dy * (Ex[ix,iy+Int32(1),iz] - ex_cur) - dt_dx * (Ey[ix+Int32(1),iy,iz] - ey_cur)

        # U stage: Z uses σ_next = σ_pml_x (pml_x)
        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            u_old = UBz[ix,iy,iz]
            UBz[ix,iy,iz] = ((_one - σ_val) * u_old + Kz) / (_one + σ_val)
            in_t = UBz[ix,iy,iz] - u_old
        else
            in_t = Kz
        end

        # T stage: Z uses σ_prev = σ_pml_y (pml_y)
        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            Bz[ix,iy,iz] = ((_one - σ_val) * Bz[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Bz[ix,iy,iz] = Bz[ix,iy,iz] + in_t
        end

        # H update: Z uses σ_own = σ_pml_z (pml_z)
        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            w_old = WBz[ix,iy,iz]
            WBz[ix,iy,iz] = m_inv * Bz[ix,iy,iz]
            Hz[ix,iy,iz] = Hz[ix,iy,iz] + (_one + σ_val) * WBz[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Hz[ix,iy,iz] = m_inv * Bz[ix,iy,iz]
        end
    end
    return nothing
end

function _cuda_pml_DE_kernel!(
    Hx, Hy, Hz,          # source H fields (read)
    Dx, Dy, Dz,          # target D fields (read/write)
    Ex, Ey, Ez,          # output E fields (read/write)
    UDx, UDy, UDz,       # U auxiliary (read/write, may be dummy)
    WDx, WDy, WDz,       # W auxiliary (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,
    eps_inv_x, eps_inv_y, eps_inv_z,  # per-voxel ε⁻¹
    dt_dx, dt_dy, dt_dz,
    iNx, iNy,
    pml_x::Int32, pml_y::Int32, pml_z::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(dt_dx)

    @inbounds begin
        hy_cur = Hy[ix,iy,iz]; hx_cur = Hx[ix,iy,iz]; hz_cur = Hz[ix,iy,iz]

        # ==================== X component ====================
        Kx = dt_dz * (Hy[ix,iy,iz-Int32(1)] - hy_cur) - dt_dy * (Hz[ix,iy-Int32(1),iz] - hz_cur)

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            u_old = UDx[ix,iy,iz]
            UDx[ix,iy,iz] = ((_one - σ_val) * u_old + Kx) / (_one + σ_val)
            in_t = UDx[ix,iy,iz] - u_old
        else
            in_t = Kx
        end

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            Dx[ix,iy,iz] = ((_one - σ_val) * Dx[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dx[ix,iy,iz] = Dx[ix,iy,iz] + in_t
        end

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            w_old = WDx[ix,iy,iz]
            WDx[ix,iy,iz] = eps_inv_x[ix_0,iy_0,iz_0] * Dx[ix,iy,iz]
            Ex[ix,iy,iz] = Ex[ix,iy,iz] + (_one + σ_val) * WDx[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ex[ix,iy,iz] = eps_inv_x[ix_0,iy_0,iz_0] * Dx[ix,iy,iz]
        end

        # ==================== Y component ====================
        Ky = dt_dx * (Hz[ix-Int32(1),iy,iz] - hz_cur) - dt_dz * (Hx[ix,iy,iz-Int32(1)] - hx_cur)

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            u_old = UDy[ix,iy,iz]
            UDy[ix,iy,iz] = ((_one - σ_val) * u_old + Ky) / (_one + σ_val)
            in_t = UDy[ix,iy,iz] - u_old
        else
            in_t = Ky
        end

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            Dy[ix,iy,iz] = ((_one - σ_val) * Dy[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dy[ix,iy,iz] = Dy[ix,iy,iz] + in_t
        end

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            w_old = WDy[ix,iy,iz]
            WDy[ix,iy,iz] = eps_inv_y[ix_0,iy_0,iz_0] * Dy[ix,iy,iz]
            Ey[ix,iy,iz] = Ey[ix,iy,iz] + (_one + σ_val) * WDy[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ey[ix,iy,iz] = eps_inv_y[ix_0,iy_0,iz_0] * Dy[ix,iy,iz]
        end

        # ==================== Z component ====================
        Kz = dt_dy * (Hx[ix,iy-Int32(1),iz] - hx_cur) - dt_dx * (Hy[ix-Int32(1),iy,iz] - hy_cur)

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            u_old = UDz[ix,iy,iz]
            UDz[ix,iy,iz] = ((_one - σ_val) * u_old + Kz) / (_one + σ_val)
            in_t = UDz[ix,iy,iz] - u_old
        else
            in_t = Kz
        end

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            Dz[ix,iy,iz] = ((_one - σ_val) * Dz[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dz[ix,iy,iz] = Dz[ix,iy,iz] + in_t
        end

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            w_old = WDz[ix,iy,iz]
            WDz[ix,iy,iz] = eps_inv_z[ix_0,iy_0,iz_0] * Dz[ix,iy,iz]
            Ez[ix,iy,iz] = Ez[ix,iy,iz] + (_one + σ_val) * WDz[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ez[ix,iy,iz] = eps_inv_z[ix_0,iy_0,iz_0] * Dz[ix,iy,iz]
        end
    end
    return nothing
end

function _cuda_pml_DE_scalar_kernel!(
    Hx, Hy, Hz,
    Dx, Dy, Dz,
    Ex, Ey, Ez,
    UDx, UDy, UDz,
    WDx, WDy, WDz,
    σ_pml_x, σ_pml_y, σ_pml_z,
    eps_inv,             # scalar ε⁻¹
    dt_dx, dt_dy, dt_dz,
    iNx, iNy,
    pml_x::Int32, pml_y::Int32, pml_z::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(eps_inv)

    @inbounds begin
        hy_cur = Hy[ix,iy,iz]; hx_cur = Hx[ix,iy,iz]; hz_cur = Hz[ix,iy,iz]

        # ==================== X component ====================
        Kx = dt_dz * (Hy[ix,iy,iz-Int32(1)] - hy_cur) - dt_dy * (Hz[ix,iy-Int32(1),iz] - hz_cur)

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            u_old = UDx[ix,iy,iz]
            UDx[ix,iy,iz] = ((_one - σ_val) * u_old + Kx) / (_one + σ_val)
            in_t = UDx[ix,iy,iz] - u_old
        else
            in_t = Kx
        end

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            Dx[ix,iy,iz] = ((_one - σ_val) * Dx[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dx[ix,iy,iz] = Dx[ix,iy,iz] + in_t
        end

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            w_old = WDx[ix,iy,iz]
            WDx[ix,iy,iz] = eps_inv * Dx[ix,iy,iz]
            Ex[ix,iy,iz] = Ex[ix,iy,iz] + (_one + σ_val) * WDx[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ex[ix,iy,iz] = eps_inv * Dx[ix,iy,iz]
        end

        # ==================== Y component ====================
        Ky = dt_dx * (Hz[ix-Int32(1),iy,iz] - hz_cur) - dt_dz * (Hx[ix,iy,iz-Int32(1)] - hx_cur)

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            u_old = UDy[ix,iy,iz]
            UDy[ix,iy,iz] = ((_one - σ_val) * u_old + Ky) / (_one + σ_val)
            in_t = UDy[ix,iy,iz] - u_old
        else
            in_t = Ky
        end

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            Dy[ix,iy,iz] = ((_one - σ_val) * Dy[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dy[ix,iy,iz] = Dy[ix,iy,iz] + in_t
        end

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            w_old = WDy[ix,iy,iz]
            WDy[ix,iy,iz] = eps_inv * Dy[ix,iy,iz]
            Ey[ix,iy,iz] = Ey[ix,iy,iz] + (_one + σ_val) * WDy[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ey[ix,iy,iz] = eps_inv * Dy[ix,iy,iz]
        end

        # ==================== Z component ====================
        Kz = dt_dy * (Hx[ix,iy-Int32(1),iz] - hx_cur) - dt_dx * (Hy[ix-Int32(1),iy,iz] - hy_cur)

        if pml_x != Int32(0)
            σ_val = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
            u_old = UDz[ix,iy,iz]
            UDz[ix,iy,iz] = ((_one - σ_val) * u_old + Kz) / (_one + σ_val)
            in_t = UDz[ix,iy,iz] - u_old
        else
            in_t = Kz
        end

        if pml_y != Int32(0)
            σ_val = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
            Dz[ix,iy,iz] = ((_one - σ_val) * Dz[ix,iy,iz] + in_t) / (_one + σ_val)
        else
            Dz[ix,iy,iz] = Dz[ix,iy,iz] + in_t
        end

        if pml_z != Int32(0)
            σ_val = σ_pml_z[Int32(2)*iz_0 - Int32(1)]
            w_old = WDz[ix,iy,iz]
            WDz[ix,iy,iz] = eps_inv * Dz[ix,iy,iz]
            Ez[ix,iy,iz] = Ez[ix,iy,iz] + (_one + σ_val) * WDz[ix,iy,iz] - (_one - σ_val) * w_old
        else
            Ez[ix,iy,iz] = eps_inv * Dz[ix,iy,iz]
        end
    end
    return nothing
end

# ------------------------------------------------------------------- #
# Per-component raw CUDA PML kernels with σ-skipping
#
# Each kernel processes ONE field component with fused curl→U→B/D→W→H/E.
# This reduces register pressure from ~50-60 (3-component) to ~20-25 per
# kernel, eliminating register spilling and improving occupancy.
#
# σ-skipping: when all 3 σ values are zero (interior voxels, 35-90% of
# domain), the kernel skips U/W auxiliary reads/writes entirely, reducing
# memory traffic from 312 to ~150-200 bytes/voxel/step.
#
# PML direction table (X component shown; Y, Z follow cyclic permutation):
#   X: curl from Ey,Ez; U uses σ_y; T uses σ_z; W/H uses σ_x
#   Y: curl from Ez,Ex; U uses σ_z; T uses σ_x; W/H uses σ_y
#   Z: curl from Ex,Ey; U uses σ_x; T uses σ_y; W/H uses σ_z
# ------------------------------------------------------------------- #

# ==================== BH per-component kernels ====================

function _cuda_pml_BH_x_kernel!(
    Ey, Ez,                          # 2 source E (read, stencil)
    Bx, Hx,                         # target B, output H (read/write)
    UBx, WBx,                       # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays (always valid, zero-filled if no PML)
    m_inv,                           # scalar μ⁻¹
    dt_dy, dt_dz,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(m_inv)
    _zero = zero(m_inv)

    @inbounds begin
        # Forward curl_x(E) = dEy/dz - dEz/dy
        K = dt_dz * (Ey[ix,iy,iz+Int32(1)] - Ey[ix,iy,iz]) - dt_dy * (Ez[ix,iy+Int32(1),iz] - Ez[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σy == _zero) & (σz == _zero) & (σx == _zero)
            # FAST PATH: interior voxel — eliminate B entirely (H += μ⁻¹·K)
            Hx[ix,iy,iz] += m_inv * K
        else
            # SLOW PATH: full PML cascade
            # U stage: X uses σ_next = σ_y
            if σy != _zero
                u_old = UBx[ix,iy,iz]
                UBx[ix,iy,iz] = ((_one - σy) * u_old + K) / (_one + σy)
                in_t = UBx[ix,iy,iz] - u_old
            else
                in_t = K
            end

            # T stage: X uses σ_prev = σ_z
            if σz != _zero
                Bx[ix,iy,iz] = ((_one - σz) * Bx[ix,iy,iz] + in_t) / (_one + σz)
            else
                Bx[ix,iy,iz] += in_t
            end

            # W/H stage: X uses σ_own = σ_x
            if σx != _zero
                w_old = WBx[ix,iy,iz]
                WBx[ix,iy,iz] = m_inv * Bx[ix,iy,iz]
                Hx[ix,iy,iz] = Hx[ix,iy,iz] + (_one + σx) * WBx[ix,iy,iz] - (_one - σx) * w_old
            else
                Hx[ix,iy,iz] = m_inv * Bx[ix,iy,iz]
            end
        end
    end
    return nothing
end

function _cuda_pml_BH_y_kernel!(
    Ez, Ex,                          # 2 source E (read, stencil)
    By, Hy,                          # target B, output H (read/write)
    UBy, WBy,                        # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays
    m_inv,                           # scalar μ⁻¹
    dt_dz, dt_dx,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(m_inv)
    _zero = zero(m_inv)

    @inbounds begin
        # Forward curl_y(E) = dEz/dx - dEx/dz
        K = dt_dx * (Ez[ix+Int32(1),iy,iz] - Ez[ix,iy,iz]) - dt_dz * (Ex[ix,iy,iz+Int32(1)] - Ex[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σz == _zero) & (σx == _zero) & (σy == _zero)
            Hy[ix,iy,iz] += m_inv * K
        else
            # U stage: Y uses σ_next = σ_z
            if σz != _zero
                u_old = UBy[ix,iy,iz]
                UBy[ix,iy,iz] = ((_one - σz) * u_old + K) / (_one + σz)
                in_t = UBy[ix,iy,iz] - u_old
            else
                in_t = K
            end

            # T stage: Y uses σ_prev = σ_x
            if σx != _zero
                By[ix,iy,iz] = ((_one - σx) * By[ix,iy,iz] + in_t) / (_one + σx)
            else
                By[ix,iy,iz] += in_t
            end

            # W/H stage: Y uses σ_own = σ_y
            if σy != _zero
                w_old = WBy[ix,iy,iz]
                WBy[ix,iy,iz] = m_inv * By[ix,iy,iz]
                Hy[ix,iy,iz] = Hy[ix,iy,iz] + (_one + σy) * WBy[ix,iy,iz] - (_one - σy) * w_old
            else
                Hy[ix,iy,iz] = m_inv * By[ix,iy,iz]
            end
        end
    end
    return nothing
end

function _cuda_pml_BH_z_kernel!(
    Ex, Ey,                          # 2 source E (read, stencil)
    Bz, Hz,                          # target B, output H (read/write)
    UBz, WBz,                        # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays
    m_inv,                           # scalar μ⁻¹
    dt_dx, dt_dy,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(m_inv)
    _zero = zero(m_inv)

    @inbounds begin
        # Forward curl_z(E) = dEx/dy - dEy/dx
        K = dt_dy * (Ex[ix,iy+Int32(1),iz] - Ex[ix,iy,iz]) - dt_dx * (Ey[ix+Int32(1),iy,iz] - Ey[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σx == _zero) & (σy == _zero) & (σz == _zero)
            Hz[ix,iy,iz] += m_inv * K
        else
            # U stage: Z uses σ_next = σ_x
            if σx != _zero
                u_old = UBz[ix,iy,iz]
                UBz[ix,iy,iz] = ((_one - σx) * u_old + K) / (_one + σx)
                in_t = UBz[ix,iy,iz] - u_old
            else
                in_t = K
            end

            # T stage: Z uses σ_prev = σ_y
            if σy != _zero
                Bz[ix,iy,iz] = ((_one - σy) * Bz[ix,iy,iz] + in_t) / (_one + σy)
            else
                Bz[ix,iy,iz] += in_t
            end

            # W/H stage: Z uses σ_own = σ_z
            if σz != _zero
                w_old = WBz[ix,iy,iz]
                WBz[ix,iy,iz] = m_inv * Bz[ix,iy,iz]
                Hz[ix,iy,iz] = Hz[ix,iy,iz] + (_one + σz) * WBz[ix,iy,iz] - (_one - σz) * w_old
            else
                Hz[ix,iy,iz] = m_inv * Bz[ix,iy,iz]
            end
        end
    end
    return nothing
end

# ==================== DE per-component kernels ====================

function _cuda_pml_DE_x_kernel!(
    Hy, Hz,                          # 2 source H (read, stencil)
    Dx, Ex,                          # target D, output E (read/write)
    UDx, WDx,                       # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays
    eps_x,                           # ε⁻¹ (scalar or per-voxel array)
    dt_dy, dt_dz,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(dt_dy)
    _zero = zero(dt_dy)

    @inbounds begin
        K = dt_dz * (Hy[ix,iy,iz-Int32(1)] - Hy[ix,iy,iz]) - dt_dy * (Hz[ix,iy-Int32(1),iz] - Hz[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σy == _zero) & (σz == _zero) & (σx == _zero)
            # FAST PATH: eliminate D — E += ε⁻¹·K
            Ex[ix,iy,iz] += _get_m(eps_x, ix_0, iy_0, iz_0) * K
        else
            if σy != _zero
                u_old = UDx[ix,iy,iz]
                UDx[ix,iy,iz] = ((_one - σy) * u_old + K) / (_one + σy)
                in_t = UDx[ix,iy,iz] - u_old
            else
                in_t = K
            end
            if σz != _zero
                Dx[ix,iy,iz] = ((_one - σz) * Dx[ix,iy,iz] + in_t) / (_one + σz)
            else
                Dx[ix,iy,iz] += in_t
            end
            m_val = _get_m(eps_x, ix_0, iy_0, iz_0)
            if σx != _zero
                w_old = WDx[ix,iy,iz]
                WDx[ix,iy,iz] = m_val * Dx[ix,iy,iz]
                Ex[ix,iy,iz] = Ex[ix,iy,iz] + (_one + σx) * WDx[ix,iy,iz] - (_one - σx) * w_old
            else
                Ex[ix,iy,iz] = m_val * Dx[ix,iy,iz]
            end
        end
    end
    return nothing
end

function _cuda_pml_DE_y_kernel!(
    Hz, Hx,                          # 2 source H (read, stencil)
    Dy, Ey,                          # target D, output E (read/write)
    UDy, WDy,                       # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays
    eps_y,                           # ε⁻¹ (scalar or per-voxel array)
    dt_dz, dt_dx,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(dt_dz)
    _zero = zero(dt_dz)

    @inbounds begin
        K = dt_dx * (Hz[ix-Int32(1),iy,iz] - Hz[ix,iy,iz]) - dt_dz * (Hx[ix,iy,iz-Int32(1)] - Hx[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σz == _zero) & (σx == _zero) & (σy == _zero)
            Ey[ix,iy,iz] += _get_m(eps_y, ix_0, iy_0, iz_0) * K
        else
            if σz != _zero
                u_old = UDy[ix,iy,iz]
                UDy[ix,iy,iz] = ((_one - σz) * u_old + K) / (_one + σz)
                in_t = UDy[ix,iy,iz] - u_old
            else
                in_t = K
            end
            if σx != _zero
                Dy[ix,iy,iz] = ((_one - σx) * Dy[ix,iy,iz] + in_t) / (_one + σx)
            else
                Dy[ix,iy,iz] += in_t
            end
            m_val = _get_m(eps_y, ix_0, iy_0, iz_0)
            if σy != _zero
                w_old = WDy[ix,iy,iz]
                WDy[ix,iy,iz] = m_val * Dy[ix,iy,iz]
                Ey[ix,iy,iz] = Ey[ix,iy,iz] + (_one + σy) * WDy[ix,iy,iz] - (_one - σy) * w_old
            else
                Ey[ix,iy,iz] = m_val * Dy[ix,iy,iz]
            end
        end
    end
    return nothing
end

function _cuda_pml_DE_z_kernel!(
    Hx, Hy,                          # 2 source H (read, stencil)
    Dz, Ez,                          # target D, output E (read/write)
    UDz, WDz,                       # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays
    eps_z,                           # ε⁻¹ (scalar or per-voxel array)
    dt_dx, dt_dy,                    # grid spacing ratios
    iNx::Int32, iNy::Int32,
)
    ix_0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy_0 = blockIdx().y
    iz_0 = blockIdx().z
    ix_0 > iNx && return nothing
    ix = ix_0 + Int32(1); iy = iy_0 + Int32(1); iz = iz_0 + Int32(1)
    _one = one(dt_dx)
    _zero = zero(dt_dx)

    @inbounds begin
        K = dt_dy * (Hx[ix,iy-Int32(1),iz] - Hx[ix,iy,iz]) - dt_dx * (Hy[ix-Int32(1),iy,iz] - Hy[ix,iy,iz])

        σx = σ_pml_x[Int32(2)*ix_0 - Int32(1)]
        σy = σ_pml_y[Int32(2)*iy_0 - Int32(1)]
        σz = σ_pml_z[Int32(2)*iz_0 - Int32(1)]

        if (σx == _zero) & (σy == _zero) & (σz == _zero)
            Ez[ix,iy,iz] += _get_m(eps_z, ix_0, iy_0, iz_0) * K
        else
            if σx != _zero
                u_old = UDz[ix,iy,iz]
                UDz[ix,iy,iz] = ((_one - σx) * u_old + K) / (_one + σx)
                in_t = UDz[ix,iy,iz] - u_old
            else
                in_t = K
            end
            if σy != _zero
                Dz[ix,iy,iz] = ((_one - σy) * Dz[ix,iy,iz] + in_t) / (_one + σy)
            else
                Dz[ix,iy,iz] += in_t
            end
            m_val = _get_m(eps_z, ix_0, iy_0, iz_0)
            if σz != _zero
                w_old = WDz[ix,iy,iz]
                WDz[ix,iy,iz] = m_val * Dz[ix,iy,iz]
                Ez[ix,iy,iz] = Ez[ix,iy,iz] + (_one + σz) * WDz[ix,iy,iz] - (_one - σz) * w_old
            else
                Ez[ix,iy,iz] = m_val * Dz[ix,iy,iz]
            end
        end
    end
    return nothing
end
