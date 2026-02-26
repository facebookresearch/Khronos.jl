# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Here lie the core timestepping kernels for the FDTD algorithm. Thanks to
# multiple dispatch, we can simply focus on the fundamental (most complicated)
# cases. We often need to use "function barriers" to ensure type stability.


"""
    step!(sim::SimulationData)

Primary timestepping routine.

Includes all the relevent steps to complete one full timestep, such as evolving
all of the fields and updating all of the sources and monitors.
"""
function step!(sim::SimulationData)

    t = round_time(sim)

    # P.3: Deactivate source arrays after all sources have shut off
    # This lets the update_field! kernel specialize to the Nothing path
    # Skip for CW sources (which have no cutoff)
    if sim.sources_active
        try
            if t > last_source_time(sim)
                sim.sources_active = false
            end
        catch
            # CW sources don't have a cutoff — sources stay active forever
        end
    end

    # CUDA Graph replay path (post-source only): the FDTD kernels
    # (curl + update + halo) are captured into two sub-graphs.
    # Monitor updates run outside the graph since they have time-varying arguments.
    if !isnothing(sim._cuda_graph_exec_H)
        CUDA.launch(sim._cuda_graph_exec_H)
        update_H_monitors!(sim, t)
        CUDA.launch(sim._cuda_graph_exec_E)
        step_chi3_correction!(sim)
        step_polarization!(sim)
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Attempt graph capture once sources have deactivated (local only)
    if !sim.sources_active && !is_distributed() && _try_capture_graphs!(sim)
        CUDA.launch(sim._cuda_graph_exec_H)
        update_H_monitors!(sim, t)
        CUDA.launch(sim._cuda_graph_exec_E)
        step_chi3_correction!(sim)
        step_polarization!(sim)
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Normal path (source-active, JIT warmup, non-CUDA backend, or distributed)
    if sim.sources_active
        update_magnetic_sources!(sim, t)
    end

    step_H_fused!(sim)

    update_H_monitors!(sim, t)

    if sim.sources_active
        update_electric_sources!(sim, t + sim.Δt / 2)
    end

    step_E_fused!(sim)

    # χ3 Kerr nonlinear correction (applied after E update, before polarization)
    step_chi3_correction!(sim)
    # E^{n+1} uses old P^n (with chi1 correction in ε_inv_eff), then P^{n+1}
    # is computed using the new E^{n+1}. This matches meep/src/step.cpp where
    # susceptibility P is updated after E, creating the correct semi-implicit coupling.
    step_polarization!(sim)

    update_E_monitors!(sim, t + sim.Δt / 2)

    increment_timestep!(sim)
end

"""
    _try_capture_graphs!(sim::SimulationData) -> Bool

Attempt to capture two CUDA graphs for the steady-state FDTD step:
  - Graph H: step_B_from_E! + update_H_from_B! (curl B kernels + update H kernels + halo exchange)
  - Graph E: step_D_from_H! + update_E_from_D! (curl D kernels + update E kernels + halo exchange)

Returns true if capture succeeded, false otherwise.
"""
function _try_capture_graphs!(sim::SimulationData)
    # Allow disabling via environment variable
    get(ENV, "KHRONOS_CUDA_GRAPHS", "1") == "0" && return false
    # Disable for complex fields (Bloch BC) — graph capture assumes fixed types
    _fields_are_complex(sim) && return false
    # Disable graph capture when multi-stream is active — standard capture
    # doesn't support cross-stream operations without STREAM_CAPTURE_MODE_RELAXED.
    # The per-component kernel speedup compensates for lack of graph replay.
    sim._use_multi_stream && return false

    try
        graph_H = CUDA.capture(; throw_error=false) do
            step_H_fused!(sim)
        end
        isnothing(graph_H) && return false

        graph_E = CUDA.capture(; throw_error=false) do
            step_E_fused!(sim)
        end
        isnothing(graph_E) && return false

        sim._cuda_graph_exec_H = CUDA.instantiate(graph_H)
        sim._cuda_graph_exec_E = CUDA.instantiate(graph_E)

        if !is_distributed() || is_root()
            @info("CUDA Graph capture successful — using graph replay for subsequent steps")
        end
        return true
    catch e
        @warn("CUDA Graph capture failed, continuing without graphs: $e")
        return false
    end
end

# FIXME for non PML
get_step_boundaries(sim) = (sim.Nx, sim.Ny, sim.Nz)

# Check if fields are complex-valued (Bloch BC)
_fields_are_complex(sim) = !isnothing(sim.chunk_data) && !isempty(sim.chunk_data) &&
    !isnothing(sim.chunk_data[1].fields.fEx) && eltype(sim.chunk_data[1].fields.fEx) <: Complex

# Check if grid is uniform (scalar spacing) — raw CUDA kernels require this
_grid_is_uniform(sim) = sim.Δx isa Real && sim.Δy isa Real && (sim.ndims <= 2 || sim.Δz isa Real)

function step_B_from_E!(sim::SimulationData)
    curl_B_kernel = sim._cached_curl_kernel
    idx_curl = 1
    for chunk in sim.chunk_data
        curl_B_kernel(
            chunk.fields.fEx,
            chunk.fields.fEy,
            chunk.fields.fEz,
            chunk.fields.fBx,
            chunk.fields.fBy,
            chunk.fields.fBz,
            chunk.fields.fCBx,
            chunk.fields.fCBy,
            chunk.fields.fCBz,
            chunk.fields.fUBx,
            chunk.fields.fUBy,
            chunk.fields.fUBz,
            chunk.geometry_data.σBx,
            chunk.geometry_data.σBy,
            chunk.geometry_data.σBz,
            chunk.boundary_data.σBx,
            chunk.boundary_data.σBy,
            chunk.boundary_data.σBz,
            sim.Δt,
            sim.Δx,
            sim.Δy,
            sim.Δz,
            idx_curl,
            ndrange = chunk.ndrange,
        )
    end

    return
end

function update_H_from_B!(sim::SimulationData)
    update_H_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    for chunk in sim.chunk_data
        update_H_kernel(
            chunk.fields.fHx,
            chunk.fields.fHy,
            chunk.fields.fHz,
            chunk.fields.fBx,
            chunk.fields.fBy,
            chunk.fields.fBz,
            chunk.fields.fWBx,
            chunk.fields.fWBy,
            chunk.fields.fWBz,
            chunk.fields.fPBx,
            chunk.fields.fPBy,
            chunk.fields.fPBz,
            sa ? chunk.fields.fSBx : nothing,
            sa ? chunk.fields.fSBy : nothing,
            sa ? chunk.fields.fSBz : nothing,
            chunk.geometry_data.μ_inv,
            chunk.geometry_data.μ_inv_x,
            chunk.geometry_data.μ_inv_y,
            chunk.geometry_data.μ_inv_z,
            chunk.boundary_data.σBx,
            chunk.boundary_data.σBy,
            chunk.boundary_data.σBz,
            ndrange = chunk.ndrange,
        )
    end

    exchange_halos!(sim, :H)
    return
end

function step_D_from_H!(sim::SimulationData)
    curl_D_kernel = sim._cached_curl_kernel
    idx_curl = -1
    for chunk in sim.chunk_data
        curl_D_kernel(
            chunk.fields.fHx,
            chunk.fields.fHy,
            chunk.fields.fHz,
            chunk.fields.fDx,
            chunk.fields.fDy,
            chunk.fields.fDz,
            chunk.fields.fCDx,
            chunk.fields.fCDy,
            chunk.fields.fCDz,
            chunk.fields.fUDx,
            chunk.fields.fUDy,
            chunk.fields.fUDz,
            chunk.geometry_data.σDx,
            chunk.geometry_data.σDy,
            chunk.geometry_data.σDz,
            chunk.boundary_data.σDx,
            chunk.boundary_data.σDy,
            chunk.boundary_data.σDz,
            sim.Δt,
            sim.Δx,
            sim.Δy,
            sim.Δz,
            idx_curl,
            ndrange = chunk.ndrange,
        )
    end

    return
end

function update_E_from_D!(sim::SimulationData)
    update_E_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    for chunk in sim.chunk_data
        update_E_kernel(
            chunk.fields.fEx,
            chunk.fields.fEy,
            chunk.fields.fEz,
            chunk.fields.fDx,
            chunk.fields.fDy,
            chunk.fields.fDz,
            chunk.fields.fWDx,
            chunk.fields.fWDy,
            chunk.fields.fWDz,
            chunk.fields.fPDx,
            chunk.fields.fPDy,
            chunk.fields.fPDz,
            sa ? chunk.fields.fSDx : nothing,
            sa ? chunk.fields.fSDy : nothing,
            sa ? chunk.fields.fSDz : nothing,
            chunk.geometry_data.ε_inv,
            chunk.geometry_data.ε_inv_x,
            chunk.geometry_data.ε_inv_y,
            chunk.geometry_data.ε_inv_z,
            chunk.boundary_data.σDx,
            chunk.boundary_data.σDy,
            chunk.boundary_data.σDz,
            ndrange = chunk.ndrange,
        )
    end

    exchange_halos!(sim, :E)
    return
end

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
    iNx,
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
    iNx,
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
    iNx,
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
    iNx,                 # x-dimension (Int32)
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
    iNx,
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
    iNx,
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

# Helper for scalar vs per-voxel material property access
@inline _get_m(m::Real, ix, iy, iz) = m
@inline _get_m(m::AbstractArray, ix, iy, iz) = m[ix, iy, iz]

# ==================== BH per-component kernels ====================

function _cuda_pml_BH_x_kernel!(
    Ey, Ez,                          # 2 source E (read, stencil)
    Bx, Hx,                         # target B, output H (read/write)
    UBx, WBx,                       # PML aux (read/write, may be dummy)
    σ_pml_x, σ_pml_y, σ_pml_z,     # 1D σ arrays (always valid, zero-filled if no PML)
    m_inv,                           # scalar μ⁻¹
    dt_dy, dt_dz,                    # grid spacing ratios
    iNx::Int32,
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
    iNx::Int32,
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
    iNx::Int32,
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
    iNx::Int32,
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
    iNx::Int32,
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
    iNx::Int32,
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

# Helper: find a valid non-Nothing PML σ array for dummy substitution
@inline function _pml_dummy_σ(b, pf)
    pf.has_pml_x ? b.σBx : (pf.has_pml_y ? b.σBy : b.σBz)
end
@inline function _pml_dummy_σD(b, pf)
    pf.has_pml_x ? b.σDx : (pf.has_pml_y ? b.σDy : b.σDz)
end

# ------------------------------------------------------------------- #
# Fused curl+update dispatch functions
#
# For chunks WITHOUT PML, use the fused step_curl_and_update! kernel
# to eliminate the B/D re-read. For PML chunks, fall back to the
# separate step_curl! + update_field! kernels.
# ------------------------------------------------------------------- #

function step_H_fused!(sim::SimulationData)
    curl_B_kernel = sim._cached_curl_kernel
    update_H_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    idx_curl = 1

    cuda_wg = parse(Int, get(ENV, "KHRONOS_CUDA_WORKGROUP_SIZE", "256"))
    # Precompute dt/dx ratios (only valid for uniform grids, used by raw CUDA paths)
    dt_dx = _grid_is_uniform(sim) ? sim.Δt / sim.Δx : zero(sim.Δt)
    dt_dy = _grid_is_uniform(sim) ? sim.Δt / sim.Δy : zero(sim.Δt)
    dt_dz = _grid_is_uniform(sim) ? sim.Δt / sim.Δz : zero(sim.Δt)
    use_ms = sim._use_multi_stream

    for (ci, chunk) in enumerate(sim.chunk_data)
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange

        if backend_engine isa CUDABackend && !_fields_are_complex(sim) && _grid_is_uniform(sim) && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.μ_inv isa Real
            # Raw CUDA path: scalar μ, B eliminated — H_new = H_old + μ⁻¹·Δt·curl(E)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), cuda_wg)
            m_inv = backend_number(g.μ_inv)
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fHx, f.fHy, f.fHz,
                    m_inv * backend_number(dt_dx), m_inv * backend_number(dt_dy), m_inv * backend_number(dt_dz),
                    iNx)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_fused_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fHx, f.fHy, f.fHz,
                    m_inv * backend_number(dt_dx), m_inv * backend_number(dt_dy), m_inv * backend_number(dt_dz),
                    iNx)
            end
        elseif !has_any_pml(chunk.spec.physics)
            # KA fused path: interior chunks with sources active or per-voxel μ
            fused_kernel = sim._cached_fused_kernel
            fused_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fBx, f.fBy, f.fBz,
                f.fHx, f.fHy, f.fHz,
                sa ? f.fSBx : nothing,
                sa ? f.fSBy : nothing,
                sa ? f.fSBz : nothing,
                nothing, nothing, nothing,  # No P for H update
                g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) &&
               _grid_is_uniform(sim) && has_any_pml(chunk.spec.physics) &&
               (!chunk.spec.physics.has_sources || !sa) &&
               !chunk.spec.physics.has_sigma_B && g.μ_inv isa Real && f.fPBx isa Nothing
            # Per-component raw CUDA PML with σ-skipping
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), cuda_wg)
            dummy3d = f.fBx

            if use_ms
                _s = sim._chunk_streams[ci]
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_BH_x_kernel!(
                    f.fEy, f.fEz, f.fBx, f.fHx,
                    isnothing(f.fUBx) ? dummy3d : f.fUBx,
                    isnothing(f.fWBx) ? dummy3d : f.fWBx,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dy), backend_number(dt_dz), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_BH_y_kernel!(
                    f.fEz, f.fEx, f.fBy, f.fHy,
                    isnothing(f.fUBy) ? dummy3d : f.fUBy,
                    isnothing(f.fWBy) ? dummy3d : f.fWBy,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dz), backend_number(dt_dx), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_BH_z_kernel!(
                    f.fEx, f.fEy, f.fBz, f.fHz,
                    isnothing(f.fUBz) ? dummy3d : f.fUBz,
                    isnothing(f.fWBz) ? dummy3d : f.fWBz,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dx), backend_number(dt_dy), iNx)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_BH_x_kernel!(
                    f.fEy, f.fEz, f.fBx, f.fHx,
                    isnothing(f.fUBx) ? dummy3d : f.fUBx,
                    isnothing(f.fWBx) ? dummy3d : f.fWBx,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dy), backend_number(dt_dz), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_BH_y_kernel!(
                    f.fEz, f.fEx, f.fBy, f.fHy,
                    isnothing(f.fUBy) ? dummy3d : f.fUBy,
                    isnothing(f.fWBy) ? dummy3d : f.fWBy,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dz), backend_number(dt_dx), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_BH_z_kernel!(
                    f.fEx, f.fEy, f.fBz, f.fHz,
                    isnothing(f.fUBz) ? dummy3d : f.fUBz,
                    isnothing(f.fWBz) ? dummy3d : f.fWBz,
                    b.σBx, b.σBy, b.σBz,
                    backend_number(g.μ_inv),
                    backend_number(dt_dx), backend_number(dt_dy), iNx)
            end
        else
            # PML KA path: separate curl + update
            curl_B_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fBx, f.fBy, f.fBz,
                f.fCBx, f.fCBy, f.fCBz,
                f.fUBx, f.fUBy, f.fUBz,
                g.σBx, g.σBy, g.σBz,
                b.σBx, b.σBy, b.σBz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
            update_H_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fBx, f.fBy, f.fBz,
                f.fWBx, f.fWBy, f.fWBz,
                f.fPBx, f.fPBy, f.fPBz,
                sa ? f.fSBx : nothing,
                sa ? f.fSBy : nothing,
                sa ? f.fSBz : nothing,
                g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                b.σBx, b.σBy, b.σBz,
                ndrange = nr,
            )
        end
    end

    if use_ms
        # GPU-side sync: make default stream wait for all chunk streams
        for i in eachindex(sim._chunk_streams)
            CUDA.synchronize(sim._chunk_streams[i])
        end
    end

    exchange_halos!(sim, :H)
    return
end

function step_E_fused!(sim::SimulationData)
    curl_D_kernel = sim._cached_curl_kernel
    update_E_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    idx_curl = -1

    cuda_wg = parse(Int, get(ENV, "KHRONOS_CUDA_WORKGROUP_SIZE", "256"))
    dt_dx = _grid_is_uniform(sim) ? sim.Δt / sim.Δx : zero(sim.Δt)
    dt_dy = _grid_is_uniform(sim) ? sim.Δt / sim.Δy : zero(sim.Δt)
    dt_dz = _grid_is_uniform(sim) ? sim.Δt / sim.Δz : zero(sim.Δt)
    use_ms = sim._use_multi_stream

    for (ci, chunk) in enumerate(sim.chunk_data)
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange

        if backend_engine isa CUDABackend && !_fields_are_complex(sim) && _grid_is_uniform(sim) && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.ε_inv_x isa AbstractArray && f.fPDx isa Nothing
            # Raw CUDA path: per-voxel ε, D eliminated — E_new = E_old + ε⁻¹·Δt·curl(H)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), cuda_wg)
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_fused_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx)
            end
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) && _grid_is_uniform(sim) && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.ε_inv isa Real && f.fPDx isa Nothing
            # Raw CUDA path: scalar ε, D eliminated — E_new = E_old + ε⁻¹·Δt·curl(H)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), cuda_wg)
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_DE_scalar_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    backend_number(g.ε_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_fused_DE_scalar_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    backend_number(g.ε_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx)
            end
        elseif !has_any_pml(chunk.spec.physics)
            # KA fused path: interior chunks with sources active
            fused_kernel = sim._cached_fused_kernel
            fused_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fDx, f.fDy, f.fDz,
                f.fEx, f.fEy, f.fEz,
                sa ? f.fSDx : nothing,
                sa ? f.fSDy : nothing,
                sa ? f.fSDz : nothing,
                f.fPDx, f.fPDy, f.fPDz,
                g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) &&
               _grid_is_uniform(sim) && has_any_pml(chunk.spec.physics) &&
               (!chunk.spec.physics.has_sources || !sa) &&
               !chunk.spec.physics.has_sigma_D && f.fPDx isa Nothing
            # Per-component raw CUDA PML with σ-skipping (handles both scalar and per-voxel ε via _get_m)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), cuda_wg)
            dummy3d = f.fDx

            # Resolve per-voxel or scalar ε⁻¹ for each component
            eps_x = g.ε_inv_x isa AbstractArray ? g.ε_inv_x : backend_number(g.ε_inv)
            eps_y = g.ε_inv_y isa AbstractArray ? g.ε_inv_y : backend_number(g.ε_inv)
            eps_z = g.ε_inv_z isa AbstractArray ? g.ε_inv_z : backend_number(g.ε_inv)

            if use_ms
                _s = sim._chunk_streams[ci]
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_DE_x_kernel!(
                    f.fHy, f.fHz, f.fDx, f.fEx,
                    isnothing(f.fUDx) ? dummy3d : f.fUDx,
                    isnothing(f.fWDx) ? dummy3d : f.fWDx,
                    b.σDx, b.σDy, b.σDz,
                    eps_x,
                    backend_number(dt_dy), backend_number(dt_dz), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_DE_y_kernel!(
                    f.fHz, f.fHx, f.fDy, f.fEy,
                    isnothing(f.fUDy) ? dummy3d : f.fUDy,
                    isnothing(f.fWDy) ? dummy3d : f.fWDy,
                    b.σDx, b.σDy, b.σDz,
                    eps_y,
                    backend_number(dt_dz), backend_number(dt_dx), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) stream=_s _cuda_pml_DE_z_kernel!(
                    f.fHx, f.fHy, f.fDz, f.fEz,
                    isnothing(f.fUDz) ? dummy3d : f.fUDz,
                    isnothing(f.fWDz) ? dummy3d : f.fWDz,
                    b.σDx, b.σDy, b.σDz,
                    eps_z,
                    backend_number(dt_dx), backend_number(dt_dy), iNx)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_DE_x_kernel!(
                    f.fHy, f.fHz, f.fDx, f.fEx,
                    isnothing(f.fUDx) ? dummy3d : f.fUDx,
                    isnothing(f.fWDx) ? dummy3d : f.fWDx,
                    b.σDx, b.σDy, b.σDz,
                    eps_x,
                    backend_number(dt_dy), backend_number(dt_dz), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_DE_y_kernel!(
                    f.fHz, f.fHx, f.fDy, f.fEy,
                    isnothing(f.fUDy) ? dummy3d : f.fUDy,
                    isnothing(f.fWDy) ? dummy3d : f.fWDy,
                    b.σDx, b.σDy, b.σDz,
                    eps_y,
                    backend_number(dt_dz), backend_number(dt_dx), iNx)
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(cuda_wg, 1, 1) _cuda_pml_DE_z_kernel!(
                    f.fHx, f.fHy, f.fDz, f.fEz,
                    isnothing(f.fUDz) ? dummy3d : f.fUDz,
                    isnothing(f.fWDz) ? dummy3d : f.fWDz,
                    b.σDx, b.σDy, b.σDz,
                    eps_z,
                    backend_number(dt_dx), backend_number(dt_dy), iNx)
            end
        else
            # PML KA path: separate curl + update
            curl_D_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fDx, f.fDy, f.fDz,
                f.fCDx, f.fCDy, f.fCDz,
                f.fUDx, f.fUDy, f.fUDz,
                g.σDx, g.σDy, g.σDz,
                b.σDx, b.σDy, b.σDz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
            update_E_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fDx, f.fDy, f.fDz,
                f.fWDx, f.fWDy, f.fWDz,
                f.fPDx, f.fPDy, f.fPDz,
                sa ? f.fSDx : nothing,
                sa ? f.fSDy : nothing,
                sa ? f.fSDz : nothing,
                g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                b.σDx, b.σDy, b.σDz,
                ndrange = nr,
            )
        end
    end

    if use_ms
        # GPU-side sync: make default stream wait for all chunk streams
        for i in eachindex(sim._chunk_streams)
            CUDA.synchronize(sim._chunk_streams[i])
        end
    end

    exchange_halos!(sim, :E)
    return
end

# ------------------------------------------------------------------- #
# Per-component curl launch functions
#
# These use step_curl_comp! which has ~30-35 registers (vs ~64 for the
# 3-component step_curl!), improving GPU occupancy from ~50% to ~75-100%
# on A100. The trade-off is 3× more kernel launches per half-step, but
# with CUDA Graphs the launch overhead is eliminated during replay.
# Used in the CUDA Graph capture path for post-source steady-state.
# ------------------------------------------------------------------- #

function step_B_from_E_comp!(sim::SimulationData)
    curl_comp = sim._cached_curl_comp_kernel
    Δt = sim.Δt; Δx = sim.Δx; Δy = sim.Δy; Δz = sim.Δz
    for chunk in sim.chunk_data
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange
        # X: curl_x = dEy/dz - dEz/dy
        curl_comp(f.fEy, f.fEz, f.fBx, f.fCBx, f.fUBx,
                  g.σBx, b.σBy, b.σBz, Δt, Δy, Δz, 1, 1, ndrange=nr)
        # Y: curl_y = dEz/dx - dEx/dz
        curl_comp(f.fEz, f.fEx, f.fBy, f.fCBy, f.fUBy,
                  g.σBy, b.σBz, b.σBx, Δt, Δx, Δz, 1, 2, ndrange=nr)
        # Z: curl_z = dEx/dy - dEy/dx
        curl_comp(f.fEx, f.fEy, f.fBz, f.fCBz, f.fUBz,
                  g.σBz, b.σBx, b.σBy, Δt, Δy, Δx, 1, 3, ndrange=nr)
    end
    return
end

function step_D_from_H_comp!(sim::SimulationData)
    curl_comp = sim._cached_curl_comp_kernel
    Δt = sim.Δt; Δx = sim.Δx; Δy = sim.Δy; Δz = sim.Δz
    for chunk in sim.chunk_data
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange
        # X: curl_x = dHy/dz - dHz/dy
        curl_comp(f.fHy, f.fHz, f.fDx, f.fCDx, f.fUDx,
                  g.σDx, b.σDy, b.σDz, Δt, Δy, Δz, -1, 1, ndrange=nr)
        # Y: curl_y = dHz/dx - dHx/dz
        curl_comp(f.fHz, f.fHx, f.fDy, f.fCDy, f.fUDy,
                  g.σDy, b.σDz, b.σDx, Δt, Δx, Δz, -1, 2, ndrange=nr)
        # Z: curl_z = dHx/dy - dHy/dx
        curl_comp(f.fHx, f.fHy, f.fDz, f.fCDz, f.fUDz,
                  g.σDz, b.σDx, b.σDy, Δt, Δy, Δx, -1, 3, ndrange=nr)
    end
    return
end

# ------------------------------------------------------------------- #
# Per-component update kernel (reduced register pressure)
#
# Splitting the fused 3-component update_field! into per-component kernels
# reduces register usage, improving GPU occupancy and memory bandwidth
# utilization. Same trade-off as step_curl_comp!: 3× more launches.
# ------------------------------------------------------------------- #

@kernel function update_field_comp!(
    A,             # field to be updated (E or H, single component)
    T,             # timestepped field (D or B, single component)
    W,             # auxiliary field (can be Nothing)
    P,             # polarizability (can be Nothing)
    S,             # source field (can be Nothing)
    m_inv,         # scalar inverse material constant (can be Nothing)
    m_inv_comp,    # per-voxel inverse material for this component (can be Nothing)
    σ_comp,        # PML conductivity for this component (can be Nothing)
    comp_dir,      # 1=x, 2=y, 3=z — selects which axis index for σ
)
    ix, iy, iz = @index(Global, NTuple)
    fidx = CartesianIndex(ix + 1, iy + 1, iz + 1)
    gidx = CartesianIndex(ix, iy, iz)

    if comp_dir == 1
        σ_val = get_σ(σ_comp, ix)
    elseif comp_dir == 2
        σ_val = get_σ(σ_comp, iy)
    else
        σ_val = get_σ(σ_comp, iz)
    end

    update_field_generic(
        A, T, W, P, S,
        get_m_inv(m_inv, m_inv_comp, gidx),
        σ_val,
        fidx,
    )
end

function update_H_from_B_comp!(sim::SimulationData)
    update_comp = sim._cached_update_comp_kernel
    sa = sim.sources_active
    for chunk in sim.chunk_data
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange
        # X component
        update_comp(f.fHx, f.fBx, f.fWBx, f.fPBx,
                    sa ? f.fSBx : nothing,
                    g.μ_inv, g.μ_inv_x, b.σBx, 1, ndrange=nr)
        # Y component
        update_comp(f.fHy, f.fBy, f.fWBy, f.fPBy,
                    sa ? f.fSBy : nothing,
                    g.μ_inv, g.μ_inv_y, b.σBy, 2, ndrange=nr)
        # Z component
        update_comp(f.fHz, f.fBz, f.fWBz, f.fPBz,
                    sa ? f.fSBz : nothing,
                    g.μ_inv, g.μ_inv_z, b.σBz, 3, ndrange=nr)
    end

    exchange_halos!(sim, :H)
    return
end

function update_E_from_D_comp!(sim::SimulationData)
    update_comp = sim._cached_update_comp_kernel
    sa = sim.sources_active
    for chunk in sim.chunk_data
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange
        # X component
        update_comp(f.fEx, f.fDx, f.fWDx, f.fPDx,
                    sa ? f.fSDx : nothing,
                    g.ε_inv, g.ε_inv_x, b.σDx, 1, ndrange=nr)
        # Y component
        update_comp(f.fEy, f.fDy, f.fWDy, f.fPDy,
                    sa ? f.fSDy : nothing,
                    g.ε_inv, g.ε_inv_y, b.σDy, 2, ndrange=nr)
        # Z component
        update_comp(f.fEz, f.fDz, f.fWDz, f.fPDz,
                    sa ? f.fSDz : nothing,
                    g.ε_inv, g.ε_inv_z, b.σDz, 3, ndrange=nr)
    end

    exchange_halos!(sim, :E)
    return
end

@kernel function step_curl!(
    @Const(Ax), @Const(Ay), @Const(Az),
    Tx, Ty, Tz,
    Cx, Cy, Cz,
    Ux, Uy, Uz,
    @Const(σDx), @Const(σDy), @Const(σDz),
    @Const(σx), @Const(σy), @Const(σz),
    Δt, Δx, Δy, Δz,
    idx_curl,
)
    ix, iy, iz = @index(Global, NTuple)
    # P.5: Shifted indices for field arrays (raw GPU array without OffsetArray).
    # Field arrays have ghost cells at raw index 1; interior starts at index 2.
    fx, fy, fz = ix + 1, iy + 1, iz + 1
    fidx = CartesianIndex(fx, fy, fz)
    gidx = CartesianIndex(ix, iy, iz)

    # X component
    Kx = Δt * curl_x!(Ay, Az, Δy, Δz, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDx, gidx, Δt)
    σ_prev = get_σ(σz, iz)
    σ_next = get_σ(σy, iy)
    generic_curl!(Kx, Cx, Ux, Tx, σD_temp, σ_next, σ_prev, fidx)

    # Y component
    Ky = Δt * curl_y!(Az, Ax, Δz, Δx, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDy, gidx, Δt)
    σ_prev = get_σ(σx, ix)
    σ_next = get_σ(σz, iz)
    generic_curl!(Ky, Cy, Uy, Ty, σD_temp, σ_next, σ_prev, fidx)

    # Z component
    Kz = Δt * curl_z!(Ax, Ay, Δx, Δy, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDz, gidx, Δt)
    σ_prev = get_σ(σy, iy)
    σ_next = get_σ(σx, ix)
    generic_curl!(Kz, Cz, Uz, Tz, σD_temp, σ_next, σ_prev, fidx)
end

# ------------------------------------------------------------------- #
# Fused curl + update kernel (no PML)
#
# For interior chunks without PML, fusing the curl and material-update
# steps into a single kernel eliminates the intermediate B/D re-read,
# saving ~13% memory traffic. The reduced argument count also lowers
# register pressure, improving occupancy.
# ------------------------------------------------------------------- #

@kernel function step_curl_and_update!(
    @Const(Ax), @Const(Ay), @Const(Az),   # source fields (E for B→H, H for D→E)
    Tx, Ty, Tz,                             # timestepped fields (B or D) - read-write
    Ox, Oy, Oz,                             # output fields (H or E) - write
    Sx, Sy, Sz,                             # source fields (optional, read+clear)
    @Const(Px), @Const(Py), @Const(Pz),   # polarization fields (optional, read)
    @Const(m_inv),                          # scalar material inverse (can be Nothing)
    @Const(m_inv_x), @Const(m_inv_y), @Const(m_inv_z),  # per-voxel material (can be Nothing)
    Δt, Δx, Δy, Δz,
    idx_curl,
)
    ix, iy, iz = @index(Global, NTuple)
    fx, fy, fz = ix + 1, iy + 1, iz + 1
    fidx = CartesianIndex(fx, fy, fz)
    gidx = CartesianIndex(ix, iy, iz)

    # X component: curl then update
    Kx = Δt * curl_x!(Ay, Az, Δy, Δz, idx_curl, fx, fy, fz)
    Tx[fidx] = Tx[fidx] + Kx
    net_x = Tx[fidx] + update_cache(Sx, fidx)
    net_x -= update_cache(Px, fidx)
    clear_source(Sx, fidx)
    Ox[fidx] = get_m_inv(m_inv, m_inv_x, gidx) * net_x

    # Y component: curl then update
    Ky = Δt * curl_y!(Az, Ax, Δz, Δx, idx_curl, fx, fy, fz)
    Ty[fidx] = Ty[fidx] + Ky
    net_y = Ty[fidx] + update_cache(Sy, fidx)
    net_y -= update_cache(Py, fidx)
    clear_source(Sy, fidx)
    Oy[fidx] = get_m_inv(m_inv, m_inv_y, gidx) * net_y

    # Z component: curl then update
    Kz = Δt * curl_z!(Ax, Ay, Δx, Δy, idx_curl, fx, fy, fz)
    Tz[fidx] = Tz[fidx] + Kz
    net_z = Tz[fidx] + update_cache(Sz, fidx)
    net_z -= update_cache(Pz, fidx)
    clear_source(Sz, fidx)
    Oz[fidx] = get_m_inv(m_inv, m_inv_z, gidx) * net_z
end

# ------------------------------------------------------------------- #
# Fused curl + update kernel (PML)
#
# For PML chunks, fusing curl cascade + material update into one kernel
# eliminates the B/D re-read between separate curl and update launches.
# Each component's T_new is kept in a register and passed directly to
# update_field_from_T, saving 12 bytes/voxel/step in global memory
# traffic and halving the number of kernel launches per PML chunk.
# KA's type specialization on Nothing eliminates dead PML code paths.
# ------------------------------------------------------------------- #

@kernel function step_curl_and_update_pml!(
    # Curl stage inputs
    @Const(Ax), @Const(Ay), @Const(Az),            # source fields (E for BH, H for DE)
    Tx, Ty, Tz,                                      # B/D fields (read-write)
    Cx, Cy, Cz,                                      # C cascade (PML, can be Nothing)
    Ux, Uy, Uz,                                      # U cascade (PML, can be Nothing)
    @Const(σDx), @Const(σDy), @Const(σDz),          # material σ (can be Nothing)
    @Const(σx), @Const(σy), @Const(σz),             # PML σ (shared between curl and update)
    Δt, Δx, Δy, Δz, idx_curl,
    # Update stage inputs
    Ox, Oy, Oz,                                      # output fields (H or E)
    Wx, Wy, Wz,                                      # W auxiliary (can be Nothing)
    @Const(Px), @Const(Py), @Const(Pz),             # P polarization (can be Nothing)
    Sx, Sy, Sz,                                      # S sources (can be Nothing)
    @Const(m_inv), @Const(m_inv_x), @Const(m_inv_y), @Const(m_inv_z),
)
    ix, iy, iz = @index(Global, NTuple)
    fx, fy, fz = ix + 1, iy + 1, iz + 1
    fidx = CartesianIndex(fx, fy, fz)
    gidx = CartesianIndex(ix, iy, iz)

    # X component: curl cascade → T_new in register → material update
    Kx = Δt * curl_x!(Ay, Az, Δy, Δz, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDx, gidx, Δt)
    T_new_x = generic_curl!(Kx, Cx, Ux, Tx, σD_temp, get_σ(σy, iy), get_σ(σz, iz), fidx)
    update_field_from_T(Ox, T_new_x, Wx, Px, Sx, get_m_inv(m_inv, m_inv_x, gidx), get_σ(σx, ix), fidx)

    # Y component: curl cascade → T_new in register → material update
    Ky = Δt * curl_y!(Az, Ax, Δz, Δx, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDy, gidx, Δt)
    T_new_y = generic_curl!(Ky, Cy, Uy, Ty, σD_temp, get_σ(σz, iz), get_σ(σx, ix), fidx)
    update_field_from_T(Oy, T_new_y, Wy, Py, Sy, get_m_inv(m_inv, m_inv_y, gidx), get_σ(σy, iy), fidx)

    # Z component: curl cascade → T_new in register → material update
    Kz = Δt * curl_z!(Ax, Ay, Δx, Δy, idx_curl, fx, fy, fz)
    σD_temp = get_σD(σDz, gidx, Δt)
    T_new_z = generic_curl!(Kz, Cz, Uz, Tz, σD_temp, get_σ(σx, ix), get_σ(σy, iy), fidx)
    update_field_from_T(Oz, T_new_z, Wz, Pz, Sz, get_m_inv(m_inv, m_inv_z, gidx), get_σ(σz, iz), fidx)
end

# ------------------------------------------------------------------- #
# Per-component curl kernels (reduced register pressure)
#
# Splitting the fused 3-component step_curl! into per-component kernels
# reduces register usage from ~64 to ~30-35, improving GPU occupancy
# from 50% to 75-100% on A100.
# ------------------------------------------------------------------- #

@kernel function step_curl_comp!(
    A1, A2,        # source fields for the two transverse components
    T,             # target field for this component
    C, U,          # PML auxiliary fields (can be Nothing)
    σD_comp,       # material conductivity for this component
    σ_next_arr,    # PML sigma for the "next" direction
    σ_prev_arr,    # PML sigma for the "prev" direction
    Δt, Δ1, Δ2,   # timestep and grid spacings for the two curl directions
    idx_curl,      # +1 or -1
    curl_dir,      # 1=x, 2=y, 3=z selects which curl formula to use
)
    ix, iy, iz = @index(Global, NTuple)
    # P.5: Shifted indices for field arrays (raw GPU array without OffsetArray)
    fx, fy, fz = ix + 1, iy + 1, iz + 1
    fidx = CartesianIndex(fx, fy, fz)
    gidx = CartesianIndex(ix, iy, iz)

    # Compute the curl for this component
    # curl_dir encodes which pair of derivatives to use:
    #   1 (x-component): dA1/dz - dA2/dy  (A1=Ay, A2=Az)
    #   2 (y-component): dA1/dx - dA2/dz  (A1=Az, A2=Ax)
    #   3 (z-component): dA1/dy - dA2/dx  (A1=Ax, A2=Ay)
    if curl_dir == 1
        K = Δt * (d_dz!(A1, Δ2, idx_curl, fx, fy, fz) - d_dy!(A2, Δ1, idx_curl, fx, fy, fz))
    elseif curl_dir == 2
        K = Δt * (d_dx!(A1, Δ1, idx_curl, fx, fy, fz) - d_dz!(A2, Δ2, idx_curl, fx, fy, fz))
    else
        K = Δt * (d_dy!(A1, Δ1, idx_curl, fx, fy, fz) - d_dx!(A2, Δ2, idx_curl, fx, fy, fz))
    end

    σD_temp = get_σD(σD_comp, gidx, Δt)

    # Extract PML sigma at the correct index for this component's directions
    if curl_dir == 1
        σ_prev = get_σ(σ_prev_arr, iz)
        σ_next = get_σ(σ_next_arr, iy)
    elseif curl_dir == 2
        σ_prev = get_σ(σ_prev_arr, ix)
        σ_next = get_σ(σ_next_arr, iz)
    else
        σ_prev = get_σ(σ_prev_arr, iy)
        σ_next = get_σ(σ_next_arr, ix)
    end

    generic_curl!(K, C, U, T, σD_temp, σ_next, σ_prev, fidx)
end

function update_magnetic_sources!(sim::SimulationData, t::Real)
    map((c) -> step_sources!(sim, c, t), (Hx(), Hy(), Hz()))
    return
end

function update_electric_sources!(sim::SimulationData, t::Real)
    map((c) -> step_sources!(sim, c, t), (Ex(), Ey(), Ez()))
    return
end

function update_H_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        # Skip monitor data that don't have .component (e.g., Near2FarMonitorData)
        hasfield(typeof(m), :component) || continue
        # Decimation: skip DFT updates on non-decimation steps
        if hasfield(typeof(m), :decimation) && m.decimation > 1 && sim.timestep % m.decimation != 0
            continue
        end
        if is_magnetic(m.component)
            update_monitor(sim, m, time)
        end
    end

    # Monitors read local chunk data; no halo exchange needed.
    return
end

function update_E_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        # Skip monitor data that don't have .component (e.g., Near2FarMonitorData)
        hasfield(typeof(m), :component) || continue
        # Decimation: skip DFT updates on non-decimation steps
        if hasfield(typeof(m), :decimation) && m.decimation > 1 && sim.timestep % m.decimation != 0
            continue
        end
        if is_electric(m.component)
            update_monitor(sim, m, time)
        end
    end

    # Monitors read local chunk data; no halo exchange needed.
    return
end

# # ------------------------------------------------------------------- #
# # Step curl methods (update T from ∇A)
# # ------------------------------------------------------------------- #

# # 2D, 2DTE, 2DTM, 3D (4)
# # D, B (2)
# # x, y, z (3)
# # σD (6)
# # PML (6)
# # Total stencils =


# """

# """
@inline update_field_from_curl(A, B, B_old, σ) = ((1 - σ) * A + B - B_old) / (1 + σ)
@inline update_field_from_curl(A, B, B_old::Nothing, σ) = ((1 - σ) * A + B) / (1 + σ)
@inline update_field_from_curl(A, B, B_old, σ::Nothing) = (A + B - B_old)
@inline update_field_from_curl(A, B, B_old::Nothing, σ::Nothing) = (A + B)

"""
    generic_curl!()

"""
@inline function generic_curl!(K, C, U, T, σD, σ_next, σ_prev, idx_array)
    #  ----------------- Most general case ----------------- #
    C_old = C[idx_array]
    C[idx_array] = update_field_from_curl(C[idx_array], K, nothing, σD)
    U_old = U[idx_array]
    U[idx_array] = update_field_from_curl(U[idx_array], C[idx_array], C_old, σ_next)
    T_new = update_field_from_curl(T[idx_array], U[idx_array], U_old, σ_prev)
    T[idx_array] = T_new
    return T_new
end

@inline function generic_curl!(K, C, U::Nothing, T, σD, σ_next::Nothing, σ_prev, idx_array)
    C_old = C[idx_array]
    C[idx_array] = update_field_from_curl(C[idx_array], K, nothing, σD)
    T_new = update_field_from_curl(T[idx_array], C[idx_array], C_old, σ_prev)
    T[idx_array] = T_new
    return T_new
end

@inline function generic_curl!(K, C, U::Nothing, T, σD, σ_next::AbstractArray, σ_prev, idx_array)
    error("Invalid setup of U fields...")
    return
end

@inline function generic_curl!(
    K,
    C::Nothing,
    U::AbstractArray,
    T::AbstractArray,
    σD::Nothing,
    σ_next,
    σ_prev,
    idx_array,
)
    U_old = U[idx_array]
    U[idx_array] = update_field_from_curl(U[idx_array], K, nothing, σ_next)
    T_new = update_field_from_curl(T[idx_array], U[idx_array], U_old, σ_prev)
    T[idx_array] = T_new
    return T_new
end

@inline function generic_curl!(
    K,
    C::Nothing,
    U::Nothing,
    T,
    σD,
    σ_next::Nothing,
    σ_prev::Nothing,
    idx_array,
)
    T_new = update_field_from_curl(T[idx_array], K, nothing, σD)
    T[idx_array] = T_new
    return T_new
end

@inline function generic_curl!(
    K,
    C::Nothing,
    U::Nothing,
    T,
    σD::Nothing,
    σ_next::Nothing,
    σ_prev,
    idx_array,
)
    T_new = update_field_from_curl(T[idx_array], K, nothing, σ_prev)
    T[idx_array] = T_new
    return T_new
end

@inline function generic_curl!(
    K,
    C::Nothing,
    U::Nothing,
    T,
    σD::Nothing,
    σ_next::AbstractArray,
    σ_prev::Nothing,
    idx_array,
)
    error("Invalid setup of U fields...")
    return
end

@inline function generic_curl!(
    K,
    C::Nothing,
    U::Nothing,
    T::AbstractArray,
    σD::Nothing,
    σ_next::Nothing,
    σ_prev::Nothing,
    idx_array,
)
    T_new = update_field_from_curl(T[idx_array], K, nothing, nothing)
    T[idx_array] = T_new
    return T_new
end

# type stability
@inline scale_by_half(x::Float64) = 0.5 * x
@inline scale_by_half(x::Float32) = 0.5f0 * x

@inline get_σ(σ::Nothing, idx_array) = nothing
@inline get_σ(σ, idx_array) = σ[2*idx_array-1]
@inline get_σD(σD::Nothing, idx_array, Δt) = nothing
@inline get_σD(σD, idx_array, Δt) = scale_by_half(Δt * σD[idx_array])

# Grid spacing dispatch: uniform (scalar) vs non-uniform (vector)
# Julia compiles separate specializations for each — zero overhead for uniform.
@inline get_inv_dx(Δ::Real, i) = inv(Δ)
@inline get_inv_dx(Δ::AbstractVector, i) = inv(Δ[i])

@inline d_dx!(A, Δx, idx_curl, ix, iy, iz) =
    get_inv_dx(Δx, ix) * (A[ix+idx_curl, iy, iz] - A[ix, iy, iz])
@inline d_dy!(A, Δy, idx_curl, ix, iy, iz) =
    get_inv_dx(Δy, iy) * (A[ix, iy+idx_curl, iz] - A[ix, iy, iz])
@inline d_dz!(A, Δz, idx_curl, ix, iy, iz) =
    get_inv_dx(Δz, iz) * (A[ix, iy, iz+idx_curl] - A[ix, iy, iz])

@inline curl_x!(Ay, Az, Δy, Δz, idx_curl, ix, iy, iz) =
    d_dz!(Ay, Δz, idx_curl, ix, iy, iz) - d_dy!(Az, Δy, idx_curl, ix, iy, iz)
@inline curl_y!(Az, Ax, Δz, Δx, idx_curl, ix, iy, iz) =
    d_dx!(Az, Δx, idx_curl, ix, iy, iz) - d_dz!(Ax, Δz, idx_curl, ix, iy, iz)
@inline curl_z!(Ax, Ay, Δx, Δy, idx_curl, ix, iy, iz) =
    d_dy!(Ax, Δy, idx_curl, ix, iy, iz) - d_dx!(Ay, Δx, idx_curl, ix, iy, iz)

# ------------------------------------------------------------------- #
# Update field methods (update T from ∇A)
# ------------------------------------------------------------------- #

# """
#     A - field to be updated (either E or H)
#     T - timestepped field (either D or B)
#     W - auxilliary field
#     P - polarizability
#     S - source field
#     σ - PML conductivites
# """

@inline update_cache(A::AbstractArray, idx_array) = A[idx_array]
@inline update_cache(A::Nothing, idx_array) = 0 # FIXME
@inline function clear_source(A::AbstractArray, idx_array)
    A[idx_array] = zero(eltype(A[idx_array]))
    return
end
@inline function clear_source(A::Nothing, idx_array)
    return
end

@inline function update_field_generic(A, T, W, P, S, m_inv, σ, idx_array)
    #  ----------------- Most general case ----------------- #
    W_old = W[idx_array] # cache
    net_field = T[idx_array]
    net_field += update_cache(S, idx_array)
    net_field -= update_cache(P, idx_array)  # D - P: polarization subtracted (D = ε_∞·E + P)
    clear_source(S, idx_array)
    W[idx_array] = m_inv * net_field
    A[idx_array] = A[idx_array] + (1 + σ) * W[idx_array] - (1 - σ) * W_old
end

@inline function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ::Nothing, idx_array)
    net_field = T[idx_array]
    net_field += update_cache(S, idx_array)
    net_field -= update_cache(P, idx_array)  # D - P: polarization subtracted (D = ε_∞·E + P)
    clear_source(S, idx_array)
    A[idx_array] = m_inv * net_field
end

@inline function update_field_generic(A, T, W, P, S, m_inv, σ::Nothing, idx_array)
    error("W fields initialized when they don't need to be...")
end

@inline function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ, idx_array)
    error("W fields not properly initialized...")
end

# update_field_from_T: same as update_field_generic but takes T_new as a
# register value instead of reading from the T array. Used by the fused
# curl+update PML kernel to eliminate the B/D re-read between stages.
@inline function update_field_from_T(A, T_val, W, P, S, m_inv, σ, idx_array)
    W_old = W[idx_array]
    net_field = T_val
    net_field += update_cache(S, idx_array)
    net_field -= update_cache(P, idx_array)  # D - P: polarization subtracted (D = ε_∞·E + P)
    clear_source(S, idx_array)
    W[idx_array] = m_inv * net_field
    A[idx_array] = A[idx_array] + (1 + σ) * W[idx_array] - (1 - σ) * W_old
end

@inline function update_field_from_T(A, T_val, W::Nothing, P, S, m_inv, σ::Nothing, idx_array)
    net_field = T_val
    net_field += update_cache(S, idx_array)
    net_field -= update_cache(P, idx_array)  # D - P: polarization subtracted (D = ε_∞·E + P)
    clear_source(S, idx_array)
    A[idx_array] = m_inv * net_field
end

@inline function update_field_from_T(A, T_val, W, P, S, m_inv, σ::Nothing, idx_array)
    error("W fields initialized when they don't need to be...")
end

@inline function update_field_from_T(A, T_val, W::Nothing, P, S, m_inv, σ, idx_array)
    error("W fields not properly initialized...")
end

@inline get_m_inv(m_inv::Nothing, m_inv_x::AbstractArray, idx_array) = m_inv_x[idx_array]
@inline get_m_inv(m_inv::Real, m_inv_x::Nothing, idx_array) = m_inv
@inline function get_m_inv(m_inv, m_inv_x, idx_array)
    error("Failed to properly specialize m_inv")
end

@kernel function update_field!(
    Ax, Ay, Az,
    @Const(Tx), @Const(Ty), @Const(Tz),
    Wx, Wy, Wz,
    @Const(Px), @Const(Py), @Const(Pz),
    Sx, Sy, Sz,
    @Const(m_inv), @Const(m_inv_x), @Const(m_inv_y), @Const(m_inv_z),
    @Const(σx), @Const(σy), @Const(σz),
)
    ix, iy, iz = @index(Global, NTuple)
    # P.5: Shifted indices for field arrays, unshifted for geometry arrays
    fidx = CartesianIndex(ix + 1, iy + 1, iz + 1)
    gidx = CartesianIndex(ix, iy, iz)

    update_field_generic(
        Ax,
        Tx,
        Wx,
        Px,
        Sx,
        get_m_inv(m_inv, m_inv_x, gidx),
        get_σ(σx, ix),
        fidx,
    )
    update_field_generic(
        Ay,
        Ty,
        Wy,
        Py,
        Sy,
        get_m_inv(m_inv, m_inv_y, gidx),
        get_σ(σy, iy),
        fidx,
    )
    update_field_generic(
        Az,
        Tz,
        Wz,
        Pz,
        Sz,
        get_m_inv(m_inv, m_inv_z, gidx),
        get_σ(σz, iz),
        fidx,
    )
end

# ------------------------------------------------------------------- #
# ADE (Auxiliary Differential Equation) polarization update kernel
# for dispersive materials (Drude/Lorentz susceptibilities).
#
# Reference: meep/src/susceptibility.cpp:188-262
# ------------------------------------------------------------------- #

"""
    update_polarization_kernel!(Px, Py, Pz, Px_prev, Py_prev, Pz_prev,
                                Ex, Ey, Ez, sigma_x, sigma_y, sigma_z,
                                gamma1_inv, gamma1, omega0_dt_sq,
                                sigma_omega0_dt_sq, drude_coeff, is_drude)

KernelAbstractions kernel for advancing one Lorentzian/Drude ADE pole by one timestep.
sigma_x/y/z are per-voxel oscillator strengths; the ADE coefficients do NOT include sigma.

The update equation for Lorentz (ω₀ ≠ 0):
    P^{n+1} = γ₁_inv * (P^n * (2 - ω₀²Δt²) - γ₁ * P^{n-1} + ω₀²Δt² * σ_voxel * E^n)

For Drude (ω₀ = 0):
    P^{n+1} = γ₁_inv * (2*P^n - γ₁*P^{n-1} + γ*2πΔt * σ_voxel * E^n)
"""
@kernel function update_polarization_kernel!(
    Px, Py, Pz,
    Px_prev, Py_prev, Pz_prev,
    @Const(Ex), @Const(Ey), @Const(Ez),
    @Const(sigma_x), @Const(sigma_y), @Const(sigma_z),
    gamma1_inv::T, gamma1::T, omega0_dt_sq::T,
    sigma_omega0_dt_sq::T, drude_coeff::T, is_drude_int::Int32,
) where {T}
    ix, iy, iz = @index(Global, NTuple)
    gidx = CartesianIndex(ix, iy, iz)
    fidx = CartesianIndex(ix + 1, iy + 1, iz + 1)  # +1 for ghost cell offset

    sx = sigma_x[gidx]
    sy = sigma_y[gidx]
    sz = sigma_z[gidx]

    # Only update voxels with non-zero sigma (dispersive material present)
    if !(sx == 0 && sy == 0 && sz == 0)
        if is_drude_int == Int32(1)
            # Drude: ω₀ = 0
            # P^{n+1} = γ₁_inv * (2*P^n - γ₁*P^{n-1} + drude_coeff * σ * E^n)
            # x-component
            if sx != 0
                px = Px[fidx]
                Px[fidx] = gamma1_inv * (2 * px - gamma1 * Px_prev[fidx] + drude_coeff * sx * Ex[fidx])
                Px_prev[fidx] = px
            end
            # y-component
            if sy != 0
                py = Py[fidx]
                Py[fidx] = gamma1_inv * (2 * py - gamma1 * Py_prev[fidx] + drude_coeff * sy * Ey[fidx])
                Py_prev[fidx] = py
            end
            # z-component
            if sz != 0
                pz = Pz[fidx]
                Pz[fidx] = gamma1_inv * (2 * pz - gamma1 * Pz_prev[fidx] + drude_coeff * sz * Ez[fidx])
                Pz_prev[fidx] = pz
            end
        else
            # Lorentz: ω₀ ≠ 0
            # P^{n+1} = γ₁_inv * (P^n * (2 - ω₀²Δt²) - γ₁*P^{n-1} + σ*ω₀²Δt² * E^n)
            coeff_p = 2 - omega0_dt_sq
            # x-component
            if sx != 0
                px = Px[fidx]
                Px[fidx] = gamma1_inv * (coeff_p * px - gamma1 * Px_prev[fidx] + sigma_omega0_dt_sq * sx * Ex[fidx])
                Px_prev[fidx] = px
            end
            # y-component
            if sy != 0
                py = Py[fidx]
                Py[fidx] = gamma1_inv * (coeff_p * py - gamma1 * Py_prev[fidx] + sigma_omega0_dt_sq * sy * Ey[fidx])
                Py_prev[fidx] = py
            end
            # z-component
            if sz != 0
                pz = Pz[fidx]
                Pz[fidx] = gamma1_inv * (coeff_p * pz - gamma1 * Pz_prev[fidx] + sigma_omega0_dt_sq * sz * Ez[fidx])
                Pz_prev[fidx] = pz
            end
        end
    end
end

"""
    accumulate_polarization_kernel!(fPDx, fPDy, fPDz, Px, Py, Pz)

Add one pole's polarization to the total P-field arrays (fPDx/y/z).
"""
@kernel function accumulate_polarization_kernel!(
    fPDx, fPDy, fPDz,
    @Const(Px), @Const(Py), @Const(Pz),
)
    ix, iy, iz = @index(Global, NTuple)
    fidx = CartesianIndex(ix + 1, iy + 1, iz + 1)
    fPDx[fidx] += Px[fidx]
    fPDy[fidx] += Py[fidx]
    fPDz[fidx] += Pz[fidx]
end

"""
    zero_polarization_kernel!(fPDx, fPDy, fPDz)

Zero out the total P-field arrays before re-accumulation.
"""
@kernel function zero_polarization_kernel!(fPDx, fPDy, fPDz)
    ix, iy, iz = @index(Global, NTuple)
    fidx = CartesianIndex(ix + 1, iy + 1, iz + 1)
    fPDx[fidx] = zero(eltype(fPDx))
    fPDy[fidx] = zero(eltype(fPDy))
    fPDz[fidx] = zero(eltype(fPDz))
end

# ------------------------------------------------------------------- #
# χ3 Kerr nonlinear correction
#
# Applied after the E-field update. Uses |E_new|² as approximation of
# |E_old|² (self-consistent single-step iteration, standard practice).
# E_corrected = E_new / (1 + χ3 * |E|²)
# ------------------------------------------------------------------- #

@kernel function _chi3_correction_kernel!(
    Ex, Ey, Ez,
    @Const(chi3),
)
    ix, iy, iz = @index(Global, NTuple)
    fx, fy, fz = ix + 1, iy + 1, iz + 1

    @inbounds begin
        chi3_val = chi3[ix, iy, iz]
        if chi3_val != zero(chi3_val)
            ex = Ex[fx, fy, fz]
            ey = Ey[fx, fy, fz]
            ez = Ez[fx, fy, fz]
            E_sq = real(ex * conj(ex) + ey * conj(ey) + ez * conj(ez))
            correction = 1 / (1 + chi3_val * E_sq)
            Ex[fx, fy, fz] = ex * correction
            Ey[fx, fy, fz] = ey * correction
            Ez[fx, fy, fz] = ez * correction
        end
    end
end

"""
    step_chi3_correction!(sim)

Apply Kerr nonlinear correction to E-fields after the linear update.
No-op if no materials have chi3.
"""
function step_chi3_correction!(sim::SimulationData)
    isnothing(sim.chunk_data) && return
    wg = parse(Int, get(ENV, "KHRONOS_WORKGROUP_SIZE", "64"))

    for chunk in sim.chunk_data
        g = chunk.geometry_data
        isnothing(g.chi3) && continue

        f = chunk.fields
        isnothing(f.fEx) && continue

        _chi3_kernel = _chi3_correction_kernel!(backend_engine, (wg,))
        _chi3_kernel(
            f.fEx, f.fEy, f.fEz,
            g.chi3,
            ndrange = chunk.ndrange,
        )
    end
end

"""
    step_polarization!(sim::SimulationData)

Advance all dispersive polarization poles by one timestep using the ADE method.
Called after `step_E_fused!` in the main `step!` loop.

For each chunk with polarization data:
1. Zero the total P-field arrays (fPDx/fPDy/fPDz)
2. For each pole: run the ADE update kernel
3. Accumulate each pole's P into the total P-field arrays
"""
function step_polarization!(sim::SimulationData)
    isnothing(sim.chunk_data) && return

    wg = parse(Int, get(ENV, "KHRONOS_WORKGROUP_SIZE", "64"))
    ade_kernel = update_polarization_kernel!(backend_engine, (wg,))
    accum_kernel = accumulate_polarization_kernel!(backend_engine, (wg,))
    zero_kernel = zero_polarization_kernel!(backend_engine, (wg,))

    for chunk in sim.chunk_data
        pd = chunk.polarization_data
        isnothing(pd) && continue
        isempty(pd.poles) && continue

        f = chunk.fields
        nr = chunk.ndrange

        # Zero total P-fields
        zero_kernel(f.fPDx, f.fPDy, f.fPDz, ndrange=nr)

        # Update each pole and accumulate
        for pole in pd.poles
            coeffs = pole.coeffs
            ade_kernel(
                pole.Px, pole.Py, pole.Pz,
                pole.Px_prev, pole.Py_prev, pole.Pz_prev,
                f.fEx, f.fEy, f.fEz,
                pole.sigma_x, pole.sigma_y, pole.sigma_z,
                backend_number(coeffs.gamma1_inv),
                backend_number(coeffs.gamma1),
                backend_number(coeffs.omega0_dt_sq),
                backend_number(coeffs.sigma_omega0_dt_sq),
                backend_number(coeffs.drude_coeff),
                Int32(coeffs.is_drude ? 1 : 0),
                ndrange=nr,
            )

            # Accumulate into total P
            accum_kernel(f.fPDx, f.fPDy, f.fPDz,
                        pole.Px, pole.Py, pole.Pz,
                        ndrange=nr)
        end
    end
end
