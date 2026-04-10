# Copyright (c) Meta Platforms, Inc. and affiliates.

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
