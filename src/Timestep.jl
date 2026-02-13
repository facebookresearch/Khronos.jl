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
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Attempt graph capture once sources have deactivated (local only)
    if !sim.sources_active && !is_distributed() && _try_capture_graphs!(sim)
        CUDA.launch(sim._cuda_graph_exec_H)
        update_H_monitors!(sim, t)
        CUDA.launch(sim._cuda_graph_exec_E)
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Normal path (source-active, JIT warmup, non-CUDA backend, or distributed)
    if sim.sources_active
        update_magnetic_sources!(sim, t)
    end

    step_B_from_E!(sim)

    update_H_from_B!(sim)

    update_H_monitors!(sim, t)

    if sim.sources_active
        update_electric_sources!(sim, t + sim.Δt / 2)
    end

    step_D_from_H!(sim)

    update_E_from_D!(sim)

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

    try
        graph_H = CUDA.capture(; throw_error=false) do
            step_B_from_E!(sim)
            update_H_from_B!(sim)
        end
        isnothing(graph_H) && return false

        graph_E = CUDA.capture(; throw_error=false) do
            step_D_from_H!(sim)
            update_E_from_D!(sim)
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

@kernel function step_curl!(
    Ax, Ay, Az,
    Tx, Ty, Tz,
    Cx, Cy, Cz,
    Ux, Uy, Uz,
    σDx, σDy, σDz,
    σx, σy, σz,
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
        if is_magnetic(m.component)
            update_monitor(sim, m, time)
        end
    end

    # Monitors read local chunk data; no halo exchange needed.
    return
end

function update_E_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
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
    T[idx_array] = update_field_from_curl(T[idx_array], U[idx_array], U_old, σ_prev)
    return
end

@inline function generic_curl!(K, C, U::Nothing, T, σD, σ_next::Nothing, σ_prev, idx_array)
    C_old = C[idx_array]
    C[idx_array] = update_field_from_curl(C[idx_array], K, nothing, σD)
    T[idx_array] = update_field_from_curl(T[idx_array], C[idx_array], C_old, σ_prev)
    return
end

@inline function generic_curl!(K, C, U::Nothing, T, σD, σ_next::AbstractArray, σ_prev, idx_array)
    error("Invalid setup of U fields...")
    return
end

@inline function generic_curl!(
    K::Real,
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
    T[idx_array] = update_field_from_curl(T[idx_array], U[idx_array], U_old, σ_prev)
    return
end

@inline function generic_curl!(
    K::Real,
    C::Nothing,
    U::Nothing,
    T,
    σD,
    σ_next::Nothing,
    σ_prev::Nothing,
    idx_array,
)
    T[idx_array] = update_field_from_curl(T[idx_array], K, nothing, σD)
    return
end

@inline function generic_curl!(
    K::Real,
    C::Nothing,
    U::Nothing,
    T,
    σD::Nothing,
    σ_next::Nothing,
    σ_prev,
    idx_array,
)
    T[idx_array] = update_field_from_curl(T[idx_array], K, nothing, σ_prev)
    return
end

@inline function generic_curl!(
    K::Real,
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
    K::Real,
    C::Nothing,
    U::Nothing,
    T::AbstractArray,
    σD::Nothing,
    σ_next::Nothing,
    σ_prev::Nothing,
    idx_array,
)
    T[idx_array] = update_field_from_curl(T[idx_array], K, nothing, nothing)
    return
end

# type stability
@inline scale_by_half(x::Float64) = 0.5 * x
@inline scale_by_half(x::Float32) = 0.5f0 * x

@inline get_σ(σ::Nothing, idx_array) = nothing
@inline get_σ(σ, idx_array) = σ[2*idx_array-1]
@inline get_σD(σD::Nothing, idx_array, Δt) = nothing
@inline get_σD(σD, idx_array, Δt) = scale_by_half(Δt * σD[idx_array])

@inline d_dx!(A, Δx, idx_curl, ix, iy, iz) =
    inv(Δx) * (A[ix+idx_curl, iy, iz] - A[ix, iy, iz])
@inline d_dy!(A, Δy, idx_curl, ix, iy, iz) =
    inv(Δy) * (A[ix, iy+idx_curl, iz] - A[ix, iy, iz])
@inline d_dz!(A, Δz, idx_curl, ix, iy, iz) =
    inv(Δz) * (A[ix, iy, iz+idx_curl] - A[ix, iy, iz])

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
    net_field += update_cache(P, idx_array)
    clear_source(S, idx_array)
    W[idx_array] = m_inv * net_field
    A[idx_array] = A[idx_array] + (1 + σ) * W[idx_array] - (1 - σ) * W_old
end

@inline function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ::Nothing, idx_array)
    net_field = T[idx_array]
    net_field += update_cache(S, idx_array)
    net_field += update_cache(P, idx_array)
    clear_source(S, idx_array)
    A[idx_array] = m_inv * net_field
end

@inline function update_field_generic(A, T, W, P, S, m_inv, σ::Nothing, idx_array)
    error("W fields initialized when they don't need to be...")
end

@inline function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ, idx_array)
    error("W fields not properly initialized...")
end

@inline get_m_inv(m_inv::Nothing, m_inv_x::AbstractArray, idx_array) = m_inv_x[idx_array]
@inline get_m_inv(m_inv::Real, m_inv_x::Nothing, idx_array) = m_inv
@inline function get_m_inv(m_inv, m_inv_x, idx_array)
    error("Failed to properly specialize m_inv")
end

@kernel function update_field!(
    Ax, Ay, Az,
    Tx, Ty, Tz,
    Wx, Wy, Wz,
    Px, Py, Pz,
    Sx, Sy, Sz,
    m_inv, m_inv_x, m_inv_y, m_inv_z,
    σx, σy, σz,
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
