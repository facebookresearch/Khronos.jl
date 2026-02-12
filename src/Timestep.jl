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

    update_magnetic_sources!(sim, t)

    step_B_from_E!(sim)

    update_H_from_B!(sim)

    update_H_monitors!(sim, t)

    update_electric_sources!(sim, t + sim.Δt / 2)

    step_D_from_H!(sim)

    update_E_from_D!(sim)

    update_E_monitors!(sim, t + sim.Δt / 2)

    increment_timestep!(sim)
end

# FIXME for non PML
get_step_boundaries(sim) = (sim.Nx, sim.Ny, sim.Nz)

function step_B_from_E!(sim::SimulationData)
    curl_B_kernel = step_curl!(backend_engine)
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
    update_H_kernel = update_field!(backend_engine)
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
            chunk.fields.fSBx,
            chunk.fields.fSBy,
            chunk.fields.fSBz,
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
    curl_D_kernel = step_curl!(backend_engine)
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
    update_E_kernel = update_field!(backend_engine)
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
            chunk.fields.fSDx,
            chunk.fields.fSDy,
            chunk.fields.fSDz,
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
    idx_array = CartesianIndex(ix, iy, iz)

    # X component
    Kx = Δt * curl_x!(Ay, Az, Δy, Δz, idx_curl, ix, iy, iz)
    σD_temp = get_σD(σDx, idx_array, Δt)
    σ_prev = get_σ(σz, iz)
    σ_next = get_σ(σy, iy)
    generic_curl!(Kx, Cx, Ux, Tx, σD_temp, σ_next, σ_prev, idx_array)

    # Y component
    Ky = Δt * curl_y!(Az, Ax, Δz, Δx, idx_curl, ix, iy, iz)
    σD_temp = get_σD(σDy, idx_array, Δt)
    σ_prev = get_σ(σx, ix)
    σ_next = get_σ(σz, iz)
    generic_curl!(Ky, Cy, Uy, Ty, σD_temp, σ_next, σ_prev, idx_array)

    # Z component
    Kz = Δt * curl_z!(Ax, Ay, Δx, Δy, idx_curl, ix, iy, iz)
    σD_temp = get_σD(σDz, idx_array, Δt)
    σ_prev = get_σ(σy, iy)
    σ_next = get_σ(σx, ix)
    generic_curl!(Kz, Cz, Uz, Tz, σD_temp, σ_next, σ_prev, idx_array)
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
    idx_array = CartesianIndex(ix, iy, iz)

    update_field_generic(
        Ax,
        Tx,
        Wx,
        Px,
        Sx,
        get_m_inv(m_inv, m_inv_x, idx_array),
        get_σ(σx, ix),
        idx_array,
    )
    update_field_generic(
        Ay,
        Ty,
        Wy,
        Py,
        Sy,
        get_m_inv(m_inv, m_inv_y, idx_array),
        get_σ(σy, iy),
        idx_array,
    )
    update_field_generic(
        Az,
        Tz,
        Wz,
        Pz,
        Sz,
        get_m_inv(m_inv, m_inv_z, idx_array),
        get_σ(σz, iz),
        idx_array,
    )
end
