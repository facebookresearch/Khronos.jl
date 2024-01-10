# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# So there are several things we can do to improve the performance of these
# kernels, depending on the hardware platform. While some of these improvements
# are trivial and expected to give good performance, others should wait to be
# implemented until benchmarks and profiling are performed.
#
# For better CUDA performance:
# * Add `@const` where necesarry
# * Leverage local memory
# * Better block and warp usage
#
# For better CPU performance:
# * Use `@inbounds` everywhere to prevent bounds checking
# * Check how the threads are reading in data, and makes sure there's no
#   contention.
# * 32 bit should be twice as fast as 64 bit... but it's not. So something is
#   wrong.
#
# For better Metal performance:
# * debug the 32 bit issue, and more performance should appear here.
#
# Some other general things to try:
# * Play with the settings of the kernel compiler
# * Check type stability all the way down (and add tests)
#
# Note that there are additional kernels in `Sources.jl`, `DFT.jl`, and
# `Monitors.jl`.
#
# Before we try *any* of these, let's come up with two or three more examples to
# run this on (with various features) and be sure to run them at multiple
# resolutions. We need to document everything. We also need to ensure everything
# has at least a basic test.


"""
    step!(sim::SimulationData)

Primary timestepping routine.

Includes all the relevent steps to complete on full timestep, such as evolving
all of the fields and updating all of the sources and monitors.
"""
function step!(sim::SimulationData)

    t = round_time(sim)

    update_magnetic_sources!(sim,t)

    step_B_from_E!(sim)

    update_H_from_B!(sim)

    update_H_monitors!(sim,t)

    update_electric_sources!(sim,t+sim.Δt/2)

    step_D_from_H!(sim)

    update_E_from_D!(sim)

    update_E_monitors!(sim,t+sim.Δt/2)

    increment_timestep!(sim)
end

# FIXME for non PML
get_step_boundaries(sim) = (sim.Nx, sim.Ny, sim.Nz)

function step_B_from_E!(sim::SimulationData)
    curl_B_kernel = step_curl!(backend_engine)
    idx_curl = 1
    ndrange = get_step_boundaries(sim)
    curl_B_kernel(
        sim.fields.fEx, sim.fields.fEy, sim.fields.fEz,
        sim.fields.fBx, sim.fields.fBy, sim.fields.fBz,
        sim.fields.fCBx, sim.fields.fCBy, sim.fields.fCBz,
        sim.fields.fUBx, sim.fields.fUBy, sim.fields.fUBz,
        sim.geometry_data.σBx, sim.geometry_data.σBy, sim.geometry_data.σBz,
        sim.boundary_data.σBx, sim.boundary_data.σBy, sim.boundary_data.σBz,
        sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
        ndrange=ndrange
    )

#      # TODO update halo
end

function update_H_from_B!(sim::SimulationData)
    update_H_kernel = update_field!(backend_engine)
    ndrange = get_step_boundaries(sim)
    update_H_kernel(
        sim.fields.fHx, sim.fields.fHy, sim.fields.fHz,
        sim.fields.fBx, sim.fields.fBy, sim.fields.fBz,
        sim.fields.fWBx, sim.fields.fWBy, sim.fields.fWBz,
        sim.fields.fPBx, sim.fields.fPBy, sim.fields.fPBz,
        sim.fields.fSBx, sim.fields.fSBy, sim.fields.fSBz,
        sim.geometry_data.μ_inv, sim.geometry_data.μ_inv_x, sim.geometry_data.μ_inv_y, sim.geometry_data.μ_inv_z,
        sim.boundary_data.σBx, sim.boundary_data.σBy, sim.boundary_data.σBz,
        ndrange = ndrange,
    )

     # TODO update halo
end

function step_D_from_H!(sim::SimulationData)
    curl_D_kernel = step_curl!(backend_engine)
    idx_curl = -1
    ndrange = get_step_boundaries(sim)
    curl_D_kernel(
        sim.fields.fHx, sim.fields.fHy, sim.fields.fHz,
        sim.fields.fDx, sim.fields.fDy, sim.fields.fDz,
        sim.fields.fCDx, sim.fields.fCDy, sim.fields.fCDz,
        sim.fields.fUDx, sim.fields.fUDy, sim.fields.fUDz,
        sim.geometry_data.σDx, sim.geometry_data.σDy, sim.geometry_data.σDz,
        sim.boundary_data.σDx, sim.boundary_data.σDy, sim.boundary_data.σDz,
        sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
        ndrange = ndrange
    )

     # TODO update halo
end

function update_E_from_D!(sim::SimulationData)
    update_E_kernel = update_field!(backend_engine)
    ndrange = get_step_boundaries(sim)
    update_E_kernel(
        sim.fields.fEx, sim.fields.fEy, sim.fields.fEz,
        sim.fields.fDx, sim.fields.fDy, sim.fields.fDz,
        sim.fields.fWDx, sim.fields.fWDy, sim.fields.fWDz,
        sim.fields.fPDx, sim.fields.fPDy, sim.fields.fPDz,
        sim.fields.fSDx, sim.fields.fSDy, sim.fields.fSDz,
        sim.geometry_data.ε_inv, sim.geometry_data.ε_inv_x, sim.geometry_data.ε_inv_y, sim.geometry_data.ε_inv_z,
        sim.boundary_data.σBx, sim.boundary_data.σBy, sim.boundary_data.σBz,
        ndrange = ndrange
    )
     # TODO update halo
     return
end

@kernel function step_curl!(
    Ax::Union{AbstractArray,Nothing}, Ay::Union{AbstractArray,Nothing}, Az::Union{AbstractArray,Nothing},
    Tx::Union{AbstractArray,Nothing}, Ty::Union{AbstractArray,Nothing}, Tz::Union{AbstractArray,Nothing},
    Cx::Union{AbstractArray,Nothing}, Cy::Union{AbstractArray,Nothing}, Cz::Union{AbstractArray,Nothing},
    Ux::Union{AbstractArray,Nothing}, Uy::Union{AbstractArray,Nothing}, Uz::Union{AbstractArray,Nothing},
    σDx::Union{AbstractArray,Nothing}, σDy::Union{AbstractArray,Nothing}, σDz::Union{AbstractArray,Nothing},
    σx::Union{AbstractArray,Nothing}, σy::Union{AbstractArray,Nothing}, σz::Union{AbstractArray,Nothing},
    Δt::Number, Δx::Number, Δy::Number, Δz::Number, idx_curl::Int
)
    ix, iy, iz = @index(Global, NTuple)
    idx_array = CartesianIndex(ix, iy, iz)

    # X component
    if (1 < ix) && (1 < iy) && (1 < iz)
        Kx = Δt * curl_x!(Ay,Az,Δy,Δz,idx_curl,ix,iy,iz)
        σD_temp = get_σD(σDx,idx_array,Δt)
        σ_prev = get_σ(σz, iz)
        σ_next = get_σ(σy, iy)
        generic_curl!(Kx, Cx, Ux, Tx, σD_temp, σ_next, σ_prev, idx_array)

    # Y component
        Ky = Δt * curl_y!(Az,Ax,Δz,Δx,idx_curl,ix,iy,iz)
        σD_temp = get_σD(σDy, idx_array, Δt)
        σ_prev = get_σ(σx, ix)
        σ_next = get_σ(σz, iz)
        generic_curl!(Ky, Cy, Uy, Ty, σD_temp, σ_next, σ_prev, idx_array)

    # Z component
        Kz = Δt * curl_z!(Ax,Ay,Δx,Δy,idx_curl,ix,iy,iz)
        σD_temp = get_σD(σDz, idx_array, Δt)
        σ_prev = get_σ(σy, iy)
        σ_next = get_σ(σx, ix)
        generic_curl!(Kz, Cz, Uz, Tz, σD_temp, σ_next, σ_prev, idx_array)
    end
end

function update_magnetic_sources!(sim::SimulationData, t::Real)
    # table-stable iterator
    map((c)->step_sources!(sim,c,t),(Hx(),Hy(),Hz()))
    return
     # TODO update halo
end

function update_electric_sources!(sim::SimulationData, t::Real)
    map((c)->step_sources!(sim,c,t),(Ex(),Ey(),Ez()))
    return
     # TODO update halo
end

function update_H_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        if is_magnetic(m.component)
            update_monitor(sim,m,time)
        end
    end
    return
end

function update_E_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        if is_electric(m.component)
            update_monitor(sim,m,time)
        end
    end
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
@inline update_field_from_curl(A,B,B_old,σ)                     = ((1 - σ) * A + B - B_old) / (1 + σ)
@inline update_field_from_curl(A,B,B_old::Nothing,σ)            = ((1 - σ) * A + B) / (1 + σ)
@inline update_field_from_curl(A,B,B_old,σ::Nothing)            = (A + B - B_old)
@inline update_field_from_curl(A,B,B_old::Nothing,σ::Nothing)   = (A + B)

"""
    generic_curl!()

"""
function generic_curl!(K, C, U, T, σD, σ_next, σ_prev, idx_array)
    #  ----------------- Most general case ----------------- #
    C_old = C[idx_array]
    C[idx_array] = update_field_from_curl(C[idx_array], K, nothing, σD)
    U_old = U[idx_array]
    U[idx_array] = update_field_from_curl(U[idx_array],C[idx_array],C_old,σ_next)
    T[idx_array] = update_field_from_curl(T[idx_array],U[idx_array],U_old,σ_prev)
    return
end

function generic_curl!(K, C, U::Nothing, T, σD, σ_next::Nothing, σ_prev, idx_array)
    C_old = C[idx_array]
    C[idx_array] = update_field_from_curl(C[idx_array], K, nothing, σD)
    T[idx_array] = update_field_from_curl(T[idx_array],C[idx_array],C_old,σ_prev)
    return
end

function generic_curl!(K, C, U::Nothing, T, σD, σ_next::AbstractArray, σ_prev, idx_array)
    error("Invalid setup of U fields...")
    return
end

function generic_curl!(K::Real, C::Nothing, U::AbstractArray, T::AbstractArray, σD::Nothing, σ_next, σ_prev, idx_array)
    U_old = U[idx_array]
    U[idx_array] = update_field_from_curl(U[idx_array],K,nothing,σ_next)
    T[idx_array] = update_field_from_curl(T[idx_array],U[idx_array],U_old,σ_prev)
    return
end

function generic_curl!(K::Real, C::Nothing, U::Nothing, T, σD, σ_next::Nothing, σ_prev::Nothing, idx_array)
    T[idx_array] = update_field_from_curl(T[idx_array],K,nothing,σD)
    return
end

function generic_curl!(K::Real, C::Nothing, U::Nothing, T, σD::Nothing, σ_next::Nothing, σ_prev, idx_array)
    T[idx_array] = update_field_from_curl(T[idx_array],K,nothing,σ_prev)
    return
end

function generic_curl!(K::Real, C::Nothing, U::Nothing, T, σD::Nothing, σ_next::AbstractArray, σ_prev::Nothing, idx_array)
    error("Invalid setup of U fields...")
    return
end

function generic_curl!(K::Real, C::Nothing, U::Nothing, T::AbstractArray, σD::Nothing, σ_next::Nothing, σ_prev::Nothing, idx_array)
    T[idx_array] = update_field_from_curl(T[idx_array],K,nothing,nothing)
    return
end

# type stability
scale_by_half(x::Float64) = 0.5 * x
scale_by_half(x::Float32) = 0.5f0 * x

get_σ(σ::Nothing, idx_array) = nothing
get_σ(σ, idx_array) =  σ[2*idx_array-1]
get_σD(σD::Nothing, idx_array, Δt) = nothing
get_σD(σD, idx_array, Δt) = scale_by_half(Δt * σD[idx_array])

@inline d_dx!(A,Δx,idx_curl,ix,iy,iz) = inv(Δx) * (A[ix+idx_curl, iy, iz] - A[ix, iy, iz])
@inline d_dy!(A,Δy,idx_curl,ix,iy,iz) = inv(Δy) * (A[ix, iy+idx_curl, iz] - A[ix, iy, iz])
@inline d_dz!(A,Δz,idx_curl,ix,iy,iz) = inv(Δz) * (A[ix, iy, iz+idx_curl] - A[ix, iy, iz])

@inline curl_x!(Ay,Az,Δy,Δz,idx_curl,ix,iy,iz)  = d_dz!(Ay,Δz,idx_curl,ix,iy,iz) - d_dy!(Az,Δy,idx_curl,ix,iy,iz)
@inline curl_y!(Az,Ax,Δz,Δx,idx_curl,ix,iy,iz)  = d_dx!(Az,Δx,idx_curl,ix,iy,iz) - d_dz!(Ax,Δz,idx_curl,ix,iy,iz)
@inline curl_z!(Ax,Ay,Δx,Δy,idx_curl,ix,iy,iz)  = d_dy!(Ax,Δy,idx_curl,ix,iy,iz) - d_dx!(Ay,Δx,idx_curl,ix,iy,iz)

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

# # 2D, 2DTE, 2DTM, 3D (4)
# # D, B (2)
# # x, y, z (3)
# # Sources (2)
# # Polarizabilities (2)
# # Conductivity (2)
# # PML (2)
# # ε, εxx+εyy+εzz, εxy ... (3)
# # Total stencils =

@inline update_cache(A::AbstractArray ,idx_array) = A[idx_array]
@inline update_cache(A::Nothing ,idx_array) = 0 # FIXME
@inline function clear_source(A::AbstractArray,idx_array)
    A[idx_array] = zero(eltype(A[idx_array]))
    return
end
@inline function clear_source(A::Nothing,idx_array)
    return
end

function update_field_generic(A, T, W, P, S, m_inv, σ, idx_array)
    #  ----------------- Most general case ----------------- #
    W_old = W[idx_array] # cache
    net_field = T[idx_array]
    net_field += update_cache(S,idx_array)
    net_field += update_cache(P,idx_array)
    clear_source(S,idx_array)
    W[idx_array] = m_inv * net_field
    A[idx_array] = A[idx_array] + (1 + σ) * W[idx_array] - (1 - σ) * W_old
end

function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ::Nothing, idx_array)
    net_field = T[idx_array]
    net_field += update_cache(S,idx_array)
    net_field += update_cache(P,idx_array)
    clear_source(S,idx_array)
    A[idx_array] = m_inv * net_field
end

function update_field_generic(A, T, W, P, S, m_inv, σ::Nothing, idx_array)
    error("W fields initialized when they don't need to be...")
end

function update_field_generic(A, T, W::Nothing, P, S, m_inv, σ, idx_array)
    error("W fields not properly initialized...")
end

get_m_inv(m_inv::Nothing,m_inv_x::AbstractArray,idx_array) = m_inv_x[idx_array]
get_m_inv(m_inv::Real,m_inv_x::Nothing,idx_array) = m_inv
function get_m_inv(m_inv,m_inv_x,idx_array)
    error("Failed to properly specialize m_inv")
end

@kernel function update_field!(
    Ax::Union{AbstractArray,Nothing}, Ay::Union{AbstractArray,Nothing}, Az::Union{AbstractArray,Nothing},
    Tx::Union{AbstractArray,Nothing}, Ty::Union{AbstractArray,Nothing}, Tz::Union{AbstractArray,Nothing},
    Wx::Union{AbstractArray,Nothing}, Wy::Union{AbstractArray,Nothing}, Wz::Union{AbstractArray,Nothing},
    Px::Union{AbstractArray,Nothing}, Py::Union{AbstractArray,Nothing}, Pz::Union{AbstractArray,Nothing},
    Sx::Union{AbstractArray,Nothing}, Sy::Union{AbstractArray,Nothing}, Sz::Union{AbstractArray,Nothing},
    m_inv::Union{Number,Nothing}, m_inv_x::Union{AbstractArray,Nothing}, m_inv_y::Union{AbstractArray,Nothing}, m_inv_z::Union{AbstractArray,Nothing},
    σx::Union{AbstractArray,Nothing}, σy::Union{AbstractArray,Nothing}, σz::Union{AbstractArray,Nothing},
)
    ix, iy, iz = @index(Global, NTuple)
    idx_array = CartesianIndex(ix, iy, iz)

    if (ix <= size(Ax)[1]) && (iy <= size(Ax)[2]) && (iz <= size(Ax)[3])
        #Ax[ix,iy,iz] =  Tx[ix,iy,iz]
        update_field_generic(Ax, Tx, Wx, Px, Sx, get_m_inv(m_inv,m_inv_x,idx_array), get_σ(σx,ix), idx_array)
    end

    if (ix <= size(Ay)[1]) && (iy <= size(Ay)[2]) && (iz <= size(Ay)[3])
        #Ay[ix,iy,iz] =  Ty[ix,iy,iz]
        update_field_generic(Ay, Ty, Wy, Py, Sy, get_m_inv(m_inv,m_inv_y,idx_array), get_σ(σy,iy), idx_array)
    end

    if (ix <= size(Az)[1]) && (iy <= size(Az)[2]) && (iz <= size(Az)[3])
        #Az[ix,iy,iz] =  Tz[ix,iy,iz]
        update_field_generic(Az, Tz, Wz, Pz, Sz, get_m_inv(m_inv,m_inv_z,idx_array), get_σ(σz,iz), idx_array)
    end
end
