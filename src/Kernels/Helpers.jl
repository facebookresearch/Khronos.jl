# Copyright (c) Meta Platforms, Inc. and affiliates.

# Helper for scalar vs per-voxel material property access
@inline _get_m(m::Real, ix, iy, iz) = m
@inline _get_m(m::AbstractArray, ix, iy, iz) = m[ix, iy, iz]

# Helper: find a valid non-Nothing PML σ array for dummy substitution
@inline function _pml_dummy_σ(b, pf)
    pf.has_pml_x ? b.σBx : (pf.has_pml_y ? b.σBy : b.σBz)
end
@inline function _pml_dummy_σD(b, pf)
    pf.has_pml_x ? b.σDx : (pf.has_pml_y ? b.σDy : b.σDz)
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
