# Copyright (c) Meta Platforms, Inc. and affiliates.

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
