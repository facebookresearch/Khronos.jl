# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Diffraction monitor: records DFT fields on a plane and decomposes into
# diffraction orders via spatial FFT. Computes power per order.

export get_diffraction_efficiencies

# ------------------------------------------------------------------- #
# Monitor initialization
# ------------------------------------------------------------------- #

"""
    init_diffraction_monitor(sim, monitor::DiffractionMonitor) -> DiffractionMonitorData

Initialize a DiffractionMonitor by creating 4 internal DFT monitors for
tangential E and H on the specified plane.
"""
function init_diffraction_monitor(sim, monitor::DiffractionMonitor)
    normal_axis = findfirst(x -> x == 0.0, monitor.size)
    isnothing(normal_axis) && error("DiffractionMonitor size must have exactly one zero dimension")

    tangential_E_comps, tangential_H_comps = if normal_axis == 1
        ([Ey(), Ez()], [Hy(), Hz()])
    elseif normal_axis == 2
        ([Ex(), Ez()], [Hx(), Hz()])
    else
        ([Ex(), Ey()], [Hx(), Hy()])
    end

    tangential_E_monitors = DFTMonitor[]
    tangential_H_monitors = DFTMonitor[]

    for comp in tangential_E_comps
        push!(tangential_E_monitors, DFTMonitor(
            component = comp,
            center = copy(monitor.center),
            size = copy(monitor.size),
            frequencies = copy(monitor.frequencies),
            decimation = monitor.decimation,
        ))
    end
    for comp in tangential_H_comps
        push!(tangential_H_monitors, DFTMonitor(
            component = comp,
            center = copy(monitor.center),
            size = copy(monitor.size),
            frequencies = copy(monitor.frequencies),
            decimation = monitor.decimation,
        ))
    end

    md = DiffractionMonitorData(
        normal_axis = normal_axis,
        tangential_E_monitors = tangential_E_monitors,
        tangential_H_monitors = tangential_H_monitors,
        frequencies = Float64.(monitor.frequencies),
        dx = Float64(sim.Δx),
        dy = Float64(sim.Δy),
        dz = Float64(sim.Δz),
        cell_size = Float64.(sim.cell_size),
    )

    monitor.monitor_data = md
    return md
end

# ------------------------------------------------------------------- #
# Diffraction order computation
# ------------------------------------------------------------------- #

"""
    get_diffraction_efficiencies(monitor::DiffractionMonitor; max_order=5)
    get_diffraction_efficiencies(md::DiffractionMonitorData; max_order=5)

Compute the power in each diffraction order by spatially Fourier-transforming
the tangential DFT fields.

Returns a Dict mapping (m, n) order tuples to vectors of power per frequency,
normalized by the total incident flux.

Diffraction orders: k_x(m) = k_x_inc + 2πm/Lx, k_y(n) = k_y_inc + 2πn/Ly.
Propagating orders have real k_z; evanescent orders are excluded.
"""
get_diffraction_efficiencies(monitor::DiffractionMonitor; kwargs...) =
    get_diffraction_efficiencies(monitor.monitor_data; kwargs...)

function get_diffraction_efficiencies(md::DiffractionMonitorData; max_order::Int=5, k_inc::Vector{Float64}=[0.0, 0.0, 0.0])
    n_freq = length(md.frequencies)
    normal_axis = md.normal_axis

    # Get DFT fields
    e1_fields = Array(md.tangential_E_monitors[1].monitor_data.fields)
    e2_fields = Array(md.tangential_E_monitors[2].monitor_data.fields)
    h1_fields = Array(md.tangential_H_monitors[1].monitor_data.fields)
    h2_fields = Array(md.tangential_H_monitors[2].monitor_data.fields)

    # Average across normal dimension
    _avg(f, d) = size(f, d) >= 2 ?
        (selectdim(f, d, 1:1) .+ selectdim(f, d, 2:2)) ./ 2 : f
    e1_fields = _avg(e1_fields, normal_axis)
    e2_fields = _avg(e2_fields, normal_axis)
    h1_fields = _avg(h1_fields, normal_axis)
    h2_fields = _avg(h2_fields, normal_axis)

    # Determine tangential grid dimensions
    other_axes = setdiff(1:3, [normal_axis])
    L1 = md.cell_size[other_axes[1]]
    L2 = md.cell_size[other_axes[2]]

    # Extract 2D tangential fields for each frequency
    results = Dict{Tuple{Int,Int}, Vector{Float64}}()

    for i_freq in 1:n_freq
        freq = md.frequencies[i_freq]
        k0 = 2π * freq  # free-space wavenumber

        # Extract 2D field slices
        et1, et2 = _extract_tangential_2d(e1_fields, e2_fields, normal_axis, i_freq)
        ht1, ht2 = _extract_tangential_2d(h1_fields, h2_fields, normal_axis, i_freq)

        n1, n2 = size(et1)

        # Compute Poynting flux in each spatial frequency (diffraction order)
        # via DFT of the tangential fields
        # S_z(m,n) ∝ Re(Et1_mn * conj(Ht2_mn) - Et2_mn * conj(Ht1_mn))
        Et1_fft = fft2_manual(et1)
        Et2_fft = fft2_manual(et2)
        Ht1_fft = fft2_manual(ht1)
        Ht2_fft = fft2_manual(ht2)

        # Normalization: divide by number of grid points
        norm_factor = 1.0 / (n1 * n2)

        for m in -max_order:max_order, n in -max_order:max_order
            # Map (m, n) to FFT indices (1-based, with wrapping)
            im = mod(m, n1) + 1
            in_ = mod(n, n2) + 1

            # Diffraction order wavevector
            kx_m = k_inc[other_axes[1]] + 2π * m / L1
            ky_n = k_inc[other_axes[2]] + 2π * n / L2
            kz_sq = k0^2 - kx_m^2 - ky_n^2

            # Skip evanescent orders
            kz_sq <= 0 && continue

            # Power in this order (Poynting flux component along normal)
            et1_mn = Et1_fft[im, in_] * norm_factor
            et2_mn = Et2_fft[im, in_] * norm_factor
            ht1_mn = Ht1_fft[im, in_] * norm_factor
            ht2_mn = Ht2_fft[im, in_] * norm_factor

            power = real(et1_mn * conj(ht2_mn) - et2_mn * conj(ht1_mn))

            key = (m, n)
            if !haskey(results, key)
                results[key] = zeros(n_freq)
            end
            results[key][i_freq] = power
        end
    end

    return results
end

# ------------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------------- #

"""Extract 2D tangential field slice for a given frequency."""
function _extract_tangential_2d(f1, f2, normal_axis, i_freq)
    if normal_axis == 3
        return (f1[:, :, 1, i_freq], f2[:, :, 1, i_freq])
    elseif normal_axis == 1
        return (f1[1, :, :, i_freq], f2[1, :, :, i_freq])
    else
        return (f1[:, 1, :, i_freq], f2[:, 1, :, i_freq])
    end
end

"""
Simple 2D DFT implementation (no FFTW dependency).
For small arrays this is fine; for large arrays FFTW would be better.
"""
function fft2_manual(x::AbstractMatrix{T}) where T
    n1, n2 = size(x)
    result = zeros(Complex{real(T)}, n1, n2)
    for k2 in 0:n2-1, k1 in 0:n1-1
        s = zero(Complex{real(T)})
        for j2 in 0:n2-1, j1 in 0:n1-1
            phase = -2π * (k1 * j1 / n1 + k2 * j2 / n2)
            s += x[j1+1, j2+1] * exp(im * phase)
        end
        result[k1+1, k2+1] = s
    end
    return result
end
