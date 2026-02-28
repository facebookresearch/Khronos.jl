# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Flux monitor: records DFT fields on a planar surface and computes
# the Poynting flux (power flow) through the surface.

export get_flux

# ------------------------------------------------------------------- #
# Monitor initialization
# ------------------------------------------------------------------- #

"""
    init_flux_monitor(sim, monitor::FluxMonitor) -> FluxMonitorData

Initialize a FluxMonitor by creating 4 internal DFT monitors for the
tangential E and H components on the specified planar surface.
"""
function init_flux_monitor(sim, monitor::FluxMonitor)
    # Determine normal axis (the axis where size == 0)
    normal_axis = findfirst(x -> x == 0.0, monitor.size)
    isnothing(normal_axis) && error("FluxMonitor size must have exactly one zero dimension")

    # Determine tangential E and H components based on normal axis
    tangential_E_comps, tangential_H_comps = if normal_axis == 1  # x-normal (YZ plane)
        ([Ey(), Ez()], [Hy(), Hz()])
    elseif normal_axis == 2  # y-normal (XZ plane)
        ([Ex(), Ez()], [Hx(), Hz()])
    else  # z-normal (XY plane)
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

    md = FluxMonitorData(
        normal_axis = normal_axis,
        tangential_E_monitors = tangential_E_monitors,
        tangential_H_monitors = tangential_H_monitors,
        frequencies = Float64.(monitor.frequencies),
        dx = Float64(sim.Δx),
        dy = Float64(sim.Δy),
        dz = Float64(sim.Δz),
    )

    monitor.monitor_data = md
    return md
end

# ------------------------------------------------------------------- #
# Flux computation
# ------------------------------------------------------------------- #

"""
    get_flux(monitor::FluxMonitor) -> Vector{Float64}
    get_flux(md::FluxMonitorData) -> Vector{Float64}

Compute the Poynting flux (power flow) through the monitor surface at each
frequency:

    S(f) = Re ∫∫ (E × H*) · n̂ dA
         = Re ∫∫ (E1 × H2* - E2 × H1*) dA

where E1, E2, H1, H2 are the two pairs of tangential field components.

Returns a vector of real-valued flux values indexed by frequency.
"""
get_flux(monitor::FluxMonitor) = get_flux(monitor.monitor_data)

function get_flux(md::FluxMonitorData)
    n_freq = length(md.frequencies)
    flux = zeros(Float64, n_freq)

    normal_axis = md.normal_axis

    # Get DFT fields from internal monitors — transfer from GPU to CPU
    e1_fields = Array(md.tangential_E_monitors[1].monitor_data.fields)
    e2_fields = Array(md.tangential_E_monitors[2].monitor_data.fields)
    h1_fields = Array(md.tangential_H_monitors[1].monitor_data.fields)
    h2_fields = Array(md.tangential_H_monitors[2].monitor_data.fields)

    # Average across the normal dimension (size=1 or 2 due to Yee stagger)
    _avg_dim(f, d) = size(f, d) >= 2 ?
        (selectdim(f, d, 1:1) .+ selectdim(f, d, 2:2)) ./ 2 : f
    e1_fields = _avg_dim(e1_fields, normal_axis)
    e2_fields = _avg_dim(e2_fields, normal_axis)
    h1_fields = _avg_dim(h1_fields, normal_axis)
    h2_fields = _avg_dim(h2_fields, normal_axis)

    # Grid spacing for the area element
    dA = if normal_axis == 1
        md.dy * md.dz
    elseif normal_axis == 2
        md.dx * md.dz
    else
        md.dx * md.dy
    end

    # Determine grid dimensions in the tangential plane
    e1_gv = md.tangential_E_monitors[1].monitor_data.gv
    e2_gv = md.tangential_E_monitors[2].monitor_data.gv
    h1_gv = md.tangential_H_monitors[1].monitor_data.gv
    h2_gv = md.tangential_H_monitors[2].monitor_data.gv

    if normal_axis == 3
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
    elseif normal_axis == 1
        n1 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
    else
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
    end

    for i_freq in 1:n_freq
        # Compute Poynting flux: S · n̂ = Re(E1 × H2* - E2 × H1*)
        s = 0.0
        for i2 in 1:n2, i1 in 1:n1
            et1, et2, ht1, ht2 = _extract_flux_fields(
                e1_fields, e2_fields, h1_fields, h2_fields,
                normal_axis, i1, i2, i_freq,
            )
            s += real(et1 * conj(ht2) - et2 * conj(ht1)) * dA
        end
        flux[i_freq] = s
    end

    return flux
end

"""
Extract tangential DFT field values at a grid point for a given frequency.
"""
@inline function _extract_flux_fields(e1, e2, h1, h2, normal_axis, i1, i2, i_freq)
    if normal_axis == 3  # z-normal
        return (e1[i1, i2, 1, i_freq], e2[i1, i2, 1, i_freq],
                h1[i1, i2, 1, i_freq], h2[i1, i2, 1, i_freq])
    elseif normal_axis == 1  # x-normal
        return (e1[1, i1, i2, i_freq], e2[1, i1, i2, i_freq],
                h1[1, i1, i2, i_freq], h2[1, i1, i2, i_freq])
    else  # y-normal
        return (e1[i1, 1, i2, i_freq], e2[i1, 1, i2, i_freq],
                h1[i1, 1, i2, i_freq], h2[i1, 1, i2, i_freq])
    end
end
