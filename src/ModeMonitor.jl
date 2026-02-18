# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Mode monitor: records DFT fields on a cross-sectional plane and computes
# the overlap integral with waveguide modes to extract S-parameters.
#
# Pattern follows Near2Far.jl: 4 internal DFT monitors for tangential E/H,
# mode profiles solved at each frequency, overlap computed in post-processing.

export compute_mode_amplitudes, get_mode_transmission

# Cache for mode profiles keyed by (size, normal_axis, geometry_id, mode_spec_hash, freqs_hash)
# Monitors with the same cross-section and frequencies can reuse solved modes.
const _mode_profile_cache = Dict{UInt64, Vector{VectorModesolver.Mode}}()

function _mode_cache_key(monitor::ModeMonitor, normal_axis::Int)
    # The mode profile depends on size, normal_axis, geometry, mode_spec params, and frequencies.
    # It does NOT depend on center (position along normal axis).
    return hash((
        monitor.size,
        normal_axis,
        objectid(monitor.mode_spec.geometry),  # same geometry vector → same modes
        monitor.mode_spec.mode_solver_resolution,
        monitor.mode_spec.solver_tolerance,
        monitor.mode_spec.num_modes,
        monitor.frequencies,
    ))
end

function clear_mode_cache!()
    empty!(_mode_profile_cache)
end

# ------------------------------------------------------------------- #
# Monitor initialization
# ------------------------------------------------------------------- #

"""
    init_mode_monitor(sim, monitor::ModeMonitor) -> ModeMonitorData

Initialize a ModeMonitor by creating 4 internal DFT monitors for the
tangential E and H components on the specified planar surface, and
solving for mode profiles at each frequency.
"""
function init_mode_monitor(sim, monitor::ModeMonitor)
    # Determine normal axis (the axis where size == 0)
    normal_axis = findfirst(x -> x == 0.0, monitor.size)
    isnothing(normal_axis) && error("ModeMonitor size must have exactly one zero dimension")

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

    # Solve for mode profiles at each frequency (or reuse from cache)
    mode_spec = monitor.mode_spec
    geometry = isempty(mode_spec.geometry) ? sim.geometry : mode_spec.geometry
    cache_key = _mode_cache_key(monitor, normal_axis)

    mode_profiles = get!(_mode_profile_cache, cache_key) do
        profiles = VectorModesolver.Mode[]
        for freq in monitor.frequencies
            mode = get_mode_profiles(;
                frequency = freq,
                mode_solver_resolution = mode_spec.mode_solver_resolution,
                mode_index = mode_spec.num_modes,
                center = copy(monitor.center),
                size = copy(monitor.size),
                solver_tolerance = mode_spec.solver_tolerance,
                geometry = geometry,
            )
            push!(profiles, mode)
        end
        profiles
    end

    md = ModeMonitorData(
        normal_axis = normal_axis,
        tangential_E_monitors = tangential_E_monitors,
        tangential_H_monitors = tangential_H_monitors,
        frequencies = Float64.(monitor.frequencies),
        mode_profiles = mode_profiles,
        geometry = geometry,
        mode_spec = mode_spec,
        dx = Float64(sim.Δx),
        dy = Float64(sim.Δy),
        dz = Float64(sim.Δz),
    )

    monitor.monitor_data = md
    return md
end

# ------------------------------------------------------------------- #
# Mode overlap integral computation
# ------------------------------------------------------------------- #

"""
    compute_mode_amplitudes(md::ModeMonitorData) -> (a_plus, a_minus)

Compute the forward (+) and backward (-) mode amplitude coefficients
using the mode overlap integral:

    a± = ∫∫ (E_sim × H_mode* ± E_mode* × H_sim) · n̂ dA / (4 * P_mode)

where P_mode = (1/2) Re ∫∫ (E_mode × H_mode*) · n̂ dA.

Returns vectors of complex amplitudes indexed by frequency.
"""
function compute_mode_amplitudes(md::ModeMonitorData)
    n_freq = length(md.frequencies)
    a_plus = zeros(ComplexF64, n_freq)
    a_minus = zeros(ComplexF64, n_freq)

    normal_axis = md.normal_axis

    # Get DFT monitor data — transfer from GPU to CPU
    e_mons = md.tangential_E_monitors
    h_mons = md.tangential_H_monitors
    e1_fields_raw = Array(e_mons[1].monitor_data.fields)
    e2_fields_raw = Array(e_mons[2].monitor_data.fields)
    h1_fields_raw = Array(h_mons[1].monitor_data.fields)
    h2_fields_raw = Array(h_mons[2].monitor_data.fields)

    # Average fields across normal dimension when size > 1
    _avg_dim(f, d) = size(f, d) >= 2 ?
        (selectdim(f, d, 1:1) .+ selectdim(f, d, 2:2)) ./ 2 : f
    e1_fields = _avg_dim(e1_fields_raw, normal_axis)
    e2_fields = _avg_dim(e2_fields_raw, normal_axis)
    h1_fields = _avg_dim(h1_fields_raw, normal_axis)
    h2_fields = _avg_dim(h2_fields_raw, normal_axis)

    # Grid dimensions (use min across components for safety)
    e1_gv = e_mons[1].monitor_data.gv
    e2_gv = e_mons[2].monitor_data.gv
    h1_gv = h_mons[1].monitor_data.gv
    h2_gv = h_mons[2].monitor_data.gv

    if normal_axis == 3       # z-normal: tangential = x, y
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        dA = md.dx * md.dy
    elseif normal_axis == 1   # x-normal: tangential = y, z
        n1 = min(e1_gv.Ny, e2_gv.Ny, h1_gv.Ny, h2_gv.Ny)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = md.dy * md.dz
    else                      # y-normal: tangential = x, z
        n1 = min(e1_gv.Nx, e2_gv.Nx, h1_gv.Nx, h2_gv.Nx)
        n2 = min(e1_gv.Nz, e2_gv.Nz, h1_gv.Nz, h2_gv.Nz)
        dA = md.dx * md.dz
    end

    # Physical base positions for coordinate mapping
    e1b = md.e1_base
    e2b = md.e2_base
    h1b = md.h1_base
    h2b = md.h2_base

    # Grid spacings for the two tangential directions
    d1, d2 = if normal_axis == 3; (md.dx, md.dy)
    elseif normal_axis == 1; (md.dy, md.dz)
    else; (md.dx, md.dz)
    end

    for i_freq in 1:n_freq
        mode = md.mode_profiles[i_freq]

        # Pre-extract mode tangential fields (2D arrays on mode solver grid)
        mode_e1_raw = _get_mode_tangential_field(mode, normal_axis, :e1)
        mode_e2_raw = _get_mode_tangential_field(mode, normal_axis, :e2)
        mode_h1_raw = _get_mode_tangential_field(mode, normal_axis, :h1)
        mode_h2_raw = _get_mode_tangential_field(mode, normal_axis, :h2)
        mode_x = collect(mode.x)
        mode_y = collect(mode.y)

        # Interpolate mode fields onto the DFT grid
        mode_e1 = zeros(ComplexF64, n1, n2)
        mode_e2 = zeros(ComplexF64, n1, n2)
        mode_h1 = zeros(ComplexF64, n1, n2)
        mode_h2 = zeros(ComplexF64, n1, n2)

        for i2 in 1:n2, i1 in 1:n1
            # Compute physical position of this DFT grid point
            if normal_axis == 3
                # z-normal: tangential coords are (x, y)
                px = e1b[1] + (i1 - 1) * d1
                py = e1b[2] + (i2 - 1) * d2
                pt = [px, py]
            elseif normal_axis == 1
                # x-normal: tangential coords are (y, z)
                py = e1b[2] + (i1 - 1) * d1
                pz = e1b[3] + (i2 - 1) * d2
                pt = [py, pz]
            else
                # y-normal: tangential coords are (x, z)
                px = e1b[1] + (i1 - 1) * d1
                pz = e1b[3] + (i2 - 1) * d2
                pt = [px, pz]
            end

            mode_e1[i1, i2] = bilinear_interpolator(mode_x, mode_y, mode_e1_raw, pt)
            mode_e2[i1, i2] = bilinear_interpolator(mode_x, mode_y, mode_e2_raw, pt)
            mode_h1[i1, i2] = bilinear_interpolator(mode_x, mode_y, mode_h1_raw, pt)
            mode_h2[i1, i2] = bilinear_interpolator(mode_x, mode_y, mode_h2_raw, pt)
        end

        # Compute mode power: P_mode = (1/2) Re ∫∫ (E_mode × H_mode*) · n̂ dA
        # For a planar surface with normal n̂, the cross product dot n̂ reduces to:
        #   (E × H*) · n̂ = e1 * conj(h2) - e2 * conj(h1)
        P_mode = 0.0
        for i2 in 1:n2, i1 in 1:n1
            P_mode += 0.5 * real(
                mode_e1[i1, i2] * conj(mode_h2[i1, i2]) -
                mode_e2[i1, i2] * conj(mode_h1[i1, i2])
            ) * dA
        end

        # Compute overlap integrals
        # a± = ∫∫ (E_sim × H_mode* ± E_mode* × H_sim) · n̂ dA / (4 * P_mode)
        overlap_plus = ComplexF64(0)
        overlap_minus = ComplexF64(0)

        for i2 in 1:n2, i1 in 1:n1
            # Extract DFT fields at this point for the current frequency
            et1, et2, ht1, ht2 = _extract_dft_fields(
                e1_fields, e2_fields, h1_fields, h2_fields,
                normal_axis, i1, i2, i_freq,
            )

            # Cross products dotted with normal:
            # (E_sim × H_mode*) · n̂ = et1 * conj(mode_h2) - et2 * conj(mode_h1)
            # (E_mode* × H_sim) · n̂ = conj(mode_e1) * ht2 - conj(mode_e2) * ht1
            sim_cross_mode = et1 * conj(mode_h2[i1, i2]) - et2 * conj(mode_h1[i1, i2])
            mode_cross_sim = conj(mode_e1[i1, i2]) * ht2 - conj(mode_e2[i1, i2]) * ht1

            overlap_plus += (sim_cross_mode + mode_cross_sim) * dA
            overlap_minus += (sim_cross_mode - mode_cross_sim) * dA
        end

        if abs(P_mode) > 1e-30
            a_plus[i_freq] = overlap_plus / (4.0 * P_mode)
            a_minus[i_freq] = overlap_minus / (4.0 * P_mode)
        end
    end

    return (a_plus, a_minus)
end

"""
    get_mode_transmission(md::ModeMonitorData) -> Vector{Float64}

Compute the forward-propagating mode power transmission |a+|² at each frequency.
"""
function get_mode_transmission(md::ModeMonitorData)
    a_plus, _ = compute_mode_amplitudes(md)
    return abs2.(a_plus)
end

# ------------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------------- #

"""
    _get_mode_tangential_field(mode, normal_axis, component_symbol)

Extract the appropriate tangential field component from a Mode object,
given the monitor's normal axis. Returns a 2D array.

For each normal axis, the tangential components are:
- x-normal (YZ plane): e1=Ey, e2=Ez, h1=Hy, h2=Hz
- y-normal (XZ plane): e1=Ex, e2=Ez, h1=Hx, h2=Hz
- z-normal (XY plane): e1=Ex, e2=Ey, h1=Hx, h2=Hy

The mode solver returns fields on (mode.x, mode.y) which correspond to
the two tangential coordinates of the monitor plane (already rotated
by get_mode_profiles).
"""
function _get_mode_tangential_field(mode::VectorModesolver.Mode, normal_axis::Int, sym::Symbol)
    # After rotation by get_mode_profiles, the mode fields are:
    # mode.Ex/Ey/Ez where x,y correspond to the two tangential directions
    # and z corresponds to the normal. The mode solver x → tangential_1,
    # mode solver y → tangential_2.
    if normal_axis == 1  # x-normal, e1=Ey, e2=Ez, h1=Hy, h2=Hz
        if sym == :e1; return _squeeze_mode_field(mode.Ey, normal_axis)
        elseif sym == :e2; return _squeeze_mode_field(mode.Ez, normal_axis)
        elseif sym == :h1; return _squeeze_mode_field(mode.Hy, normal_axis)
        else; return _squeeze_mode_field(mode.Hz, normal_axis)
        end
    elseif normal_axis == 2  # y-normal, e1=Ex, e2=Ez, h1=Hx, h2=Hz
        if sym == :e1; return _squeeze_mode_field(mode.Ex, normal_axis)
        elseif sym == :e2; return _squeeze_mode_field(mode.Ez, normal_axis)
        elseif sym == :h1; return _squeeze_mode_field(mode.Hx, normal_axis)
        else; return _squeeze_mode_field(mode.Hz, normal_axis)
        end
    else  # z-normal, e1=Ex, e2=Ey, h1=Hx, h2=Hy
        if sym == :e1; return _squeeze_mode_field(mode.Ex, normal_axis)
        elseif sym == :e2; return _squeeze_mode_field(mode.Ey, normal_axis)
        elseif sym == :h1; return _squeeze_mode_field(mode.Hx, normal_axis)
        else; return _squeeze_mode_field(mode.Hy, normal_axis)
        end
    end
end

"""
    _squeeze_mode_field(f, normal_axis::Int)

Squeeze a mode field array to 2D by collapsing the singleton normal dimension,
then transpose so the result is indexed as `(mode.x, mode.y)`.

The mode solver internally stores fields in `(y, x)` order (rows = y-axis,
columns = x-axis). After collapsing the singleton normal dimension we must
transpose to match the `(mode.x, mode.y)` convention expected by
`bilinear_interpolator(mode_x, mode_y, field, point)`.

After `get_mode_profiles` applies `permutedims`, the singleton dimension
position depends on the normal axis:
- x-normal (YZ plane): singleton at dim 1 → f[1, :, :]
- y-normal (XZ plane): singleton at dim 2 → f[:, 1, :]
- z-normal (XY plane): singleton at dim 3 → f[:, :, 1]
"""
function _squeeze_mode_field(f, normal_axis::Int)
    if ndims(f) == 3
        squeezed = if normal_axis == 1
            f[1, :, :]
        elseif normal_axis == 2
            f[:, 1, :]
        else
            f[:, :, 1]
        end
        return permutedims(squeezed)
    elseif ndims(f) == 2
        return permutedims(f)
    else
        return reshape(f, 1, length(f))
    end
end

"""
    _extract_dft_fields(e1, e2, h1, h2, normal_axis, i1, i2, i_freq)

Extract the tangential DFT field values at a grid point, handling the
collapsed normal dimension.
"""
function _extract_dft_fields(e1, e2, h1, h2, normal_axis, i1, i2, i_freq)
    if normal_axis == 3
        et1 = ComplexF64(e1[i1, i2, 1, i_freq])
        et2 = ComplexF64(e2[i1, i2, 1, i_freq])
        ht1 = ComplexF64(h1[i1, i2, 1, i_freq])
        ht2 = ComplexF64(h2[i1, i2, 1, i_freq])
    elseif normal_axis == 1
        et1 = ComplexF64(e1[1, i1, i2, i_freq])
        et2 = ComplexF64(e2[1, i1, i2, i_freq])
        ht1 = ComplexF64(h1[1, i1, i2, i_freq])
        ht2 = ComplexF64(h2[1, i1, i2, i_freq])
    else  # normal_axis == 2
        et1 = ComplexF64(e1[i1, 1, i2, i_freq])
        et2 = ComplexF64(e2[i1, 1, i2, i_freq])
        ht1 = ComplexF64(h1[i1, 1, i2, i_freq])
        ht2 = ComplexF64(h2[i1, 1, i2, i_freq])
    end
    return (et1, et2, ht1, ht2)
end
