# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Mode monitor: records DFT fields on a cross-sectional plane and computes
# the overlap integral with waveguide modes to extract S-parameters.
#
# Pattern follows Near2Far.jl: 4 internal DFT monitors for tangential E/H,
# mode profiles solved at each frequency, overlap computed in post-processing.

export compute_mode_amplitudes, get_mode_transmission

# Cache for mode profiles keyed by cross-section geometry and frequency set.
# Monitors at the same position with different mode indices share a single cache entry.
# Value: (max_nev, all_modes) where all_modes[freq_idx][mode_idx] = Mode.
const _mode_profile_cache = Dict{UInt64, Tuple{Int, Vector{Vector{VectorModesolver.Mode}}}}()

function _mode_cache_key(monitor::ModeMonitor, normal_axis::Int)
    # The mode profile depends on size, normal_axis, geometry, mode_spec params, and frequencies.
    # It does NOT depend on num_modes (mode index), so monitors requesting different mode
    # indices at the same cross-section share work.
    #
    # Cache grouping via mode_group:
    #   :auto (default)  → key includes monitor center, so each position gets its own solve.
    #                      This is the safe default: no risk of sharing across different
    #                      cross-sections.
    #   any other Symbol → key includes the symbol instead of center, so ALL monitors with
    #                      the same mode_group share a single solve.  Use this when you know
    #                      multiple monitors see the same waveguide cross-section (e.g.
    #                      input/output of a straight waveguide).
    group = monitor.mode_spec.mode_group
    group_key = group === :auto ? monitor.center : group

    return hash((
        group_key,
        monitor.size,
        normal_axis,
        objectid(monitor.mode_spec.geometry),  # same geometry vector → same modes
        monitor.mode_spec.mode_solver_resolution,
        monitor.mode_spec.solver_tolerance,
        monitor.mode_spec.target_neff,
        monitor.mode_spec.num_mode_freqs,
        monitor.frequencies,
    ))
end

function clear_mode_cache!()
    empty!(_mode_profile_cache)
    empty!(_mode_max_nev)
end

# Pre-computed max mode index per cache key.  Populated by `prescan_mode_monitors!`
# before any `init_mode_monitor` calls so that the first solve for a cache key
# already requests enough modes for all monitors sharing that key.
const _mode_max_nev = Dict{UInt64, Int}()

"""
    prescan_mode_monitors!(monitors)

Pre-scan all `ModeMonitor`s and record, for each cache key, the maximum
`num_modes` (mode index) requested.  This allows the first call to
`init_mode_monitor` to solve for all needed modes at once, avoiding
repeated re-solves as progressively higher mode indices are encountered.
"""
function prescan_mode_monitors!(monitors)
    empty!(_mode_max_nev)
    for m in monitors
        if m isa ModeMonitor
            normal_axis = findfirst(x -> x == 0.0, m.size)
            isnothing(normal_axis) && continue
            key = _mode_cache_key(m, normal_axis)
            prev = get(_mode_max_nev, key, 0)
            _mode_max_nev[key] = max(prev, m.mode_spec.num_modes)
        end
    end
end

# ------------------------------------------------------------------- #
# Frequency interpolation for mode profiles
# ------------------------------------------------------------------- #

"""
    _interpolate_mode_profiles(coarse_freqs, coarse_modes, target_freqs)

Interpolate mode profiles solved at `coarse_freqs` to all `target_freqs`.
Uses cubic spline interpolation on each pixel of each field component.
"""
function _interpolate_mode_profiles(
    coarse_freqs::Vector{Float64},
    coarse_modes::Vector{VectorModesolver.Mode},
    target_freqs::Vector{Float64},
)
    n_coarse = length(coarse_freqs)
    n_target = length(target_freqs)

    # Get field shape from the first mode (all coarse modes share the same grid)
    ref = coarse_modes[1]
    field_shape = size(ref.Ex)  # e.g. (nx, ny, 1) or (1, nx, ny)
    n_pixels = prod(field_shape)

    # Stack coarse data
    neff_coarse = [m.neff for m in coarse_modes]
    lambda_coarse = [m.λ for m in coarse_modes]

    field_names = (:Ex, :Ey, :Ez, :Hx, :Hy, :Hz)

    # Flatten each field to a 1D vector of n_pixels for each coarse frequency
    # coarse_flat[fn] is (n_coarse, n_pixels) matrix
    coarse_flat = Dict{Symbol, Matrix{ComplexF64}}()
    for fn in field_names
        mat = zeros(ComplexF64, n_coarse, n_pixels)
        for ic in 1:n_coarse
            f = getfield(coarse_modes[ic], fn)
            mat[ic, :] .= vec(f)
        end
        coarse_flat[fn] = mat
    end

    # Phase alignment: eigenvectors have arbitrary global phase at each frequency.
    # Align all modes to have the same phase as the first mode at the pixel with
    # maximum amplitude in Ex (the dominant transverse component for TE modes).
    ref_ex = coarse_flat[:Ex]
    # Find the pixel with the largest amplitude across all coarse modes
    avg_amplitude = zeros(n_pixels)
    for ip in 1:n_pixels
        avg_amplitude[ip] = sum(abs.(ref_ex[:, ip]))
    end
    ref_pixel = argmax(avg_amplitude)

    # Compute phase correction for each coarse mode relative to the first
    ref_phase = angle(ref_ex[1, ref_pixel])
    for ic in 2:n_coarse
        phase_diff = angle(ref_ex[ic, ref_pixel]) - ref_phase
        correction = exp(-im * phase_diff)
        for fn in field_names
            coarse_flat[fn][ic, :] .*= correction
        end
    end

    # Interpolate neff and lambda
    neff_itp = Interpolations.interpolate((coarse_freqs,), neff_coarse, Interpolations.Gridded(Interpolations.Linear()))
    lambda_itp = Interpolations.interpolate((coarse_freqs,), lambda_coarse, Interpolations.Gridded(Interpolations.Linear()))

    # Interpolate each pixel of each field component
    target_flat = Dict{Symbol, Matrix{ComplexF64}}()
    for fn in field_names
        src = coarse_flat[fn]
        dst = zeros(ComplexF64, n_target, n_pixels)
        for ip in 1:n_pixels
            re_vals = real.(src[:, ip])
            im_vals = imag.(src[:, ip])
            re_itp = Interpolations.interpolate((coarse_freqs,), re_vals, Interpolations.Gridded(Interpolations.Linear()))
            im_itp = Interpolations.interpolate((coarse_freqs,), im_vals, Interpolations.Gridded(Interpolations.Linear()))
            for it in 1:n_target
                dst[it, ip] = re_itp(target_freqs[it]) + im * im_itp(target_freqs[it])
            end
        end
        target_flat[fn] = dst
    end

    # Build Mode objects for each target frequency
    profiles = VectorModesolver.Mode[]
    for it in 1:n_target
        mode = VectorModesolver.Mode(
            λ = lambda_itp(target_freqs[it]),
            neff = neff_itp(target_freqs[it]),
            x = collect(ref.x),
            y = collect(ref.y),
            Ex = reshape(target_flat[:Ex][it, :], field_shape),
            Ey = reshape(target_flat[:Ey][it, :], field_shape),
            Ez = reshape(target_flat[:Ez][it, :], field_shape),
            Hx = reshape(target_flat[:Hx][it, :], field_shape),
            Hy = reshape(target_flat[:Hy][it, :], field_shape),
            Hz = reshape(target_flat[:Hz][it, :], field_shape),
        )
        push!(profiles, mode)
    end

    return profiles
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

    # Solve for mode profiles using sweep solver (or reuse from cache).
    # The cache stores ALL modes up to max_nev, so monitors requesting
    # different mode indices at the same cross-section share work.
    mode_spec = monitor.mode_spec
    geometry = isempty(mode_spec.geometry) ? sim.geometry : mode_spec.geometry
    cache_key = _mode_cache_key(monitor, normal_axis)
    needed_nev = mode_spec.num_modes  # mode index this monitor needs

    cached = get(_mode_profile_cache, cache_key, nothing)

    if !isnothing(cached) && cached[1] >= needed_nev
        # Cache hit with enough modes already solved
        all_modes = cached[2]
    else
        # Cache miss or need more modes than previously solved.
        # Use pre-scanned max_nev so we solve for ALL modes any monitor
        # sharing this cache key will need — avoiding redundant re-solves.
        prev_nev = isnothing(cached) ? 0 : cached[1]
        prescanned = get(_mode_max_nev, cache_key, needed_nev)
        max_nev = max(needed_nev, prev_nev, prescanned)

        all_freqs = Float64.(monitor.frequencies)
        n_freqs = length(all_freqs)
        n_coarse = mode_spec.num_mode_freqs

        if n_coarse < 2 || n_coarse >= n_freqs
            # Full solve at every frequency using sweep solver
            all_modes = get_mode_profiles_sweep(
                frequencies = all_freqs,
                max_mode_index = max_nev,
                mode_solver_resolution = mode_spec.mode_solver_resolution,
                center = copy(monitor.center),
                size = copy(monitor.size),
                solver_tolerance = mode_spec.solver_tolerance,
                geometry = geometry,
                target_neff = mode_spec.target_neff,
            )
        else
            # Coarse solve + per-mode-index interpolation
            coarse_indices = round.(Int, range(1, n_freqs, length = n_coarse))
            coarse_freqs = all_freqs[coarse_indices]

            @info("  Mode interpolation: solving at $n_coarse of $n_freqs frequencies")

            coarse_modes = get_mode_profiles_sweep(
                frequencies = coarse_freqs,
                max_mode_index = max_nev,
                mode_solver_resolution = mode_spec.mode_solver_resolution,
                center = copy(monitor.center),
                size = copy(monitor.size),
                solver_tolerance = mode_spec.solver_tolerance,
                geometry = geometry,
                target_neff = mode_spec.target_neff,
            )

            # Interpolate each mode index separately (phase alignment is per-mode)
            all_modes = [Vector{VectorModesolver.Mode}(undef, max_nev) for _ in 1:n_freqs]
            for mi in 1:max_nev
                mi_coarse = [coarse_modes[fi][mi] for fi in 1:n_coarse]
                mi_interp = _interpolate_mode_profiles(coarse_freqs, mi_coarse, all_freqs)
                for fi in 1:n_freqs
                    all_modes[fi][mi] = mi_interp[fi]
                end
            end
        end

        _mode_profile_cache[cache_key] = (max_nev, all_modes)
    end

    # Select the mode index this monitor needs
    mode_profiles = VectorModesolver.Mode[]
    for fi in eachindex(monitor.frequencies)
        n_available = length(all_modes[fi])
        if n_available < needed_nev
            error("Mode solver found only $n_available modes at frequency " *
                  "$(monitor.frequencies[fi]), but mode index $needed_nev was requested. " *
                  "Try reducing num_modes or setting target_neff in ModeSpec.")
        end
        push!(mode_profiles, all_modes[fi][needed_nev])
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
function compute_mode_amplitudes(md::ModeMonitorData; verbose::Bool = false)
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

    if verbose
        println("  [compute_mode_amplitudes] verbose diagnostics:")
        println("    normal_axis = $normal_axis")
        println("    n1 = $n1, n2 = $n2, dA = $dA")
        println("    d1 = $d1, d2 = $d2")
        println("    DFT field shapes: e1=$(size(e1_fields)), e2=$(size(e2_fields)), h1=$(size(h1_fields)), h2=$(size(h2_fields))")
        println("    e1_base = $(md.e1_base), h2_base = $(md.h2_base)")
        # Check max field values
        println("    max|e1_fields| = $(maximum(abs.(e1_fields)))")
        println("    max|h2_fields| = $(maximum(abs.(h2_fields)))")
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

        if verbose && (i_freq == 1 || i_freq == (n_freq + 1) ÷ 2 || i_freq == n_freq)
            println("    freq[$i_freq] = $(md.frequencies[i_freq]):")
            println("      P_mode = $P_mode")
            println("      |overlap_plus| = $(abs(overlap_plus))")
            println("      |a+| = $(abs(a_plus[i_freq]))")
            println("      mode_e1 max = $(maximum(abs.(mode_e1)))")
            println("      mode_h2 max = $(maximum(abs.(mode_h2)))")
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
