# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    Monitors.jl

Time domain monitors etc.
"""

# ------------------------------------------------------------ #
# Interface functions
# ------------------------------------------------------------ #

function init_monitors(sim::SimulationData, monitors)
    for m in monitors
        push!(sim.monitor_data, init_monitors(sim, m))
    end
    return
end

@inline function init_monitors(sim::SimulationData, ::Nothing)
    return
end

# ------------------------------------------------------------ #
# Time Monitor
# ------------------------------------------------------------ #

function init_monitors(sim::SimulationData, monitor::TimeMonitor)
    # compute grid volume from dimensions
    gv = GridVolume(
        sim,
        Volume(center = monitor.center, size = monitor.size),
        monitor.component,
    )

    # default to simulation timestep
    Δt = isnothing(monitor.Δt) ? sim.Δt : monitor.Δt

    # compute length of array
    N = trunc(Int, monitor.length / Δt)

    # preallocate
    monitor.monitor_data = TimeMonitorData{backend_array,complex_backend_array}(
        monitor.component,
        zeros(N, (get_gridvolume_dims(gv)[1:sim.ndims])...), # TODO extend to 3D
        gv,
        monitor.length,
        Δt,
        [1],
    )

    return monitor.monitor_data
end

#TODO add update and kernel functions for time monitor.

# ------------------------------------------------------------ #
# DFT Monitor
# ------------------------------------------------------------ #

function init_monitors(sim::SimulationData, monitor::DFTMonitor)
    # continuous space volume
    vol = Volume(center = monitor.center, size = monitor.size)

    # compute grid volume from dimensions
    gv = GridVolume(sim, vol, monitor.component)

    # Pre-compute constants
    min_corner = SVector{3}(get_min_corner(vol)...)
    max_corner = SVector{3}(get_max_corner(vol)...)
    vol_size = SVector{3}(vol.size...)
    Δ = SVector{3}(
        isnothing(sim.Δx) ? 0.0 : Float64(sim.Δx),
        isnothing(sim.Δy) ? 0.0 : Float64(sim.Δy),
        isnothing(sim.Δz) ? 0.0 : Float64(sim.Δz),
    )
    scale_factor = sim.Δt / sqrt(backend_number(2) * backend_number(π)) * monitor.decimation

    # The interpolation weight is separable: w(x,y) = w_x(x) * w_y(y).
    # Pre-compute 1D weight arrays per axis, then outer-product them.
    # This reduces O(Nx*Ny) expensive weight calls to O(Nx+Ny).
    origin = get_component_origin(sim, gv.component)
    gv_origin = get_min_corner(gv)

    wx = Vector{Float64}(undef, gv.Nx)
    for ix in 1:gv.Nx
        px = origin[1] + (ix + gv_origin[1] - 2) * sim.Δx
        p = SVector(px, 0.0, 0.0)
        lo = SVector(min_corner[1], 0.0, 0.0)
        hi = SVector(max_corner[1], 0.0, 0.0)
        vs = SVector(vol_size[1], 0.0, 0.0)
        d = SVector(Δ[1], 0.0, 0.0)
        wx[ix] = _compute_interpolation_weight_fast(p, lo, hi, vs, 1, d)
    end

    wy = Vector{Float64}(undef, gv.Ny)
    for iy in 1:gv.Ny
        py = origin[2] + (iy + gv_origin[2] - 2) * sim.Δy
        p = SVector(py, 0.0, 0.0)
        lo = SVector(min_corner[2], 0.0, 0.0)
        hi = SVector(max_corner[2], 0.0, 0.0)
        vs = SVector(vol_size[2], 0.0, 0.0)
        d = SVector(Δ[2], 0.0, 0.0)
        wy[iy] = _compute_interpolation_weight_fast(p, lo, hi, vs, 1, d)
    end

    # Build 2D scale array via outer product
    scale = zeros(gv.Nx, gv.Ny)
    @inbounds for iy in 1:gv.Ny
        sy = wy[iy] * scale_factor
        for ix in 1:gv.Nx
            scale[ix, iy] = wx[ix] * sy
        end
    end

    # preallocate
    monitor.monitor_data = DFTMonitorData{backend_array,complex_backend_array}(
        monitor.component,
        complex_backend_array(
            # Create an array that is (Nx, Ny, Nz, Nf) in size, where Nf is the
            # number of frequency points.
            zeros((get_gridvolume_dims(gv))..., length(monitor.frequencies)),
        ),
        scale,
        gv,
        backend_array(monitor.frequencies),
        monitor.decimation,
    )

    return monitor.monitor_data
end

function update_monitor(sim::SimulationData, monitor::DFTMonitorData, time::Real)
    ndrange = (monitor.gv.Nx, monitor.gv.Ny, monitor.gv.Nz)

    if isnothing(sim.chunk_data) || length(sim.chunk_data) == 1
        # Single chunk: use sim.fields directly (shared reference)
        # P.5: Field arrays are raw (no OffsetArray), offset includes +1 for ghost cell
        sim._cached_dft_kernel(
            monitor.fields,
            get_fields_from_component(sim, monitor.component),
            monitor.frequencies,
            monitor.gv.start_idx[1],
            monitor.gv.start_idx[2],
            monitor.gv.start_idx[3],
            complex_backend_number(-im * 2 * π * time),
            ndrange = ndrange,
        )
    else
        # Multi-chunk: find the chunk containing the monitor and read from it
        for chunk in sim.chunk_data
            chunk_field = _get_chunk_field(chunk, monitor.component)
            isnothing(chunk_field) && continue

            chunk_gv = chunk.spec.grid_volume
            chunk_comp_gv = _get_chunk_component_gv(sim, chunk_gv, monitor.component)

            # Compute overlap between monitor and chunk in global index space
            overlap_start = max.(monitor.gv.start_idx, chunk_comp_gv.start_idx)
            overlap_end = min.(monitor.gv.end_idx, chunk_comp_gv.end_idx)

            # Skip if no overlap
            any(overlap_end .< overlap_start) && continue

            overlap_size = overlap_end .- overlap_start .+ 1

            # Monitor-local base: which monitor index does the overlap start at
            mon_base = overlap_start .- monitor.gv.start_idx

            # Chunk-local offset: where in the chunk's field array
            # P.5: +1 for ghost cell in raw field arrays (no OffsetArray)
            chunk_offset = overlap_start .- chunk_gv.start_idx .+ 1

            update_dft_kernel_chunk = sim._cached_dft_chunk_kernel
            update_dft_kernel_chunk(
                monitor.fields,
                chunk_field,
                monitor.frequencies,
                mon_base[1], mon_base[2], mon_base[3],
                chunk_offset[1], chunk_offset[2], chunk_offset[3],
                complex_backend_number(-im * 2 * π * time),
                ndrange = tuple(overlap_size...),
            )
        end
    end
end

@kernel function update_dft_monitor_chunk!(
    monitor_fields::AbstractArray,
    chunk_fields::AbstractArray,
    frequencies::AbstractArray,
    mon_base_x::Int,
    mon_base_y::Int,
    mon_base_z::Int,
    chunk_offset_x::Int,
    chunk_offset_y::Int,
    chunk_offset_z::Int,
    time_fac::Number,
)
    ix, iy, iz = @index(Global, NTuple)

    # Map kernel index to monitor index and chunk field index
    mx = ix + mon_base_x
    my = iy + mon_base_y
    mz = iz + mon_base_z
    cx = ix + chunk_offset_x
    cy = iy + chunk_offset_y
    cz = iz + chunk_offset_z

    for k in eachindex(frequencies)
        monitor_fields[mx, my, mz, k] += (
            exp(frequencies[k] * time_fac) *
            chunk_fields[cx, cy, cz]
        )
    end
end

@kernel function update_dft_monitor!(
    monitor_fields::AbstractArray,
    sim_fields::AbstractArray,
    frequencies::AbstractArray,
    offset_x::Int,
    offset_y::Int,
    offset_z::Int,
    time_fac::Number,
)
    ix, iy, iz = @index(Global, NTuple)

    for k in eachindex(frequencies)
        monitor_fields[ix, iy, iz, k] += (
            exp(frequencies[k] * time_fac) *
            sim_fields[ix+offset_x, iy+offset_y, iz+offset_z]
        )
    end
end

get_dft_fields(monitor::DFTMonitor) = monitor.monitor_data.fields

# ------------------------------------------------------------ #
# Near2Far Monitor
# ------------------------------------------------------------ #

function init_monitors(sim::SimulationData, monitor::Near2FarMonitor)
    md = init_near2far_monitor(sim, monitor)

    # Initialize the internal DFT monitors and push their data into sim
    for dft_mon in md.tangential_E_monitors
        push!(sim.monitor_data, init_monitors(sim, dft_mon))
    end
    for dft_mon in md.tangential_H_monitors
        push!(sim.monitor_data, init_monitors(sim, dft_mon))
    end

    # Compute physical base positions (position of dft[1,1,1]) for each component.
    # These are the exact Yee grid positions where the DFT fields are sampled.
    Δ = SVector(sim.Δx, sim.Δy, sim.Δz)
    for (i, dft_mon) in enumerate(md.tangential_E_monitors)
        origin = get_component_origin(sim, dft_mon.monitor_data.component)
        gv = dft_mon.monitor_data.gv
        base = origin .+ (SVector{3,Float64}(gv.start_idx...) .- 1) .* Δ
        if i == 1
            md.e1_base = collect(base)
        else
            md.e2_base = collect(base)
        end
    end
    for (i, dft_mon) in enumerate(md.tangential_H_monitors)
        origin = get_component_origin(sim, dft_mon.monitor_data.component)
        gv = dft_mon.monitor_data.gv
        base = origin .+ (SVector{3,Float64}(gv.start_idx...) .- 1) .* Δ
        if i == 1
            md.h1_base = collect(base)
        else
            md.h2_base = collect(base)
        end
    end

    return md
end
