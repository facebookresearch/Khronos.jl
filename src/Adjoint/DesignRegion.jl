# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
    DesignRegion

A parameterized region of the simulation whose permittivity is controlled by
design parameters ρ ∈ [0,1]. Used for density-based topology optimization.

The design grid is independent of the simulation (Yee) grid. Bilinear
interpolation maps design parameters onto each Yee grid component location,
and the transpose of this interpolation is used in the adjoint gradient
contraction step.
"""
@with_kw mutable struct DesignRegion
    volume::Volume
    design_parameters::Vector{Float64}
    grid_size::Tuple{Int,Int}          # (Nx_design, Ny_design)
    ε_min::Float64 = 1.0               # permittivity when ρ=0
    ε_max::Float64 = 12.0              # permittivity when ρ=1

    # Computed during init_design_region!
    # Sparse interpolation weights from design grid -> each Yee component
    interp_weights_Ex::Union{Nothing, Vector{Tuple{Int,Int,Float64}}} = nothing
    interp_weights_Ey::Union{Nothing, Vector{Tuple{Int,Int,Float64}}} = nothing
    interp_weights_Ez::Union{Nothing, Vector{Tuple{Int,Int,Float64}}} = nothing

    # Grid volumes for the design region on each Yee component
    gv_Ex::Union{Nothing, GridVolume} = nothing
    gv_Ey::Union{Nothing, GridVolume} = nothing
    gv_Ez::Union{Nothing, GridVolume} = nothing

    # DFT monitors installed on the design region (for gradient computation)
    design_monitors::Union{Nothing, Vector{DFTMonitor}} = nothing
end

"""
    _design_grid_coords(dr::DesignRegion)

Return the 1D coordinate arrays for the design grid.
"""
function _design_grid_coords(dr::DesignRegion)
    vol = dr.volume
    Nx, Ny = dr.grid_size
    x_min = vol.center[1] - vol.size[1] / 2
    x_max = vol.center[1] + vol.size[1] / 2
    y_min = vol.center[2] - vol.size[2] / 2
    y_max = vol.center[2] + vol.size[2] / 2

    xs = Nx > 1 ? collect(range(x_min, x_max, length=Nx)) : [vol.center[1]]
    ys = Ny > 1 ? collect(range(y_min, y_max, length=Ny)) : [vol.center[2]]

    return xs, ys
end

"""
    _compute_bilinear_weights(design_xs, design_ys, sim, component, dr)

Compute bilinear interpolation weights mapping design grid nodes to Yee grid
positions for a specific field component. Returns a vector of
(yee_linear_idx, design_linear_idx, weight) tuples.
"""
function _compute_bilinear_weights(
    design_xs::Vector{Float64},
    design_ys::Vector{Float64},
    sim::SimulationData,
    component::Field,
    dr::DesignRegion,
)
    gv = GridVolume(sim, dr.volume, component)

    Δx = _scalar_spacing(sim.Δx)
    Δy = _scalar_spacing(sim.Δy)
    origin = get_component_origin(sim, component)
    gv_origin = get_min_corner(gv)

    Nx_design = length(design_xs)
    Ny_design = length(design_ys)

    weights = Tuple{Int,Int,Float64}[]

    for iy_yee in 1:gv.Ny, ix_yee in 1:gv.Nx
        # Physical position of this Yee grid point
        px = origin[1] + (ix_yee + gv_origin[1] - 2) * Δx
        py = origin[2] + (iy_yee + gv_origin[2] - 2) * Δy

        yee_idx = (iy_yee - 1) * gv.Nx + ix_yee

        # Find the bounding design grid cell
        # Clamp to design grid bounds
        if Nx_design == 1
            ix_lo = ix_hi = 1
            wx = 1.0
        else
            dx_design = design_xs[2] - design_xs[1]
            fx = (px - design_xs[1]) / dx_design + 1.0
            fx = clamp(fx, 1.0, Float64(Nx_design))
            ix_lo = min(floor(Int, fx), Nx_design - 1)
            ix_hi = ix_lo + 1
            wx = fx - ix_lo  # weight for ix_hi
        end

        if Ny_design == 1
            iy_lo = iy_hi = 1
            wy = 1.0
        else
            dy_design = design_ys[2] - design_ys[1]
            fy = (py - design_ys[1]) / dy_design + 1.0
            fy = clamp(fy, 1.0, Float64(Ny_design))
            iy_lo = min(floor(Int, fy), Ny_design - 1)
            iy_hi = iy_lo + 1
            wy = fy - iy_lo
        end

        # Four corners of the bilinear interpolation
        corners = [
            ((iy_lo - 1) * Nx_design + ix_lo, (1 - wx) * (1 - wy)),
            ((iy_lo - 1) * Nx_design + ix_hi, wx * (1 - wy)),
            ((iy_hi - 1) * Nx_design + ix_lo, (1 - wx) * wy),
            ((iy_hi - 1) * Nx_design + ix_hi, wx * wy),
        ]

        for (design_idx, w) in corners
            if abs(w) > 1e-15
                push!(weights, (yee_idx, design_idx, w))
            end
        end
    end

    return weights
end

"""
    init_design_region!(sim::SimulationData, dr::DesignRegion)

Initialize a DesignRegion: compute interpolation weights from the design grid
to each Yee grid component. Must be called after prepare_simulation!.
"""
function init_design_region!(sim::SimulationData, dr::DesignRegion)
    design_xs, design_ys = _design_grid_coords(dr)

    dr.gv_Ex = GridVolume(sim, dr.volume, Ex())
    dr.gv_Ey = GridVolume(sim, dr.volume, Ey())
    dr.gv_Ez = GridVolume(sim, dr.volume, Ez())

    dr.interp_weights_Ex = _compute_bilinear_weights(design_xs, design_ys, sim, Ex(), dr)
    dr.interp_weights_Ey = _compute_bilinear_weights(design_xs, design_ys, sim, Ey(), dr)
    dr.interp_weights_Ez = _compute_bilinear_weights(design_xs, design_ys, sim, Ez(), dr)

    return dr
end

"""
    _interpolate_design_to_yee(rho, weights, n_yee)

Interpolate design parameters onto the Yee grid using pre-computed weights.
Returns a 1D array of interpolated values at each Yee grid point.
"""
function _interpolate_design_to_yee(
    rho::Vector{Float64},
    weights::Vector{Tuple{Int,Int,Float64}},
    n_yee::Int,
)
    result = zeros(Float64, n_yee)
    for (yee_idx, design_idx, w) in weights
        result[yee_idx] += rho[design_idx] * w
    end
    return result
end

"""
    _rho_to_epsilon_inv(rho_interp, ε_min, ε_max)

Map interpolated design parameters [0,1] to inverse permittivity.
Uses linear interpolation: ε(ρ) = ε_min + ρ*(ε_max - ε_min).
"""
function _rho_to_epsilon_inv(rho_interp::Vector{Float64}, ε_min::Float64, ε_max::Float64)
    ε = ε_min .+ rho_interp .* (ε_max - ε_min)
    return 1.0 ./ ε
end

"""
    update_design!(sim::SimulationData, dr::DesignRegion, rho::Vector{Float64})

Update the simulation's permittivity arrays based on new design parameters.
Interpolates rho to each Yee component, converts to ε_inv, and uploads to GPU.
"""
function update_design!(sim::SimulationData, dr::DesignRegion, rho::Vector{Float64})
    dr.design_parameters .= rho

    # Interpolate to each Yee component
    n_Ex = dr.gv_Ex.Nx * dr.gv_Ex.Ny
    n_Ey = dr.gv_Ey.Nx * dr.gv_Ey.Ny
    n_Ez = dr.gv_Ez.Nx * dr.gv_Ez.Ny

    rho_Ex = _interpolate_design_to_yee(rho, dr.interp_weights_Ex, n_Ex)
    rho_Ey = _interpolate_design_to_yee(rho, dr.interp_weights_Ey, n_Ey)
    rho_Ez = _interpolate_design_to_yee(rho, dr.interp_weights_Ez, n_Ez)

    # Convert to inverse permittivity
    ε_inv_Ex = _rho_to_epsilon_inv(rho_Ex, dr.ε_min, dr.ε_max)
    ε_inv_Ey = _rho_to_epsilon_inv(rho_Ey, dr.ε_min, dr.ε_max)
    ε_inv_Ez = _rho_to_epsilon_inv(rho_Ez, dr.ε_min, dr.ε_max)

    # Reshape to 2D grid
    ε_inv_Ex_2d = reshape(ε_inv_Ex, dr.gv_Ex.Nx, dr.gv_Ex.Ny)
    ε_inv_Ey_2d = reshape(ε_inv_Ey, dr.gv_Ey.Nx, dr.gv_Ey.Ny)
    ε_inv_Ez_2d = reshape(ε_inv_Ez, dr.gv_Ez.Nx, dr.gv_Ez.Ny)

    # Write to simulation geometry arrays (and per-chunk arrays)
    _write_design_to_sim!(sim, dr, ε_inv_Ex_2d, ε_inv_Ey_2d, ε_inv_Ez_2d)

    return nothing
end

"""
    _write_design_to_sim!(sim, dr, ε_inv_Ex, ε_inv_Ey, ε_inv_Ez)

Write the design region's ε_inv values into the simulation's geometry arrays.
Handles both single-chunk and multi-chunk cases.
"""
function _write_design_to_sim!(
    sim::SimulationData,
    dr::DesignRegion,
    ε_inv_Ex::AbstractMatrix,
    ε_inv_Ey::AbstractMatrix,
    ε_inv_Ez::AbstractMatrix,
)
    # For each chunk, find the overlapping subvolume and update
    for chunk in sim.chunk_data
        chunk_gv = chunk.spec.grid_volume

        for (gv, ε_inv_new, ε_inv_arr) in [
            (dr.gv_Ex, ε_inv_Ex, chunk.geometry_data.ε_inv_x),
            (dr.gv_Ey, ε_inv_Ey, chunk.geometry_data.ε_inv_y),
            (dr.gv_Ez, ε_inv_Ez, chunk.geometry_data.ε_inv_z),
        ]
            isnothing(ε_inv_arr) && continue

            # Compute overlap between design region GridVolume and chunk
            overlap_start = max.(gv.start_idx, chunk_gv.start_idx)
            overlap_end = min.(gv.end_idx, chunk_gv.end_idx)

            # Skip if no overlap
            any(overlap_end .< overlap_start) && continue

            # Design region local indices
            dr_start = overlap_start .- gv.start_idx .+ 1
            dr_end = overlap_end .- gv.start_idx .+ 1

            # Chunk local indices for geometry arrays (NO ghost cell offset;
            # geometry arrays are ndrange-sized, not ndrange+2 like field arrays)
            chunk_start = overlap_start .- chunk_gv.start_idx .+ 1
            chunk_end = overlap_end .- chunk_gv.start_idx .+ 1

            # Clamp to actual array dimensions to avoid out-of-bounds
            arr_dims = collect(size(ε_inv_arr))
            chunk_start = max.(chunk_start, 1)
            chunk_end = min.(chunk_end, arr_dims)
            any(chunk_end .< chunk_start) && continue

            # Copy the slice to CPU, modify, upload
            if sim.ndims == 2
                slice_gpu = backend_array(backend_number.(ε_inv_new[dr_start[1]:dr_end[1], dr_start[2]:dr_end[2]]))
                # Direct array write for CPU backend, copyto! for GPU
                ε_inv_arr[chunk_start[1]:chunk_end[1], chunk_start[2]:chunk_end[2]] .= slice_gpu
            else
                # 3D case: design region is 2D, extruded in z
                slice_gpu = backend_array(backend_number.(ε_inv_new[dr_start[1]:dr_end[1], dr_start[2]:dr_end[2]]))
                for iz_chunk in chunk_start[3]:chunk_end[3]
                    ε_inv_arr[chunk_start[1]:chunk_end[1], chunk_start[2]:chunk_end[2], iz_chunk] .= slice_gpu
                end
            end
        end
    end

    return nothing
end

"""
    install_design_region_monitors!(sim, dr, frequencies)

Install DFT monitors for the forward and adjoint fields at the design region.
Three monitors are created (one for each E-field component: Ex, Ey, Ez)
using the existing DFT infrastructure.
"""
function install_design_region_monitors!(
    sim::SimulationData,
    dr::DesignRegion,
    frequencies::Vector{Float64},
)
    dr.design_monitors = DFTMonitor[]

    for comp in [Ex(), Ey(), Ez()]
        mon = DFTMonitor(
            component = comp,
            center = dr.volume.center,
            size = dr.volume.size,
            frequencies = frequencies,
        )
        push!(dr.design_monitors, mon)
        # Initialize the monitor and push its data into sim.monitor_data
        push!(sim.monitor_data, init_monitors(sim, mon))
    end

    return dr.design_monitors
end
