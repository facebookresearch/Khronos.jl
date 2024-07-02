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

    # Get grid weights
    scale = create_array_from_gridvolume(sim, gv)
    for i_3 in axes(scale, 3)
        for i_2 in axes(scale, 2)
            for i_1 in axes(scale, 1)
                point = grid_volume_idx_to_point(sim, gv, [i_1, i_2, i_3])
                scale[i_1, i_2] =
                    compute_interpolation_weight(
                        point,
                        vol,
                        sim.ndims,
                        sim.Δx,
                        sim.Δy,
                        sim.Δz,
                    ) * sim.Δt / sqrt(2.0 * π) * monitor.decimation
            end
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
    update_dft_kernel = update_dft_monitor!(backend_engine)

    ndrange = (monitor.gv.Nx, monitor.gv.Ny, monitor.gv.Nz)
    update_dft_kernel(
        monitor.fields,
        get_fields_from_component(sim, monitor.component),
        monitor.frequencies,
        monitor.gv.start_idx[1] - 1,
        monitor.gv.start_idx[2] - 1,
        monitor.gv.start_idx[3] - 1,
        complex_backend_number(-im * 2 * π * time),
        ndrange = ndrange,
    )
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
