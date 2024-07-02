# (c) Meta Platforms, Inc. and affiliates.
#
# Functions to visualize the simulation domain, spatial/temporal source
# profiles, and monitor data.

using CairoMakie
using Statistics

export plot2D, plot_monitor, plot_timesource, plot_source

Makie.inline!(true)

"""
    plot2D(
    sim::SimulationData,
    component::Field,
    vol::Volume;
    plot_geometry::Bool = false,
    symmetric_field_scaling::Bool = true,
)

Plot each of the three 2D cross sections (XY, XZ, YZ) of a simulation (`sim`) as
described by `vol`.
"""
function plot2D(
    sim::SimulationData,
    component::Union{Field,Nothing},
    vol::Volume;
    plot_geometry::Bool = false,
    symmetric_field_scaling::Bool = true,
)
    prepare_simulation!(sim)

    f = Figure()

    vol_size = vol.size

    # Store all the fields plot configurations in a running dict for convenience.
    field_args = Dict()

    # Prepare fields cross sections
    if !isnothing(component)
        # process requested field components
        (XY_slice, XZ_slice, YZ_slice) = _pull_field_slices(sim, component, vol)
        vmin = min(minimum(YZ_slice), minimum(XZ_slice), minimum(XY_slice))
        vmax = max(maximum(YZ_slice), maximum(XZ_slice), maximum(XY_slice))
        vmax = max(abs(vmin), vmax)
        vmin = -vmax
        fields_alpha = 1.0

        field_args[:transparency] = true
        field_args[:colormap] = :bluesreds
        if symmetric_field_scaling
            field_args[:colorrange] = (vmin, vmax)
        end
    end

    # Prepare geometry cross sections
    if plot_geometry
        # for now use εx... although we should generalize this
        (eps_XY_slice, eps_XZ_slice, eps_YZ_slice) = _pull_field_slices(sim, εx(), vol)
        eps_vmin = min(minimum(eps_YZ_slice), minimum(eps_XZ_slice), minimum(eps_XY_slice))
        eps_vmax = max(maximum(eps_YZ_slice), maximum(eps_XZ_slice), maximum(eps_XY_slice))
        fields_alpha = 0.4
    end

    # XY cross section
    ax_xy = Axis(f[1, 1], title = "XY", xlabel = "X", ylabel = "Y", aspect = DataAspect())
    if plot_geometry
        xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(eps_XY_slice)[1])
        ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(eps_XY_slice)[2])
        heatmap!(
            ax_xy,
            xs,
            ys,
            eps_XY_slice,
            colormap = :binary,
            colorrange = (eps_vmin, eps_vmax),
        )
    end
    if !isnothing(component)
        xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(XY_slice)[1])
        ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(XY_slice)[2])
        hm = heatmap!(ax_xy, xs, ys, XY_slice; field_args...)
        #Colorbar(f[:, 2], hm)
    end

    # XZ cross section
    ax_xz = Axis(f[1, 2], title = "XZ", xlabel = "X", ylabel = "Z", aspect = DataAspect())
    if plot_geometry
        xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(eps_XZ_slice)[1])
        zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(eps_XZ_slice)[2])
        heatmap!(
            ax_xz,
            xs,
            zs,
            eps_XZ_slice,
            colormap = :binary,
            colorrange = (eps_vmin, eps_vmax),
        )
    end
    if !isnothing(component)
        xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(XZ_slice)[1])
        zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(XZ_slice)[2])
        hm = heatmap!(ax_xz, xs, zs, XZ_slice; field_args...)
        #Colorbar(f[:, 4], hm)
    end

    # YZ cross section
    ax_yz = Axis(f[1, 3], title = "YZ", xlabel = "Y", ylabel = "Z", aspect = DataAspect())
    if plot_geometry
        ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(eps_YZ_slice)[1])
        zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(eps_YZ_slice)[2])
        heatmap!(
            ax_yz,
            ys,
            zs,
            eps_YZ_slice,
            colormap = :binary,
            colorrange = (eps_vmin, eps_vmax),
        )
    end
    if !isnothing(component)
        ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(YZ_slice)[1])
        zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(YZ_slice)[2])
        hm = heatmap!(ax_yz, ys, zs, YZ_slice; field_args...)
        #Colorbar(f[:, end+1], hm)
    end

    return f
end

"""
    plot_monitor(monitor::DFTMonitor, frequency_idx::Int)

Plot the field response of a `monitor` at a specified `frequency_idx`.
"""
function plot_monitor(monitor::DFTMonitor, frequency_idx::Int)
    freq_data = Base.Array(get_dft_fields(monitor)[:, :, :, frequency_idx])
    freq_data = _pull_2D_slice_from_3D(freq_data)

    freq_data = real(freq_data)

    f = Figure()
    ax_xy = Axis(f[1, 1], aspect = DataAspect())

    fields_vmin = minimum(freq_data)
    fields_vmax = maximum(freq_data)
    fields_vmax = max(abs(fields_vmin), abs(fields_vmax))
    fields_vmin = -fields_vmax

    #TODO check if max==min

    hm = heatmap!(
        ax_xy,
        freq_data,
        colormap = :bluesreds,
        colorrange = (fields_vmin, fields_vmax),
    )
    Colorbar(f[:, end+1], hm)

    resize_to_layout!(f)
    return f
end

"""
    plot_timesource(sim::SimulationData, time_source::TimeSource, frequencies)

Plot the time and frequency response of a time-dependent source.
"""
function plot_timesource(sim::SimulationData, time_source::TimeSource, frequencies)
    prepare_simulation!(sim)

    # Pull the time-series data
    t = range(start = 0, stop = get_cutoff(time_source), step = sim.Δt)
    src_amplitude = zeros(size(t))
    for n in eachindex(t)
        src_amplitude[n] = real(eval_time_source(time_source, t[n]))
    end

    # Compute the DTFT
    fourier_amplitude = DTFT(t, real.(src_amplitude), frequencies)
    PSD = abs.(fourier_amplitude) .^ 2
    PSD = PSD / maximum(PSD)

    # Set up the plot
    f = Figure(; size = (800, 400))
    ax1 = Axis(f[1, 1], xlabel = "Time (s)", ylabel = "Source amplitude")
    lines!(ax1, t, real.(src_amplitude), color = :red)
    ax2 = Axis(
        f[1, 2],
        yscale = log10,
        xlabel = "Wavelength",
        ylabel = "Power spectral density (a.u.)",
    )
    lines!(ax2, 1 ./ frequencies, PSD, color = :blue)

    resize_to_layout!(f)

    return f
end

"""
    plot_timesource(sim::SimulationData, source::Source, frequencies)

Plot the time and frequency response of a time-dependent source.
"""
function plot_timesource(sim::SimulationData, source::Source, frequencies)
    plot_timesource(sim, get_time_profile(source), frequencies)
end

function plot_source(sim::SimulationData, source::Source)
    prepare_simulation!(sim)
    source_volume = Volume(center = source.center, size = source.size)
    transverse_components = get_plane_transverse_fields(source_volume)

    f = Figure()
    num_cols = 2

    for (i, current_field) in enumerate(transverse_components)
        row = div(i - 1, num_cols) + 1
        col = (i - 1) % num_cols + 1

        ax = Axis(f[row, col*2-1])  # Position axis in every second slot for heatmaps
        x_range, y_range = _get_slice_axes(source.fields[current_field], source_volume)
        field_profile = _pull_2D_slice_from_3D(source.fields[current_field])
        hm = heatmap!(ax, real(field_profile), colormap = :bluesreds)
        cb = Colorbar(f[row, col*2], hm, width = Relative(1 / 8))
    end

    return f

end

# ---------------------------------------------------- #
# Utility functions
# ---------------------------------------------------- #

"""
    _pull_2D_slice_from_3D(data::AbstractArray)::AbstractArray

Given a 3D array with one singeleton dimension, indexes out the corresponding
plane (e.g. for plotting). Similar to a `squeeze`, but safer.
"""
function _pull_2D_slice_from_3D(data::AbstractArray)::AbstractArray
    if size(data)[1] == 1
        data = data[1, :, :]
    elseif size(data)[2] == 1
        data = data[:, 1, :]
    elseif size(data)[3] == 1
        data = data[:, :, 1]
    else
        error("Invalid 3D array (size: $(size(data)))")
    end

end

function _get_slice_axes(data::AbstractArray, vol::Volume)
    if size(data)[1] == 1
        # YZ plane
        idx_x = 2
        idx_y = 3
    elseif size(data)[2] == 1
        idx_x = 1
        idx_y = 3
    elseif size(data)[3] == 1
        idx_x = 1
        idx_y = 2
    else
        error("Invalid 3D array (size: $(size(data)))")
    end
    function _custom_range(idx_dim)
        return collect(
            range(
                -vol.size[idx_dim] / 2.0 + vol.center[idx_dim],
                vol.size[idx_dim] / 2.0 + vol.center[idx_dim],
                length = size(data)[idx_dim],
            ),
        )
    end
    x_range = _custom_range(idx_x)
    y_range = _custom_range(idx_y)
    return (x_range, y_range)
end

function _get_plane_ranges(gv::GridVolume)
    x_range = gv.start_idx[1]:gv.end_idx[1]
    y_range = gv.start_idx[2]:gv.end_idx[2]
    z_range = gv.start_idx[3]:gv.end_idx[3]
    return (x_range, y_range, z_range)
end

function _pull_fields_from_device(sim::SimulationData, component::Field)
    current_fields = get_fields_from_component(sim, component)
    array_range = get_component_voxel_count(sim, component)
    # Index out the ghost cells and collect to the host
    current_fields = Base.Array(
        collect(current_fields[1:array_range[1], 1:array_range[2], 1:array_range[3]]),
    )
end

function _pull_field_slices(sim::SimulationData, component::Field, vol::Volume)
    current_fields = _pull_fields_from_device(sim, component)

    vol_x = Volume(vol.center, [0, vol.size[2], vol.size[3]])
    vol_y = Volume(vol.center, [vol.size[1], 0, vol.size[3]])
    vol_z = Volume(vol.center, [vol.size[1], vol.size[2], 0])

    gv_x = GridVolume(sim, vol_x, component)
    gv_y = GridVolume(sim, vol_y, component)
    gv_z = GridVolume(sim, vol_z, component)

    YZ_slice = mean(current_fields[_get_plane_ranges(gv_x)...], dims = 1)[1, :, :]
    XZ_slice = mean(current_fields[_get_plane_ranges(gv_y)...], dims = 2)[:, 1, :]
    XY_slice = mean(current_fields[_get_plane_ranges(gv_z)...], dims = 3)[:, :, 1]

    return (XY_slice, XZ_slice, YZ_slice)
end
