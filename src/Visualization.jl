# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using CairoMakie
using Statistics

Makie.inline!(true)

function plot2D(
    sim::SimulationData,
    component::Field,
    center::Vector{Real},
    size::Vector{Real},
)
    prepare_simulation!(sim)
end

function _get_plane_ranges(gv::GridVolume)
    x_range = gv.start_idx[1]:gv.end_idx[1]
    y_range = gv.start_idx[2]:gv.end_idx[2]
    z_range = gv.start_idx[3]:gv.end_idx[3]
    return (x_range, y_range, z_range)
end

function _pull_field_slices(sim::SimulationData, component::Field, vol::Volume)
    current_fields = Base.Array(get_fields_from_component(sim, component))

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

"""
    plot2D(
    sim::SimulationData,
    component::Field,
    vol::Volume;
    plot_geometry::Bool = false,
    symmetric_field_scaling::Bool = true,
)

TBW
"""
function plot2D(
    sim::SimulationData,
    component::Field,
    vol::Volume;
    plot_geometry::Bool = false,
    symmetric_field_scaling::Bool = true,
)
    prepare_simulation!(sim)

    vol_size = vol.size

    # process requested field components
    (XY_slice, XZ_slice, YZ_slice) = _pull_field_slices(sim, component, vol)
    vmin = min(minimum(YZ_slice), minimum(XZ_slice), minimum(XY_slice))
    vmax = max(maximum(YZ_slice), maximum(XZ_slice), maximum(XY_slice))
    vmax = max(abs(vmin), vmax)
    vmin = -vmax
    fields_alpha = 1.0

    if plot_geometry
        # for now use εx... although we should generalize this
        (eps_XY_slice, eps_XZ_slice, eps_YZ_slice) = _pull_field_slices(sim, εx(), vol)
        eps_vmin = min(minimum(eps_YZ_slice), minimum(eps_XZ_slice), minimum(eps_XY_slice))
        eps_vmax = max(maximum(eps_YZ_slice), maximum(eps_XZ_slice), maximum(eps_XY_slice))
        fields_alpha = 0.4
    end

    # if component isa ε
    #     colormap = :binary
    # else:
    #     colormap = :bluesreds
    # end
    # bluesreds

    field_args = Dict()
    field_args[:transparency] = true
    field_args[:colormap] = :bluesreds
    if symmetric_field_scaling
        field_args[:colorrange] = (vmin, vmax)
    end

    f = Figure(size = (1200, 100))

    ax_xy = Axis(f[1, 1], title = "XY", xlabel = "X", ylabel = "Y", aspect = DataAspect())
    if plot_geometry
        heatmap!(ax_xy, eps_XY_slice, colormap = :binary, colorrange = (eps_vmin, eps_vmax))
    end
    xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(XY_slice)[1])
    ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(XY_slice)[2])
    hm = heatmap!(ax_xy, xs, ys, XY_slice; field_args...)
    Colorbar(f[:, 2], hm)

    ax_xz = Axis(f[1, 3], title = "XZ", xlabel = "X", ylabel = "Z", aspect = DataAspect())
    if plot_geometry
        heatmap!(ax_xz, eps_XZ_slice, colormap = :binary, colorrange = (eps_vmin, eps_vmax))
    end
    xs = range(-vol_size[1] / 2, vol_size[1] / 2, length = size(XZ_slice)[1])
    zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(XZ_slice)[2])
    hm = heatmap!(ax_xz, xs, zs, XZ_slice; field_args...)
    Colorbar(f[:, 4], hm)

    ax_yz = Axis(f[1, 5], title = "YZ", xlabel = "Y", ylabel = "Z", aspect = DataAspect())
    if plot_geometry
        heatmap!(ax_yz, eps_YZ_slice, colormap = :binary, colorrange = (eps_vmin, eps_vmax))
    end
    ys = range(-vol_size[2] / 2, vol_size[2] / 2, length = size(YZ_slice)[1])
    zs = range(-vol_size[3] / 2, vol_size[3] / 2, length = size(YZ_slice)[2])
    hm = heatmap!(ax_yz, ys, zs, YZ_slice; field_args...)
    Colorbar(f[:, end+1], hm)
    return f
end

"""
    plot_monitor(monitor::DFTMonitor, frequency_idx::Int)

TBW
"""
function plot_monitor(monitor::DFTMonitor, frequency_idx::Int)
    freq_data = get_dft_fields(monitor)[:, :, :, frequency_idx]
    if size(freq_data)[1] == 1
        freq_data = freq_data[1, :, :]
    elseif size(freq_data)[2] == 1
        freq_data = freq_data[:, 1, :]
    elseif size(freq_data)[3] == 1
        freq_data = freq_data[:, :, 1]
    else
        error("Invalid 3D array (size: $(size(freq_data)))")
    end

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
    plot_timesource(::SimulationData, time_source::ContinuousWaveData)

TBW
"""
function plot_timesource(::SimulationData, time_source::ContinuousWaveData)
    error("Unable to plot the response of a CW source.")
end

"""
    plot_timesource(sim::SimulationData, time_source::TimeSource, frequencies)

TBW
"""
function plot_timesource(sim::SimulationData, time_source::TimeSource, frequencies)
    prepare_simulation!(sim)

    # Pull the time-series data
    t = range(start = 0, stop = get_cutoff(time_source), step = sim.Δt)
    src_amplitude = zeros(size(t))
    for n in eachindex(t)
        src_amplitude[n] = eval_time_source(time_source, t[n])
    end

    # Compute the DTFT
    fourier_amplitude = DTFT(t, real.(src_amplitude), frequencies)
    PSD = abs.(fourier_amplitude) .^ 2
    PSD = PSD / maximum(PSD)

    # Set up the plot
    f = Figure(resolution = (800, 400))
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

function plot_source(sim::SimulationData, source::Source)

end
