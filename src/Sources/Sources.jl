# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

include("./TimeSources.jl")
include("./SpatialSources.jl")

# ---------------------------------------------------------- #
# Useful routines
# ---------------------------------------------------------- #

"""
    get_source_from_field_component(sim, component)

We need a convenient way to access the source array that is associated with a
particular component, from that component itself.
"""
get_source_from_field_component(sim::SimulationData, ::Ex) = sim.fields.fSDx
get_source_from_field_component(sim::SimulationData, ::Ey) = sim.fields.fSDy
get_source_from_field_component(sim::SimulationData, ::Ez) = sim.fields.fSDz
get_source_from_field_component(sim::SimulationData, ::Hx) = sim.fields.fSBx
get_source_from_field_component(sim::SimulationData, ::Hy) = sim.fields.fSBy
get_source_from_field_component(sim::SimulationData, ::Hz) = sim.fields.fSBz

# ---------------------------------------------------------- #
# Source data initialization
# ---------------------------------------------------------- #

function add_sources(sim::SimulationData, sources::Vector{<:Source})
    sim.source_data = reduce(vcat, [(assemble_sources(sim, s)) for s in sources])
    sim.source_components = reduce(vcat, [get_source_components(s) for s in sources])
end

function add_sources(sim::SimulationData, sources::Union{Nothing,Vector{<:Any}})
    error("No sources added...")
end

function assemble_sources(sim::SimulationData, source::Source)

    src_volume = get_source_volume(source)
    source_objects = SourceData{complex_backend_array}[]
    for component in get_source_components(source)
        # create the GridVolume
        gv = GridVolume(sim, src_volume, component)

        # Fill in the amplitude data
        amplitude_data =
            zeros(complex_backend_number, ([gv.Nx, gv.Ny, gv.Nz][1:sim.ndims])...)
        for i in CartesianIndices(amplitude_data)
            point = grid_volume_idx_to_point(sim, gv, i)
            weight = compute_interpolation_weight(
                point,
                src_volume,
                sim.ndims,
                sim.Δx,
                sim.Δy,
                sim.Δz,
            )
            amplitude_data[i] =
                weight *
                get_amplitude(source) *
                get_source_profile(source, point, component)
        end
        push!(
            source_objects,
            SourceData{complex_backend_array}(
                amplitude_data,
                get_time_profile(source),
                gv,
                component,
            ),
        )
    end
    return source_objects
end

# ---------------------------------------------------------- #
# Source stepping routines
# ---------------------------------------------------------- #

get_source_offset(gv) = (gv.start_idx[1] - 1, gv.start_idx[2] - 1, gv.start_idx[3] - 1)

function step_source!(
    current_source::SourceData,
    sim::SimulationData,
    component::Union{E,H},
    t::Real,
)
    if current_source.component isa typeof(component)
        source_kernel = update_source!(backend_engine)

        current_source_array = get_source_from_field_component(sim, component)
        spatial_amplitude = current_source.amplitude_data
        scalar_amplitude = eval_time_source(current_source.time_src, t)
        offset_index = get_source_offset(current_source.gv)
        source_kernel(
            current_source_array,
            spatial_amplitude,
            scalar_amplitude,
            offset_index...,
            ndrange = size(spatial_amplitude),
        )
    end
    return
end

function step_sources!(sim::SimulationData, component::Union{E,H}, t::Real)
    # Loop through all sources looking for current components
    if !isnothing(sim.source_data)
        map((cs) -> step_source!(cs, sim, component, t), sim.source_data)
    end
    return
end

# The current_source array currently spans the entire domain, whereas the
# spatial_amplitude array only spans the user-specified grid volume. In order for
# things to match up, we need to offset things appropriately. In this case, we'll
# index into the smaller array (spatial_amplitude) and offset the larger array.
@kernel function update_source!(
    current_source::AbstractArray,
    spatial_amplitude::AbstractArray,
    scalar_amplitude::Number,
    offset_ix::Int,
    offset_iy::Int,
    offset_iz::Int,
)
    ix, iy, iz = @index(Global, NTuple)
    current_source[ix+offset_ix, iy+offset_iy, iz+offset_iz] +=
        real(scalar_amplitude * spatial_amplitude[ix, iy, iz])
end
