# Copyright (c) Meta Platforms, Inc. and affiliates.

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

# Chunk-level source field accessors
get_source_from_field_component(chunk::ChunkData, ::Ex) = chunk.fields.fSDx
get_source_from_field_component(chunk::ChunkData, ::Ey) = chunk.fields.fSDy
get_source_from_field_component(chunk::ChunkData, ::Ez) = chunk.fields.fSDz
get_source_from_field_component(chunk::ChunkData, ::Hx) = chunk.fields.fSBx
get_source_from_field_component(chunk::ChunkData, ::Hy) = chunk.fields.fSBy
get_source_from_field_component(chunk::ChunkData, ::Hz) = chunk.fields.fSBz

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

    # Pre-compute constants outside the per-component loop
    amplitude = get_amplitude(source)
    min_corner = SVector{3}(get_min_corner(src_volume)...)
    max_corner = SVector{3}(get_max_corner(src_volume)...)
    vol_size = SVector{3}(src_volume.size...)
    Δx = isnothing(sim.Δx) ? 0.0 : Float64(sim.Δx)
    Δy = isnothing(sim.Δy) ? 0.0 : Float64(sim.Δy)
    Δz = isnothing(sim.Δz) ? 0.0 : Float64(sim.Δz)
    Δ = SVector{3}(Δx, Δy, Δz)
    ndims = sim.ndims::Int

    for component in get_source_components(source)
        # create the GridVolume
        gv = GridVolume(sim, src_volume, component)

        # Pre-compute per-component constants
        pol_factor = _get_source_pol_factor(source, component)

        # Pre-compute the component origin and grid volume origin ONCE
        # (avoids per-voxel type-unstable access to sim.Δx/Δy/Δz/cell_size/cell_center)
        origin = SVector{3,Float64}(get_component_origin(sim, gv.component)...)
        gv_origin = get_min_corner(gv)
        cc3 = Float64(sim.cell_center[3])

        # Build amplitude data on CPU, then bulk-transfer to GPU.
        # Uses a function barrier (_fill_amplitude_data!) so Julia can specialize
        # on the concrete source type and avoid dynamic dispatch per voxel.
        dims = ([gv.Nx, gv.Ny, gv.Nz][1:ndims])
        amplitude_data_cpu = zeros(Complex{Float64}, dims...)
        _fill_amplitude_data!(amplitude_data_cpu, source, component, pol_factor,
            amplitude, origin, gv_origin, cc3, Δx, Δy, Δz,
            min_corner, max_corner, vol_size, ndims, Δ)
        amplitude_data = complex_backend_array(complex_backend_number.(amplitude_data_cpu))
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

"""
    _fill_amplitude_data!(data, source, component, ...)

Function barrier: Julia specializes this on the concrete type of `source`,
eliminating dynamic dispatch overhead in the inner loop (~5x speedup for
PlaneWaveSource).
"""
function _fill_amplitude_data!(
    amplitude_data_cpu::AbstractArray{Complex{Float64}},
    source, component, pol_factor, amplitude,
    origin::SVector{3,Float64}, gv_origin, cc3::Float64,
    Δx::Float64, Δy::Float64, Δz::Float64,
    min_corner::SVector{3,Float64}, max_corner::SVector{3,Float64},
    vol_size::SVector{3,Float64}, ndims::Int, Δ::SVector{3,Float64},
)
    gv_start_1 = gv_origin[1]
    gv_start_2 = gv_origin[2]
    gv_start_3 = gv_origin[3]
    for i in CartesianIndices(amplitude_data_cpu)
        t = Tuple(i)
        ix = t[1]
        iy = length(t) >= 2 ? t[2] : 0
        iz = length(t) >= 3 ? t[3] : 0
        # Inline grid_volume_idx_to_point to avoid type-unstable sim field access
        px = origin[1] + (ix + gv_start_1 - 2) * Δx
        py = origin[2] + (iy + gv_start_2 - 2) * Δy
        pz_raw = origin[3] + (iz + gv_start_3 - 2) * Δz
        pz = (isinf(pz_raw) || isnan(pz_raw)) ? cc3 : pz_raw
        point = SVector(px, py, pz)
        weight = _compute_interpolation_weight_fast(
            point, min_corner, max_corner, vol_size, ndims, Δ,
        )
        profile = _get_source_profile_fast(source, point, component, pol_factor)
        amplitude_data_cpu[i] = weight * amplitude * profile
    end
end

# ── Fast source profile helpers ─────────────────────────────────────────────
# These pre-compute loop-invariant values and avoid per-voxel allocations.

# Default: pol_factor is unused, just return 1.0
_get_source_pol_factor(::Source, ::Field) = 1.0
_get_source_pol_factor(source::PlaneWaveSourceData, component::Field) =
    get_planewave_polarization_scaling(source, component)

# PlaneWaveSource: only the phase varies per-voxel; pol_factor is pre-computed
@inline function _get_source_profile_fast(
    source::PlaneWaveSourceData, point::SVector{3,Float64},
    ::Field, pol_factor::Number,
)
    kv = source.k_vector
    kx = Float64(kv[1]); ky = Float64(kv[2]); kz = Float64(kv[3])
    phase = (point[1] * kx + point[2] * ky + point[3] * kz) * Float64(source.k)
    return pol_factor * exp(-im * phase)
end

# UniformSource: constant profile
@inline function _get_source_profile_fast(
    ::UniformSourceData, ::SVector{3,Float64}, ::Field, ::Number,
)
    return 1.0
end

# Fallback for other source types: convert SVector to Vector (allocates, but
# only for uncommon source types like GaussianBeam/EquivalentSource)
@inline function _get_source_profile_fast(
    source::Source, point::SVector{3,Float64}, component::Field, ::Number,
)
    return get_source_profile(source, collect(point), component)
end

# ---------------------------------------------------------- #
# Source stepping routines
# ---------------------------------------------------------- #

# P.5: Field arrays are raw (no OffsetArray), so offset includes +1 for ghost cell
get_source_offset(gv) = (gv.start_idx[1], gv.start_idx[2], gv.start_idx[3])

function step_source!(
    current_source::SourceData,
    sim::SimulationData,
    component::Union{E,H},
    t::Real,
)
    if current_source.component isa typeof(component)
        source_kernel = sim._cached_source_kernel

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

"""
    step_source_chunk!(current_source, chunk, component, t)

Step a source within a specific chunk's field arrays.
For single-chunk, the offset is global. For multi-chunk, the offset is adjusted
to chunk-local coordinates.
"""
function step_source_chunk!(
    current_source::SourceData,
    chunk::ChunkData,
    sim::SimulationData,
    component::Union{E,H},
    t::Real,
    single_chunk::Bool,
)
    if current_source.component isa typeof(component)
        source_kernel = sim._cached_source_kernel

        current_source_array = get_source_from_field_component(chunk, component)
        isnothing(current_source_array) && return

        spatial_amplitude = current_source.amplitude_data
        scalar_amplitude = eval_time_source(current_source.time_src, t)

        if single_chunk
            # Single chunk: use global offset (field arrays span full domain)
            offset_index = get_source_offset(current_source.gv)
        else
            # Multi-chunk: adjust offset to chunk-local coordinates
            # P.5: +1 for ghost cell in raw field arrays (no OffsetArray)
            chunk_start = chunk.spec.grid_volume.start_idx
            offset_index = (
                current_source.gv.start_idx[1] - chunk_start[1] + 1,
                current_source.gv.start_idx[2] - chunk_start[2] + 1,
                current_source.gv.start_idx[3] - chunk_start[3] + 1,
            )
        end

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
    isnothing(sim.chunk_data) && return

    single_chunk = length(sim.chunk_data) == 1
    for chunk in sim.chunk_data
        for cs in chunk.source_data
            step_source_chunk!(cs, chunk, sim, component, t, single_chunk)
        end
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
