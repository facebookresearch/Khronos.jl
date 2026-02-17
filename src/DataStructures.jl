# Copyright (c) Meta Platforms, Inc. and affiliates.

# -------------------------------------------------------- #
# 
# -------------------------------------------------------- #

abstract type Direction end
struct X <: Direction end
struct Y <: Direction end
struct Z <: Direction end

abstract type Field end
abstract type Electric <: Field end
abstract type Magnetic <: Field end
struct Center <: Field end

abstract type B <: Magnetic end
abstract type H <: Magnetic end
abstract type D <: Electric end
abstract type E <: Electric end
abstract type ε <: Electric end
abstract type μ <: Magnetic end

struct Ex <: E end
struct Ey <: E end
struct Ez <: E end
struct Dx <: D end
struct Dy <: D end
struct Dz <: D end
struct Hx <: H end
struct Hy <: H end
struct Hz <: H end
struct Bx <: B end
struct By <: B end
struct Bz <: B end

struct εx <: ε end
struct εy <: ε end
struct εz <: ε end
struct εxy <: ε end
struct εxz <: ε end
struct εyz <: ε end

struct μx <: μ end
struct μy <: μ end
struct μz <: μ end
struct μxy <: μ end
struct μxz <: μ end
struct μyz <: μ end

abstract type Dimension end
abstract type TwoD <: Dimension end
abstract type TwoD_TE <: TwoD end
abstract type TwoD_TM <: TwoD end
abstract type Cylindrical <: Dimension end # Not yet implemented
abstract type ThreeD <: Dimension end

# -------------------------------------------------------- #
# utils.jl
# -------------------------------------------------------- #

@with_kw struct Volume
    center::Vector{Real} = [0.0, 0.0, 0.0]
    size::Vector{Real} = [0.0, 0.0, 0.0]
end

@with_kw struct GridVolume
    component::Field
    start_idx::Vector{Int}
    end_idx::Vector{Int}
    Nx::Int
    Ny::Int
    Nz::Int
end

# -------------------------------------------------------- #
# Fields.jl
# -------------------------------------------------------- #

@with_kw struct Fields{T}
    # Primary fields
    fEx::Union{T,Nothing} = nothing
    fEy::Union{T,Nothing} = nothing
    fEz::Union{T,Nothing} = nothing
    fHx::Union{T,Nothing} = nothing
    fHy::Union{T,Nothing} = nothing
    fHz::Union{T,Nothing} = nothing
    fBx::Union{T,Nothing} = nothing
    fBy::Union{T,Nothing} = nothing
    fBz::Union{T,Nothing} = nothing
    fDx::Union{T,Nothing} = nothing
    fDy::Union{T,Nothing} = nothing
    fDz::Union{T,Nothing} = nothing

    # B Auxilliary fields
    fCBx::Union{T,Nothing} = nothing
    fCBy::Union{T,Nothing} = nothing
    fCBz::Union{T,Nothing} = nothing
    fUBx::Union{T,Nothing} = nothing
    fUBy::Union{T,Nothing} = nothing
    fUBz::Union{T,Nothing} = nothing
    fWBx::Union{T,Nothing} = nothing
    fWBy::Union{T,Nothing} = nothing
    fWBz::Union{T,Nothing} = nothing
    fSBx::Union{T,Nothing} = nothing
    fSBy::Union{T,Nothing} = nothing
    fSBz::Union{T,Nothing} = nothing
    fPBx::Union{T,Nothing} = nothing
    fPBy::Union{T,Nothing} = nothing
    fPBz::Union{T,Nothing} = nothing

    # D Auxilliary fields
    fCDx::Union{T,Nothing} = nothing
    fCDy::Union{T,Nothing} = nothing
    fCDz::Union{T,Nothing} = nothing
    fUDx::Union{T,Nothing} = nothing
    fUDy::Union{T,Nothing} = nothing
    fUDz::Union{T,Nothing} = nothing
    fWDx::Union{T,Nothing} = nothing
    fWDy::Union{T,Nothing} = nothing
    fWDz::Union{T,Nothing} = nothing
    fSDx::Union{T,Nothing} = nothing
    fSDy::Union{T,Nothing} = nothing
    fSDz::Union{T,Nothing} = nothing
    fPDx::Union{T,Nothing} = nothing
    fPDy::Union{T,Nothing} = nothing
    fPDz::Union{T,Nothing} = nothing
end
export Fields

# -------------------------------------------------------- #
# Boundaries.jl
# -------------------------------------------------------- #

@with_kw struct BoundaryData{T}
    # either the PML conductivity or the normal conductivity
    σBx::Union{T,Nothing} = nothing
    σBy::Union{T,Nothing} = nothing
    σBz::Union{T,Nothing} = nothing
    σDx::Union{T,Nothing} = nothing
    σDy::Union{T,Nothing} = nothing
    σDz::Union{T,Nothing} = nothing

    # currently not used
    κBx::Union{T,Nothing} = nothing
    κBy::Union{T,Nothing} = nothing
    κBz::Union{T,Nothing} = nothing
    κDx::Union{T,Nothing} = nothing
    κDy::Union{T,Nothing} = nothing
    κDz::Union{T,Nothing} = nothing

    # currently not used
    αBx::Union{T,Nothing} = nothing
    αBy::Union{T,Nothing} = nothing
    αBz::Union{T,Nothing} = nothing
    αDx::Union{T,Nothing} = nothing
    αDy::Union{T,Nothing} = nothing
    αDz::Union{T,Nothing} = nothing
end



# -------------------------------------------------------- #
# Sources.jl
# -------------------------------------------------------- #

abstract type TimeSource end
abstract type Source end

@with_kw struct SourceData{T}
    amplitude_data::T
    time_src::TimeSource
    gv::GridVolume
    component::Field
end

# -------------------------------------------------------- #
# Geometry.jl
# -------------------------------------------------------- #

"""
TODO: for now, we assume we can allocate and assign on the device.
We may need to change that later.
"""
@with_kw struct GeometryData{T,A}
    ε_inv::Union{T,Nothing} = nothing
    ε_inv_x::Union{A,Nothing} = nothing
    ε_inv_y::Union{A,Nothing} = nothing
    ε_inv_z::Union{A,Nothing} = nothing
    ε_inv_xy::Union{A,Nothing} = nothing
    ε_inv_xz::Union{A,Nothing} = nothing
    ε_inv_yz::Union{A,Nothing} = nothing

    σD::Union{A,Nothing} = nothing
    σDx::Union{A,Nothing} = nothing
    σDy::Union{A,Nothing} = nothing
    σDz::Union{A,Nothing} = nothing

    μ_inv::Union{T,Nothing} = nothing
    μ_inv_x::Union{A,Nothing} = nothing
    μ_inv_y::Union{A,Nothing} = nothing
    μ_inv_z::Union{A,Nothing} = nothing
    μ_inv_xy::Union{A,Nothing} = nothing
    μ_inv_xz::Union{A,Nothing} = nothing
    μ_inv_yz::Union{A,Nothing} = nothing

    σB::Union{A,Nothing} = nothing
    σBx::Union{A,Nothing} = nothing
    σBy::Union{A,Nothing} = nothing
    σBz::Union{A,Nothing} = nothing
end

@with_kw struct Material{N}
    ε::Union{Nothing,N} = nothing
    εx::Union{Nothing,N} = nothing
    εy::Union{Nothing,N} = nothing
    εz::Union{Nothing,N} = nothing
    εxy::Union{Nothing,N} = nothing
    εxz::Union{Nothing,N} = nothing
    εyz::Union{Nothing,N} = nothing

    σD::Union{Nothing,N} = nothing
    σDx::Union{Nothing,N} = nothing
    σDy::Union{Nothing,N} = nothing
    σDz::Union{Nothing,N} = nothing

    μ::Union{Nothing,N} = nothing
    μx::Union{Nothing,N} = nothing
    μy::Union{Nothing,N} = nothing
    μz::Union{Nothing,N} = nothing
    μxy::Union{Nothing,N} = nothing
    μxz::Union{Nothing,N} = nothing
    μyz::Union{Nothing,N} = nothing

    σB::Union{Nothing,N} = nothing
    σBx::Union{Nothing,N} = nothing
    σBy::Union{Nothing,N} = nothing
    σBz::Union{Nothing,N} = nothing

    susceptibilities::Vector{Susceptibility} = Susceptibility[]
end

@with_kw struct Object
    shape::Shape
    material::Material
end

# -------------------------------------------------------- #
# Polarizability.jl
# -------------------------------------------------------- #

"""
    PolarizationPoleState{T}

Per-pole polarization state for ADE (Auxiliary Differential Equation) updates.
Each pole of each susceptibility has its own pair of current/previous P arrays.
"""
struct PolarizationPoleState{T}
    Px::T              # current polarization x-component
    Py::T              # current polarization y-component
    Pz::T              # current polarization z-component
    Px_prev::T         # previous-step polarization x-component
    Py_prev::T         # previous-step polarization y-component
    Pz_prev::T         # previous-step polarization z-component
    sigma_x::T         # per-voxel coupling strength (0 in free space) for x
    sigma_y::T         # per-voxel coupling strength for y
    sigma_z::T         # per-voxel coupling strength for z
    coeffs::ADECoefficients  # pre-computed update coefficients
end

"""
    PolarizationData{T}

Container for all dispersive polarization data associated with a chunk.
"""
struct PolarizationData
    poles::Vector{<:PolarizationPoleState}
end

# -------------------------------------------------------- #
# Monitors.jl
# -------------------------------------------------------- #
abstract type Monitor end
abstract type MonitorData end

@with_kw struct TimeMonitor{N} <: Monitor
    component::Field # TODO extend to vector of components
    center::Vector{Int}
    size::Vector{Int}
    length::N
    Δt::Union{Nothing,N} = nothing
end

@with_kw struct TimeMonitorData{N,T} <: MonitorData
    component::Field
    fields::T
    gv::GridVolume
    length::Union{Nothing,Int64}
    Δt::Union{Nothing,N}
    time_index::Vector{Int64} = [1] # oof ugly closure for mutation
end

# -------------------------------------------------------- #
# DFT.jl
# -------------------------------------------------------- #

@with_kw mutable struct DFTMonitor <: Monitor
    component::Field # TODO extend to vector of components
    center::Vector{<:Number}
    size::Vector{<:Number}
    frequencies::Vector{<:Number}
    decimation::Int64 = 1
    monitor_data::Union{Nothing,<:MonitorData} = nothing
end

@with_kw struct DFTMonitorData{T,A} <: MonitorData
    component::Field
    fields::A
    scale::T
    gv::GridVolume
    frequencies::T
    decimation::Int64 = 1
end

# -------------------------------------------------------- #
# Near2Far.jl
# -------------------------------------------------------- #

"""
    Near2FarMonitorData

Internal data structure for near-to-far field computation. Stores references
to the 4 tangential DFT monitors (2 E + 2 H) and observation point info.
"""
@with_kw mutable struct Near2FarMonitorData <: MonitorData
    normal_axis::Int                              # 1=x, 2=y, 3=z
    normal_sign::Float64                          # +1 or -1
    tangential_E_monitors::Vector{DFTMonitor}     # 2 tangential E-field DFT monitors
    tangential_H_monitors::Vector{DFTMonitor}     # 2 tangential H-field DFT monitors
    frequencies::Vector{Float64}
    medium_eps::Float64 = 1.0
    medium_mu::Float64 = 1.0
    observation_points::Union{Nothing, Matrix{Float64}} = nothing  # Nx3
    theta::Union{Nothing, Vector{Float64}} = nothing
    phi::Union{Nothing, Vector{Float64}} = nothing
    r::Float64 = 1e6
    dx::Float64 = 0.0   # grid spacing (for Yee offset corrections in near2far)
    dy::Float64 = 0.0
    dz::Float64 = 0.0
    # Physical position of each component's first DFT grid point [x,y,z].
    # Computed from get_component_origin + (start_idx - 1) * Δ.
    # These give exact Yee-grid-aware coordinates for near2far integration.
    e1_base::Vector{Float64} = Float64[]
    e2_base::Vector{Float64} = Float64[]
    h1_base::Vector{Float64} = Float64[]
    h2_base::Vector{Float64} = Float64[]
end

"""
    Near2FarMonitor

Monitor that records tangential E/H fields on a planar surface and computes
far-field radiation patterns via the surface equivalence principle.

Two modes:
1. Angular mode: specify `theta`/`phi` arrays → far-field at distance `r`
2. Point mode: specify `observation_points` (Nx3 matrix) → full Green's function
"""
@with_kw mutable struct Near2FarMonitor <: Monitor
    center::Vector{<:Number}
    size::Vector{<:Number}          # one dimension must be 0 (planar surface)
    frequencies::Vector{<:Number}
    observation_points::Union{Nothing, Matrix{<:Number}} = nothing
    theta::Union{Nothing, Vector{<:Number}} = nothing
    phi::Union{Nothing, Vector{<:Number}} = nothing
    r::Float64 = 1e6
    normal_dir::Symbol = :+         # :+ or :- outward normal direction
    medium_eps::Float64 = 1.0
    medium_mu::Float64 = 1.0
    decimation::Int64 = 1
    monitor_data::Union{Nothing, Near2FarMonitorData} = nothing
end

# -------------------------------------------------------- #
# Chunking.jl
# -------------------------------------------------------- #

"""
    PhysicsFlags

Captures which physics features are active in a rectangular region.
Drives which arrays to allocate and which kernel variant compiles.
"""
struct PhysicsFlags
    has_epsilon::Bool
    has_mu::Bool
    has_sigma_D::Bool
    has_sigma_B::Bool
    has_pml_x::Bool
    has_pml_y::Bool
    has_pml_z::Bool
    has_sources::Bool
    has_monitors::Bool
    has_polarizability::Bool
end

function PhysicsFlags(;
    has_epsilon::Bool = false,
    has_mu::Bool = false,
    has_sigma_D::Bool = false,
    has_sigma_B::Bool = false,
    has_pml_x::Bool = false,
    has_pml_y::Bool = false,
    has_pml_z::Bool = false,
    has_sources::Bool = false,
    has_monitors::Bool = false,
    has_polarizability::Bool = false,
)
    PhysicsFlags(
        has_epsilon, has_mu, has_sigma_D, has_sigma_B,
        has_pml_x, has_pml_y, has_pml_z, has_sources, has_monitors,
        has_polarizability,
    )
end

has_any_pml(pf::PhysicsFlags) = pf.has_pml_x || pf.has_pml_y || pf.has_pml_z

"""
    ChunkSpec

Describes one rectangular chunk of the simulation domain.
"""
struct ChunkSpec
    id::Int
    volume::Volume
    grid_volume::GridVolume
    physics::PhysicsFlags
    neighbor_ids::Vector{Int}
    device_id::Int
end

"""
    ChunkPlan

Complete decomposition of the simulation domain into chunks.
"""
struct ChunkPlan
    chunks::Vector{ChunkSpec}
    adjacency::Vector{Tuple{Int,Int,Int}}
    total_chunks::Int
end

"""
    HaloConnection

Describes a ghost cell send/recv pair between two chunks.
"""
struct HaloConnection
    src_chunk_id::Int
    dst_chunk_id::Int
    axis::Int
    src_range::NTuple{3,UnitRange{Int}}
    dst_range::NTuple{3,UnitRange{Int}}
end

"""
    ChunkData

Runtime data per chunk. Parallels Meep's `structure_chunk` + `fields_chunk`.
"""
mutable struct ChunkData{N,T,CT,BT}
    spec::ChunkSpec
    fields::Fields{BT}
    geometry_data::GeometryData{N,T}
    boundary_data::BoundaryData{T}
    source_data::Vector{SourceData{CT}}
    monitor_data::Vector{MonitorData}
    polarization_data::Union{Nothing,PolarizationData}
    halo_send::Vector{HaloConnection}
    halo_recv::Vector{HaloConnection}
    halo_send_buffers::Vector{AbstractArray}
    halo_recv_buffers::Vector{AbstractArray}
    halo_send_gpu_staging::Vector{AbstractArray}
    halo_recv_gpu_staging::Vector{AbstractArray}
    ndrange::NTuple{3,Int}
end

# -------------------------------------------------------- #

function get_Nz(cell_size, resolution)
    if size(cell_size)[1] < 3
        return 0
    end

    if cell_size[3] == 0.0
        return 0
    end

    return floor(Int, cell_size[3] * resolution)
end

function get_Δz(cell_size, Nz)
    if Nz == 0
        return Inf
    end

    return cell_size[3] / (Nz)
end

@with_kw mutable struct SimulationData{N,T,CN,CT,BT}
    # N: Real number type
    # T: Real array type
    # CN: Complex number type
    # CT: Complex array type
    cell_size::Vector{N} = N.(cell_size)
    cell_center::Any
    resolution::Any
    sources::Any
    boundaries::Union{Vector{Vector{N}},Nothing} = nothing
    geometry::Union{Vector{Object},Nothing} = nothing
    monitors::Union{Vector{Monitor},Nothing} = nothing
    num_chunks::Union{Int,Symbol,Nothing} = nothing  # nothing=single chunk, :auto, or explicit Int

    Nx::Union{Int,Nothing} = floor(Int, cell_size[1] * resolution)
    Ny::Union{Int,Nothing} = floor(Int, cell_size[2] * resolution)
    Nz::Union{Int,Nothing} = get_Nz(cell_size, resolution)
    is_prepared::Bool = false
    Courant::N = 0.5
    Δx::Union{Vector{N},N,Nothing} = N.(cell_size[1] / (Nx))
    Δy::Union{Vector{N},N,Nothing} = N.(cell_size[2] / (Ny))
    Δz::Union{Vector{N},N,Nothing} = N.(get_Δz(cell_size, Nz))
    Δt::Union{N,Nothing} = minimum((Δx, Δy, Δz)) * Courant
    ndims::Int = (Nz == 0) ? 2 : 3
    dimensionality::Type{<:Dimension} = ThreeD

    # Data to be generated upon initialization
    fields::Union{Fields{BT},Nothing} = nothing
    source_data::Union{Vector{SourceData{CT}},Nothing} = nothing
    source_components::Union{Vector{<:Field},Nothing} = nothing
    boundary_data::Union{BoundaryData{T},Nothing} = nothing
    geometry_data::Union{GeometryData{N,T},Nothing} = nothing
    monitor_data::Vector{MonitorData} = MonitorData[] # todo figure out parameterizing abstract types
    timestep::Int = 0
    sources_active::Bool = true  # P.3: flag to disable source arrays after shutoff

    # P.2: Cached kernel objects to avoid repeated construction from non-const global
    _cached_curl_kernel::Any = nothing
    _cached_update_kernel::Any = nothing
    _cached_curl_comp_kernel::Any = nothing  # Per-component split curl kernel
    _cached_update_comp_kernel::Any = nothing  # Per-component split update kernel
    _cached_fused_kernel::Any = nothing       # Fused curl+update kernel (no PML)
    _cached_fused_pml_kernel::Any = nothing  # Fused curl+update kernel (PML)
    _cached_source_kernel::Any = nothing
    _cached_dft_kernel::Any = nothing
    _cached_dft_chunk_kernel::Any = nothing

    # Chunking support
    chunk_plan::Union{ChunkPlan,Nothing} = nothing
    chunk_data::Union{Vector{ChunkData{N,T,CT,BT}},Nothing} = nothing
    chunk_rank_assignment::Union{Vector{Int},Nothing} = nothing

    # Precomputed halo copy operations for fast exchange (avoids per-step field lookups)
    _halo_ops_H::Vector{Any} = Any[]
    _halo_ops_E::Vector{Any} = Any[]

    # CUDA Graph replay for post-source steady-state stepping
    _cuda_graph_exec_H::Any = nothing  # CuGraphExec for H half-step (curl B + update H + halo)
    _cuda_graph_exec_E::Any = nothing  # CuGraphExec for E half-step (curl D + update E + halo)
end

# Convenience wrapper
Simulation(; kwargs...) = SimulationData{
    backend_number,
    backend_array,
    complex_backend_number,
    complex_backend_array,
    AbstractArray,
}(;
    kwargs...,
)
