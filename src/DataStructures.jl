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
# Subpixel smoothing
# -------------------------------------------------------- #

"""Subpixel smoothing mode for geometry rasterization."""
abstract type SubpixelSmoothing end

"""No smoothing — standard point sampling (default)."""
struct NoSmoothing <: SubpixelSmoothing end

"""Volume averaging — isotropic ε̄ = f·ε₁ + (1-f)·ε₂."""
struct VolumeAveraging <: SubpixelSmoothing end

"""
Anisotropic smoothing (Farjadpour et al. 2006).
Uses interface normal to compute direction-dependent effective ε.
Achieves second-order convergence at dielectric interfaces.
"""
struct AnisotropicSmoothing <: SubpixelSmoothing end

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

abstract type BoundaryCondition end
struct PML <: BoundaryCondition end
struct Periodic <: BoundaryCondition end
@with_kw struct Bloch <: BoundaryCondition
    k::Float64 = 0.0
end
struct PECBoundary <: BoundaryCondition end
struct PMCBoundary <: BoundaryCondition end

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

    chi3::Union{A,Nothing} = nothing   # per-voxel Kerr χ3 coefficient
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

    chi3::Union{Nothing,N} = nothing  # Kerr nonlinearity coefficient

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
    LayerSpec

A single dielectric layer in a planar stack for layered near-to-far projection.

Fields:
- `z_min`: bottom z-coordinate of the layer
- `z_max`: top z-coordinate of the layer
- `eps`: relative permittivity
- `mu`: relative permeability (default 1.0)
"""
struct LayerSpec
    z_min::Float64
    z_max::Float64
    eps::Float64
    mu::Float64
end
LayerSpec(z_min, z_max, eps; mu=1.0) = LayerSpec(z_min, z_max, eps, mu)

"""
    LayerStack

Ordered stack of dielectric layers for layered near-to-far field projection.
Layers should be ordered from bottom to top.

When attached to a `Near2FarMonitor`, far-field projection uses the transfer
matrix method to account for Fresnel reflection/transmission at every interface,
enabling correct far-field computation even when the monitor is inside a
high-index medium (e.g., GaN) and the observation is in air.
"""
struct LayerStack
    layers::Vector{LayerSpec}
end

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
    layer_stack::Union{Nothing, LayerStack} = nothing
end

"""
    Near2FarMonitor

Monitor that records tangential E/H fields on a planar surface and computes
far-field radiation patterns via the surface equivalence principle.

Two modes:
1. Angular mode: specify `theta`/`phi` arrays → far-field at distance `r`
2. Point mode: specify `observation_points` (Nx3 matrix) → full Green's function

When `layer_stack` is provided, the far-field projection uses the transfer matrix
method to propagate through the layer stack (e.g., from GaN into air), enabling
correct far-field projection without simulating the air region in FDTD.
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
    layer_stack::Union{Nothing, LayerStack} = nothing
    monitor_data::Union{Nothing, Near2FarMonitorData} = nothing
end

# -------------------------------------------------------- #
# ModeMonitor.jl
# -------------------------------------------------------- #

"""
    ModeSpec

Configuration for the mode solver used by ModeMonitor.
"""
@with_kw struct ModeSpec
    num_modes::Int = 1
    target_neff::Float64 = 0.0
    geometry::Vector{Object} = Object[]
    mode_solver_resolution::Int = 50
    solver_tolerance::Float64 = 1e-6
    num_mode_freqs::Int = 5   # coarse frequency points for mode interpolation (0 = solve every freq)
    mode_group::Symbol = :auto  # cache grouping — monitors with the same mode_group share mode solves
                                # :auto (default) = key by monitor center (safe, one solve per position)
                                # any other Symbol (e.g. :wg1) = all monitors with that symbol share
end

"""
    ModeMonitorData

Internal data structure for mode overlap computation. Stores references
to 4 tangential DFT monitors (2 E + 2 H), solved mode profiles, and
grid information needed for the overlap integral.
"""
@with_kw mutable struct ModeMonitorData <: MonitorData
    normal_axis::Int                              # 1=x, 2=y, 3=z
    tangential_E_monitors::Vector{DFTMonitor}     # 2 tangential E-field DFT monitors
    tangential_H_monitors::Vector{DFTMonitor}     # 2 tangential H-field DFT monitors
    frequencies::Vector{Float64}
    mode_profiles::Vector{VectorModesolver.Mode}  # solved modes (1 per freq)
    geometry::Vector{Object}                      # for mode solving
    mode_spec::ModeSpec
    dx::Float64 = 0.0
    dy::Float64 = 0.0
    dz::Float64 = 0.0
    # Physical position of each component's first DFT grid point [x,y,z].
    e1_base::Vector{Float64} = Float64[]
    e2_base::Vector{Float64} = Float64[]
    h1_base::Vector{Float64} = Float64[]
    h2_base::Vector{Float64} = Float64[]
end

"""
    ModeMonitor

Monitor that records tangential E/H fields on a cross-sectional plane and
computes the overlap integral with a waveguide mode to extract complex
mode amplitudes (forward/backward S-parameters).

One dimension of `size` must be zero (planar cross-section).
"""
@with_kw mutable struct ModeMonitor <: Monitor
    center::Vector{<:Number}
    size::Vector{<:Number}          # one dimension must be 0
    frequencies::Vector{<:Number}
    mode_spec::ModeSpec = ModeSpec()
    decimation::Int64 = 1
    monitor_data::Union{Nothing, ModeMonitorData} = nothing
end

# -------------------------------------------------------- #
# FluxMonitor.jl
# -------------------------------------------------------- #

"""
    FluxMonitorData

Internal data for flux computation. Stores references to 4 tangential DFT
monitors (2 E + 2 H) and grid info for Poynting flux integration.
"""
@with_kw mutable struct FluxMonitorData <: MonitorData
    normal_axis::Int                              # 1=x, 2=y, 3=z
    tangential_E_monitors::Vector{DFTMonitor}     # 2 tangential E-field DFT monitors
    tangential_H_monitors::Vector{DFTMonitor}     # 2 tangential H-field DFT monitors
    frequencies::Vector{Float64}
    dx::Float64 = 0.0
    dy::Float64 = 0.0
    dz::Float64 = 0.0
end

"""
    FluxMonitor

Monitor that records tangential E/H fields on a planar surface and computes
the Poynting flux (power flow) through the surface at each frequency.

One dimension of `size` must be zero (planar surface).
"""
@with_kw mutable struct FluxMonitor <: Monitor
    center::Vector{<:Number}
    size::Vector{<:Number}          # one dimension must be 0
    frequencies::Vector{<:Number}
    decimation::Int64 = 1
    monitor_data::Union{Nothing, FluxMonitorData} = nothing
end

# -------------------------------------------------------- #
# DiffractionMonitor.jl
# -------------------------------------------------------- #

@with_kw mutable struct DiffractionMonitorData <: MonitorData
    normal_axis::Int
    tangential_E_monitors::Vector{DFTMonitor}
    tangential_H_monitors::Vector{DFTMonitor}
    frequencies::Vector{Float64}
    dx::Float64 = 0.0
    dy::Float64 = 0.0
    dz::Float64 = 0.0
    cell_size::Vector{Float64} = Float64[]
end

@with_kw mutable struct DiffractionMonitor <: Monitor
    center::Vector{<:Number}
    size::Vector{<:Number}          # one dimension must be 0
    frequencies::Vector{<:Number}
    decimation::Int64 = 1
    monitor_data::Union{Nothing, DiffractionMonitorData} = nothing
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
    phase_factor::ComplexF64   # Bloch phase: 1.0 for periodic, exp(i·k·L) for Bloch
end

# Convenience constructor for real (non-Bloch) connections
HaloConnection(src, dst, axis, sr, dr) = HaloConnection(src, dst, axis, sr, dr, ComplexF64(1.0))

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

# Grid spacing dispatch for CFL computation
@inline _min_Δ(Δ::Real) = Δ
@inline _min_Δ(Δ::AbstractVector) = minimum(Δ)

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
    absorbers::Union{Vector,Nothing} = nothing  # per-axis absorber config: [nothing, [nothing, Absorber(...)], nothing]
    boundary_conditions::Union{Vector{Vector{BoundaryCondition}},Nothing} = nothing  # per-axis [minus, plus] BC types
    symmetry::Tuple{Int,Int,Int} = (0, 0, 0)  # per-axis: 0=none, +1=even (PMC for E), -1=odd (PEC for E)
    subpixel_smoothing::SubpixelSmoothing = NoSmoothing()
    geometry::Union{Vector{Object},Nothing} = nothing
    monitors::Union{Vector{Monitor},Nothing} = nothing
    num_chunks::Union{Int,Symbol,Nothing} = nothing  # nothing=single chunk, :auto, or explicit Int

    Nx::Union{Int,Nothing} = floor(Int, cell_size[1] * resolution)
    Ny::Union{Int,Nothing} = floor(Int, cell_size[2] * resolution)
    Nz::Union{Int,Nothing} = get_Nz(cell_size, resolution)
    is_prepared::Bool = false
    Courant::N = 0.5
    Δx::Union{AbstractVector,N,Nothing} = N.(cell_size[1] / (Nx))
    Δy::Union{AbstractVector,N,Nothing} = N.(cell_size[2] / (Ny))
    Δz::Union{AbstractVector,N,Nothing} = N.(get_Δz(cell_size, Nz))
    Δt::Union{N,Nothing} = minimum((_min_Δ(Δx), _min_Δ(Δy), _min_Δ(Δz))) * Courant
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

    # Multi-stream concurrent chunk dispatch
    _chunk_streams::Vector{Any} = Any[]
    _use_multi_stream::Bool = false

    # Cached CUDA dispatch constants (computed once in prepare_simulation!)
    _cached_cuda_wg_x::Int32 = Int32(32)
    _cached_cuda_wg_y::Int32 = Int32(8)
    _cached_dt_dx::Any = nothing
    _cached_dt_dy::Any = nothing
    _cached_dt_dz::Any = nothing
    _cached_grid_is_uniform::Bool = false
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
