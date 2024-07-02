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

    #TODO: add info relevant for polarizability
end

@with_kw struct Object
    shape::Shape
    material::Material
end

# -------------------------------------------------------- #
# Polarizability.jl
# -------------------------------------------------------- #

# TODO

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
# Simulation.jl
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
