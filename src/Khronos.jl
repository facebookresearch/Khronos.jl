# Copyright (c) Meta Platforms, Inc. and affiliates.

module Khronos

using Einsum
using GeometryPrimitives
using Parameters
using KernelAbstractions
using Revise
using Logging
using LinearAlgebra
using OffsetArrays
using StaticArrays
using MPI
using NCCL
import VectorModesolver
import Interpolations
import FFTW

macro status(exs)
    @logmsg(0, exs)
end
macro verbose(exs)
    @logmsg(-10, exs)
end

verbose_logger = ConsoleLogger(stderr, -10)

include("load_deps.jl")

include("Susceptibility.jl")
include("DataStructures.jl")
include("utils.jl")
include("Geometry.jl")
include("Mode.jl")
include("Boundaries.jl")
include("Sources/Sources.jl")
include("Chunking.jl")
include("Distributed.jl")
include("Fields.jl")
include("Monitors.jl")
include("Memory.jl")
include("Timestep.jl")
include("Near2Far.jl")
include("ModeMonitor.jl")
include("FluxMonitor.jl")
include("DiffractionMonitor.jl")
include("Simulation.jl")
include("Batch.jl")
include("Visualization.jl")

export Simulation
export SubpixelSmoothing, NoSmoothing, VolumeAveraging, AnisotropicSmoothing
export Periodic, Bloch, PECBoundary, PMCBoundary

end
