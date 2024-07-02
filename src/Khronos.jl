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
import VectorModesolver

macro status(exs)
    @logmsg(0, exs)
end
macro verbose(exs)
    @logmsg(-10, exs)
end

verbose_logger = ConsoleLogger(stderr, -10)

include("load_deps.jl")

include("DataStructures.jl")
include("utils.jl")
include("Geometry.jl")
include("Mode.jl")
include("Boundaries.jl")
include("Sources/Sources.jl")
include("Fields.jl")
include("DFT.jl")
include("Monitors.jl")
include("Timestep.jl")
include("Simulation.jl")
include("Visualization.jl")

export Simulation

end
