# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

module Khronos

using GeometryPrimitives
using Parameters
using KernelAbstractions
using Revise
using Logging
using LinearAlgebra
using OffsetArrays

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
include("Boundaries.jl")
include("Sources/Sources.jl")
include("Fields.jl")
include("Geometry.jl")
include("DFT.jl")
include("Monitors.jl")
include("Timestep.jl")
include("Simulation.jl")
include("Visualization.jl")

export Simulation

end
