# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using Test

@testset "Khronos.jl" begin

    include("test_interpolation.jl")
    include("test_grid_volume.jl")
    include("test_simulation.jl")

end
