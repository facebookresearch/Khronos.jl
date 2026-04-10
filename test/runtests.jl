# Copyright (c) Meta Platforms, Inc. and affiliates.

using Test

@testset "Khronos.jl" begin

    include("test_interpolation.jl")
    include("test_grid_volume.jl")
    include("test_materials.jl")
    include("test_mode.jl")
    include("test_sources.jl")
    include("test_timestep.jl")
    include("test_visualization.jl")
    include("test_volumes.jl")
    include("test_chunking.jl")
    include("test_dispersive.jl")
    include("test_near2far.jl")
    include("test_batch.jl")
    include("test_mode_monitor.jl")
    include("test_absorber.jl")
    include("test_subpixel.jl")
    include("test_gds.jl")

end
