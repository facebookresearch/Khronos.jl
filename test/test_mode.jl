# (c) Meta Platforms, Inc. and affiliates.
#
# Test the functionality of the mode functions

import Khronos
using Test


function _setup_vector_field(vector_field::AbstractArray)
    vector_field[:, :, :, 1] .= 0.0
    vector_field[:, :, :, 2] .= 1.0
    vector_field[:, :, :, 3] .= 2.0
    return vector_field
end

@testset "mode" begin
    # Check that things shuffle like we expect
    vector_field = rand(10, 20, 13, 3)
    vector_field = _setup_vector_field(vector_field)
    XZ_field = similar(vector_field)
    XZ_field[:, :, :, 1] .= 0.0
    XZ_field[:, :, :, 2] .= 2.0
    XZ_field[:, :, :, 3] .= -1.0
    @test isapprox(XZ_field, Khronos.rotate_XY_to_XZ(vector_field))

    vector_field = rand(10, 20, 13, 3)
    vector_field = _setup_vector_field(vector_field)

    YZ_field = similar(vector_field)
    YZ_field[:, :, :, 1] .= 2.0
    YZ_field[:, :, :, 2] .= 0.0
    YZ_field[:, :, :, 3] .= -1.0
    @test isapprox(YZ_field, Khronos.rotate_XY_to_YZ(vector_field))

    # All of the mode rotation functions should undo each other.
    vector_field = rand(10, 20, 13, 3)
    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XY_to_XZ |> Khronos.rotate_XZ_to_XY,
    )
    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XZ_to_XY |> Khronos.rotate_XY_to_XZ,
    )

    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XZ_to_YZ |> Khronos.rotate_YZ_to_XZ,
    )
    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_YZ_to_XZ |> Khronos.rotate_XZ_to_YZ,
    )

    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XY_to_YZ |> Khronos.rotate_YZ_to_XY,
    )
    @test isapprox(
        vector_field,
        vector_field |> Khronos.rotate_YZ_to_XY |> Khronos.rotate_XY_to_YZ,
    )

    # Some sanity checks.
    @test !isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XZ_to_XY |> Khronos.rotate_YZ_to_XY,
    )
    @test !isapprox(
        vector_field,
        vector_field |> Khronos.rotate_XZ_to_XY |> Khronos.rotate_XZ_to_XY,
    )
end
