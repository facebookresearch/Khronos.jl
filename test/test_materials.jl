# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Test the functionality of all the material functions.

import Khronos
using Test

frequency = 1.0 / 1.55

@testset "materials" begin

    ε = 12.4
    simple_isotropic = Khronos.Material(ε = ε)
    simple_isotropic_tensor = Khronos.get_ε_at_frequency(simple_isotropic, frequency)
    @test isapprox(simple_isotropic_tensor[1, 1], ε)
    @test isapprox(simple_isotropic_tensor[2, 2], ε)
    @test isapprox(simple_isotropic_tensor[2, 2], ε)

    εx = 1.0
    εy = 2.0
    εz = 3.0
    simple_anisotropic = Khronos.Material(εx = εx, εy = εy, εz = εz)
    simple_anisotropic_tensor = Khronos.get_ε_at_frequency(simple_anisotropic, frequency)
    @test isapprox(simple_anisotropic_tensor[1, 1], εx)
    @test isapprox(simple_anisotropic_tensor[2, 2], εy)
    @test isapprox(simple_anisotropic_tensor[3, 3], εz)

    complex_ε = 12.0 + im * 0.1
    complex_mat = Khronos.fit_complex_material(complex_ε, frequency)
    simple_complex_tensor = Khronos.get_ε_at_frequency(complex_mat, frequency)
    @test isapprox(simple_complex_tensor[1, 1], complex_ε)
    @test isapprox(simple_complex_tensor[2, 2], complex_ε)
    @test isapprox(simple_complex_tensor[2, 2], complex_ε)

end
