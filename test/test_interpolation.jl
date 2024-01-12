# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# test the interpolation and restriction functions inside utils.jl. these are
# primarily used to create monitor and source regions, where interpolation and
# restriction is needed for smooth behavior.
#
# all interpolation and restriction operations should preserve a constant
# integral as one moves the integrating window. We use this property to verify
# the weights are being calculated correctly.

import Khronos
using Test

const USE_GPU = false;
Khronos.choose_backend()

Δx = 0.1;
Δy = 0.1;
Δz = nothing;
ndims = 2
Lx = 1.0
Ly = 1.4
x = -Lx/2:Δx:Lx/2
y = -Ly/2:Δy:Ly/2
w = zeros(size(x, 1), size(y, 1))

function restrict_to_array!(arr, volume)
    for (iy, y_val) in enumerate(y)
        for (ix, x_val) in enumerate(x)
            arr[ix, iy] = Khronos.compute_interpolation_weight([x_val, y_val, 0.0], volume, ndims, Δx, Δy, Δz)
        end
    end
end

@testset "Single-point and line restrictions" begin
    pt_volume = Khronos.Volume(center=[0, 0, 0], size=[0, 0, 0])
    restrict_to_array!(w, pt_volume)
    @test sum(w) ≈ 1.0 atol = 1e-15
    
    pt_volume = Khronos.Volume(center=[-0.12, -0.26, 0], size=[0, 0, 0])
    restrict_to_array!(w, pt_volume)
    @test sum(w) ≈ 1.0 atol = 1e-15

    line_x_volume = Khronos.Volume(center=[0.14, -0.21, 0], size=[5Δx, 0, 0])
    restrict_to_array!(w, line_x_volume)
    @test sum(w) * Δx ≈ 5Δx atol = 1e-15

    line_y_volume = Khronos.Volume(center=[0.14, -0.21, 0], size=[0, 5Δy, 0])
    restrict_to_array!(w, line_y_volume)
    @test (sum(w)) * Δy ≈ 5Δy atol = 1e-15
end

@testset "Full-volume restrictions" begin
    sx = 0.4
    sy = 0.4
    for ix in -1.6:0.4:1.6
        square_volume = Khronos.Volume(center=[ix * Δx, ix * Δx, 0], size=[sx, sy, 0])
        restrict_to_array!(w, square_volume)
        @test (sum(w)) * Δx * Δy ≈ sx * sy atol = 1e-15
    end


    sx = 0.4
    sy = 0.5
    for ix in -1.6:0.4:1.6
        square_volume = Khronos.Volume(center=[ix * Δx, ix * Δx, 0], size=[sx, sy, 0])
        restrict_to_array!(w, square_volume)
        @test (sum(w)) * Δx * Δy ≈ sx * sy atol = 1e-15
    end
end

@testset "Subpixel-volume restrictions" begin
    sx = 5.0
    sy = 0.5
    for ix in -1.6:0.4:1.6
        line_x_volume = Khronos.Volume(center=[0.14, ix * Δy, 0], size=[sx * Δx, sy * Δy, 0])
        restrict_to_array!(w, line_x_volume)
        @test sum(w) ≈ sx * sy atol = 1e-14
    end

    sx = 0.4
    sy = 6.0
    for ix in -0.8:0.4:0.8
        line_x_volume = Khronos.Volume(center=[ix * Δx, 0.14, 0], size=[sx * Δx, sy * Δy, 0])
        restrict_to_array!(w, line_x_volume)
        @test sum(w) ≈ sx * sy atol = 1e-14
    end
end
