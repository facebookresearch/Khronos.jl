# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using Test

function build_sim()
    return fdtd.Simulation(
        cell_size = [10.0, 10.0, 10.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        sources = [
            fdtd.UniformSource(
                time_profile = fdtd.ContinuousWaveSource(fcen = 1.0),
                component = fdtd.Ez(),
                center = [0.023, 0.784, 0.631],
                size = [0.0, 0.0, 0.0],
            ),
        ],
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )
end

# ------------------------------------------------ #
# Test abstract kernels
# ------------------------------------------------ #

# For now, let's just make sure we don't have an NaNs.

@testset "Curl B from E" begin
    sim = build_sim()
    fdtd.prepare_simulation!(sim)
    fdtd.step_B_from_E!(sim)
    for field_component in ([sim.fields.fBx, sim.fields.fBy, sim.fields.fBz])
        @test !any(isnan.(field_component))
    end
    return
end

@testset "Update H from B" begin
    sim = build_sim()
    fdtd.prepare_simulation!(sim)
    fdtd.update_H_from_B!(sim)
    for field_component in ([sim.fields.fHx, sim.fields.fHy, sim.fields.fHz])
        @test !any(isnan.(field_component))
    end
    return
end

@testset "Curl D from H" begin
    sim = build_sim()
    fdtd.prepare_simulation!(sim)
    fdtd.step_D_from_H!(sim)
    for field_component in ([sim.fields.fDx, sim.fields.fDy, sim.fields.fDz])
        @test !any(isnan.(field_component))
    end
    return
end

@testset "Update E from D" begin
    sim = build_sim()
    fdtd.prepare_simulation!(sim)
    fdtd.step_D_from_H!(sim)
    for field_component in ([sim.fields.fEx, sim.fields.fEy, sim.fields.fEz])
        @test !any(isnan.(field_component))
    end
    return
end

# ------------------------------------------------ #
# Test sub kernels
# ------------------------------------------------ #
