# (c) Meta Platforms, Inc. and affiliates.
#
# Test the functionality of all the sources.

import Khronos
using Test
using CairoMakie

function build_sim(sources::Vector{<:Khronos.Source}, monitors)
    return Khronos.Simulation(
        cell_size = [4.0, 4.0, 4.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        sources = sources,
        monitors = monitors,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    )
end

@testset "Visualization" begin
    CW_pt_source = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ]

    pulse_pt_source = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 1.0, fwidth = 0.1),
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ]

    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0, 0, 0],
            size = [1.0, 1.0, 0.0],
            frequencies = [1.0],
        ),
    ]

    @testset "plot2D" begin
        # Simply ensure no errors are thrown
        sim = build_sim(CW_pt_source, [])
        Khronos.run(sim, until = 0.1)
        @test_nowarn Khronos.plot2D(
            sim,
            Khronos.Ex(),
            Khronos.Volume([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        )
    end

    @testset "plot DFT monitors" begin
        # Simply ensure no errors are thrown
        sim = build_sim(pulse_pt_source, monitors)
        Khronos.run(sim, until = 0.1)
        @test_nowarn Khronos.plot_monitor(sim.monitors[1], 1)
    end

    @testset "plot Gaussian pulse sources" begin
        sim = build_sim(pulse_pt_source, [])
        Khronos.run(sim, until = 0.1)

        frequencies = 0.8:0.05:1.2
        @test_nowarn Khronos.plot_timesource(sim, pulse_pt_source[1], frequencies)
    end

end
