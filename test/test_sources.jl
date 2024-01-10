# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using Test

# ------------------------------------------------ #
# Test spatial sources
# ------------------------------------------------ #

function build_sim(sources::Vector{<:fdtd.Source})
    return fdtd.Simulation(
        cell_size = [10.,10.,10.],
        cell_center = [0.,0.,0.],
        resolution = 10,
        sources = sources,
        boundaries = [[1.,1.],[1.,1.],[1.,1.]]
    )
end

function test_source_size(sim::fdtd.SimulationData, component::fdtd.Field, source_size::Int)
    fdtd.prepare_simulation!(sim)
    A = fdtd.get_source_from_field_component(sim, component)
    
    # ensure that the array starts out as zeros
    @test length(A[A .!= 0]) == 0
    
    fdtd.update_electric_sources!(sim,1)
    fdtd.update_magnetic_sources!(sim,1)
    
    # test the final result
    @test length(A[A .!= 0]) == source_size
end

@testset "point sources" begin
    for component in [fdtd.Ex(), fdtd.Ey(), fdtd.Ez(), fdtd.Hx(), fdtd.Hy(), fdtd.Hz()]
        point_source = [fdtd.UniformSource(
            time_profile = fdtd.ContinuousWaveSource(fcen=1.),
            component = component,
            center = [0.023, 0.784, 0.631],
            size = [0., 0., 0.])
        ]
        sim = build_sim(point_source)
        # point is in the middle of the voxel, so it should touch 8 corners
        test_source_size(sim, component, 8)
    end
end

@testset "line sources" begin
    for dir in [1,2,3]
        for component in [fdtd.Ex(), fdtd.Ey(), fdtd.Ez(), fdtd.Hx(), fdtd.Hy(), fdtd.Hz()]
            size = [0., 0., 0.]
            size[dir] = Inf
            point_source = [fdtd.UniformSource(
                time_profile = fdtd.ContinuousWaveSource(fcen=1.),
                component = component,
                center = [0.023, 0.784, 0.631],
                size = size)
            ]
            sim = build_sim(point_source)
            # The line is in the middle of the voxel, so it should touch four surrounding nodes
            num_points = fdtd.get_component_voxel_count(sim,component)[dir] * 4
            test_source_size(sim, component, num_points)
        end
    end
end

# TODO Plane sources
# TODO Volume sources

# ------------------------------------------------ #
# Test temporal sources
# ------------------------------------------------ #
@testset "Continuous sources" begin
    cw_src = fdtd.ContinuousWaveSource(fcen=1.)
    t = range(0.0,10.0,step=0.05)
    y = zeros(size(t))
    for (t_idx, current_t) ∈ enumerate(t)
        y[t_idx] = fdtd.eval_time_source(cw_src,current_t)
    end
    # For now, just make sure we don't have any NaNs.
    @test !any(isnan.(y))
    return
end