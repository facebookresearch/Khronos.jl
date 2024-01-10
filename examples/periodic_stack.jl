import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)

function build_periodic_stack()
    sources = [
        fdtd.PlaneWaveSource(
            time_profile = fdtd.GaussianPulseSource(
               fcen = 1 / λ, fwidth = 0.2 * 1 / λ),
            center = [0,0,0],
            size = [Inf, Inf, 0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
        )
    ]

    mat_low = fdtd.Material(ε=1.5)
    mat_mid = fdtd.Material(ε=2.5)
    mat_high = fdtd.Material(ε=3.5)

    materials = [
        mat_low, mat_mid, mat_low, mat_high, mat_mid, mat_high
    ]
    thicknesses = [
        0.5, 1.0, 0.75, 1.0, 0.25, 0.5
    ]

    z_cur = -2.5
    geometry = []
    for (current_mat, current_thick) in zip(materials,thicknesses)
        z_cur += current_thick / 2.0
        println(z_cur)
        append!(
            geometry,
            [fdtd.Object(
                Cuboid([0.0, 0.0, z_cur], [4.0, 4.0, current_thick]), current_mat)
            ]
            )
        z_cur += current_thick / 2.0
    end
    dpml = 2.5
    sim = fdtd.Simulation(
        cell_size = [6.0,6.0,10.0],
        cell_center = [0.0,0.0,0.0],
        resolution = 20,
        geometry = geometry,
        sources = sources,
        boundaries = [[dpml,dpml],[dpml,dpml],[dpml,dpml]],
    )

    return sim
end

sim = build_periodic_stack()

fdtd.prepare_simulation!(sim)
t_end  = 10.0;
fdtd.run(sim,until=t_end)

scene = fdtd.plot2D(sim, fdtd.Ex(), fdtd.Volume([0.,0.,0.],[4.0,4.0,8.]), plot_geometry=false)
