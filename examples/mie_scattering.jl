using Plots
import fdtd

fdtd.environment!(false, Float64, 3)

function mie_scattering()
    r = 1.0 # radius of sphere
    wvl_min = 2 * π * r / 10
    wvl_max = 2 * π * r / 2
    frq_min = 1 / wvl_max
    frq_max = 1 / wvl_min
    frq_cen = 0.5 * (frq_min + frq_max)
    dfrq = frq_max - frq_min
    nfrq = 100

    ## at least 8 pixels per smallest wavelength, i.e. floor(8/wvl_min)
    resolution = 25

    dpml = 0.5 * wvl_max
    dair = 0.5 * wvl_max

    pml_layers = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

    s = 2 * (dpml + dair + r)
    cell_size = [s, s, s]

    gauss_pulse = fdtd.GaussianPulseSource(1.0, 0.2)

    time_src = gauss_pulse

    sources = [
        fdtd.UniformSource(
            time_profile = time_src,
            component = fdtd.Ez(),
            center = [-4.0, 0.0, 0.0],
            size = [0.0, Inf, Inf],
        ),
        fdtd.UniformSource(
            time_profile = time_src,
            component = fdtd.Hy(),
            center = [-2.0, 0.0, 0.0],
            size = [0.0, Inf, Inf],
            amplitude = -1,
        ),
    ]

    a = fdtd.Cuboid([0.0, 0.0], [2.0, 4.0])
    Si = fdtd.Material(ε = 3, σD = 4)
    air = fdtd.Material(ε = 1)
    geometry = [fdtd.Object(fdtd.Ball([0.0, 0.0, 0.0], 1.0), Si)]

    sim = fdtd.Simulation{fdtd.Data.Array}(
        cell_size = [5.0, 5.0, 5.0],
        cell_center = [0.0, 0.0, 0.0],
        resolution = 10,
        sources = sources,
        #monitors = monitors,
        boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        geometry = geometry,
    )
    fdtd.prepare_simulation(sim)

    return sim

end

sim = mie_scattering()
