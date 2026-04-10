#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for the khronos.meep wrapper.

These tests validate that the meep-compatible API constructs all objects
correctly and produces the expected parameter values. They do NOT require
Julia or CUDA — they test the Python layer only.
"""

import math
import sys
import warnings

# ------------------------------------------------------------------- #
# Test helpers
# ------------------------------------------------------------------- #

_pass = 0
_fail = 0


def check(name, condition):
    global _pass, _fail
    if condition:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {name}")


def approx(a, b, tol=1e-10):
    return abs(a - b) < tol


# ------------------------------------------------------------------- #
# Tests
# ------------------------------------------------------------------- #

def test_constants():
    import khronos.meep as mp

    # Field components are unique integers
    components = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
    check("field components unique", len(set(components)) == 6)

    # Direction constants
    check("X != Y != Z", mp.X != mp.Y and mp.Y != mp.Z)

    # Parity
    check("TE == EVEN_Z", mp.TE == mp.EVEN_Z)
    check("TM == ODD_Z", mp.TM == mp.ODD_Z)

    # inf
    check("inf > 1e10", mp.inf > 1e10)

    # ALL
    check("ALL == -1", mp.ALL == -1)
    check("AUTOMATIC == -1", mp.AUTOMATIC == -1)


def test_vector3():
    import khronos.meep as mp

    v = mp.Vector3(1, 2, 3)
    check("Vector3 x", v.x == 1)
    check("Vector3 y", v.y == 2)
    check("Vector3 z", v.z == 3)

    # Arithmetic
    v2 = mp.Vector3(4, 5, 6)
    s = v + v2
    check("add", s.x == 5 and s.y == 7 and s.z == 9)

    d = v2 - v
    check("sub", d.x == 3 and d.y == 3 and d.z == 3)

    m = v * 2
    check("mul scalar", m.x == 2 and m.y == 4 and m.z == 6)

    m2 = 3 * v
    check("rmul scalar", m2.x == 3 and m2.y == 6 and m2.z == 9)

    # Dot product
    check("dot", v.dot(v2) == 32)

    # Cross product
    c = v.cross(v2)
    check("cross x", c.x == 2 * 6 - 3 * 5)  # -3
    check("cross y", c.y == 3 * 4 - 1 * 6)  # 6
    check("cross z", c.z == 1 * 5 - 2 * 4)  # -3

    # Norm
    check("norm", approx(v.norm(), math.sqrt(14)))

    # Unit
    u = v.unit()
    check("unit norm", approx(u.norm(), 1.0))

    # Negation
    n = -v
    check("neg", n.x == -1 and n.y == -2 and n.z == -3)

    # Default
    v0 = mp.Vector3()
    check("default zero", v0.x == 0 and v0.y == 0 and v0.z == 0)

    # Indexing
    check("index 0", v[0] == 1)
    check("index 1", v[1] == 2)
    check("index 2", v[2] == 3)

    # close
    check("close", v.close(mp.Vector3(1, 2, 3)))
    check("not close", not v.close(mp.Vector3(1, 2, 4)))

    # scale
    vs = v.scale(2)
    check("scale", vs.x == 2 and vs.y == 4 and vs.z == 6)


def test_medium():
    import khronos.meep as mp

    # Basic
    m = mp.Medium(epsilon=12)
    check("epsilon", m.epsilon == 12)

    # From index
    m2 = mp.Medium(index=3.5)
    check("index->epsilon", approx(m2.epsilon, 12.25))

    # Predefined
    check("vacuum eps", mp.vacuum.epsilon == 1.0)
    check("air eps", mp.air.epsilon == 1.0)
    check("metal is PEC", hasattr(mp.metal, 'epsilon'))

    # Susceptibility
    lor = mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=1.0)
    check("LorentzianSusc", lor.frequency == 1.0)

    drude = mp.DrudeSusceptibility(frequency=1.0, gamma=0.1, sigma=1.0)
    check("DrudeSusc", drude.frequency == 1.0)

    # Medium with susceptibilities
    m3 = mp.Medium(epsilon=2.0, E_susceptibilities=[lor])
    check("medium with susc", len(m3.E_susceptibilities) == 1)


def test_geometry():
    import khronos.meep as mp

    # Block
    b = mp.Block(size=mp.Vector3(mp.inf, 1, 2),
                 center=mp.Vector3(0, 0, 0),
                 material=mp.Medium(epsilon=12))
    check("Block size", b.size.y == 1)
    check("Block material", b.material.epsilon == 12)

    # Sphere
    s = mp.Sphere(radius=0.5, center=mp.Vector3(1, 0, 0))
    check("Sphere radius", s.radius == 0.5)
    check("Sphere center", s.center.x == 1)

    # Cylinder
    c = mp.Cylinder(radius=1.0, height=2.0, axis=mp.Vector3(0, 0, 1))
    check("Cylinder radius", c.radius == 1.0)
    check("Cylinder height", c.height == 2.0)

    # Prism
    verts = [mp.Vector3(-1, -1), mp.Vector3(1, -1),
             mp.Vector3(1, 1), mp.Vector3(-1, 1)]
    p = mp.Prism(vertices=verts, height=0.5, axis=mp.Vector3(0, 0, 1))
    check("Prism vertices", len(p.vertices) == 4)
    check("Prism height", p.height == 0.5)


def test_sources():
    import khronos.meep as mp

    # GaussianSource
    gs = mp.GaussianSource(frequency=0.15, fwidth=0.1)
    check("GaussianSource freq", gs.frequency == 0.15)
    check("GaussianSource fwidth", gs.fwidth == 0.1)

    # From wavelength
    gs2 = mp.GaussianSource(wavelength=1.55, fwidth=0.1)
    check("GaussianSource wavelength", approx(gs2.frequency, 1.0 / 1.55))

    # ContinuousSource
    cs = mp.ContinuousSource(frequency=0.15)
    check("ContinuousSource freq", cs.frequency == 0.15)

    # Source (point dipole)
    src = mp.Source(gs, component=mp.Ez, center=mp.Vector3(-7))
    check("Source component", src.component == mp.Ez)
    check("Source center", src.center.x == -7)
    check("Source size (default point)", src.size.x == 0 and src.size.y == 0)

    # Source (line source)
    src2 = mp.Source(gs, component=mp.Ez,
                     center=mp.Vector3(-7), size=mp.Vector3(0, 1))
    check("Source size line", src2.size.y == 1)

    # EigenModeSource
    ems = mp.EigenModeSource(
        src=gs, center=mp.Vector3(-5), size=mp.Vector3(0, 2),
        eig_band=1, eig_parity=mp.ODD_Z,
    )
    check("EigenModeSource band", ems.eig_band == 1)
    check("EigenModeSource parity", ems.eig_parity == mp.ODD_Z)
    check("EigenModeSource center", ems.center.x == -5)


def test_boundaries():
    import khronos.meep as mp

    # PML
    pml = mp.PML(1.0)
    check("PML thickness", pml.thickness == 1.0)
    check("PML direction default", pml.direction == mp.ALL)

    # PML directional
    pml_x = mp.PML(0.5, direction=mp.X)
    check("PML direction X", pml_x.direction == mp.X)

    # PML sided
    pml_low = mp.PML(0.5, side=mp.Low)
    check("PML side Low", pml_low.side == mp.Low)

    # Absorber
    ab = mp.Absorber(2.0)
    check("Absorber thickness", ab.thickness == 2.0)
    check("Absorber is PML subclass", isinstance(ab, mp.PML))


def test_flux_regions():
    import khronos.meep as mp

    fr = mp.FluxRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2))
    check("FluxRegion center", fr.center.x == 5)
    check("FluxRegion size", fr.size.y == 2)

    mr = mp.ModeRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2))
    check("ModeRegion", isinstance(mr, mp.FluxRegion))


def test_volume():
    import khronos.meep as mp

    vol = mp.Volume(center=mp.Vector3(0, 0), size=mp.Vector3(16, 8))
    check("Volume center", vol.center.x == 0)
    check("Volume size", vol.size.x == 16 and vol.size.y == 8)


def test_simulation_construction():
    import khronos.meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8),
        resolution=10,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                           component=mp.Ez, center=mp.Vector3(-7))],
    )
    check("Simulation cell_size", sim.cell_size.x == 16 and sim.cell_size.y == 8)
    check("Simulation resolution", sim.resolution == 10)
    check("Simulation geometry", len(sim.geometry) == 1)
    check("Simulation sources", len(sim.sources) == 1)
    check("Simulation boundary_layers", len(sim.boundary_layers) == 1)
    check("Simulation fields None", sim.fields is None)
    check("Simulation Courant default", sim.Courant == 0.5)


def test_add_flux():
    import khronos.meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8),
        resolution=10,
    )

    # fcen/df/nfreq form
    flux = sim.add_flux(0.15, 0.1, 50,
                        mp.FluxRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2)))
    check("add_flux nfreqs", flux.nfreqs == 50)

    freqs = mp.get_flux_freqs(flux)
    check("flux_freqs length", len(freqs) == 50)
    check("flux_freqs first", approx(freqs[0], 0.1))
    check("flux_freqs last", approx(freqs[-1], 0.2))

    # Array form
    freq_arr = [0.1, 0.15, 0.2]
    flux2 = sim.add_flux(freq_arr,
                         mp.FluxRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2)))
    check("add_flux array", flux2.nfreqs == 3)


def test_add_dft_fields():
    import khronos.meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8),
        resolution=10,
    )

    dft = sim.add_dft_fields([mp.Ez], 0.15, 0, 1,
                             where=mp.Volume(center=mp.Vector3(), size=mp.Vector3(16, 8)))
    check("add_dft_fields nfreqs", dft.nfreqs == 1)
    check("add_dft_fields components", dft.components == [mp.Ez])

    # Multiple components
    dft2 = sim.add_dft_fields([mp.Ez, mp.Hx], 0.15, 0.1, 10,
                              where=mp.Volume(center=mp.Vector3(), size=mp.Vector3(16, 8)))
    check("add_dft_fields multi-comp", len(dft2.components) == 2)
    check("add_dft_fields multi-freq", dft2.nfreqs == 10)


def test_add_mode_monitor():
    import khronos.meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8),
        resolution=10,
    )

    mon = sim.add_mode_monitor(0.15, 0.1, 10,
                               mp.ModeRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2)))
    check("add_mode_monitor nfreqs", mon.nfreqs == 10)
    check("add_mode_monitor is DftFlux", isinstance(mon, mp.DftFlux))


def test_reset():
    import khronos.meep as mp

    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)
    flux = sim.add_flux(0.15, 0.1, 10,
                        mp.FluxRegion(center=mp.Vector3(5), size=mp.Vector3(0, 2)))

    sim.reset_meep()
    check("reset fields", sim.fields is None)
    check("reset _is_initialized", not sim._is_initialized)


def test_symmetries():
    import khronos.meep as mp

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = mp.Simulation(
            cell_size=mp.Vector3(16, 8),
            resolution=10,
            symmetries=[mp.Mirror(mp.Y)],
        )
        check("symmetry warning", len(w) >= 1)

    check("symmetries stored", len(sim.symmetries) == 1)
    check("Mirror direction", sim.symmetries[0].direction == mp.Y)


def test_step_functions_warn():
    import khronos.meep as mp

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.at_every(10, None)
        mp.at_beginning(None)
        mp.at_end(None)
        mp.after_sources(None)
        mp.output_epsilon(None)
        mp.output_efield_z(None)
        check("step func warnings", len(w) >= 6)


def test_stop_conditions():
    import khronos.meep as mp

    sc1 = mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-3)
    check("stop_when_fields_decayed", sc1.decay_by == 1e-3)
    check("stop_when_fields_decayed dt", sc1.dt == 50)

    sc2 = mp.stop_when_dft_decayed(tol=1e-5)
    check("stop_when_dft_decayed", sc2.tol == 1e-5)

    sc3 = mp.stop_when_energy_decayed(dt=50, decay_by=1e-11)
    check("stop_when_energy_decayed", sc3.decay_by == 1e-11)


def test_harminv_warning():
    import khronos.meep as mp

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = mp.Harminv(mp.Ez, mp.Vector3(), 0.15, 0.1)
        check("Harminv warning", len(w) >= 1)
    check("Harminv freqs empty", h.freqs == [])


def test_units():
    from khronos.meep._units import (
        meep_to_khronos_length, meep_to_khronos_freq, meep_to_khronos_time,
        khronos_to_meep_length, khronos_to_meep_freq, khronos_to_meep_time,
        set_length_scale, get_length_scale,
    )

    # Default: a = 1 μm (meep units = Khronos units)
    check("default scale", get_length_scale() == 1.0)
    check("length 1:1", meep_to_khronos_length(5) == 5)
    check("freq 1:1", meep_to_khronos_freq(0.15) == 0.15)
    check("time 1:1", meep_to_khronos_time(100) == 100)

    # Roundtrip
    check("length roundtrip", approx(khronos_to_meep_length(meep_to_khronos_length(7)), 7))
    check("freq roundtrip", approx(khronos_to_meep_freq(meep_to_khronos_freq(0.3)), 0.3))
    check("time roundtrip", approx(khronos_to_meep_time(meep_to_khronos_time(50)), 50))

    # Custom scale: a = 0.5 μm
    set_length_scale(0.5)
    check("custom length", meep_to_khronos_length(2) == 1.0)
    check("custom freq", approx(meep_to_khronos_freq(0.3), 0.6))
    check("custom time", approx(meep_to_khronos_time(100), 50))

    # Reset
    set_length_scale(1.0)
    check("reset scale", get_length_scale() == 1.0)


def test_material_grid():
    import khronos.meep as mp

    mg = mp.MaterialGrid(
        grid_size=mp.Vector3(20, 20),
        medium1=mp.air,
        medium2=mp.Medium(epsilon=12),
    )
    check("MaterialGrid grid_size", mg.grid_size.x == 20)
    check("MaterialGrid medium1", mg.medium1.epsilon == 1.0)
    check("MaterialGrid medium2", mg.medium2.epsilon == 12.0)

    # Update weights (requires numpy)
    try:
        import numpy as np
        mg.update_weights(np.ones(400))
        check("MaterialGrid update_weights", mg.weights is not None)
    except ImportError:
        print("  SKIP: MaterialGrid update_weights (numpy not available)")


def test_adjoint_module():
    from khronos.meep import adjoint as mpa

    # Check all expected names exist
    check("DesignRegion", hasattr(mpa, 'DesignRegion'))
    check("OptimizationProblem", hasattr(mpa, 'OptimizationProblem'))
    check("EigenmodeCoefficient", hasattr(mpa, 'EigenmodeCoefficient'))
    check("FourierFields", hasattr(mpa, 'FourierFields'))
    check("conic_filter", hasattr(mpa, 'conic_filter'))
    check("tanh_projection", hasattr(mpa, 'tanh_projection'))
    check("constraint_solid", hasattr(mpa, 'constraint_solid'))
    check("constraint_void", hasattr(mpa, 'constraint_void'))
    check("gray_indicator", hasattr(mpa, 'gray_indicator'))


def test_adjoint_design_region():
    import khronos.meep as mp
    from khronos.meep import adjoint as mpa

    mg = mp.MaterialGrid(
        grid_size=mp.Vector3(20, 20),
        medium1=mp.air,
        medium2=mp.Medium(epsilon=12),
    )
    dr = mpa.DesignRegion(
        mg,
        volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1)),
    )
    check("DesignRegion center", dr.center.x == 0)
    check("DesignRegion size", dr.size.x == 1)
    check("DesignRegion design_parameters", dr.design_parameters is mg)
    check("DesignRegion num_design_params", dr.num_design_params == 400)


def test_adjoint_objective_construction():
    import khronos.meep as mp
    from khronos.meep import adjoint as mpa

    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)

    ec = mpa.EigenmodeCoefficient(
        sim,
        volume=mp.Volume(center=mp.Vector3(5), size=mp.Vector3(0, 2)),
        mode=1,
        forward=True,
    )
    check("EigenmodeCoefficient mode", ec.mode == 1)
    check("EigenmodeCoefficient forward", ec.forward is True)

    ff = mpa.FourierFields(
        sim,
        volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(2, 2)),
        component=mp.Ez,
    )
    check("FourierFields component", ff.component == mp.Ez)


def test_filters():
    """Test filter functions (require numpy)."""
    try:
        import numpy as np
    except ImportError:
        print("  SKIP: filters (numpy not available)")
        return

    from khronos.meep import adjoint as mpa

    # tanh_projection
    x = np.linspace(0, 1, 100)
    proj = mpa.tanh_projection(x, beta=8, eta=0.5)
    check("tanh_projection shape", proj.shape == x.shape)
    check("tanh_projection range", proj.min() >= 0 and proj.max() <= 1)
    check("tanh_projection midpoint", approx(float(mpa.tanh_projection(
        np.array([0.5]), 8, 0.5)[0]), 0.5, tol=0.01))

    # conic_filter
    Nx, Ny = 20, 20
    rho = np.random.rand(Nx, Ny)
    filtered = mpa.conic_filter(rho, radius=3, Lx=Nx, Ly=Ny, resolution=1)
    check("conic_filter shape", filtered.shape == rho.shape)
    check("conic_filter smoothed", filtered.std() < rho.std())

    # gray_indicator
    gi = mpa.gray_indicator(rho)
    check("gray_indicator scalar", np.isscalar(gi) or gi.size == 1)

    # heaviside_projection
    hp = mpa.heaviside_projection(x, beta=8, eta=0.5)
    check("heaviside_projection shape", hp.shape == x.shape)


def test_interpolate():
    import khronos.meep as mp

    result = mp.interpolate(3, [0, 10])
    check("interpolate length", len(result) == 5)  # 2 endpoints + 3 intermediate
    check("interpolate values", result == [0, 2.5, 5.0, 7.5, 10])


def test_bend_flux_pattern():
    """Test the canonical two-run normalization pattern (construction only)."""
    import khronos.meep as mp

    # Parameters
    resolution = 10
    sx, sy = 16, 8
    dpml = 1
    fcen, df, nfreq = 0.15, 0.1, 50

    # Straight waveguide
    geometry = [mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ez,
                         center=mp.Vector3(-0.5 * sx + dpml),
                         size=mp.Vector3(0, 1))]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
    )

    # Add monitors
    refl = sim.add_flux(fcen, df, nfreq,
                        mp.FluxRegion(center=mp.Vector3(-0.5 * sx + dpml + 0.5),
                                      size=mp.Vector3(0, 2)))
    tran = sim.add_flux(fcen, df, nfreq,
                        mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml),
                                      size=mp.Vector3(0, 2)))

    check("bend_flux refl freqs", refl.nfreqs == nfreq)
    check("bend_flux tran freqs", tran.nfreqs == nfreq)
    check("bend_flux sim monitors", len(sim._flux_monitors) == 2)

    # After run would call:
    # straight_refl_data = sim.get_flux_data(refl)
    # straight_tran_flux = mp.get_fluxes(tran)
    # sim.reset_meep()
    # sim.load_minus_flux_data(refl, straight_refl_data)

    sim.reset_meep()
    check("bend_flux reset", sim.fields is None)
    check("bend_flux monitors preserved", len(sim._flux_monitors) == 2)


def test_tfsf_source():
    """Test TFSFSource construction (Khronos extension)."""
    import khronos.meep as mp

    gs = mp.GaussianSource(0.15, fwidth=0.1)
    tfsf = mp.TFSFSource(
        src=gs,
        center=mp.Vector3(0, 0),
        size=mp.Vector3(4, 4),
        direction=1,
        injection_axis=2,
        pol_angle=0.0,
    )
    check("TFSFSource center", tfsf.center.x == 0)
    check("TFSFSource size", tfsf.size.x == 4)
    check("TFSFSource direction", tfsf.direction == 1)
    check("TFSFSource injection_axis", tfsf.injection_axis == 2)
    check("TFSFSource src", tfsf.src is gs)


def test_diffraction_monitor():
    """Test DftDiffraction and add_diffraction_monitor (Khronos extension)."""
    import khronos.meep as mp

    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)

    dm = sim.add_diffraction_monitor(
        0.15, 0.1, 10,
        center=mp.Vector3(0, 0, 2),
        size=mp.Vector3(4, 4, 0),
    )
    check("DftDiffraction type", isinstance(dm, mp.DftDiffraction))
    check("DftDiffraction nfreqs", dm.nfreqs == 10)
    check("DftDiffraction center", dm.center.x == 0)
    check("DftDiffraction stored", len(sim._diffraction_monitors) == 1)


def test_layer_stack():
    """Test LayerSpec and LayerStack construction (Khronos extension)."""
    import khronos.meep as mp

    ls1 = mp.LayerSpec(z_min=0, z_max=0.5, eps=2.4**2)
    ls2 = mp.LayerSpec(z_min=0.5, z_max=mp.inf, eps=1.0)
    check("LayerSpec eps", ls1.eps == 2.4**2)
    check("LayerSpec mu default", ls1.mu == 1.0)

    stack = mp.LayerStack([ls1, ls2])
    check("LayerStack length", len(stack.layers) == 2)

    # Use in add_near2far
    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)
    n2f = sim.add_near2far(
        0.15, 0.1, 5,
        mp.Near2FarRegion(center=mp.Vector3(0, 0, 1), size=mp.Vector3(4, 4, 0)),
        layer_stack=stack,
        theta=[0.0, 0.1, 0.2],
        phi=[0.0],
        proj_distance=1e4,
    )
    check("near2far layer_stack", n2f.layer_stack is stack)
    check("near2far theta", n2f.theta == [0.0, 0.1, 0.2])
    check("near2far proj_distance", n2f.proj_distance == 1e4)


def test_backend_precision():
    """Test backend and precision params (Khronos extension)."""
    import khronos.meep as mp

    # Default
    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)
    check("default backend", sim.backend == "cuda")
    check("default precision", sim.precision == "float32")

    # Custom
    sim2 = mp.Simulation(
        cell_size=mp.Vector3(16, 8), resolution=10,
        backend="cpu", precision="float64",
    )
    check("cpu backend", sim2.backend == "cpu")
    check("float64 precision", sim2.precision == "float64")


def test_eps_averaging_modes():
    """Test extended eps_averaging with volume mode (Khronos extension)."""
    import khronos.meep as mp

    sim1 = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10,
                         eps_averaging=True)
    check("eps_averaging True", sim1.eps_averaging is True)

    sim2 = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10,
                         eps_averaging="volume")
    check("eps_averaging volume", sim2.eps_averaging == "volume")

    sim3 = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10,
                         eps_averaging=False)
    check("eps_averaging False", sim3.eps_averaging is False)


def test_nonuniform_grid():
    """Test non-uniform grid params (Khronos extension)."""
    import khronos.meep as mp

    dl = [0.1] * 20 + [0.05] * 40 + [0.1] * 20
    sim = mp.Simulation(
        cell_size=mp.Vector3(16, 8), resolution=10,
        grid_dl_x=dl,
    )
    check("grid_dl_x stored", sim.grid_dl_x is dl)
    check("grid_dl_x length", len(sim.grid_dl_x) == 80)
    check("grid_dl_y None", sim.grid_dl_y is None)


def test_estimate_memory_exists():
    """Test that estimate_memory method exists."""
    import khronos.meep as mp

    sim = mp.Simulation(cell_size=mp.Vector3(16, 8), resolution=10)
    check("estimate_memory exists", hasattr(sim, 'estimate_memory'))
    check("estimate_memory callable", callable(sim.estimate_memory))


def test_batch_functions_exist():
    """Test that batch execution functions exist (Khronos extension)."""
    import khronos.meep as mp

    check("run_batch exists", hasattr(mp, 'run_batch'))
    check("run_batch_concurrent exists", hasattr(mp, 'run_batch_concurrent'))
    check("run_batch_multi_gpu exists", hasattr(mp, 'run_batch_multi_gpu'))
    check("run_batch callable", callable(mp.run_batch))
    check("run_batch_concurrent callable", callable(mp.run_batch_concurrent))
    check("run_batch_multi_gpu callable", callable(mp.run_batch_multi_gpu))


def test_lee_functions_exist():
    """Test that LEE computation functions exist (Khronos extension)."""
    import khronos.meep as mp

    check("compute_LEE exists", hasattr(mp, 'compute_LEE'))
    check("compute_incoherent_LEE exists", hasattr(mp, 'compute_incoherent_LEE'))
    check("compute_LEE callable", callable(mp.compute_LEE))
    check("compute_incoherent_LEE callable", callable(mp.compute_incoherent_LEE))


def test_gdsii_import_exists():
    """Test that import_gdsii function exists (Khronos extension)."""
    import khronos.meep as mp

    check("import_gdsii exists", hasattr(mp, 'import_gdsii'))
    check("import_gdsii callable", callable(mp.import_gdsii))


def test_full_api_surface():
    """Verify all expected meep API names exist in khronos.meep."""
    import khronos.meep as mp

    expected_names = [
        # Constants
        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "Dx", "Dy", "Dz", "Bx", "By", "Bz",
        "Dielectric", "Permeability",
        "X", "Y", "Z", "ALL", "AUTOMATIC", "inf",
        "TE", "TM", "EVEN_Z", "ODD_Z", "NO_PARITY",
        "Low", "High",
        # Types
        "Vector3", "Volume",
        "Medium", "LorentzianSusceptibility", "DrudeSusceptibility",
        "MaterialGrid",
        "Block", "Sphere", "Cylinder", "Prism",
        "FluxRegion", "ModeRegion",
        "Mirror", "Rotate2", "Rotate4",
        "GaussianSource", "ContinuousSource", "CustomSource",
        "Source", "EigenModeSource", "GaussianBeamSource",
        "PML", "Absorber",
        "Simulation",
        "DftFlux", "DftFields",
        "FluxData", "EigenmodeData",
        # Functions
        "get_fluxes", "get_flux_freqs",
        "stop_when_fields_decayed", "stop_when_dft_decayed",
        "at_beginning", "at_end", "at_every",
        "after_sources", "output_epsilon", "output_efield_z",
        "Harminv", "quiet", "verbosity",
        "interpolate",
        # Predefined materials
        "vacuum", "air", "metal",
        "perfect_electric_conductor", "perfect_magnetic_conductor",
        # Adjoint
        "adjoint",
        # Khronos extensions
        "TFSFSource",
        "LayerSpec", "LayerStack", "import_gdsii",
        "DftDiffraction", "get_diffraction_efficiencies", "get_diffraction_freqs",
        "run_batch", "run_batch_concurrent", "run_batch_multi_gpu",
        "compute_LEE", "compute_incoherent_LEE",
    ]

    missing = [name for name in expected_names if not hasattr(mp, name)]
    check(f"API surface ({len(expected_names)} names)", len(missing) == 0)
    if missing:
        print(f"    Missing: {missing}")


# ------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------- #

def main():
    global _pass, _fail

    tests = [
        ("constants", test_constants),
        ("vector3", test_vector3),
        ("medium", test_medium),
        ("geometry", test_geometry),
        ("sources", test_sources),
        ("boundaries", test_boundaries),
        ("flux_regions", test_flux_regions),
        ("volume", test_volume),
        ("simulation_construction", test_simulation_construction),
        ("add_flux", test_add_flux),
        ("add_dft_fields", test_add_dft_fields),
        ("add_mode_monitor", test_add_mode_monitor),
        ("reset", test_reset),
        ("symmetries", test_symmetries),
        ("step_functions_warn", test_step_functions_warn),
        ("stop_conditions", test_stop_conditions),
        ("harminv_warning", test_harminv_warning),
        ("units", test_units),
        ("material_grid", test_material_grid),
        ("adjoint_module", test_adjoint_module),
        ("adjoint_design_region", test_adjoint_design_region),
        ("adjoint_objective_construction", test_adjoint_objective_construction),
        ("filters", test_filters),
        ("interpolate", test_interpolate),
        ("bend_flux_pattern", test_bend_flux_pattern),
        # Khronos extension tests
        ("tfsf_source", test_tfsf_source),
        ("diffraction_monitor", test_diffraction_monitor),
        ("layer_stack", test_layer_stack),
        ("backend_precision", test_backend_precision),
        ("eps_averaging_modes", test_eps_averaging_modes),
        ("nonuniform_grid", test_nonuniform_grid),
        ("estimate_memory_exists", test_estimate_memory_exists),
        ("batch_functions_exist", test_batch_functions_exist),
        ("lee_functions_exist", test_lee_functions_exist),
        ("gdsii_import_exists", test_gdsii_import_exists),
        # Must be last
        ("full_api_surface", test_full_api_surface),
    ]

    for name, func in tests:
        try:
            func()
        except Exception as e:
            _fail += 1
            print(f"  FAIL: {name} raised {type(e).__name__}: {e}")

    print()
    print(f"Results: {_pass} passed, {_fail} failed")
    if _fail > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
