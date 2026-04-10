# Khronos Python API Reference (meep-compatible)

The Python API mirrors [meep's Python interface](https://meep.readthedocs.io/). Replace `import meep as mp` with `import khronos.meep as mp` and most scripts work as-is.

## Vector3

```python
mp.Vector3(x=0, y=0, z=0)
v.x, v.y, v.z              # component access
v.norm()                    # magnitude
v.dot(other)                # dot product
v.cross(other)              # cross product
v + w, v - w, v * scalar    # arithmetic
```

## Materials

```python
mp.Medium(epsilon=1, mu=1, epsilon_diag=..., epsilon_offdiag=...,
          D_conductivity=0, B_conductivity=0,
          E_susceptibilities=[], H_susceptibilities=[],
          valid_freq_range=mp.FreqRange(), chi3=0)

# Dispersive models
mp.LorentzianSusceptibility(frequency=0, gamma=0, sigma=0, sigma_diag=..., sigma_offdiag=...)
mp.DrudeSusceptibility(frequency=0, gamma=0, sigma=0)

# Predefined
mp.vacuum, mp.air, mp.metal
mp.perfect_electric_conductor, mp.perfect_magnetic_conductor
```

## Geometry

```python
mp.Block(center=Vector3(), size=Vector3(), material=mp.Medium(), e1=..., e2=..., e3=...)
mp.Sphere(center=Vector3(), radius=0, material=mp.Medium())
mp.Cylinder(center=Vector3(), radius=0, height=inf, axis=mp.Vector3(0,0,1), material=mp.Medium())
mp.Cone(center=..., radius=..., radius2=0, height=..., axis=..., material=...)
mp.Ellipsoid(center=..., size=..., material=...)
mp.Prism(vertices=[Vector3(...)], height=inf, axis=mp.Vector3(0,0,1), material=...)
mp.Wedge(center=..., radius=..., height=..., axis=..., wedge_angle=..., material=...)
```

**Khronos extensions:**

```python
mp.import_gdsii(filename, layer, zmin, zmax, material)   # GDSII import
mp.LayerStack([mp.LayerSpec(zmin, zmax, eps, mu=1), ...]) # for near2far through layers
```

## Sources

```python
# Time profiles
mp.GaussianSource(frequency, fwidth=0, width=0, start_time=0, cutoff=5, wavelength=None)
mp.ContinuousSource(frequency, start_time=0, end_time=1e20, slowness=3, wavelength=None)
mp.CustomSource(src_func, start_time=-1e20, end_time=1e20, center_frequency=0, fwidth=0)

# Spatial source (point, line, plane, or volume)
mp.Source(src, component, center=Vector3(), size=Vector3(), amplitude=1.0)

# Mode source (waveguide eigenmode)
mp.EigenModeSource(src, center=..., size=..., eig_band=1, direction=mp.AUTOMATIC,
                   eig_kpoint=mp.Vector3(), eig_parity=mp.NO_PARITY, eig_resolution=0)

# Gaussian beam
mp.GaussianBeamSource(src, center=..., size=..., beam_x0=..., beam_kdir=...,
                       beam_w0=..., beam_E0=...)
```

**Khronos extensions:**

```python
# Total-field/scattered-field box
mp.TFSFSource(src, center=..., size=..., k_vector=..., polarization_angle=0)
```

## Boundaries

```python
mp.PML(thickness, direction=mp.ALL, side=mp.ALL, R_asymptotic=1e-15,
       mean_stretch=1.0, pml_profile=None)
mp.Absorber(thickness, direction=mp.ALL, side=mp.ALL)
```

## Simulation

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(),           # required
    resolution=10,                     # pixels per meep unit
    geometry=[...],                    # list of geometric objects
    sources=[...],                     # list of sources
    boundary_layers=[mp.PML(1.0)],     # absorbing boundaries
    symmetries=[],                     # symmetry objects (stored, not enforced)
    default_material=mp.Medium(),      # background material
    eps_averaging=True,                # subpixel smoothing (True, "anisotropic", "volume", False)
    dimensions=3,                      # 2 or 3 (auto-detected from cell_size)
    Courant=0.5,                       # stability factor
    k_point=False,                     # Bloch k-vector or False
    geometry_center=mp.Vector3(),      # cell center offset
    # --- Khronos extensions ---
    backend="cuda",                    # "cuda", "cpu", or "metal"
    precision="float32",               # "float32" or "float64"
    grid_dl_x=None,                    # non-uniform grid spacings (list of floats)
    grid_dl_y=None,
    grid_dl_z=None,
)
```

## Adding Monitors

Monitors are added *before* calling `run()`:

```python
# Flux monitor (power through a surface)
flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=..., size=...))

# Mode monitor (same as flux, for eigenmode decomposition)
mode = sim.add_mode_monitor(fcen, df, nfreq, mp.FluxRegion(center=..., size=...))

# DFT field monitor (frequency-domain field snapshots)
fields = sim.add_dft_fields([mp.Ez, mp.Hx], fcen, df, nfreq,
                             where=mp.Volume(center=..., size=...))

# Near-to-far monitor (far-field radiation patterns)
n2f = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(center=..., size=...),
                        # Khronos extensions:
                        theta=[...], phi=[...], proj_distance=1e6,
                        layer_stack=mp.LayerStack([...]))
```

**Khronos extensions:**

```python
# Diffraction order monitor
diff = sim.add_diffraction_monitor(fcen, df, nfreq, center=..., size=...)
```

## Running

```python
sim.run(until=200)                              # fixed time
sim.run(until_after_sources=200)                # after sources shut off
sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-5))

# Khronos extensions
sim.run(until=200, num_gpus=4)                  # multi-GPU domain decomposition
mem = sim.estimate_memory(verbose=True)         # GPU memory estimate before run
```

## Stop Conditions

```python
mp.stop_when_dft_decayed(tol=1e-11, minimum_run_time=0, maximum_run_time=None)
mp.stop_when_fields_decayed(dt, component, pt, decay_by)  # approximated via DFT convergence
mp.stop_when_energy_decayed(dt=50, decay_by=1e-11)        # approximated via DFT convergence
```

## Data Extraction

```python
# Flux
fluxes = mp.get_fluxes(flux_monitor)           # list of flux values per frequency
freqs = mp.get_flux_freqs(flux_monitor)        # list of frequencies

# Normalization (2-run pattern)
fdata = sim.get_flux_data(flux_monitor)        # save flux data
sim.load_minus_flux_data(flux_monitor, fdata)  # subtract on next run

# Eigenmode decomposition
coeffs = sim.get_eigenmode_coefficients(mode_monitor, [1, 2])
alpha = coeffs.alpha                            # shape: (bands, freqs, 2) — fwd/bwd

# DFT field array
arr = sim.get_dft_array(field_monitor, mp.Ez, freq_index)  # complex numpy array
```

**Khronos extensions:**

```python
efficiencies = mp.get_diffraction_efficiencies(diff_monitor)
freqs = mp.get_diffraction_freqs(diff_monitor)
```

## Field Components & Constants

```python
# Electric field
mp.Ex, mp.Ey, mp.Ez

# Magnetic field
mp.Hx, mp.Hy, mp.Hz

# Material
mp.Dielectric, mp.Permeability

# Directions
mp.X, mp.Y, mp.Z, mp.ALL, mp.AUTOMATIC
mp.Low, mp.High

# Parity
mp.NO_PARITY, mp.EVEN_Z, mp.ODD_Z, mp.EVEN_Y, mp.ODD_Y, mp.TE, mp.TM

# Special
mp.inf
```

## Batch Simulation (Khronos Extension)

```python
# Sequential batch (same geometry, different sources)
results = mp.run_batch(sim, [sources_1, sources_2, ...], until=200)

# Concurrent batch (multi-stream GPU, single GPU)
results = mp.run_batch_concurrent(sim, source_configs, max_concurrent=4)

# Multi-GPU batch (distribute across GPUs)
results = mp.run_batch_multi_gpu(sim, source_configs, n_gpus=4)
```

## Light Extraction Efficiency (Khronos Extension)

```python
lee = mp.compute_LEE(power, theta, phi, cone_half_angle=90.0)
lee = mp.compute_incoherent_LEE(batch_results, theta, phi, cone_half_angle=90.0)
```

## Adjoint Optimization

```python
import khronos.meep.adjoint as mpa

# Design region (parameterized geometry)
design_region = mpa.DesignRegion(
    mp.MaterialGrid(mp.Vector3(Nx, Ny), mp.air, mp.Medium(epsilon=12)),
    center=mp.Vector3(), size=mp.Vector3(2, 2),
)

# Objective function
obj = mpa.EigenmodeCoefficient(sim, mp.Volume(center=..., size=...), mode=1)

# Optimization problem
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[obj],
    objective_arguments=[...],
    design_regions=[design_region],
    frequencies=[fcen],
)

# Evaluate and get gradient
f, dJ_du = opt()  # forward + adjoint in one call

# Topology optimization filters
filtered = mpa.conic_filter(weights, radius, Nx, Ny, resolution)
projected = mpa.tanh_projection(filtered, beta=10, eta=0.5)
```

## Visualization (Khronos Extension)

```python
mp.plot2D(sim)                    # 2D simulation layout
mp.plot_eps(sim)                  # epsilon distribution
mp.plot_boundaries(sim)           # PML boundaries
mp.plot_sources(sim)              # source regions
mp.plot_monitors(sim)             # monitor regions
mp.plot_fields(sim, component)    # DFT field data
```

## Unsupported Features

The following meep features accept arguments for API compatibility but are **not functional** in Khronos:

- **Step functions** (`at_every`, `at_beginning`, `at_end`, etc.) — accepted but ignored with a warning
- **Output functions** (`output_epsilon`, `output_efield_z`, etc.) — no-op
- **Live field access** (`get_field_point`, `get_array` for field components) — not available; use DFT monitors
- **HDF5 output** (`output_dft`, `dump_structure`, `dump_fields`) — not supported
- **`plot2D`/`plot3D`** on Simulation object — use the module-level `mp.plot2D(sim)` instead
- **Symmetry enforcement** — symmetries are stored but the simulation runs the full domain
