# Khronos.jl — Julia API Reference

## Backends

```julia
Khronos.choose_backend(Khronos.CUDADevice(), Float32)   # GPU (NVIDIA)
Khronos.choose_backend(Khronos.CPUDevice(), Float64)     # CPU
Khronos.choose_backend(Khronos.MetalDevice(), Float32)   # GPU (Apple)
```

## Geometry

Shapes come from `GeometryPrimitives.jl`:

```julia
using GeometryPrimitives

Cuboid([cx, cy, cz], [sx, sy, sz])   # center, size
Sphere([cx, cy, cz], radius)
Cylinder([cx, cy, cz], radius, height, axis)
```

Wrap shapes with materials:

```julia
Khronos.Object(
    shape = Cuboid([0, 0, 0], [1, 1, 1]),
    material = Khronos.Material(ε = 12.0),
)
```

Material supports anisotropic permittivity (`εx`, `εy`, `εz`), conductivity (`σD`, `σB`), permeability (`μ`), Kerr nonlinearity (`chi3`), and dispersive poles (via `susceptibilities`).

## Sources

```julia
# Time profiles
Khronos.GaussianPulseSource(fcen = 0.15, fwidth = 0.1)
Khronos.ContinuousWaveSource(fcen = 0.15)

# Spatial sources
Khronos.UniformSource(time_profile = ..., component = Khronos.Ez(), center = [...], size = [...])
Khronos.PlaneWaveSource(time_profile = ..., center = [...], size = [...], k_vector = [...])
Khronos.GaussianBeamSource(time_profile = ..., center = [...], size = [...], beam_w0 = 1.0)
Khronos.ModeSource(time_profile = ..., center = [...], size = [...])
```

## Field Components

```julia
Khronos.Ex(), Khronos.Ey(), Khronos.Ez()
Khronos.Hx(), Khronos.Hy(), Khronos.Hz()
```

## Monitors

```julia
# DFT field monitor (frequency-domain fields)
Khronos.DFTMonitor(component = Khronos.Ez(), center = [...], size = [...], frequencies = [...])

# Flux monitor (Poynting vector through a surface)
Khronos.FluxMonitor(center = [...], size = [...], frequencies = [...])

# Near-to-far field monitor
Khronos.Near2FarMonitor(center = [...], size = [...], frequencies = [...],
                         theta = [...], phi = [...], r = 1e6)

# Diffraction order monitor
Khronos.DiffractionMonitor(center = [...], size = [...], frequencies = [...])
```

## Boundary Conditions

```julia
Khronos.PML()                    # Perfectly matched layer
Khronos.Periodic()               # Periodic boundary
Khronos.Bloch(k = 0.5)           # Bloch-periodic
Khronos.PECBoundary()            # Perfect electric conductor
Khronos.PMCBoundary()            # Perfect magnetic conductor
```

The `boundaries` parameter sets PML thickness per axis: `[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`.

## Subpixel Smoothing

```julia
Khronos.NoSmoothing()            # No subpixel averaging
Khronos.VolumeAveraging()        # Volume-averaged epsilon
Khronos.AnisotropicSmoothing()   # Anisotropic smoothing (most accurate)
```

## Running

```julia
Khronos.run(sim, until = 200.0)
Khronos.run(sim, until_after_sources = 100.0)
Khronos.run(sim, until_after_sources = Khronos.stop_when_dft_decayed(
    tolerance = 1e-4, minimum_runtime = 50.0, maximum_runtime = 1000.0,
))
```

## Data Extraction

```julia
# Flux through a surface
flux = Khronos.get_flux(sim.monitors[1].monitor_data)

# DFT fields (returns complex array)
fields = sim.monitors[2].monitor_data.fields
```

## Multi-GPU

```julia
# Automatic domain decomposition across N GPUs
Khronos.plan_chunks(sim, 4)
Khronos.run(sim, until = 200.0)
```
