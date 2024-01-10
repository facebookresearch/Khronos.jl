# Benchmarks

Here we describe the following performance benchmarks used to detect performance regressions:

* `dipole.jl` - A simple dipole in vacumm, run at multiple resolutions. (Future work could include symmetries and different kinds of boundary conditions, near2far transforms, dft monitors.)
* `periodic_stack.jl` - A 1D periodic stack (simulated in 3D), run at multiple resolutions. Consists of a planewave source, PML, and dielectric materials throughout the domain. (Future work could include periodic boundaries, dispersive materials, anisotropic materials, subpixel smoothing, DFT monitors, time monitors.)
* `metallic_sphere.jl` - A metallic sphere in vaccum, run at multiple resolutions. (Future work could DFT and time monitors, subpixel smoothing, and dispersive materials.)

All benchmarks detect the current hardware and flags the user if the performance of the bechmark is significantly different than what's expected. Each benchmark simulates the **timestepping rate** (voxels/second). Future work should also check for additional/different allocations.

The file, `benchmark_utils.jl` contains various refactored routines that are repeated throughout the benchmarks.

Future benchmarks include
* `grating_coupler.jl` - A simple silicon photonics grating coupler simulation (pulled from SiEPIC).
* `directional_coupler.jl` - A simple silicon photonics directional coupler simulation (pulled from SiEPIC).
* `metalens.jl` - A simple metalens simulation (pulled from the Tidy3D paper).
* `uLED.jl` - A simple uLED pixel.