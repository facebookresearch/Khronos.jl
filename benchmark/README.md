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

## Usage

To run the suite of benchmarks on your platform, simply run:

```bash
julia run_benchmarks.jl
```

To (optionally) specify a particular hardware platform (either `CUDA`, `METAL` or `CPU`), use the `--backend` flag:

```bash
julia run_benchmarks.jl --backend=CUDA
```

To (optionally) specify the arithemtic precision (either `Float32` or `Float64`), use the `--precision` flag:

```bash
julia run_benchmarks.jl --backend=CUDA --precision=Float64
```

To (optionally) _profile_ the current hardware and save the results, us the `--profile` flag:

```bash
julia run_benchmarks.jl --backend=CUDA --precision=Float64 --profile
```

## Saving profiling results

All profiling results can be saved to the benchmark's corresponding yaml file. For example, all `dipole.jl` results will be saved in `dipole.yml`.

Whenever adding a new hardware platform or precision configuration, you must manually add in the appropriate tests to the yaml file. This allows you to cherrypick specific simulation parameters best geared for that particular hardware platform.

For example, since an NVIDIA H100 GPU has significantly more VRAM than an NVIDIA V100, you'll want to setup tests with larger domains, resolutions, etc.

Similarly, all profile configurations accept a "tolerance" parameter, which can be used to specify how sensitive a change in performance needs to be before alerting a user. For example, a tolerance of 1.1 indicates a change of 10% will raise a warning, encouraging the user to record the change.