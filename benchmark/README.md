# Benchmarks

Here we describe the performance benchmarks used to detect performance regressions. Each benchmark measures the **timestepping rate** (voxels/second) and compares it against per-hardware baselines stored in YAML files.

## Basic FDTD benchmarks

These test core FDTD kernel performance with minimal physics:

* `dipole.jl` — A point dipole in vacuum with PML. Sweeps resolution and domain size.
* `periodic_stack.jl` — A 1D periodic dielectric stack (simulated in 3D) with planewave source and PML.
* `sphere.jl` — Planewave scattering off a dielectric sphere. Tests geometry rasterization (Ball primitive).

## Silicon photonics benchmarks

These represent realistic silicon photonics workloads on the standard SOI platform (220 nm Si, SiO2 cladding, C-band 1.55 μm):

* `waveguide_mode.jl` — Straight Si waveguide with ModeSource excitation. Tests eigenmode solve + injection, subpixel smoothing at waveguide boundaries.
* `directional_coupler.jl` — Two parallel Si waveguides with a narrow coupling gap. Tests fine geometry features, evanescent coupling region, ModeSource.
* `mmi_coupler.jl` — 2×2 multimode interferometer (MMI) coupler. Tests multiple Cuboid geometry objects, ModeSource, TE z-symmetry.
* `ring_coupler.jl` — Waveguide-to-ring resonator coupler. Tests Cylinder geometry, absorber boundaries, overlapping geometry with priority ordering, ModeSource.

## Advanced physics benchmarks

These test specialized FDTD features beyond basic dielectric simulations:

* `periodic_bloch.jl` — Photonic crystal slab with Bloch-periodic boundary conditions. Tests complex-valued fields, phase-shifted halos (a fundamentally different kernel path from PML).
* `uled.jl` — Blue micro-LED with dispersive silver (Drude model). Tests ADE (auxiliary differential equation) update kernels, TruncatedCone geometry, Cylinder geometry, multi-layer material stacks.
* `metalens.jl` — Pancharatnam-Berry phase metalens with thousands of rotated TiO2 pillars. Stress-tests geometry rasterization and raw GPU throughput at large voxel counts.

## File structure

Each benchmark consists of:
- A `.jl` file containing the simulation setup and benchmark loop
- A `.yml` file containing per-hardware baseline timestep rates and tolerances

The utility module `benchmark_utils.jl` provides shared infrastructure: backend detection, hardware key generation, and baseline comparison logic.

## Usage

To run the full suite of benchmarks on your platform:

```bash
julia run_benchmarks.jl
```

To specify a particular hardware backend (`CUDA`, `Metal`, or `CPU`):

```bash
julia run_benchmarks.jl --backend=CUDA
```

To specify the arithmetic precision (`Float32` or `Float64`):

```bash
julia run_benchmarks.jl --backend=CUDA --precision=Float64
```

To run only specific benchmarks (comma-separated names):

```bash
julia run_benchmarks.jl --select=dipole,metalens,ring_coupler
```

To profile the current hardware and save results to the YAML baseline files:

```bash
julia run_benchmarks.jl --backend=CUDA --precision=Float64 --profile
```

## Saving profiling results

All profiling results are saved to each benchmark's corresponding YAML file (e.g., `dipole.yml`, `metalens.yml`). When adding a new hardware platform or precision configuration, you must manually add the appropriate test entries to the YAML file. This allows you to tailor simulation parameters (domain size, resolution) to each hardware platform's capabilities.

Each benchmark entry accepts a `tolerance` parameter controlling the sensitivity threshold. For example, a tolerance of `1.1` means a 10% performance drop will raise a warning.

## Feature coverage matrix

| Benchmark | PML | Bloch | Geometry | ModeSource | Dispersive | Symmetry |
|-----------|-----|-------|----------|------------|------------|----------|
| dipole | ✓ | | point only | | | |
| periodic_stack | ✓ | | Cuboid | | | |
| sphere | ✓ | | Ball | | | |
| waveguide_mode | ✓ | | Cuboid (SOI) | ✓ | | |
| directional_coupler | ✓ | | Cuboid (SOI) | ✓ | | |
| mmi_coupler | ✓ | | Cuboid (SOI) | ✓ | | ✓ |
| ring_coupler | ✓ | | Cylinder + Cuboid | ✓ | | |
| periodic_bloch | ✓ | ✓ | Cylinder + Cuboid | | | |
| uled | ✓ | | TruncatedCone + Cylinder + Cuboid | | ✓ (Drude) | |
| metalens | ✓ | | rotated Cuboid (×N²) | | | |
