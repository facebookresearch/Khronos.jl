# Khronos.jl

![Continuous integration](https://github.com/facebookresearch/Khronos.jl/actions/workflows/ci.yml/badge.svg)

Khronos is a GPU-accelerated Maxwell FDTD solver written entirely in Julia.

## Key Features

- GPU acceleration compatible with NVIDIA CUDA, AMD ROCm, Intel OneAPI, and Apple Metal (via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl))
- 100% Julia code (works with Windows, Mac, and Linux)
- Specifiable precision (e.g. `Float64`, `Float32`)
- 2D (TE/TM) and 3D simulation support
- Parameterizable geometry via GeometryPrimitives.jl with subpixel smoothing
- GDS-II layout import for photonic device geometries
- Diagonally anisotropic permittivity ($\varepsilon$) and permeability ($\mu$), and electric/magnetic conductivity terms ($\sigma_D$ / $\sigma_B$) for either the permittivity or permeability, respectively.
- Dispersive materials via Drude and Lorentzian susceptibility poles (auxiliary differential equation method)
- $\chi^{(3)}$ Kerr nonlinearity
- Predefined current sources (including planewaves, Gaussian beams, and total-field/scattered-field sources) and arbitrary current sources.
- _Equivalent_ sources from predefined electric and magnetic fields (e.g. to inject modes computed from a mode solver).
- Continuous-wave and Gaussian-pulse time profiles
- Arbitrary 1D, 2D, or 3D (rectilinear) DFT monitors
- Flux, mode, and diffraction monitors
- Near-to-far field transformation with GPU-accelerated Green's function evaluation
- Layered-medium far-field projection via transfer matrix method (e.g. project from inside GaN to air without simulating the air region)
- Perfectly matched layer (PML) and absorber boundaries
- Periodic and Bloch boundary conditions
- Multi-GPU support via MPI/NCCL-based domain decomposition with halo exchange
- Concurrent GPU batch execution via multi-stream parallelism
- Adjoint-method inverse design with automatic differentiation (Zygote/ChainRules integration)
- Composable kernel framework with fused PML kernels and per-component specialization
- Meep-compatible Python API (`import khronos.meep as mp`)
- Predefined simulation runtime functions, including run for an arbitrary time and run until the DFT fields have converged.
- Simple plotting of visualization cross sections and overlaid field response, DFT monitors, and source frequency responses.
- Benchmark tooling to track performance regressions/enhancements on arbitrary hardware

## Current limitations

- All simulations require PML or absorber boundaries.
- Uniform gridding.

## License
Khronos is licensed under the [MIT license](https://github.com/facebookresearch/Khronos.jl/blob/main/LICENSE).
