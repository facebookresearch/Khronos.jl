# Khronos.jl

![Continuous integration](https://github.com/facebookresearch/Khronos.jl/actions/workflows/ci.yml/badge.svg)

Khronos is a GPU-accelerated Maxwell FDTD solver written entirely in Julia.

## Key Features

- GPU acceleration compatible with NVIDIA CUDA, AMD ROCm, Intel OneAPI, and Apple Metal (via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl))
- 100% Julia code (works with Windows, Mac, and Linux)
- Specifiable precision (e.g. `Float64`, `Float32`)
- Parameterizable geometry via GeometryPrimitives.jl
- Diagonally anisotropic permittivity ($\varepsilon$) and permeability ($\mu$), and electric/magnetic conductivity terms ($\sigma_D$ / $\sigma_B$) for either the permittivity or permeability, respectively.
- Predefined current sources (including planewaves and Gaussian beams) and arbitrary curret sources.
- _Equivalent_ sources from predefined electric and magnetic fields (e.g. to inject modes computed from a mode solver).
- Continuous-wave and Guassian-pulse time profiles
- Arbitrary 1D, 2D, or 3D (rectilinear) DFT monitors
- Perfectly matched layer (PML) absorbing boundaries
- Predefined simulation runtime functions, including run for an arbitray time and run until the DFT fields have converged.
- Simple plotting of visualization cross sections and overlayed field response, DFT monitors, and source frequency responses.
- Benchmark tooling to track performance regressions/enhancements on arbitrary hardware
- Composable kernel framework allows for the flexible definition of thousands of different kernels

## Current limitations

- All simulations require PML.
- Single GPU only.
- Only linear materials without any polarizabilities.
- Uniform gridding.
