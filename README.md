# Khronos.jl

![Continuous integration](https://github.com/facebookresearch/Khronos.jl/actions/workflows/ci.yml/badge.svg)

A GPU-accelerated, differentiable, Maxwell FDTD solver using Julia.

## Key Features

### Performance
- GPU acceleration compatible with NVIDIA CUDA, AMD ROCm, Intel OneAPI, and Apple Metal (via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl))
- Works with Windows, Mac, and Linux
- Over 1 TCell/second on A100 and H100 GPU architectures!

### Geometry
- Parameterizable geometry via GeometryPrimitives.jl

### Materials
- Diagonally anisotropic permittivity ($\varepsilon$) and permeability ($\mu$)
- Electric/magnetic conductivity terms ($\sigma_D$/$\sigma_B$) for either the permittivity or permeability, respectively.

### Sources
- Predefined current sources, including arbitrary planewaves and Gaussian beams.
- _Equivalent_ sources from predefined electric and magnetic fields (e.g. to inject modes computed from a mode solver).
- Arbitrary current source definitions
- Source _restriction_ (analagous to subpixel smoothing, and the transpose of interpolation operations) which allows continuous movement of source volumes.
- Continuous-wave and Guassian-pulse time profiles

### DFT Monitors
- Arbitrary 1D, 2D, or 3D (rectilinear) DFT monitors
- Monitor _restriction_ (analagous to subpixel smoothing, and the transpose of interpolation operations) which allows continuous movement of monitor regions.

### Boundary Conditions and Boundary Layers
- Perfectly matched layer (PML) absorbing boundaries

### Simulation
- Predefined simulation runtime functions, including run until an arbitray time and run until the DFT fields have converged.

### Visualization
- Simple plotting of visualization cross sections and overlayed field response
- Plot the response of a DFT monitor
- Plot the 

### General
- Benchmark tooling to track performance regressions/enhancements on arbitrary hardware
- CI testing
- Basic examples
- Composable kernel framework allows for the flexible definition of thousands of different kernels
