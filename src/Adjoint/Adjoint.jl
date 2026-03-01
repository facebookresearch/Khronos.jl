# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adjoint submodule for topology optimization using the hybrid
# time-/frequency-domain adjoint method.
#
# Reference: Hammond et al., "High-performance hybrid time/frequency-domain
# topology optimization for large-scale photonics inverse design," Optics
# Express (2022).
#
# Usage:
#   using Khronos
#
#   # 1. Set up simulation, geometry, sources as usual
#   sim = Simulation(...)
#
#   # 2. Define a design region (density parameterization)
#   dr = DesignRegion(volume=..., grid_size=(Nx, Ny), ε_min=..., ε_max=...)
#   init_design_region!(sim, dr)
#
#   # 3. Define objective (what to optimize)
#   obj = EigenmodeCoefficient(volume=..., mode_num=1, forward=true)
#   # or: obj = FourierFieldsObjective(volume=..., component=Ez())
#
#   # 4. Build optimization problem
#   opt = OptimizationProblem(
#       sim=sim,
#       objective_functions=[rho -> sum(abs2.(rho))],  # scalar objective
#       objective_arguments=[obj],
#       design_regions=[dr],
#       frequencies=[fcen],
#   )
#
#   # 5. Evaluate objective + gradient
#   f0, gradient = opt(rho)     # 1 forward + 1 adjoint sim
#
#   # 6. Validate with finite differences
#   fd_grad, idx = calculate_fd_gradient(opt, rho; n_samples=10)

using LinearAlgebra
using Statistics: mean
import FFTW: fft, ifft
import ForwardDiff
import ChainRulesCore

include("DesignRegion.jl")
include("ObjectiveQuantities.jl")
include("AdjointSourceScale.jl")
include("FilteredSource.jl")
include("Filters.jl")
include("GradientKernel.jl")
include("OptimizationProblem.jl")
include("ChainRulesIntegration.jl")

# Design region
export DesignRegion, update_design!, init_design_region!

# Optimization
export OptimizationProblem, forward_run!, adjoint_run!
export calculate_gradient!, calculate_fd_gradient

# Objective functions
export EigenmodeCoefficient, FourierFieldsObjective

# Adjoint utilities
export adj_src_scale, create_adjoint_time_profile
export FilteredSource

# Filters and projections
export conic_filter, tanh_projection, cylindrical_filter, gaussian_filter
export exponential_erosion, exponential_dilation
export heaviside_erosion, heaviside_dilation
export gray_indicator, get_eta_from_conic, get_conic_radius_from_eta_e

# ChainRules integration
export simulate
