# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adjoint submodule for topology optimization using the hybrid
# time-/frequency-domain adjoint method.
#
# Reference: Hammond et al., "High-performance hybrid time/frequency-domain
# topology optimization for large-scale photonics inverse design," Optics
# Express (2022).

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

export DesignRegion, update_design!, init_design_region!
export OptimizationProblem, forward_run!, adjoint_run!, calculate_gradient!
export EigenmodeCoefficient, FourierFieldsObjective
export adj_src_scale, create_adjoint_time_profile
export FilteredSource
export conic_filter, tanh_projection, cylindrical_filter, gaussian_filter
export simulate
