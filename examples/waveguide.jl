# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import fdtd
using CairoMakie
using GeometryPrimitives

fdtd.choose_backend(fdtd.CPUDevice(), Float64)