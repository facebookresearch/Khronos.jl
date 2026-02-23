# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Julia bridge: lazy initialization of the Khronos.jl runtime.

The Julia process starts on first access to `get_khronos()`, not at import time.
"""
import os

_jl = None
_Khronos = None
_GP = None  # GeometryPrimitives


def _init_julia():
    global _jl, _Khronos, _GP
    if _jl is not None:
        return

    import juliacall
    _jl = juliacall.newmodule("KhronosWrapper")

    # Set project to Khronos.jl root
    khronos_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    khronos_root = os.path.abspath(khronos_root)
    _jl.seval(f'using Pkg; Pkg.activate("{khronos_root}")')
    _jl.seval("import Khronos")
    _jl.seval("using GeometryPrimitives")
    _Khronos = _jl.Khronos
    _GP = _jl.seval("GeometryPrimitives")


def get_jl():
    """Get the Julia module for direct evaluation."""
    _init_julia()
    return _jl


def get_khronos():
    """Get the Khronos Julia module."""
    _init_julia()
    return _Khronos


def get_gp():
    """Get the GeometryPrimitives Julia module."""
    _init_julia()
    return _GP
