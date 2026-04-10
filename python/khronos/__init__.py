# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Khronos: GPU-accelerated FDTD with meep-compatible Python API.

Quick start (meep API):
    import khronos.meep as mp
    sim = mp.Simulation(...)
    sim.run(until_after_sources=200)

Sysimage (optional, for fast startup):
    $ khronos-build-sysimage        # one-time, ~10 min
    # All subsequent imports will be fast
"""

__version__ = "0.2.0"


def build_sysimage(**kwargs):
    """Build a PackageCompiler sysimage for fast Julia startup.

    See `khronos-build-sysimage --help` for CLI usage.
    """
    from ._sysimage import build_sysimage as _build
    return _build(**kwargs)


def sysimage_info():
    """Print information about the current sysimage."""
    from ._sysimage import sysimage_info as _info
    return _info()
