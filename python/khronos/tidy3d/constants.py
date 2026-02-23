# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Physical constants and unit conversion utilities."""

import math

# Speed of light in vacuum (m/s)
C_0 = 299792458.0

# Vacuum permittivity (F/m)
EPSILON_0 = 8.8541878128e-12

# Vacuum permeability (H/m)
MU_0 = 1.2566370621219e-6

# Impedance of free space (Ohm)
ETA_0 = math.sqrt(MU_0 / EPSILON_0)

# Reduced Planck constant (J·s)
HBAR = 1.0545718176e-34

# Boltzmann constant (J/K)
K_B = 1.380649e-23

# Elementary charge (C)
Q_e = 1.602176634e-19

# Infinity
inf = float("inf")

# ------------------------------------------------------------------- #
# Unit conversion: tidy3d (SI, meters) ↔ Khronos (natural, μm)
# ------------------------------------------------------------------- #

# Khronos uses μm as the length unit with c = 1
UNIT_LENGTH = 1e-6  # 1 μm in meters


def to_length(meters):
    """Convert meters → μm (Khronos length unit)."""
    return meters / UNIT_LENGTH


def to_freq(hz):
    """Convert Hz → Khronos frequency (c/μm)."""
    return hz * UNIT_LENGTH / C_0


def to_time(seconds):
    """Convert seconds → Khronos time (μm/c)."""
    return seconds * C_0 / UNIT_LENGTH


def from_length(um):
    """Convert μm → meters."""
    return um * UNIT_LENGTH


def from_freq(khronos_freq):
    """Convert Khronos frequency → Hz."""
    return khronos_freq * C_0 / UNIT_LENGTH


def from_time(khronos_time):
    """Convert Khronos time → seconds."""
    return khronos_time * UNIT_LENGTH / C_0
