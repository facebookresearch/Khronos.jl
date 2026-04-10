# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Meep ↔ Khronos unit conversion.

Meep uses dimensionless units where c = 1 and the user chooses a length
scale `a`. By convention, a = 1 μm (the most common choice).

All meep quantities are expressed as multiples of `a`:
  - length:    x_meep = x_physical / a
  - frequency: f_meep = a / λ  (= a * f_physical / c)
  - time:      t_meep = t_physical * c / a

Khronos uses μm as its internal length unit with c = 1:
  - length:    x_khronos = x_physical / (1 μm) [in μm]
  - frequency: f_khronos = f_meep / a_um (natural freq in c/μm)
  - time:      t_khronos = t_meep * a_um (natural time in μm/c)

When a = 1 μm (default), meep units and Khronos units coincide.
"""

# Default length scale: 1 meep unit = 1 μm
_a_um = 1.0


def set_length_scale(a_um):
    """Set the meep length scale in micrometers.

    Parameters
    ----------
    a_um : float
        1 meep distance unit equals ``a_um`` micrometers. Default 1.0.
    """
    global _a_um
    _a_um = float(a_um)


def get_length_scale():
    """Return the current meep length scale in micrometers."""
    return _a_um


def meep_to_khronos_length(x_meep):
    """Convert meep length → Khronos length (μm)."""
    return x_meep * _a_um


def meep_to_khronos_freq(f_meep):
    """Convert meep frequency → Khronos frequency (c/μm).

    In meep, frequency = a/λ. In Khronos, frequency is in natural units (c/μm).
    When a = 1 μm these are identical.
    """
    return f_meep / _a_um


def meep_to_khronos_time(t_meep):
    """Convert meep time → Khronos time (μm/c)."""
    return t_meep * _a_um


def khronos_to_meep_length(x_khronos):
    """Convert Khronos length (μm) → meep length."""
    return x_khronos / _a_um


def khronos_to_meep_freq(f_khronos):
    """Convert Khronos frequency (c/μm) → meep frequency."""
    return f_khronos * _a_um


def khronos_to_meep_time(t_khronos):
    """Convert Khronos time (μm/c) → meep time."""
    return t_khronos / _a_um
