# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Meep-compatible constants, enums, and predefined materials."""

import math

# ------------------------------------------------------------------- #
# Field components (integer enums matching meep conventions)
# ------------------------------------------------------------------- #

Ex = 0
Ey = 1
Ez = 2
Hx = 3
Hy = 4
Hz = 5
Dx = 6
Dy = 7
Dz = 8
Bx = 9
By = 10
Bz = 11
Dielectric = 12
Permeability = 13

# Convenience aliases
Er = Ex
Ep = Ey
Hr = Hx
Hp = Hy
Dr = Dx
Dp = Dy
Br = Bx
Bp = By

ALL_COMPONENTS = Dielectric

# ------------------------------------------------------------------- #
# Direction enums
# ------------------------------------------------------------------- #

X = 0
Y = 1
Z = 2
R = 4
P = 5
NO_DIRECTION = -1

# Side enums
Low = 0
High = 1

# ------------------------------------------------------------------- #
# Parity / symmetry enums
# ------------------------------------------------------------------- #

NO_PARITY = 0
EVEN_Z = 1
ODD_Z = 2
EVEN_Y = 4
ODD_Y = 8
TE = EVEN_Z
TM = ODD_Z

# ------------------------------------------------------------------- #
# Special values
# ------------------------------------------------------------------- #

AUTOMATIC = -1
CYLINDRICAL = -2
ALL = -1

inf = 1.0e20

# ------------------------------------------------------------------- #
# Khronos component name mapping
# ------------------------------------------------------------------- #

# Map meep integer component IDs to Khronos Julia component constructor names
_COMPONENT_TO_KHRONOS = {
    Ex: "Ex",
    Ey: "Ey",
    Ez: "Ez",
    Hx: "Hx",
    Hy: "Hy",
    Hz: "Hz",
}

# Map meep integer component IDs to polarization vectors (for sources)
_COMPONENT_TO_POLARIZATION = {
    Ex: ([1.0, 0.0, 0.0], True),   # (vector, is_electric)
    Ey: ([0.0, 1.0, 0.0], True),
    Ez: ([0.0, 0.0, 1.0], True),
    Hx: ([1.0, 0.0, 0.0], False),
    Hy: ([0.0, 1.0, 0.0], False),
    Hz: ([0.0, 0.0, 1.0], False),
}

def is_electric(component):
    """Return True if the component is an electric field."""
    return component in (Ex, Ey, Ez, Dx, Dy, Dz)

def is_magnetic(component):
    """Return True if the component is a magnetic field."""
    return component in (Hx, Hy, Hz, Bx, By, Bz)

def component_direction(component):
    """Return the direction index (0=x, 1=y, 2=z) of a field component."""
    return component % 3
