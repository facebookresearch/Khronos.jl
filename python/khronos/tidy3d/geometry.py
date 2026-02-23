# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Geometry primitives: Box, Sphere, Cylinder."""

from .constants import to_length, inf


class Box:
    """Axis-aligned rectangular box."""

    def __init__(self, center=(0, 0, 0), size=(0, 0, 0)):
        self.center = tuple(center)
        self.size = tuple(size)

    def _to_khronos(self, gp):
        c = [to_length(x) for x in self.center]
        s = [to_length(x) if x != inf else 1e6 for x in self.size]
        return gp.Cuboid(c, s)


class Sphere:
    """Sphere defined by center and radius."""

    def __init__(self, center=(0, 0, 0), radius=1.0):
        self.center = tuple(center)
        self.radius = radius

    def _to_khronos(self, gp):
        c = [to_length(x) for x in self.center]
        r = to_length(self.radius)
        return gp.Ball(c, r)


class Cylinder:
    """Cylinder with center, radius, length, and axis."""

    def __init__(self, center=(0, 0, 0), radius=1.0, length=1.0, axis=2):
        self.center = tuple(center)
        self.radius = radius
        self.length = length
        self.axis = axis

    def _to_khronos(self, gp):
        c = [to_length(x) for x in self.center]
        r = to_length(self.radius)
        h = to_length(self.length)
        axis_vec = [0.0, 0.0, 0.0]
        axis_vec[self.axis] = 1.0
        return gp.Cylinder(c, r, h, axis_vec)
