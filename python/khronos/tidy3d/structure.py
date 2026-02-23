# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Structure: pairs geometry with medium."""


class Structure:
    """A physical structure in the simulation domain."""

    def __init__(self, geometry, medium, name=None):
        self.geometry = geometry
        self.medium = medium
        self.name = name

    def _to_khronos(self, K, gp):
        shape = self.geometry._to_khronos(gp)
        material = self.medium._to_khronos(K)
        return K.Object(shape, material)
