# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Material definitions: Medium, Lorentz, Drude, PEC."""

import math
from .constants import to_freq


class Medium:
    """Isotropic non-dispersive medium."""

    def __init__(self, permittivity=1.0, conductivity=0.0, name=None, allow_gain=False):
        self.permittivity = permittivity
        self.conductivity = conductivity
        self.name = name
        self.allow_gain = allow_gain

    @classmethod
    def from_nk(cls, n, k, freq, **kwargs):
        """Create medium from refractive index n and extinction coefficient k."""
        eps_real = n**2 - k**2
        eps_imag = 2 * n * k
        # sigma = eps_imag * omega = eps_imag * 2*pi*freq
        sigma = eps_imag * 2 * math.pi * freq
        return cls(permittivity=eps_real, conductivity=sigma, **kwargs)

    def refractive_index(self, wavelength):
        """Get real refractive index at a given wavelength (SI, meters)."""
        return math.sqrt(max(self.permittivity, 1.0))

    def _to_khronos(self, K):
        kwargs = {"ε": float(self.permittivity)}
        if self.conductivity != 0:
            kwargs["σD"] = float(self.conductivity)
        return K.Material(**kwargs)


class Lorentz(Medium):
    """Lorentz dispersive medium.

    coeffs: list of (delta_epsilon, frequency_Hz, delta_Hz) tuples.
    """

    def __init__(self, eps_inf=1.0, coeffs=None, **kwargs):
        super().__init__(permittivity=eps_inf, **kwargs)
        self.eps_inf = eps_inf
        self.coeffs = coeffs or []

    def refractive_index(self, wavelength):
        return math.sqrt(max(self.eps_inf, 1.0))

    def _to_khronos(self, K):
        susceptibilities = []
        for de, f, delta in self.coeffs:
            omega_0 = to_freq(f)
            gamma = to_freq(2 * delta)
            sigma = float(de)
            susceptibilities.append(K.LorentzianSusceptibility(omega_0, gamma, sigma))
        return K.Material(ε=float(self.eps_inf), susceptibilities=susceptibilities)


class Drude(Medium):
    """Drude dispersive medium.

    coeffs: list of (frequency_Hz, delta_Hz) tuples.
    """

    def __init__(self, eps_inf=1.0, coeffs=None, **kwargs):
        super().__init__(permittivity=eps_inf, **kwargs)
        self.eps_inf = eps_inf
        self.coeffs = coeffs or []

    def refractive_index(self, wavelength):
        return math.sqrt(max(self.eps_inf, 1.0))

    def _to_khronos(self, K):
        susceptibilities = []
        for f, delta in self.coeffs:
            gamma = to_freq(2 * delta)
            sigma = to_freq(f) ** 2  # Drude sigma = omega_p^2
            susceptibilities.append(K.DrudeSusceptibility(gamma, sigma))
        return K.Material(ε=float(self.eps_inf), susceptibilities=susceptibilities)


class Sellmeier(Medium):
    """Sellmeier dispersive medium.

    coeffs: list of (B, C_um2) tuples (B dimensionless, C in μm²).
    """

    def __init__(self, coeffs=None, **kwargs):
        super().__init__(permittivity=1.0, **kwargs)
        self.coeffs = coeffs or []

    @classmethod
    def from_dispersion(cls, n, dn_dwvl, freq, **kwargs):
        """Create Sellmeier from n and dn/dλ at a reference frequency."""
        wvl = 3e8 / freq  # wavelength in meters
        # Single-term Sellmeier fit
        B = n**2 - 1 + dn_dwvl * wvl * 2 * n
        C = (wvl * 1e6) ** 2 * 0.01  # rough estimate in μm²
        return cls(coeffs=[(B, C)], **kwargs)

    def refractive_index(self, wavelength):
        wvl_um = wavelength * 1e6  # convert meters to μm
        eps = 1.0
        for B, C in self.coeffs:
            eps += B * wvl_um**2 / (wvl_um**2 - C)
        return math.sqrt(max(eps, 1.0))

    def _to_khronos(self, K):
        # Convert Sellmeier to Lorentzian: each term B*λ²/(λ²-C) maps to
        # a Lorentzian with omega_0 = 2*pi*c/sqrt(C), sigma = B
        susceptibilities = []
        for B, C in self.coeffs:
            if C > 0:
                omega_0 = 1.0 / math.sqrt(C)  # in c/μm units (natural)
                susceptibilities.append(K.LorentzianSusceptibility(omega_0, 0.0, float(B)))
        eps_inf = 1.0
        return K.Material(ε=float(eps_inf), susceptibilities=susceptibilities)


class PECMedium:
    """Perfect Electric Conductor (approximated with very high conductivity)."""

    def __init__(self):
        pass

    def refractive_index(self, wavelength):
        return 1e6

    def _to_khronos(self, K):
        return K.Material(ε=1.0, σD=1e6)


# Singleton
PEC = PECMedium()
