# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Topology optimization filters and projections.

These are pure-Python (numpy/autograd) functions that do NOT touch the
FDTD engine. They implement standard topology optimization techniques:
density filtering, threshold projection, and length-scale constraints.

All functions are compatible with autograd for automatic differentiation.
"""

try:
    import numpy as np
except ImportError:
    np = None

try:
    from autograd import numpy as npa
    _HAS_AUTOGRAD = True
except ImportError:
    npa = np
    _HAS_AUTOGRAD = False


# ------------------------------------------------------------------- #
# Convolution filters
# ------------------------------------------------------------------- #

def _get_kernel(radius, Lx, Ly, resolution, kernel_func):
    """Build a 2D convolution kernel on the design grid."""
    Nx = int(round(Lx * resolution)) + 1
    Ny = int(round(Ly * resolution)) + 1
    dx = Lx / max(Nx - 1, 1)
    dy = Ly / max(Ny - 1, 1)

    # Kernel radius in pixels
    rx = int(round(radius / dx))
    ry = int(round(radius / dy))

    # Build kernel
    x = np.arange(-rx, rx + 1) * dx
    y = np.arange(-ry, ry + 1) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    kernel = kernel_func(R, radius)
    kernel[R > radius] = 0
    kernel /= kernel.sum()
    return kernel


def _convolve2d(x, kernel, periodic_axes=None):
    """2D convolution with optional periodic boundary handling."""
    from scipy.signal import fftconvolve
    Nx, Ny = x.shape
    kx, ky = kernel.shape
    px, py = kx // 2, ky // 2

    # Pad input
    if periodic_axes is not None and len(periodic_axes) > 0:
        padded = np.pad(x, ((px, px), (py, py)), mode="wrap")
    else:
        padded = np.pad(x, ((px, px), (py, py)), mode="edge")

    # Use autograd-compatible convolution if available
    if _HAS_AUTOGRAD:
        # Manual convolution for autograd compatibility
        result = npa.zeros_like(x)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                result = result + kernel[i, j] * padded[i:i+Nx, j:j+Ny]
        return result
    else:
        result = fftconvolve(padded, kernel, mode="valid")
        return result[:Nx, :Ny]


def conic_filter(x, radius, Lx, Ly, resolution, periodic_axes=None):
    """Apply a conic (linear-decay) density filter.

    Parameters
    ----------
    x : 2D array
        Design weights (Nx, Ny).
    radius : float
        Filter radius in meep units.
    Lx, Ly : float
        Design region dimensions.
    resolution : float
        Grid resolution.
    periodic_axes : list, optional
        Axes with periodic boundary conditions.

    Returns
    -------
    Filtered weights (same shape as x).
    """
    kernel = _get_kernel(radius, Lx, Ly, resolution,
                         lambda R, r: npa.maximum(r - R, 0))
    return _convolve2d(x, kernel, periodic_axes)


def cylindrical_filter(x, radius, Lx, Ly, resolution, periodic_axes=None):
    """Apply a cylindrical (uniform disk) density filter."""
    kernel = _get_kernel(radius, Lx, Ly, resolution,
                         lambda R, r: np.ones_like(R))
    return _convolve2d(x, kernel, periodic_axes)


def gaussian_filter(x, sigma, Lx, Ly, resolution, periodic_axes=None):
    """Apply a Gaussian density filter."""
    radius = 3 * sigma  # truncate at 3 sigma
    kernel = _get_kernel(radius, Lx, Ly, resolution,
                         lambda R, r: np.exp(-R**2 / (2 * sigma**2)))
    return _convolve2d(x, kernel, periodic_axes)


# ------------------------------------------------------------------- #
# Projection operators
# ------------------------------------------------------------------- #

def tanh_projection(x, beta, eta):
    """Hyperbolic tangent threshold projection.

    Projects continuous design weights toward binary (0/1) values.

    Parameters
    ----------
    x : array
        Filtered weights in [0, 1].
    beta : float
        Projection strength. 0 = no projection, inf = Heaviside step.
    eta : float
        Projection threshold (typically 0.5).

    Returns
    -------
    Projected weights in (0, 1).
    """
    return (npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))) / (
        npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta))
    )


def smoothed_projection(rho_filtered, beta, eta, resolution=None):
    """Smoothed projection that remains differentiable as beta → ∞."""
    return tanh_projection(rho_filtered, beta, eta)


def heaviside_projection(x, beta, eta):
    """Heaviside-approximation projection (same as tanh_projection)."""
    return tanh_projection(x, beta, eta)


# ------------------------------------------------------------------- #
# Length scale constraints
# ------------------------------------------------------------------- #

def _eroded_dilated(x, radius, Lx, Ly, resolution, eta, periodic_axes=None):
    """Apply filter + projection at a specific threshold."""
    filtered = conic_filter(x, radius, Lx, Ly, resolution, periodic_axes)
    return tanh_projection(filtered, 1e6, eta)  # sharp projection


def indicator_solid(x, c, eta_e, filter_f, threshold_f, resolution,
                    periodic_axes=None):
    """Indicator function for solid features (weight = 1 regions)."""
    filtered = filter_f(x)
    projected_e = threshold_f(filtered, eta_e)
    projected_i = threshold_f(filtered, 0.5)
    return npa.mean(npa.abs(projected_e - projected_i))


def indicator_void(x, c, eta_d, filter_f, threshold_f, resolution,
                   periodic_axes=None):
    """Indicator function for void features (weight = 0 regions)."""
    filtered = filter_f(x)
    projected_d = threshold_f(filtered, eta_d)
    projected_i = threshold_f(filtered, 0.5)
    return npa.mean(npa.abs(projected_d - projected_i))


def constraint_solid(x, c, eta_e, filter_f, threshold_f, resolution,
                     periodic_axes=None):
    """Minimum solid feature size constraint.

    Returns a value ≤ 0 when the constraint is satisfied.
    """
    ind = indicator_solid(x, c, eta_e, filter_f, threshold_f, resolution,
                          periodic_axes)
    return ind - c


def constraint_void(x, c, eta_d, filter_f, threshold_f, resolution,
                    periodic_axes=None):
    """Minimum void feature size constraint.

    Returns a value ≤ 0 when the constraint is satisfied.
    """
    ind = indicator_void(x, c, eta_d, filter_f, threshold_f, resolution,
                         periodic_axes)
    return ind - c


def gray_indicator(x):
    """Measure of how binary (non-gray) the design is.

    Returns a value in [0, 1] where 0 = fully binary and 1 = fully gray.
    """
    return npa.mean(4 * x * (1 - x))


def length_indicator(x, filter_f, threshold_f, resolution,
                     periodic_axes=None):
    """Length scale indicator (NOT IMPLEMENTED — returns 0)."""
    return 0.0


def get_conic_radius_from_eta_e(b, eta_e):
    """Get the conic filter radius from minimum feature length and erosion threshold.

    Parameters
    ----------
    b : float
        Minimum feature size.
    eta_e : float
        Erosion threshold (e.g. 0.75 for dilated, 0.25 for eroded).

    Returns
    -------
    Filter radius.
    """
    return b / (2 * np.sqrt(eta_e * (1 - eta_e)))


def get_eta_from_conic(b, R):
    """Get the erosion/dilation threshold from feature length and filter radius.

    Parameters
    ----------
    b : float
        Minimum feature size.
    R : float
        Filter radius.

    Returns
    -------
    Threshold eta.
    """
    arg = b / (2 * R)
    if arg >= 1:
        return 0.5
    return 0.5 * (1 - np.sqrt(1 - arg**2))


def get_threshold_wang(delta, sigma):
    """Wang threshold for connectivity constraints."""
    return delta / sigma
