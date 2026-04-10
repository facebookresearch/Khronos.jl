# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Vector3 and geometry classes matching the meep Python API.
"""

import math
from .constants import inf, X, Y, Z, ALL, AUTOMATIC
from ._units import meep_to_khronos_length


def _jl_vec(lst):
    """Convert a Python list of numbers to a Julia Vector{Float64}.

    PythonCall converts Python lists to PyList{Any}, which doesn't match
    Julia's AbstractVector{<:Real}. This helper creates a proper Julia vector.
    """
    from .._bridge import get_jl
    jl = get_jl()
    return jl.Vector[jl.Float64](lst)


# ------------------------------------------------------------------- #
# Vector3
# ------------------------------------------------------------------- #

class Vector3:
    """Three-component vector, compatible with meep.Vector3."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return NotImplemented

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    # Arithmetic
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x / other, self.y / other, self.z / other)
        return NotImplemented

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __abs__(self):
        return self.norm()

    # Methods
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cdot(self, other):
        return (self.conj()).dot(other)

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unit(self):
        n = self.norm()
        if n == 0:
            return Vector3()
        return self / n

    def conj(self):
        return Vector3(self.x, self.y, self.z)  # real vector

    def scale(self, s):
        return self * s

    def close(self, other, tol=1e-7):
        return (self - other).norm() < tol

    def rotate(self, axis, theta):
        """Rotate around axis (Vector3) by angle theta (radians)."""
        u = axis.unit()
        c = math.cos(theta)
        s = math.sin(theta)
        d = u.dot(self)
        cr = u.cross(self)
        return self * c + cr * s + u * (d * (1 - c))


# ------------------------------------------------------------------- #
# Volume
# ------------------------------------------------------------------- #

class Volume:
    """A rectangular volume in the simulation domain."""

    def __init__(self, center=None, size=None, dims=2, is_cylindrical=False,
                 vertices=None):
        self.center = center if center is not None else Vector3()
        self.size = size if size is not None else Vector3()
        self.dims = dims
        self.is_cylindrical = is_cylindrical
        self.vertices = vertices

    @property
    def pt1(self):
        return self.center - self.size * 0.5

    @property
    def pt2(self):
        return self.center + self.size * 0.5


# ------------------------------------------------------------------- #
# Materials
# ------------------------------------------------------------------- #

class Medium:
    """Dielectric material, compatible with meep.Medium.

    Can be specified via ``epsilon`` (permittivity) or ``index`` (refractive index).
    """

    def __init__(self, epsilon=None, index=None,
                 epsilon_diag=None, epsilon_offdiag=None,
                 mu=None, mu_diag=None, mu_offdiag=None,
                 E_susceptibilities=None, H_susceptibilities=None,
                 D_conductivity=None, D_conductivity_diag=None,
                 B_conductivity=None, B_conductivity_diag=None,
                 chi2=None, chi3=None,
                 E_chi2_diag=None, E_chi3_diag=None,
                 H_chi2_diag=None, H_chi3_diag=None,
                 valid_freq_range=None):
        # Resolve epsilon from index or explicit value
        if index is not None:
            self.epsilon_diag = Vector3(index**2, index**2, index**2)
        elif epsilon is not None:
            self.epsilon_diag = Vector3(epsilon, epsilon, epsilon)
        elif epsilon_diag is not None:
            if isinstance(epsilon_diag, Vector3):
                self.epsilon_diag = epsilon_diag
            else:
                self.epsilon_diag = Vector3(*epsilon_diag)
        else:
            self.epsilon_diag = Vector3(1, 1, 1)

        self.epsilon_offdiag = epsilon_offdiag or Vector3(0, 0, 0)
        if not isinstance(self.epsilon_offdiag, Vector3):
            self.epsilon_offdiag = Vector3(*self.epsilon_offdiag)

        # Permeability
        if mu is not None:
            self.mu_diag = Vector3(mu, mu, mu)
        elif mu_diag is not None:
            self.mu_diag = mu_diag if isinstance(mu_diag, Vector3) else Vector3(*mu_diag)
        else:
            self.mu_diag = Vector3(1, 1, 1)
        self.mu_offdiag = mu_offdiag or Vector3(0, 0, 0)
        if not isinstance(self.mu_offdiag, Vector3):
            self.mu_offdiag = Vector3(*self.mu_offdiag)

        # Susceptibilities
        self.E_susceptibilities = E_susceptibilities or []
        self.H_susceptibilities = H_susceptibilities or []

        # Conductivity
        if D_conductivity is not None:
            self.D_conductivity_diag = Vector3(D_conductivity, D_conductivity, D_conductivity)
        elif D_conductivity_diag is not None:
            self.D_conductivity_diag = D_conductivity_diag if isinstance(D_conductivity_diag, Vector3) else Vector3(*D_conductivity_diag)
        else:
            self.D_conductivity_diag = Vector3(0, 0, 0)

        if B_conductivity is not None:
            self.B_conductivity_diag = Vector3(B_conductivity, B_conductivity, B_conductivity)
        elif B_conductivity_diag is not None:
            self.B_conductivity_diag = B_conductivity_diag if isinstance(B_conductivity_diag, Vector3) else Vector3(*B_conductivity_diag)
        else:
            self.B_conductivity_diag = Vector3(0, 0, 0)

        # Nonlinear susceptibilities
        self.chi2 = chi2
        self.chi3 = chi3
        self.E_chi2_diag = E_chi2_diag
        self.E_chi3_diag = E_chi3_diag
        self.H_chi2_diag = H_chi2_diag
        self.H_chi3_diag = H_chi3_diag
        self.valid_freq_range = valid_freq_range

    @property
    def epsilon(self):
        """Isotropic permittivity (returns xx component)."""
        return self.epsilon_diag.x

    def _to_khronos(self, K):
        kwargs = {}

        # Permittivity
        ex, ey, ez = self.epsilon_diag
        if ex == ey == ez:
            kwargs["ε"] = float(ex)
        else:
            kwargs["εx"] = float(ex)
            kwargs["εy"] = float(ey)
            kwargs["εz"] = float(ez)

        # Conductivity
        sx, sy, sz = self.D_conductivity_diag
        if sx == sy == sz:
            if sx != 0:
                kwargs["σD"] = float(sx)
        else:
            if sx != 0:
                kwargs["σDx"] = float(sx)
            if sy != 0:
                kwargs["σDy"] = float(sy)
            if sz != 0:
                kwargs["σDz"] = float(sz)

        # Permeability
        mx, my, mz = self.mu_diag
        if mx == my == mz:
            if mx != 1.0:
                kwargs["μ"] = float(mx)
        else:
            kwargs["μx"] = float(mx)
            kwargs["μy"] = float(my)
            kwargs["μz"] = float(mz)

        # Susceptibilities
        if self.E_susceptibilities:
            susceptibilities = []
            for s in self.E_susceptibilities:
                susceptibilities.append(s._to_khronos(K))
            kwargs["susceptibilities"] = susceptibilities

        return K.Material(**kwargs)


# Susceptibility types
class Susceptibility:
    """Base susceptibility class."""

    def __init__(self, sigma_diag=None, sigma_offdiag=None, sigma=None):
        if sigma is not None:
            self.sigma_diag = Vector3(sigma, sigma, sigma)
        elif sigma_diag is not None:
            self.sigma_diag = sigma_diag if isinstance(sigma_diag, Vector3) else Vector3(*sigma_diag)
        else:
            self.sigma_diag = Vector3(1, 1, 1)
        self.sigma_offdiag = sigma_offdiag or Vector3(0, 0, 0)


class LorentzianSusceptibility(Susceptibility):
    """Lorentzian resonance."""

    def __init__(self, frequency=0, gamma=0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency  # in meep units
        self.gamma = gamma  # in meep units

    def _to_khronos(self, K):
        from ._units import meep_to_khronos_freq
        # Julia's compute_ade_coefficients applies 2π internally,
        # so pass plain frequency (not angular frequency)
        omega_0 = meep_to_khronos_freq(self.frequency)
        gam = meep_to_khronos_freq(self.gamma)
        sig = float(self.sigma_diag.x)
        return K.LorentzianSusceptibility(omega_0, gam, sig)


class DrudeSusceptibility(Susceptibility):
    """Drude conductivity model."""

    def __init__(self, frequency=0, gamma=0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.gamma = gamma

    def _to_khronos(self, K):
        from ._units import meep_to_khronos_freq
        # Julia's compute_ade_coefficients applies 2π internally,
        # so pass plain frequency (not angular frequency)
        omega_p = meep_to_khronos_freq(self.frequency)
        gam = meep_to_khronos_freq(self.gamma)
        return K.DrudeSusceptibility(gam, omega_p**2)


# ------------------------------------------------------------------- #
# Predefined materials
# ------------------------------------------------------------------- #

vacuum = Medium(epsilon=1)
air = Medium(epsilon=1)
metal = Medium(epsilon=1, D_conductivity=1e7)
perfect_electric_conductor = Medium(epsilon=1, D_conductivity=1e7)
perfect_magnetic_conductor = Medium(mu=1, B_conductivity=1e7)


# ------------------------------------------------------------------- #
# Geometric Objects
# ------------------------------------------------------------------- #

class GeometricObject:
    """Base class for all geometric objects."""

    def __init__(self, material=None, center=None, epsilon_func=None, label=None):
        self.material = material or Medium()
        self.center = center if center is not None else Vector3()
        if not isinstance(self.center, Vector3):
            self.center = Vector3(*self.center)
        self.epsilon_func = epsilon_func
        self.label = label

    def _to_khronos(self, K, gp):
        """Convert to a Khronos Object (geometry + material pair)."""
        geom = self._to_khronos_geom(gp)
        mat = self.material._to_khronos(K)
        return K.Object(geom, mat)


class Block(GeometricObject):
    """Rectangular block (axis-aligned or rotated)."""

    def __init__(self, size=None, center=None, material=None,
                 e1=None, e2=None, e3=None, **kwargs):
        super().__init__(material=material, center=center, **kwargs)
        self.size = size if size is not None else Vector3()
        if not isinstance(self.size, Vector3):
            self.size = Vector3(*self.size)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def _to_khronos_geom(self, gp):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        s = _jl_vec([meep_to_khronos_length(x) if x < inf else 1e6
             for x in self.size])
        return gp.Cuboid(c, s)

    def contains(self, x, y, z):
        """Test if point (x,y,z) is inside this block."""
        cx, cy, cz = self.center.x, self.center.y, self.center.z
        sx = self.size.x if self.size.x < inf else 1e20
        sy = self.size.y if self.size.y < inf else 1e20
        sz = self.size.z if self.size.z < inf else 1e20
        return (abs(x - cx) <= sx / 2 and
                abs(y - cy) <= sy / 2 and
                abs(z - cz) <= sz / 2)


class Sphere(GeometricObject):
    """Sphere."""

    def __init__(self, radius, center=None, material=None, **kwargs):
        super().__init__(material=material, center=center, **kwargs)
        self.radius = radius

    def _to_khronos_geom(self, gp):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        r = meep_to_khronos_length(self.radius)
        return gp.Ball(c, r)

    def contains(self, x, y, z):
        """Test if point (x,y,z) is inside this sphere."""
        dx = x - self.center.x
        dy = y - self.center.y
        dz = z - self.center.z
        return dx*dx + dy*dy + dz*dz <= self.radius * self.radius


class Cylinder(GeometricObject):
    """Cylinder with specified radius, height, and axis."""

    def __init__(self, radius, center=None, material=None,
                 height=None, axis=None, **kwargs):
        super().__init__(material=material, center=center, **kwargs)
        self.radius = radius
        self.height = height if height is not None else inf
        if axis is None:
            self.axis = Vector3(0, 0, 1)
        elif isinstance(axis, (int, float)):
            # Handle axis as direction enum
            v = [0, 0, 0]
            v[int(axis)] = 1
            self.axis = Vector3(*v)
        elif isinstance(axis, Vector3):
            self.axis = axis
        else:
            self.axis = Vector3(*axis)

    def _to_khronos_geom(self, gp):
        c = _jl_vec([meep_to_khronos_length(x) for x in self.center])
        r = meep_to_khronos_length(self.radius)
        h = meep_to_khronos_length(self.height) if self.height < inf else 1e6
        axis_vec = _jl_vec([float(self.axis.x), float(self.axis.y), float(self.axis.z)])
        return gp.Cylinder(c, r, h, axis_vec)

    def contains(self, x, y, z):
        """Test if point (x,y,z) is inside this cylinder."""
        cx, cy, cz = self.center.x, self.center.y, self.center.z
        ax = self.axis
        # Project point onto axis
        dx, dy, dz = x - cx, y - cy, z - cz
        along = dx * ax.x + dy * ax.y + dz * ax.z
        h = self.height if self.height < inf else 1e20
        if abs(along) > h / 2:
            return False
        # Radial distance
        rx = dx - along * ax.x
        ry = dy - along * ax.y
        rz = dz - along * ax.z
        return rx*rx + ry*ry + rz*rz <= self.radius * self.radius


class Prism(GeometricObject):
    """Extruded polygon."""

    def __init__(self, vertices, height=None, axis=None,
                 center=None, material=None, sidewall_angle=0, **kwargs):
        super().__init__(material=material, center=center, **kwargs)
        self.vertices = [Vector3(*v) if not isinstance(v, Vector3) else v
                         for v in vertices]
        self.height = height if height is not None else inf
        if axis is None:
            self.axis = Vector3(0, 0, 1)
        elif isinstance(axis, Vector3):
            self.axis = axis
        else:
            self.axis = Vector3(*axis)
        self.sidewall_angle = sidewall_angle

    def _to_khronos_geom(self, gp):
        # Determine extrusion axis
        if self.axis.z != 0:
            ax = 2
        elif self.axis.y != 0:
            ax = 1
        else:
            ax = 0

        transverse = [i for i in range(3) if i != ax]
        h = meep_to_khronos_length(self.height) if self.height < inf else 1e6

        verts_um = [
            (meep_to_khronos_length(v[transverse[0]]),
             meep_to_khronos_length(v[transverse[1]]))
            for v in self.vertices
        ]
        cx = sum(v[0] for v in verts_um) / len(verts_um)
        cy = sum(v[1] for v in verts_um) / len(verts_um)

        center_3d = [0.0, 0.0, 0.0]
        center_3d[transverse[0]] = cx
        center_3d[transverse[1]] = cy
        center_3d[ax] = meep_to_khronos_length(self.center[ax])

        axis_vec = [0.0, 0.0, 0.0]
        axis_vec[ax] = 1.0

        return gp.Prism(verts_um, h, _jl_vec(axis_vec), _jl_vec(center_3d))

    def contains(self, x, y, z):
        """Test if point (x,y,z) is inside this prism."""
        # Determine extrusion axis
        if self.axis.z != 0:
            ax_idx = 2
        elif self.axis.y != 0:
            ax_idx = 1
        else:
            ax_idx = 0
        trans = [i for i in range(3) if i != ax_idx]
        pt = [x, y, z]

        # Height check along extrusion axis
        c_ax = self.center[ax_idx]
        h = self.height if self.height < inf else 1e20
        if abs(pt[ax_idx] - c_ax) > h / 2:
            return False

        # 2D point-in-polygon (ray casting) in the transverse plane
        px, py = pt[trans[0]], pt[trans[1]]
        verts = [(v[trans[0]], v[trans[1]]) for v in self.vertices]
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            yi, yj = verts[i][1], verts[j][1]
            xi, xj = verts[i][0], verts[j][0]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


class Cone(GeometricObject):
    """Cone (truncated). NOT SUPPORTED in Khronos."""

    def __init__(self, radius, radius2=0, center=None, material=None,
                 height=None, axis=None, **kwargs):
        raise NotImplementedError(
            "Cone geometry is not supported by the Khronos backend."
        )


class Ellipsoid(Block):
    """Ellipsoid (not supported — mapped to bounding box with warning)."""

    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "Ellipsoid is not natively supported by Khronos. "
            "It will be approximated as its bounding box.",
            stacklevel=2,
        )
        super().__init__(**kwargs)


class Wedge(GeometricObject):
    """Wedge. NOT SUPPORTED in Khronos."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "Wedge geometry is not supported by the Khronos backend."
        )


# ------------------------------------------------------------------- #
# FluxRegion / ModeRegion / Near2FarRegion
# ------------------------------------------------------------------- #

class FluxRegion:
    """Region for flux measurement."""

    def __init__(self, center=None, size=None, direction=AUTOMATIC,
                 weight=1.0, volume=None):
        if volume is not None:
            self.center = volume.center
            self.size = volume.size
        else:
            self.center = center if center is not None else Vector3()
            self.size = size if size is not None else Vector3()
        if not isinstance(self.center, Vector3):
            self.center = Vector3(*self.center)
        if not isinstance(self.size, Vector3):
            self.size = Vector3(*self.size)
        self.direction = direction
        self.weight = weight


class ModeRegion(FluxRegion):
    """Region for mode decomposition (alias of FluxRegion)."""
    pass


class Near2FarRegion(FluxRegion):
    """Region for near-to-far-field transformation (alias of FluxRegion)."""
    pass


class ForceRegion(FluxRegion):
    """Region for force computation."""
    pass


class EnergyRegion(FluxRegion):
    """Region for energy density computation."""
    pass


# ------------------------------------------------------------------- #
# MaterialGrid (for topology optimization)
# ------------------------------------------------------------------- #

class MaterialGrid:
    """Continuously varying material for topology optimization.

    Parameters
    ----------
    grid_size : Vector3
        Number of grid points (Nx, Ny, Nz).
    medium1 : Medium
        Material at weight = 0.
    medium2 : Medium
        Material at weight = 1.
    weights : numpy array, optional
        Initial design weights in [0, 1].
    grid_type : str
        Interpolation type. Default "U_DEFAULT".
    do_averaging : bool
        Whether to use subpixel averaging.
    beta : float
        Projection strength (0 = no projection).
    eta : float
        Projection threshold (0.5 = symmetric).
    damping : float
        Artificial dissipation for intermediate values.
    """

    def __init__(self, grid_size, medium1, medium2, weights=None,
                 grid_type="U_DEFAULT", do_averaging=False,
                 beta=0, eta=0.5, damping=0):
        if isinstance(grid_size, Vector3):
            self.grid_size = grid_size
        else:
            self.grid_size = Vector3(*grid_size)
        self.medium1 = medium1
        self.medium2 = medium2
        if weights is not None:
            import numpy as _np
            self.weights = _np.array(weights)
        else:
            self.weights = None
        self.grid_type = grid_type
        self.do_averaging = do_averaging
        self.beta = beta
        self.eta = eta
        self.damping = damping
        self.Nx = int(self.grid_size.x)
        self.Ny = int(self.grid_size.y)
        self.Nz = int(self.grid_size.z) if self.grid_size.z > 0 else 1

    def update_weights(self, x):
        """Update the design weights."""
        import numpy as _np
        self.weights = _np.array(x)


# ------------------------------------------------------------------- #
# Symmetry classes
# ------------------------------------------------------------------- #

class Symmetry:
    """Base symmetry class."""

    def __init__(self, direction=X, phase=1.0):
        self.direction = direction
        self.phase = phase


class Mirror(Symmetry):
    """Mirror symmetry."""
    pass


class Rotate2(Symmetry):
    """2-fold rotational symmetry."""
    pass


class Rotate4(Symmetry):
    """4-fold rotational symmetry."""
    pass


# ------------------------------------------------------------------- #
# Matrix and Lattice (for completeness)
# ------------------------------------------------------------------- #

class Matrix:
    """3x3 matrix."""

    def __init__(self, c1=None, c2=None, c3=None, diag=None, offdiag=None):
        if diag is not None:
            if not isinstance(diag, Vector3):
                diag = Vector3(*diag)
            self.c1 = Vector3(diag.x, 0, 0)
            self.c2 = Vector3(0, diag.y, 0)
            self.c3 = Vector3(0, 0, diag.z)
        else:
            self.c1 = c1 or Vector3(1, 0, 0)
            self.c2 = c2 or Vector3(0, 1, 0)
            self.c3 = c3 or Vector3(0, 0, 1)


class Lattice:
    """Lattice specification (for periodic structures)."""

    def __init__(self, size=None, basis_size=None,
                 basis1=None, basis2=None, basis3=None):
        self.size = size or Vector3(1, 1, 1)
        self.basis_size = basis_size or Vector3(1, 1, 1)
        self.basis1 = basis1 or Vector3(1, 0, 0)
        self.basis2 = basis2 or Vector3(0, 1, 0)
        self.basis3 = basis3 or Vector3(0, 0, 1)


# ------------------------------------------------------------------- #
# Utility functions
# ------------------------------------------------------------------- #

def interpolate(n, nums):
    """Linearly interpolate n points between each pair in nums."""
    if len(nums) < 2 or n < 1:
        return list(nums)
    result = []
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        for j in range(n + 1):
            t = j / (n + 1)
            if isinstance(a, Vector3):
                result.append(a * (1 - t) + b * t)
            else:
                result.append(a + (b - a) * t)
    result.append(nums[-1])
    return result


# ------------------------------------------------------------------- #
# LayerSpec / LayerStack (Khronos extension — not in meep)
# ------------------------------------------------------------------- #

class LayerSpec:
    """A single dielectric layer for near2far transfer-matrix projection.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    z_min : float
        Lower z-bound of the layer in meep units.
    z_max : float
        Upper z-bound (use mp.inf for semi-infinite).
    eps : float
        Relative permittivity.
    mu : float
        Relative permeability (default 1.0).
    """

    def __init__(self, z_min, z_max, eps, mu=1.0):
        self.z_min = z_min
        self.z_max = z_max
        self.eps = eps
        self.mu = mu


class LayerStack:
    """Stack of dielectric layers for near2far transfer-matrix projection.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    layers : list of LayerSpec
        Ordered list of layers (bottom to top).
    """

    def __init__(self, layers):
        self.layers = list(layers)

    def _to_khronos(self, K):
        from ._units import meep_to_khronos_length
        k_layers = []
        for ls in self.layers:
            z_min = meep_to_khronos_length(ls.z_min) if ls.z_min < 1e18 else 1e20
            z_max = meep_to_khronos_length(ls.z_max) if ls.z_max < 1e18 else 1e20
            k_layers.append(K.LayerSpec(z_min, z_max, float(ls.eps), mu=float(ls.mu)))
        return K.LayerStack(k_layers)


# ------------------------------------------------------------------- #
# GDSII import (Khronos extension — not in meep)
# ------------------------------------------------------------------- #

def import_gdsii(filename, cell_name, layer_map, axis=2):
    """Import geometry objects from a GDSII file.

    NOT available in meep — this is a Khronos extension.

    Parameters
    ----------
    filename : str
        Path to the .gds file.
    cell_name : str
        Name of the GDSII cell to extract.
    layer_map : dict
        Maps ``(layer, datatype)`` tuples to ``(z_min, z_max, Medium)``
        tuples. The Medium is converted to a Khronos Material.
    axis : int
        Extrusion axis (0=x, 1=y, 2=z). Default 2.

    Returns
    -------
    list
        List of Khronos geometry objects (ready to add to Simulation.geometry).
    """
    from .._bridge import get_khronos, get_gp
    from ._units import meep_to_khronos_length
    K = get_khronos()
    gp = get_gp()

    lib = K.read_gds(filename)
    k_layer_map = {}
    for (layer, dtype), (z_min, z_max, medium) in layer_map.items():
        k_layer_map[(layer, dtype)] = (
            meep_to_khronos_length(z_min),
            meep_to_khronos_length(z_max),
            medium._to_khronos(K),
        )
    objects = K.gds_to_objects(lib, cell_name, k_layer_map, axis=axis + 1)
    return list(objects)
