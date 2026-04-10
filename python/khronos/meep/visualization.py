# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Visualization module for khronos.meep simulations.

Provides meep-style 2D cross-section plots showing geometry (epsilon),
PML boundaries, sources, monitors, and optional field overlays. Works
purely in Python without Julia — uses the geometry objects' containment
methods to sample epsilon.

Usage::

    import khronos.meep as mp

    sim = mp.Simulation(...)
    sim.add_flux(...)

    # Setup view (before running)
    mp.plot2D(sim, output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(16, 0, 8)))

    # After running, overlay fields
    mp.plot2D(sim, output_plane=..., fields=field_data)
"""

import math

# numpy and matplotlib are optional — imported lazily inside functions
# so the module can be loaded without them installed.

from .geom import Vector3, Volume, Block, Sphere, Cylinder, Prism, GeometricObject
from .constants import X, Y, Z, ALL, Low, High, inf
from .source import GaussianBeamSource, EigenModeSource, Source

# ------------------------------------------------------------------- #
# Default plotting parameters (matching meep conventions)
# ------------------------------------------------------------------- #

default_eps_parameters = {
    "cmap": "binary",
    "alpha": 1.0,
    "interpolation": "none",
    "contour": False,
    "resolution": None,      # override sim.resolution for epsilon sampling
}

default_field_parameters = {
    "cmap": "inferno",
    "alpha": 0.6,
    "interpolation": "none",
    "post_process": None,  # set to |field|^2 at runtime (needs numpy)
}

default_source_parameters = {
    "facecolor": "orange",
    "edgecolor": "orange",
    "alpha": 0.15,
    "hatch": "//",
    "linewidth": 2,
    "linecolor": "orange",
}

default_monitor_parameters = {
    "facecolor": "blue",
    "edgecolor": "blue",
    "alpha": 0.15,
    "hatch": "//",
    "linewidth": 2,
    "linecolor": "blue",
}

default_boundary_parameters = {
    "facecolor": "green",
    "edgecolor": "green",
    "alpha": 0.12,
    "hatch": "",
    "linewidth": 0.5,
    "linestyle": "--",
}


# ------------------------------------------------------------------- #
# Epsilon grid sampler
# ------------------------------------------------------------------- #

def get_epsilon_grid(sim, output_plane, resolution=None):
    """Sample permittivity on a 2D grid.

    Tries to use the Julia backend (Khronos.sample_geometry_slice) first,
    which uses the same rasterization algorithm as the FDTD engine. Falls
    back to a pure-Python sampler when Julia is not available.

    Parameters
    ----------
    sim : Simulation
        The simulation object with .geometry, .default_material.
    output_plane : Volume
        2D plane to sample (one dimension of size must be 0).
    resolution : int, optional
        Pixels per meep unit. Defaults to sim.resolution.

    Returns
    -------
    eps_data : 2D numpy array
    extent : list [umin, umax, vmin, vmax] for imshow
    """
    res = resolution or sim.resolution
    c = output_plane.center
    s = output_plane.size

    # Determine plane and coordinate ranges
    if s.x == 0:
        plane, fixed_val = "yz", c.x
        u_range = (c.y - s.y / 2, c.y + s.y / 2)
        v_range = (c.z - s.z / 2, c.z + s.z / 2)
        normal = ":x"
    elif s.y == 0:
        plane, fixed_val = "xz", c.y
        u_range = (c.x - s.x / 2, c.x + s.x / 2)
        v_range = (c.z - s.z / 2, c.z + s.z / 2)
        normal = ":y"
    elif s.z == 0:
        plane, fixed_val = "xy", c.z
        u_range = (c.x - s.x / 2, c.x + s.x / 2)
        v_range = (c.y - s.y / 2, c.y + s.y / 2)
        normal = ":z"
    else:
        raise ValueError("output_plane must have one zero-sized dimension")

    extent = [u_range[0], u_range[1], v_range[0], v_range[1]]

    # Try Julia backend first — uses the same algorithm as the FDTD engine
    eps_data = _get_epsilon_grid_julia(sim, normal, fixed_val, u_range, v_range, res)
    if eps_data is not None:
        return eps_data, extent

    # Fallback: pure Python sampler
    return _get_epsilon_grid_python(sim, plane, fixed_val, u_range, v_range, res), extent


def _get_epsilon_grid_julia(sim, normal, fixed_val, u_range, v_range, resolution):
    """Use Khronos.sample_geometry_slice via Julia for accurate epsilon."""
    try:
        from .._bridge import get_khronos, get_gp
        from ._units import meep_to_khronos_length
        K = get_khronos()

        # Build Khronos geometry objects if not already done
        if not hasattr(sim, '_khronos_geometry_cache') or sim._khronos_geometry_cache is None:
            gp = get_gp()
            k_geom = []
            for obj in sim.geometry:
                if hasattr(obj, '_to_khronos'):
                    k_geom.append(obj._to_khronos(K, gp))
            sim._khronos_geometry_cache = k_geom

        import numpy as np
        # Convert ranges to Khronos units (µm)
        u_range_k = (meep_to_khronos_length(u_range[0]), meep_to_khronos_length(u_range[1]))
        v_range_k = (meep_to_khronos_length(v_range[0]), meep_to_khronos_length(v_range[1]))
        fixed_k = meep_to_khronos_length(fixed_val)

        # Call Julia's sample_geometry_slice
        from juliacall import Main as jl
        normal_sym = jl.seval(normal)
        eps_data, _, _ = K.sample_geometry_slice(
            sim._khronos_geometry_cache,
            normal_sym,
            fixed_k,
            x_range=u_range_k,
            y_range=v_range_k,
            resolution=int(resolution),
        )
        return np.array(eps_data)
    except Exception:
        return None


def _get_epsilon_grid_python(sim, plane, fixed_val, u_range, v_range, res):
    """Pure-Python epsilon grid sampler (fallback when Julia unavailable).

    Slower than the Julia backend but works without any external dependencies.
    """
    import numpy as np

    Nu = max(2, int((u_range[1] - u_range[0]) * res))
    Nv = max(2, int((v_range[1] - v_range[0]) * res))

    us = np.linspace(u_range[0], u_range[1], Nu)
    vs = np.linspace(v_range[0], v_range[1], Nv)

    default_eps = sim.default_material.epsilon if hasattr(sim.default_material, 'epsilon') else 1.0
    eps_data = np.full((Nu, Nv), default_eps, dtype=np.float32)

    # Build meshgrid for vectorized containment
    UU, VV = np.meshgrid(us, vs, indexing='ij')

    # Iterate geometry: last object wins (meep convention)
    for obj in sim.geometry:
        if not isinstance(obj, GeometricObject):
            continue
        mat_eps = obj.material.epsilon if hasattr(obj, 'material') and obj.material is not None else default_eps

        if isinstance(obj, Block):
            # Vectorized AABB containment
            cx, cy, cz = obj.center.x, obj.center.y, obj.center.z
            sx_b = obj.size.x if obj.size.x < inf else 1e20
            sy_b = obj.size.y if obj.size.y < inf else 1e20
            sz_b = obj.size.z if obj.size.z < inf else 1e20
            if plane == "xy":
                if abs(fixed_val - cz) > sz_b / 2:
                    continue
                mask = (np.abs(UU - cx) <= sx_b / 2) & (np.abs(VV - cy) <= sy_b / 2)
            elif plane == "xz":
                if abs(fixed_val - cy) > sy_b / 2:
                    continue
                mask = (np.abs(UU - cx) <= sx_b / 2) & (np.abs(VV - cz) <= sz_b / 2)
            else:  # yz
                if abs(fixed_val - cx) > sx_b / 2:
                    continue
                mask = (np.abs(UU - cy) <= sy_b / 2) & (np.abs(VV - cz) <= sz_b / 2)
            eps_data[mask] = mat_eps

        elif isinstance(obj, Prism) and hasattr(obj, 'vertices'):
            # Prism: check height along axis, then vectorized 2D point-in-polygon
            from matplotlib.path import Path as MplPath

            verts_xy = [(v.x, v.y) for v in obj.vertices]
            vx = [v[0] for v in verts_xy]
            vy = [v[1] for v in verts_xy]
            bb_xmin, bb_xmax = min(vx), max(vx)
            bb_ymin, bb_ymax = min(vy), max(vy)
            c_ax = obj.center.z if obj.axis.z != 0 else (obj.center.y if obj.axis.y != 0 else obj.center.x)
            h = obj.height if obj.height < inf else 1e20

            if plane == "xy":
                if abs(fixed_val - c_ax) > h / 2:
                    continue
                # Vectorized: build Path from polygon vertices and test all grid points
                poly_path = MplPath(verts_xy)
                # Only test within bounding box for speed
                u_idx = np.where((us >= bb_xmin) & (us <= bb_xmax))[0]
                v_idx = np.where((vs >= bb_ymin) & (vs <= bb_ymax))[0]
                if len(u_idx) > 0 and len(v_idx) > 0:
                    sub_u, sub_v = np.meshgrid(us[u_idx], vs[v_idx], indexing='ij')
                    pts = np.column_stack([sub_u.ravel(), sub_v.ravel()])
                    inside = poly_path.contains_points(pts).reshape(sub_u.shape)
                    for ii, ui in enumerate(u_idx):
                        for jj, vi in enumerate(v_idx):
                            if inside[ii, jj]:
                                eps_data[ui, vi] = mat_eps

            elif plane == "xz":
                # For XZ plane, test if fixed_val (y) is inside the polygon's y-range
                if fixed_val < bb_ymin or fixed_val > bb_ymax:
                    continue
                if abs(fixed_val - obj.center.y) > max(abs(bb_ymax), abs(bb_ymin)) + 1:
                    continue
                # Find x-range where polygon intersects y=fixed_val
                # and fill z-range from zmin to zmin+height
                for iu in range(Nu):
                    x = us[iu]
                    if x < bb_xmin or x > bb_xmax:
                        continue
                    if obj.contains(x, fixed_val, c_ax):
                        for iv in range(Nv):
                            z = vs[iv]
                            if abs(z - c_ax) <= h / 2:
                                eps_data[iu, iv] = mat_eps

            else:  # yz
                for iv in range(Nv):
                    for iu in range(Nu):
                        if obj.contains(fixed_val, us[iu], vs[iv]):
                            eps_data[iu, iv] = mat_eps

        elif hasattr(obj, 'contains'):
            # Generic fallback (Sphere, Cylinder, etc.)
            for iv in range(Nv):
                for iu in range(Nu):
                    if plane == "xy":
                        x, y, z = us[iu], vs[iv], fixed_val
                    elif plane == "xz":
                        x, y, z = us[iu], fixed_val, vs[iv]
                    else:
                        x, y, z = fixed_val, us[iu], vs[iv]
                    if obj.contains(x, y, z):
                        eps_data[iu, iv] = mat_eps

    return eps_data


# ------------------------------------------------------------------- #
# Sub-plot functions
# ------------------------------------------------------------------- #

def _get_plane_axes(output_plane):
    """Return (xlabel, ylabel, plane_type) for the output plane."""
    s = output_plane.size
    if s.x == 0:
        return "y (µm)", "z (µm)", "yz"
    elif s.y == 0:
        return "x (µm)", "z (µm)", "xz"
    else:
        return "x (µm)", "y (µm)", "xy"


def _vol_to_rect(volume, plane_type):
    """Convert a Volume to a (u_center, v_center, u_size, v_size) in the plane."""
    c = volume.center
    s = volume.size
    if plane_type == "xy":
        return c.x, c.y, s.x, s.y
    elif plane_type == "xz":
        return c.x, c.z, s.x, s.z
    else:  # yz
        return c.y, c.z, s.y, s.z


def plot_eps(sim, ax, output_plane, eps_parameters=None):
    """Plot permittivity as a 2D heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    params = {**default_eps_parameters, **(eps_parameters or {})}
    res = params.pop("resolution", None)
    contour = params.pop("contour", False)

    eps_data, extent = get_epsilon_grid(sim, output_plane, resolution=res)

    if contour:
        ax.contour(eps_data.T, levels=np.unique(eps_data),
                   origin="lower", extent=extent, colors="gray",
                   linewidths=0.5)
    else:
        ax.imshow(eps_data.T, origin="lower", extent=extent,
                  aspect="auto", cmap=params.get("cmap", "binary"),
                  alpha=params.get("alpha", 1.0),
                  interpolation=params.get("interpolation", "none"))


def plot_boundaries(sim, ax, output_plane, boundary_parameters=None):
    """Draw PML boundary regions as semi-transparent rectangles."""
    import matplotlib.patches as patches
    params = {**default_boundary_parameters, **(boundary_parameters or {})}
    _, _, plane_type = _get_plane_axes(output_plane)

    cell_c = sim.geometry_center if hasattr(sim, 'geometry_center') else Vector3()
    cell_s = sim.cell_size

    for pml in sim.boundary_layers:
        thickness = pml.thickness
        direction = pml.direction
        side = pml.side

        if direction == ALL:
            dirs = [X, Y, Z]
        else:
            dirs = [direction]

        for d in dirs:
            if side == ALL:
                sides = [Low, High]
            else:
                sides = [side]

            for s in sides:
                # Compute PML slab volume
                vol_c = [cell_c.x, cell_c.y, cell_c.z]
                vol_s = [cell_s.x, cell_s.y, cell_s.z]
                half = [cell_s.x / 2, cell_s.y / 2, cell_s.z / 2]

                if s == Low:
                    vol_c[d] = cell_c[d] - half[d] + thickness / 2
                else:
                    vol_c[d] = cell_c[d] + half[d] - thickness / 2
                vol_s[d] = thickness

                vol = Volume(center=Vector3(*vol_c), size=Vector3(*vol_s))
                uc, vc, us, vs = _vol_to_rect(vol, plane_type)

                if us > 0 and vs > 0:
                    rect = patches.Rectangle(
                        (uc - us / 2, vc - vs / 2), us, vs,
                        linewidth=params["linewidth"],
                        linestyle=params.get("linestyle", "--"),
                        edgecolor=params["edgecolor"],
                        facecolor=params["facecolor"],
                        alpha=params["alpha"],
                        hatch=params.get("hatch", ""),
                    )
                    ax.add_patch(rect)


def plot_sources(sim, ax, output_plane, source_parameters=None):
    """Draw source regions with direction arrows."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    params = {**default_source_parameters, **(source_parameters or {})}
    _, _, plane_type = _get_plane_axes(output_plane)

    for src in sim.sources:
        c = src.center
        s = src.size
        uc, vc, us, vs = _vol_to_rect(
            Volume(center=c, size=s), plane_type)

        # Draw source region
        if us > 0 and vs > 0:
            rect = patches.Rectangle(
                (uc - us / 2, vc - vs / 2), us, vs,
                linewidth=params["linewidth"],
                edgecolor=params["edgecolor"],
                facecolor=params["facecolor"],
                alpha=params["alpha"],
                hatch=params.get("hatch", ""),
            )
            ax.add_patch(rect)
        elif us > 0:
            ax.plot([uc - us / 2, uc + us / 2], [vc, vc],
                    color=params["linecolor"], linewidth=params["linewidth"])
        elif vs > 0:
            ax.plot([uc, uc], [vc - vs / 2, vc + vs / 2],
                    color=params["linecolor"], linewidth=params["linewidth"])
        else:
            ax.plot(uc, vc, "o", color=params["linecolor"], ms=5, zorder=5)

        # Direction arrow for GaussianBeamSource
        if isinstance(src, GaussianBeamSource) and hasattr(src, 'beam_kdir'):
            kdir = src.beam_kdir
            arrow_len = 2.0
            if plane_type == "xz":
                du, dv = kdir.x * arrow_len, kdir.z * arrow_len
            elif plane_type == "xy":
                du, dv = kdir.x * arrow_len, kdir.y * arrow_len
            else:
                du, dv = kdir.y * arrow_len, kdir.z * arrow_len
            ax.annotate("", xy=(uc + du, vc + dv), xytext=(uc, vc),
                        arrowprops=dict(arrowstyle="->", color=params["linecolor"], lw=2))

        # Beam waist circle for GaussianBeamSource in XY plane
        if isinstance(src, GaussianBeamSource) and plane_type == "xy":
            w0 = src.beam_w0 if hasattr(src, 'beam_w0') else 5.0
            circle = plt.Circle((uc, vc), w0, fill=False,
                                color=params["linecolor"], linewidth=1.5,
                                linestyle="--")
            ax.add_patch(circle)


def plot_monitors(sim, ax, output_plane, monitor_parameters=None):
    """Draw monitor regions."""
    import matplotlib.patches as patches
    params = {**default_monitor_parameters, **(monitor_parameters or {})}
    _, _, plane_type = _get_plane_axes(output_plane)

    monitors = sim._all_dft_objects if hasattr(sim, '_all_dft_objects') else []
    for mon in monitors:
        if not hasattr(mon, 'center') or not hasattr(mon, 'size'):
            continue
        c = mon.center
        s = mon.size
        uc, vc, us, vs = _vol_to_rect(
            Volume(center=Vector3(*c) if not isinstance(c, Vector3) else c,
                   size=Vector3(*s) if not isinstance(s, Vector3) else s),
            plane_type)

        if us > 0 and vs > 0:
            rect = patches.Rectangle(
                (uc - us / 2, vc - vs / 2), us, vs,
                linewidth=params["linewidth"],
                edgecolor=params["edgecolor"],
                facecolor=params["facecolor"],
                alpha=params["alpha"],
                hatch=params.get("hatch", ""),
            )
            ax.add_patch(rect)
        elif us > 0:
            ax.plot([uc - us / 2, uc + us / 2], [vc, vc],
                    color=params["linecolor"], linewidth=params["linewidth"],
                    linestyle="--")
        elif vs > 0:
            ax.plot([uc, uc], [vc - vs / 2, vc + vs / 2],
                    color=params["linecolor"], linewidth=params["linewidth"],
                    linestyle="--")

        # Direction arrow for flux monitors (points in +normal direction)
        from .dft import DftFlux
        if isinstance(mon, DftFlux):
            arrow_len = 1.5
            if us == 0 and vs > 0:
                ax.annotate("", xy=(uc + arrow_len, vc), xytext=(uc, vc),
                            arrowprops=dict(arrowstyle="->", color=params["linecolor"], lw=1.5))


def plot_fields(ax, field_data, output_plane, field_parameters=None):
    """Overlay field data on the axes.

    Parameters
    ----------
    ax : matplotlib Axes
    field_data : 2D numpy array
        Complex or real field data.
    output_plane : Volume
        The 2D plane the data corresponds to.
    field_parameters : dict, optional
    """
    params = {**default_field_parameters, **(field_parameters or {})}
    import numpy as np
    post_process = params.pop("post_process", None) or (lambda x: np.abs(x) ** 2)

    data = post_process(field_data)
    if data.ndim > 2:
        data = data.squeeze()
    if data.ndim != 2:
        return

    c = output_plane.center
    s = output_plane.size
    if s.x == 0:
        extent = [c.y - s.y/2, c.y + s.y/2, c.z - s.z/2, c.z + s.z/2]
    elif s.y == 0:
        extent = [c.x - s.x/2, c.x + s.x/2, c.z - s.z/2, c.z + s.z/2]
    else:
        extent = [c.x - s.x/2, c.x + s.x/2, c.y - s.y/2, c.y + s.y/2]

    vmax = np.percentile(data[data > 0], 95) if np.any(data > 0) else 1.0
    ax.imshow(data.T, origin="lower", extent=extent,
              aspect="auto", cmap=params.get("cmap", "inferno"),
              alpha=params.get("alpha", 0.6),
              interpolation=params.get("interpolation", "none"),
              vmin=0, vmax=vmax)


# ------------------------------------------------------------------- #
# Main entry point
# ------------------------------------------------------------------- #

def plot2D(sim, ax=None, output_plane=None, fields=None,
           eps_parameters=None, boundary_parameters=None,
           source_parameters=None, monitor_parameters=None,
           field_parameters=None,
           show_epsilon=True, show_sources=True, show_monitors=True,
           show_boundary_layers=True, labels=False, **kwargs):
    """Plot a 2D cross-section of the simulation.

    Draws geometry (epsilon), PML boundaries, sources, monitors, and
    optionally overlays field data. Works without Julia — geometry is
    sampled directly from Python objects.

    Parameters
    ----------
    sim : Simulation
        The simulation object.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.
    output_plane : Volume
        2D cross-section plane. One dimension of size must be 0.
        If None, uses the XY plane at z=0.
    fields : numpy array, optional
        2D field data to overlay.
    show_epsilon : bool
        Draw the geometry (permittivity). Default True.
    show_sources : bool
        Draw source regions. Default True.
    show_monitors : bool
        Draw monitor regions. Default True.
    show_boundary_layers : bool
        Draw PML boundaries. Default True.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if output_plane is None:
        # Default: XY plane at z=0
        output_plane = Volume(
            center=sim.geometry_center if hasattr(sim, 'geometry_center') else Vector3(),
            size=Vector3(sim.cell_size.x, sim.cell_size.y, 0),
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    xlabel, ylabel, plane_type = _get_plane_axes(output_plane)

    # Layer 1: Epsilon (geometry)
    if show_epsilon:
        plot_eps(sim, ax, output_plane, eps_parameters)

    # Layer 2: PML boundaries
    if show_boundary_layers:
        plot_boundaries(sim, ax, output_plane, boundary_parameters)

    # Layer 3: Sources
    if show_sources:
        plot_sources(sim, ax, output_plane, source_parameters)

    # Layer 4: Monitors
    if show_monitors:
        plot_monitors(sim, ax, output_plane, monitor_parameters)

    # Layer 5: Fields (optional overlay)
    if fields is not None:
        plot_fields(ax, fields, output_plane, field_parameters)

    # Axis formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    # Set limits from output plane
    c = output_plane.center
    s = output_plane.size
    if plane_type == "xy":
        ax.set_xlim(c.x - s.x / 2, c.x + s.x / 2)
        ax.set_ylim(c.y - s.y / 2, c.y + s.y / 2)
    elif plane_type == "xz":
        ax.set_xlim(c.x - s.x / 2, c.x + s.x / 2)
        ax.set_ylim(c.z - s.z / 2, c.z + s.z / 2)
    else:
        ax.set_xlim(c.y - s.y / 2, c.y + s.y / 2)
        ax.set_ylim(c.z - s.z / 2, c.z + s.z / 2)

    return ax
