#!/usr/bin/env python3
"""
Grating Coupler Coupling Efficiency — khronos.meep API
======================================================

Simulates the fiber-to-waveguide coupling efficiency of the UBC SiEPIC
EBeam ``ebeam_gc_te1550`` grating coupler using the khronos.meep Python API.

Geometry is loaded from the foundry GDS file using gdspy. A Gaussian beam
source (SMF-28, 15° from normal) illuminates the grating, and a flux monitor
at the waveguide output measures broadband coupling efficiency (1500–1600 nm).

Produces figures:
  - fig0_setup_xz.png:  XZ setup view (geometry + PML + source + monitor)
  - fig1_setup_xy.png:  XY setup view
  - fig2_field_xz.png:  Steady-state |Ey|² with geometry contours (XZ)
  - fig3_field_xy.png:  Steady-state |Ey|² with geometry contours (XY)
  - fig4_coupling.png:  Broadband coupling efficiency

Usage::

    source ~/.venv/bin/activate
    python grating_coupler.py
"""

import math
import os
import sys

import gdspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
import numpy as np

# Add khronos.meep to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import khronos.meep as mp

# ================================================================== #
# Physical parameters (UBC SiEPIC EBeam PDK)
# ================================================================== #

n_si   = 3.48
n_sio2 = 1.44
si     = mp.Medium(epsilon=n_si**2)
sio2   = mp.Medium(epsilon=n_sio2**2)

si_thickness  = 0.22   # 220 nm core
box_thickness = 2.0    # 2 µm BOX (truncated; real is 3 µm)
clad_height   = 2.5    # air above Si

wvl_center = 1.55
wvl_min, wvl_max = 1.50, 1.60
fcen   = 1.0 / wvl_center
fmin   = 1.0 / wvl_max
fmax   = 1.0 / wvl_min
fwidth = fmax - fmin
nfreq  = 101

# Fiber: 15° from surface normal (standard SiPh convention)
fiber_angle_deg = 15.0
fiber_waist     = 5.2   # MFD/2 for SMF-28
fiber_angle_rad = math.radians(fiber_angle_deg)

# GDS
gds_file = os.path.expanduser(
    "~/ubc/ubcpdk/gds/EBeam/ebeam_gc_te1550.gds")

# ================================================================== #
# Load GDS polygons with gdspy
# ================================================================== #

print("Loading GDS with gdspy...")
gds_lib  = gdspy.GdsLibrary(infile=gds_file)
gds_cell = gds_lib.cells["ebeam_gc_te1550"]
polys_by_spec = gds_cell.get_polygons(by_spec=True)

si_polygons = polys_by_spec.get((1, 0), [])
print(f"  Cell: ebeam_gc_te1550, layer (1,0): {len(si_polygons)} polygons")

# Compute bounding box
all_pts = np.concatenate(si_polygons)
gds_xmin, gds_ymin = all_pts.min(axis=0)
gds_xmax, gds_ymax = all_pts.max(axis=0)
print(f"  Extent: x=[{gds_xmin:.2f}, {gds_xmax:.2f}], "
      f"y=[{gds_ymin:.2f}, {gds_ymax:.2f}]")

# ================================================================== #
# Build geometry
# ================================================================== #

# Convert GDS polygons to mp.Prism objects (extruded along z)
geometry = []

# 1. Silicon grating coupler from GDS (layer 1,0)
for pg in si_polygons:
    vertices = [mp.Vector3(v[0], v[1]) for v in pg]
    geometry.append(mp.Prism(
        vertices=vertices,
        height=si_thickness,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(z=si_thickness / 2),
        material=si,
    ))

# 2. Waveguide extension beyond GDS (extends from x≈0 into +x PML)
#    The GDS waveguide ends at x≈0. We need to extend it so the mode
#    propagates into the flux monitor and out through PML.
wg_width = 0.5  # 500 nm strip
wg_extension = 5.0  # extend 5 µm beyond GDS edge
geometry.append(mp.Block(
    center=mp.Vector3(gds_xmax + wg_extension / 2, 0, si_thickness / 2),
    size=mp.Vector3(wg_extension, wg_width, si_thickness),
    material=si,
))

# 3. BOX layer (SiO2 spanning all x/y)
geometry.append(mp.Block(
    center=mp.Vector3(0, 0, -box_thickness / 2),
    size=mp.Vector3(mp.inf, mp.inf, box_thickness),
    material=sio2,
))

print(f"  Geometry: {len(geometry)} objects "
      f"({len(si_polygons)} GDS + 1 wg extension + 1 BOX)")

# ================================================================== #
# Simulation domain
# ================================================================== #

dpml       = 1.0
resolution = 30  # pixels/µm
margin     = 2.0

# Domain encompasses the full GC + waveguide extension + margins
sx = (gds_xmax + wg_extension) - gds_xmin + 2 * margin
sy = (gds_ymax - gds_ymin) + 2 * margin
sz = box_thickness + si_thickness + clad_height

# Center the domain
cell_center_x = (gds_xmin + gds_xmax + wg_extension) / 2
cell_center_z = (clad_height + si_thickness - box_thickness) / 2

cell_size   = mp.Vector3(sx, sy, sz)
cell_center = mp.Vector3(cell_center_x, 0, cell_center_z)

# ================================================================== #
# Source: Gaussian beam (15° from normal)
# ================================================================== #

# Source plane must be below PML. PML top = cell_center_z + sz/2 - dpml.
pml_top   = cell_center_z + sz / 2 - dpml
source_z  = min(si_thickness + 1.5, pml_top - 0.5)

# Beam focus at grating teeth center (x ≈ -24 µm)
grating_center_x = -24.0
# Source plane shifted to account for angle
source_x = grating_center_x - (source_z - si_thickness) * math.tan(fiber_angle_rad)

# k-vector: mostly -z (downward), slight +x (toward waveguide)
beam_kdir = mp.Vector3(
    math.sin(fiber_angle_rad),   # 0.259
    0,
    -math.cos(fiber_angle_rad),  # -0.966
)

sources = [
    mp.GaussianBeamSource(
        src=mp.GaussianSource(fcen, fwidth=fwidth),
        center=mp.Vector3(source_x, 0, source_z),
        size=mp.Vector3(15, 15, 0),
        beam_x0=mp.Vector3(0, 0, 0),
        beam_kdir=beam_kdir,
        beam_w0=fiber_waist,
        beam_E0=mp.Vector3(0, 1, 0),  # TE polarization
    ),
]

# ================================================================== #
# Monitor at waveguide output
# ================================================================== #

# Place monitor ~1 µm before the +x PML boundary
mon_x     = gds_xmax + wg_extension / 2
mon_center = mp.Vector3(mon_x, 0, si_thickness / 2)
mon_size   = mp.Vector3(0, wg_width * 3, si_thickness * 4)

# ================================================================== #
# Build simulation
# ================================================================== #

print(f"\nSimulation setup:")
print(f"  Domain:     {sx:.1f} × {sy:.1f} × {sz:.1f} µm")
print(f"  Resolution: {resolution} px/µm")
print(f"  PML:        {dpml} µm")
print(f"  Source:     ({source_x:.2f}, 0, {source_z:.2f}), "
      f"k=({beam_kdir.x:.3f}, {beam_kdir.y:.3f}, {beam_kdir.z:.3f})")
print(f"  Monitor:    ({mon_x:.2f}, 0, {si_thickness/2:.2f})")

# ================================================================== #
# Run simulation (requires Julia + CUDA)
# ================================================================== #

try:
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry_center=cell_center,
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        default_material=mp.air,
        eps_averaging=False,
        backend="cuda",
    )

    output_flux = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mon_center, size=mon_size),
    )

    dft_xz = sim.add_dft_fields(
        [mp.Ey], fcen, 0, 1,
        where=mp.Volume(
            center=mp.Vector3(cell_center_x, 0, cell_center_z),
            size=mp.Vector3(sx - 2 * dpml, 0, sz - 2 * dpml),
        ),
    )
    dft_xy = sim.add_dft_fields(
        [mp.Ey], fcen, 0, 1,
        where=mp.Volume(
            center=mp.Vector3(cell_center_x, 0, si_thickness / 2),
            size=mp.Vector3(sx - 2 * dpml, sy - 2 * dpml, 0),
        ),
    )

    print("\nRunning 3D FDTD simulation...")
    sim.run(
        until_after_sources=mp.stop_when_dft_decayed(tol=1e-4),
    )
    print("Simulation complete.\n")

    # Extract results
    freqs       = np.array(mp.get_flux_freqs(output_flux))
    flux        = np.array(mp.get_fluxes(output_flux))
    wavelengths = 1.0 / freqs

    peak_flux = np.max(np.abs(flux))
    eff       = np.abs(flux) / peak_flux if peak_flux > 0 else np.zeros_like(flux)
    eff_db    = 10 * np.log10(np.clip(eff, 1e-10, None))

    peak_idx = int(np.argmax(np.abs(flux)))
    half_max = peak_flux / 2
    in_band  = wavelengths[np.abs(flux) >= half_max]
    bw_nm    = abs(in_band[-1] - in_band[0]) * 1000 if len(in_band) >= 2 else 0

    print(f"  Peak flux:    {peak_flux:.4e}")
    print(f"  Peak at:      {wavelengths[peak_idx]*1000:.1f} nm")
    print(f"  3dB BW:       {bw_nm:.1f} nm")

    # Extract DFT fields
    field_xz = sim.get_dft_array(dft_xz, mp.Ey, 0)
    field_xy = sim.get_dft_array(dft_xy, mp.Ey, 0)

    # Save all results
    np.savetxt(os.path.join(outdir, "gc_te1550_results.csv"),
               np.column_stack([wavelengths * 1000, flux, eff, eff_db]),
               header="wavelength_nm,flux,rel_eff,dB",
               delimiter=",", fmt="%.4f,%.6e,%.6f,%.2f")

    if field_xz is not None and field_xz.size > 0:
        np.save(os.path.join(outdir, "field_xz.npy"), field_xz)
    if field_xy is not None and field_xy.size > 0:
        np.save(os.path.join(outdir, "field_xy.npy"), field_xy)

    print(f"  Data saved to {outdir}")

    # Generate field overlay figures
    _plot_fields_and_coupling()

except Exception as e:
    print(f"\n  Simulation requires Julia+CUDA: {type(e).__name__}: {e}")
    print("  Setup figures were generated. Run the Julia driver for FDTD,")
    print("  then re-run this script to produce field overlay plots.")

    # If saved data exists from a previous Julia run, plot it
    results_file = os.path.join(outdir, "gc_te1550_results.csv")
    if os.path.exists(results_file):
        print("\n  Found previous results — generating plots...")
        _plot_fields_and_coupling()


def _plot_fields_and_coupling():
    """Generate field overlay and coupling efficiency figures from saved data."""
    results_file = os.path.join(outdir, "gc_te1550_results.csv")
    if not os.path.exists(results_file):
        print("  No results file found. Skipping field/coupling plots.")
        return

    data = np.loadtxt(results_file, delimiter=",", skiprows=1)
    wl_nm = data[:, 0]
    flux_data = data[:, 1]
    eff_data = data[:, 2]
    eff_db_data = data[:, 3]

    pk = int(np.argmax(np.abs(flux_data)))
    half = np.max(np.abs(flux_data)) / 2
    ib = wl_nm[np.abs(flux_data) >= half]
    bw = abs(ib[-1] - ib[0]) if len(ib) >= 2 else 0

    # Field data
    xz_extent = [x_lo + dpml, x_hi - dpml, z_lo + dpml, z_hi - dpml]
    xy_extent = [x_lo + dpml, x_hi - dpml, y_lo + dpml, y_hi - dpml]

    # XZ field overlay
    xz_file = os.path.join(outdir, "field_xz.npy")
    if os.path.exists(xz_file):
        fxz = np.abs(np.load(xz_file).squeeze()) ** 2
        if fxz.ndim == 2:
            fig2, ax2 = plt.subplots(1, 1, figsize=(16, 5))
            draw_geometry_xz(ax2)
            im = ax2.imshow(fxz.T, origin="lower", extent=xz_extent,
                            cmap="inferno", aspect="auto", alpha=0.7,
                            vmax=np.percentile(fxz, 95))
            plt.colorbar(im, ax=ax2, label="|Ey|²", shrink=0.8)
            draw_pml(ax2, "xz", dpml)
            draw_monitor(ax2, "xz", mon_center, mon_size)
            ax2.set_xlim(x_lo, x_hi)
            ax2.set_ylim(z_lo, z_hi)
            ax2.set_xlabel("x (µm)")
            ax2.set_ylabel("z (µm)")
            ax2.set_title(f"|Ey|² at λ={wvl_center*1000:.0f} nm — XZ (y=0)")
            ax2.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "fig2_field_xz.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig2_field_xz.png")

    # XY field overlay
    xy_file = os.path.join(outdir, "field_xy.npy")
    if os.path.exists(xy_file):
        fxy = np.abs(np.load(xy_file).squeeze()) ** 2
        if fxy.ndim == 2:
            fig3, ax3 = plt.subplots(1, 1, figsize=(16, 8))
            draw_geometry_xy(ax3)
            im = ax3.imshow(fxy.T, origin="lower", extent=xy_extent,
                            cmap="inferno", aspect="auto", alpha=0.7,
                            vmax=np.percentile(fxy, 95))
            plt.colorbar(im, ax=ax3, label="|Ey|²", shrink=0.8)
            draw_pml(ax3, "xy", dpml)
            draw_monitor(ax3, "xy", mon_center, mon_size)
            ax3.set_xlim(x_lo, x_hi)
            ax3.set_ylim(y_lo, y_hi)
            ax3.set_xlabel("x (µm)")
            ax3.set_ylabel("y (µm)")
            ax3.set_title(f"|Ey|² at λ={wvl_center*1000:.0f} nm — XY (z=Si/2)")
            ax3.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "fig3_field_xy.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig3_field_xy.png")

    # Coupling efficiency
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    ax4a.plot(wl_nm, eff_data, "b-", linewidth=2)
    ax4a.fill_between(wl_nm, 0, eff_data, alpha=0.15, color="blue")
    ax4a.axhline(0.5, color="red", ls="--", lw=0.8, label="-3 dB")
    ax4a.set_xlabel("Wavelength (nm)")
    ax4a.set_ylabel("Relative Coupling Efficiency")
    ax4a.set_xlim(1500, 1600)
    ax4a.set_ylim(0, 1.05)
    ax4a.grid(True, alpha=0.3)
    ax4a.legend()
    ax4a.set_title("Linear")

    ax4b.plot(wl_nm, eff_db_data, "r-", linewidth=2)
    ax4b.axhline(-3, color="blue", ls="--", lw=0.8, label="-3 dB")
    ax4b.set_xlabel("Wavelength (nm)")
    ax4b.set_ylabel("Coupling (dB, relative)")
    ax4b.set_xlim(1500, 1600)
    ax4b.set_ylim(-15, 1)
    ax4b.grid(True, alpha=0.3)
    ax4b.legend()
    ax4b.set_title("dB")

    fig4.suptitle(
        f"ebeam_gc_te1550 — Peak: {wl_nm[pk]:.1f} nm, "
        f"3dB BW: {bw:.1f} nm — "
        f"Fiber {fiber_angle_deg:.0f}° from normal, {resolution} px/µm",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_coupling.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig4_coupling.png")


print(f"\nDone. All outputs in: {outdir}")

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "grating_coupler_output")
os.makedirs(outdir, exist_ok=True)

# ================================================================== #
# Visualization helpers (meep-style)
# ================================================================== #

# Domain bounds
x_lo = cell_center_x - sx / 2
x_hi = cell_center_x + sx / 2
y_lo = -sy / 2
y_hi = sy / 2
z_lo = cell_center_z - sz / 2
z_hi = cell_center_z + sz / 2


def draw_pml(ax, plane, dpml):
    """Draw PML regions as green semi-transparent rectangles."""
    kw = dict(linewidth=0.5, edgecolor="green", facecolor="green",
              alpha=0.15, linestyle="--")
    if plane == "xz":
        ax.add_patch(Rectangle((x_lo, z_lo), dpml, sz, **kw))            # left
        ax.add_patch(Rectangle((x_hi - dpml, z_lo), dpml, sz, **kw))     # right
        ax.add_patch(Rectangle((x_lo, z_lo), sx, dpml, **kw))            # bottom
        ax.add_patch(Rectangle((x_lo, z_hi - dpml), sx, dpml, **kw))     # top
    elif plane == "xy":
        ax.add_patch(Rectangle((x_lo, y_lo), dpml, sy, **kw))
        ax.add_patch(Rectangle((x_hi - dpml, y_lo), dpml, sy, **kw))
        ax.add_patch(Rectangle((x_lo, y_lo), sx, dpml, **kw))
        ax.add_patch(Rectangle((x_lo, y_hi - dpml), sx, dpml, **kw))


def draw_source(ax, plane, src):
    """Draw source plane and beam direction arrow."""
    cx, cy, cz = src.center.x, src.center.y, src.center.z
    sx_s, sy_s, sz_s = src.size.x, src.size.y, src.size.z
    kdir = src.beam_kdir

    if plane == "xz":
        # Source plane as orange line
        if sz_s == 0:  # horizontal plane
            ax.plot([cx - sx_s/2, cx + sx_s/2], [cz, cz],
                    color="orange", linewidth=2, label="Source")
        # Direction arrow
        arrow_len = 3.0
        ax.annotate("", xy=(cx + arrow_len * kdir.x, cz + arrow_len * kdir.z),
                    xytext=(cx, cz),
                    arrowprops=dict(arrowstyle="->", color="orange", lw=2))
        # Beam waist circle
        ax.plot(cx, cz, "o", color="orange", ms=6, zorder=5)
    elif plane == "xy":
        if sz_s == 0:
            circle = plt.Circle((cx, cy), fiber_waist,
                                fill=False, color="orange", linewidth=1.5,
                                linestyle="--", label="Beam waist")
            ax.add_patch(circle)
            ax.plot(cx, cy, "o", color="orange", ms=5, zorder=5)


def draw_monitor(ax, plane, center, size, label="Monitor"):
    """Draw flux monitor as blue dashed line with arrow."""
    cx, cy, cz = center.x, center.y, center.z
    sx_m, sy_m, sz_m = size.x, size.y, size.z

    if plane == "xz" and sx_m == 0:  # YZ plane monitor shown in XZ as vertical line
        z_lo_m = cz - sz_m / 2
        z_hi_m = cz + sz_m / 2
        ax.plot([cx, cx], [z_lo_m, z_hi_m],
                color="blue", linewidth=2, linestyle="--", label=label)
        ax.annotate("", xy=(cx + 1.5, cz), xytext=(cx, cz),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
    elif plane == "xy" and sx_m == 0:
        y_lo_m = cy - sy_m / 2
        y_hi_m = cy + sy_m / 2
        ax.plot([cx, cx], [y_lo_m, y_hi_m],
                color="blue", linewidth=2, linestyle="--", label=label)


def draw_geometry_xz(ax, y_cut=0):
    """Draw geometry cross-section at y=y_cut in XZ plane."""
    # BOX layer
    ax.axhspan(-box_thickness, 0, color="#4a90d9", alpha=0.2, label="SiO₂ (BOX)")
    # Si layer — draw individual GDS polygons that intersect y=y_cut
    for pg in si_polygons:
        ys = pg[:, 1]
        if ys.min() <= y_cut <= ys.max():
            xs = pg[:, 0]
            ax.fill_between([xs.min(), xs.max()], 0, si_thickness,
                            color="#e74c3c", alpha=0.4)
    # Waveguide extension
    ax.fill_between([gds_xmax, gds_xmax + wg_extension],
                    0, si_thickness, color="#e74c3c", alpha=0.4)


def draw_geometry_xy(ax, z_cut=0.11):
    """Draw geometry cross-section at z=z_cut in XY plane."""
    # Only draw Si polygons that span the z_cut
    if 0 <= z_cut <= si_thickness:
        for pg in si_polygons:
            patch = plt.Polygon(pg, closed=True, fc="#e74c3c", ec="darkred",
                                alpha=0.3, linewidth=0.3)
            ax.add_patch(patch)
        # Waveguide extension
        wg_rect = Rectangle(
            (gds_xmax, -wg_width / 2), wg_extension, wg_width,
            fc="#e74c3c", ec="darkred", alpha=0.3, linewidth=0.3)
        ax.add_patch(wg_rect)


# ================================================================== #
# Figure 0: Setup view — XZ
# ================================================================== #

print("\nGenerating setup figures...")

fig0, ax0 = plt.subplots(1, 1, figsize=(16, 5))
draw_geometry_xz(ax0)
draw_pml(ax0, "xz", dpml)
draw_source(ax0, "xz", sources[0])
draw_monitor(ax0, "xz", mon_center, mon_size)
ax0.set_xlim(x_lo, x_hi)
ax0.set_ylim(z_lo, z_hi)
ax0.set_xlabel("x (µm)")
ax0.set_ylabel("z (µm)")
ax0.set_title("Simulation Setup — XZ (y=0)")
ax0.set_aspect("equal")
ax0.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig0_setup_xz.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  fig0_setup_xz.png")

# ================================================================== #
# Figure 1: Setup view — XY
# ================================================================== #

fig1, ax1 = plt.subplots(1, 1, figsize=(16, 8))
draw_geometry_xy(ax1)
draw_pml(ax1, "xy", dpml)
draw_source(ax1, "xy", sources[0])
draw_monitor(ax1, "xy", mon_center, mon_size)
ax1.set_xlim(x_lo, x_hi)
ax1.set_ylim(y_lo, y_hi)
ax1.set_xlabel("x (µm)")
ax1.set_ylabel("y (µm)")
ax1.set_title("Simulation Setup — XY (z=Si/2)")
ax1.set_aspect("equal")
ax1.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig1_setup_xy.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  fig1_setup_xy.png")

# ================================================================== #
# Run simulation
# ================================================================== #

print("\nRunning 3D FDTD simulation...")
sim.run(
    until_after_sources=mp.stop_when_dft_decayed(tol=1e-4),
)
print("Simulation complete.\n")

# ================================================================== #
# Extract results
# ================================================================== #

freqs       = np.array(mp.get_flux_freqs(output_flux))
flux        = np.array(mp.get_fluxes(output_flux))
wavelengths = 1.0 / freqs

peak_flux = np.max(np.abs(flux))
eff       = np.abs(flux) / peak_flux if peak_flux > 0 else np.zeros_like(flux)
eff_db    = 10 * np.log10(np.clip(eff, 1e-10, None))

peak_idx = int(np.argmax(np.abs(flux)))
half_max = peak_flux / 2
in_band  = wavelengths[np.abs(flux) >= half_max]
bw_nm    = abs(in_band[-1] - in_band[0]) * 1000 if len(in_band) >= 2 else 0

print(f"  Peak flux:    {peak_flux:.4e}")
print(f"  Peak at:      {wavelengths[peak_idx]*1000:.1f} nm")
print(f"  3dB BW:       {bw_nm:.1f} nm")
print(f"  NaN in flux:  {np.any(np.isnan(flux))}")

# Extract DFT fields
field_xz = sim.get_dft_array(dft_xz, mp.Ey, 0)
field_xy = sim.get_dft_array(dft_xy, mp.Ey, 0)

# Save coupling data
data_file = os.path.join(outdir, "gc_te1550_results.csv")
np.savetxt(data_file,
           np.column_stack([wavelengths * 1000, flux, eff, eff_db]),
           header="wavelength_nm,flux,rel_eff,dB",
           delimiter=",", fmt="%.4f,%.6e,%.6f,%.2f")
print(f"  Data: {data_file}")

# ================================================================== #
# Figure 2: Field XZ with geometry contours
# ================================================================== #

if field_xz is not None and field_xz.size > 0:
    intensity_xz = np.abs(field_xz.squeeze()) ** 2
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 5))

    # Field heatmap
    xz_extent = [x_lo + dpml, x_hi - dpml, z_lo + dpml, z_hi - dpml]
    if intensity_xz.ndim == 2:
        im = ax2.imshow(intensity_xz.T, origin="lower", extent=xz_extent,
                        cmap="inferno", aspect="auto",
                        vmax=np.percentile(intensity_xz, 95))
        plt.colorbar(im, ax=ax2, label="|Ey|²", shrink=0.8)

    # Geometry contours
    draw_geometry_xz(ax2)
    draw_pml(ax2, "xz", dpml)
    draw_monitor(ax2, "xz", mon_center, mon_size)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(z_lo, z_hi)
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("z (µm)")
    ax2.set_title(f"|Ey|² at λ={wvl_center*1000:.0f} nm — XZ (y=0)")
    ax2.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_field_xz.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_field_xz.png")

# ================================================================== #
# Figure 3: Field XY with geometry contours
# ================================================================== #

if field_xy is not None and field_xy.size > 0:
    intensity_xy = np.abs(field_xy.squeeze()) ** 2
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 8))

    xy_extent = [x_lo + dpml, x_hi - dpml, y_lo + dpml, y_hi - dpml]
    if intensity_xy.ndim == 2:
        im = ax3.imshow(intensity_xy.T, origin="lower", extent=xy_extent,
                        cmap="inferno", aspect="auto",
                        vmax=np.percentile(intensity_xy, 95))
        plt.colorbar(im, ax=ax3, label="|Ey|²", shrink=0.8)

    draw_geometry_xy(ax3)
    draw_pml(ax3, "xy", dpml)
    draw_monitor(ax3, "xy", mon_center, mon_size)

    ax3.set_xlim(x_lo, x_hi)
    ax3.set_ylim(y_lo, y_hi)
    ax3.set_xlabel("x (µm)")
    ax3.set_ylabel("y (µm)")
    ax3.set_title(f"|Ey|² at λ={wvl_center*1000:.0f} nm — XY (z=Si/2)")
    ax3.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_field_xy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_field_xy.png")

# ================================================================== #
# Figure 4: Coupling efficiency
# ================================================================== #

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

wl_nm = wavelengths * 1000

ax4a.plot(wl_nm, eff, "b-", linewidth=2)
ax4a.fill_between(wl_nm, 0, eff, alpha=0.15, color="blue")
ax4a.axhline(0.5, color="red", ls="--", lw=0.8, label="-3 dB")
ax4a.set_xlabel("Wavelength (nm)")
ax4a.set_ylabel("Relative Coupling Efficiency")
ax4a.set_xlim(1500, 1600)
ax4a.set_ylim(0, 1.05)
ax4a.grid(True, alpha=0.3)
ax4a.legend()
ax4a.set_title("Linear")

ax4b.plot(wl_nm, eff_db, "r-", linewidth=2)
ax4b.axhline(-3, color="blue", ls="--", lw=0.8, label="-3 dB")
ax4b.set_xlabel("Wavelength (nm)")
ax4b.set_ylabel("Coupling (dB, relative)")
ax4b.set_xlim(1500, 1600)
ax4b.set_ylim(-15, 1)
ax4b.grid(True, alpha=0.3)
ax4b.legend()
ax4b.set_title("dB")

fig4.suptitle(
    f"ebeam_gc_te1550 — Peak: {wl_nm[peak_idx]:.1f} nm, "
    f"3dB BW: {bw_nm:.1f} nm — "
    f"Fiber {fiber_angle_deg:.0f}° from normal, {resolution} px/µm",
    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig4_coupling.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  fig4_coupling.png")

print(f"\nDone. All outputs in: {outdir}")
