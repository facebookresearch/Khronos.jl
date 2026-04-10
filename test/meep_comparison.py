#!/usr/bin/env python3
"""
Compare meep's adj_src_scale with Khronos's to identify discrepancies.
Uses the same parameters as test_adjoint_1pixel.jl.
"""
import numpy as np
import meep as mp
import meep.adjoint as mpa

# Parameters matching test_adjoint_1pixel.jl
fcen = 0.5
fwidth = 0.15
resolution = 20
pml = 1.0
pad = 0.5
design_Lx = 1.0
cell_y = 4.0
cell_x = design_Lx + 2 * pad + 2 * pml
eps_lo = 1.0
eps_hi = 4.0

cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(pml)]

geometry = [mp.Block(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(cell_x + 1, cell_y + 1, 0),
    material=mp.Medium(epsilon=eps_lo),
)]

sources = [mp.Source(
    src=mp.GaussianSource(fcen, fwidth=fwidth),
    component=mp.Ez,
    center=mp.Vector3(-(design_Lx / 2 + pad), 0, 0),
    size=mp.Vector3(0, 0, 0),
    amplitude=1.0,
)]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# Create a simple FourierFields objective
output_mon_x = design_Lx / 2 + pad / 2
mon_vol = mp.Volume(center=mp.Vector3(output_mon_x, 0, 0), size=mp.Vector3(0, 0, 0))
obj = mpa.FourierFields(sim, mon_vol, mp.Ez)

# Register and run
obj.register_monitors([fcen])
sim.run(until=30.0)

# Evaluate
E_obj = obj()
print(f"\n=== MEEP adj_src_scale comparison ===")
print(f"E_obj shape: {E_obj.shape}")
print(f"E_obj[0]: {E_obj[0]}")
print(f"|E_obj[0]|^2: {np.abs(E_obj[0])**2}")

# Compute adj_src_scale
scale = obj._adj_src_scale(include_resolution=False)
scale_dV = obj._adj_src_scale(include_resolution=True)
print(f"\nAdj source scale:")
print(f"  scale (no dV):  {scale[0]}")
print(f"  |scale (no dV)|: {np.abs(scale[0])}")
print(f"  scale (with dV): {scale_dV[0]}")
print(f"  |scale (with dV)|: {np.abs(scale_dV[0])}")
print(f"  using_real_fields: {sim.using_real_fields()}")

# Compute individual components
dt = sim.fields.dt
T = sim.meep_time()
print(f"\n  dt = {dt}")
print(f"  T_sim = {T}")
print(f"  n_steps = {int(T/dt)}")

# iomega
omega = 2 * np.pi * fcen
iomega = (1.0 - np.exp(-1j * omega * dt)) / dt
print(f"  iomega = {iomega}")
print(f"  |iomega| = {np.abs(iomega)}")

# fwd_dtft
src = obj._create_time_profile()
y = np.array([src.swigobj.current(t, dt) for t in np.arange(0, T, dt)])
fwd_dtft = np.sum(dt / np.sqrt(2*np.pi) * np.exp(1j * 2 * np.pi * fcen * np.arange(len(y)) * dt) * y)
print(f"  fwd_dtft = {fwd_dtft}")
print(f"  |fwd_dtft| = {np.abs(fwd_dtft)}")

# Also compute DTFT of the dipole (not current)
y_dipole = np.array([src.swigobj.current(t, dt) for t in np.arange(0, T, dt)])
# Actually, meep's current() = d(dipole)/dt via finite difference
# Let's compute dipole directly
y_dipole_vals = np.array([complex(src.swigobj.dipole(t)) for t in np.arange(0, T, dt)])
dtft_dipole = np.sum(dt / np.sqrt(2*np.pi) * np.exp(1j * 2 * np.pi * fcen * np.arange(len(y_dipole_vals)) * dt) * y_dipole_vals)
print(f"  dtft(dipole) = {dtft_dipole}")
print(f"  |dtft(dipole)| = {np.abs(dtft_dipole)}")

# adj_src_phase
src_center_dtft = np.sum(dt / np.sqrt(2*np.pi) * np.exp(1j * 2 * np.pi * fcen * np.arange(len(y)) * dt) * y)
fwidth_scale = np.exp(-2j * np.pi * 5 / 0.1)
adj_src_phase = np.exp(1j * np.angle(src_center_dtft)) * fwidth_scale
print(f"  adj_src_phase = {adj_src_phase}")
print(f"  |adj_src_phase| = {np.abs(adj_src_phase)}")
print(f"  fwidth_scale = {fwidth_scale}")

# Manual scale computation
ndims = sim._infer_dimensions(sim.k_point)
dV = 1.0 / resolution**ndims
scale_manual = dV * iomega / fwd_dtft / adj_src_phase
if sim.using_real_fields():
    scale_manual *= 2
print(f"\n  Manual scale (matching meep): {scale_manual}")
print(f"  |manual scale|: {np.abs(scale_manual)}")
print(f"  Meep's scale:                {scale_dV[0]}")
print(f"  |meep scale|:  {np.abs(scale_dV[0])}")
print(f"  Match: {np.isclose(scale_manual, scale_dV[0])}")

# Key: What does meep's current() vs Khronos's eval_time_source look like?
print(f"\nSource waveform comparison:")
print(f"  meep current(0) = {y[0]}")
print(f"  meep current(dt) = {y[1]}")
print(f"  meep dipole(0) = {y_dipole_vals[0]}")
print(f"  meep dipole(dt) = {y_dipole_vals[1]}")
print(f"  (dipole(dt) - dipole(0))/dt = {(y_dipole_vals[1] - y_dipole_vals[0])/dt}")
print(f"  current(0) should match the above: {np.isclose(y[0], (y_dipole_vals[1] - y_dipole_vals[0])/dt)}")
