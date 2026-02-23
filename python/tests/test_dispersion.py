#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Test: dielectric slab transmission using the tidy3d-compatible API.

Ports examples/dielectric_slab.jl to the Python wrapper. Validates that
the wrapper correctly translates parameters and produces correct physics.
"""

import khronos.tidy3d as td
import khronos.web as web
import numpy as np


def fresnel_slab_transmission(freq_hz, n_slab, thickness_m):
    """Analytical Fresnel transmission through a dielectric slab."""
    r12 = (1.0 - n_slab) / (1.0 + n_slab)
    t12 = 2.0 / (1.0 + n_slab)
    r21 = -r12
    t21 = 2.0 * n_slab / (1.0 + n_slab)
    k = 2 * np.pi * freq_hz / td.C_0
    phase = n_slab * k * thickness_m
    T = abs(t12 * t21 * np.exp(1j * phase) / (1.0 + r12 * r21 * np.exp(2j * phase))) ** 2
    return T


def main():
    # Physical parameters
    n_slab = 2.0
    eps_slab = n_slab ** 2
    slab_thickness = 0.5e-6  # meters

    # Frequency range
    lda_center = 1.0e-6  # 1 μm
    freq_center = td.C_0 / lda_center
    freq_min = td.C_0 / 1.5e-6
    freq_max = td.C_0 / 0.6e-6
    fwidth = 0.5 * (freq_max - freq_min)

    n_freqs = 51
    monitor_freqs = np.linspace(freq_min, freq_max, n_freqs).tolist()

    # Domain
    buffer = 1.5e-6
    cell_xy = 0.1e-6

    source_z = -slab_thickness / 2 - buffer / 2
    monitor_z = slab_thickness / 2 + buffer / 2

    domain_z = slab_thickness + 2 * buffer

    # --- Reference simulation (empty) ---
    sim_ref = td.Simulation(
        size=(cell_xy, cell_xy, domain_z),
        center=(0, 0, 0),
        sources=[
            td.PlaneWave(
                source_time=td.GaussianPulse(freq0=freq_center, fwidth=fwidth),
                center=(0, 0, source_z),
                size=(td.inf, td.inf, 0),
                direction="+",
            ),
        ],
        monitors=[
            td.FluxMonitor(
                center=(0, 0, monitor_z),
                size=(td.inf, td.inf, 0),
                freqs=monitor_freqs,
                name="flux",
            ),
        ],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=40),
        run_time=100 / freq_center,
    )

    print("Running reference simulation...")
    data_ref = web.run(sim_ref, task_name="reference")
    flux_ref = data_ref["flux"].flux

    # --- Slab simulation ---
    sim_slab = td.Simulation(
        size=(cell_xy, cell_xy, domain_z),
        center=(0, 0, 0),
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(cell_xy * 10, cell_xy * 10, slab_thickness)),
                medium=td.Medium(permittivity=eps_slab),
            ),
        ],
        sources=[
            td.PlaneWave(
                source_time=td.GaussianPulse(freq0=freq_center, fwidth=fwidth),
                center=(0, 0, source_z),
                size=(td.inf, td.inf, 0),
                direction="+",
            ),
        ],
        monitors=[
            td.FluxMonitor(
                center=(0, 0, monitor_z),
                size=(td.inf, td.inf, 0),
                freqs=monitor_freqs,
                name="flux",
            ),
        ],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=40),
        run_time=100 / freq_center,
    )

    print("Running slab simulation...")
    data_slab = web.run(sim_slab, task_name="slab")
    flux_slab = data_slab["flux"].flux

    # Compute transmission
    T_sim = flux_slab / flux_ref

    # Analytical reference
    T_analytical = np.array([
        fresnel_slab_transmission(f, n_slab, slab_thickness) for f in monitor_freqs
    ])

    # Compare
    wavelengths = td.C_0 / np.array(monitor_freqs) * 1e6  # μm
    print("\n" + "=" * 60)
    print("RESULTS: Transmission comparison")
    print("=" * 60)

    max_error = 0.0
    for i in range(0, n_freqs, 10):
        err = abs(T_sim[i] - T_analytical[i])
        max_error = max(max_error, err)
        print(f"  λ = {wavelengths[i]:.4f} μm: T_sim = {T_sim[i]:.4f}, "
              f"T_analytical = {T_analytical[i]:.4f}, error = {err:.6f}")

    max_error = max(abs(T_sim[i] - T_analytical[i]) for i in range(n_freqs))
    print(f"\nMax absolute error: {max_error:.6f}")
    if max_error < 0.08:
        print("✓ PASS: Simulation matches Fresnel theory within 8%")
    else:
        print("✗ FAIL: Error exceeds 8% threshold")

    print("=" * 60)


if __name__ == "__main__":
    main()
