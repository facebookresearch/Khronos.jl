# Khronos.jl Example Coverage Plan

This document catalogs every example and tutorial across the four reference
FDTD solvers (Khronos.jl, Meep, Tidy3D, fdtdx), identifies coverage gaps,
and specifies the examples Khronos should implement to maximize testing of
feature parity, correctness, performance, and coverage.

---

## Current Khronos.jl Examples

Khronos currently has **6 examples**, **3 benchmarks**, and **8 test files**.

| # | File | Physics | Dim | Sources | Monitors | Materials | Validation |
|---|------|---------|-----|---------|----------|-----------|------------|
| 1 | `examples/dipole.jl` | Point dipole radiation | 3D | UniformSource (CW) | None | Vacuum | Visual only |
| 2 | `examples/planewave.jl` | Planewave propagation | 3D | PlaneWaveSource (CW) | None | Vacuum | Visual only |
| 3 | `examples/sphere.jl` | Mie scattering | 3D | PlaneWaveSource (CW) | None | Lossy dielectric (Ball) | Visual only |
| 4 | `examples/waveguide.jl` | Waveguide mode excitation | 3D | ModeSource | None | Si + SiO₂ (Cuboid) | Visual only |
| 5 | `examples/gaussian_beam.jl` | Oblique Gaussian beam | 3D | GaussianBeamSource (pulse) | DFTMonitor | Vacuum | Visual only |
| 6 | `examples/periodic_slab.jl` | Fabry-Perot transmission | 3D | PlaneWaveSource (pulse) | DFTMonitor (100 freqs) | Dielectric slab | **Analytic comparison** |

**Key gaps in current examples:**
- All examples are 3D (no 2D examples despite `TwoD_TE`/`TwoD_TM` support)
- Only 2 of 6 use monitors
- Only 1 has quantitative validation (periodic_slab vs. Fabry-Perot theory)
- No examples use: periodic BCs, Bloch BCs, PEC/PMC, flux monitors, mode
  monitors, near-to-far, dispersive materials, nonlinear materials,
  symmetry, multi-GPU, cylindrical coordinates, or adjoint optimization
- No performance benchmarks with cross-solver comparison
- No 2-run normalization workflow (incident + scattered)

---

## Reference Solver Example Inventory

### Meep — 33 Examples

| Category | Count | Examples |
|----------|-------|---------|
| Waveguides / Basics | 3 | straight-waveguide, bent-waveguide, bend-flux |
| Resonators / Cavities | 5 | ring, holey-wvg-cavity, holey-wvg-bands, metal-cavity-ldos, solve-cw |
| Gratings / Diffraction | 6 | binary_grating, binary_grating_n2f, binary_grating_oblique, binary_grating_phasemap, polarization_grating, finite_grating |
| Scattering | 3 | mie_scattering, differential_cross_section, cylinder_cross_section |
| Reflectance | 3 | refl-angular, refl-quartz, refl_angular_bfast |
| Sources / Planewaves | 2 | oblique-planewave, oblique-source |
| Mode Decomposition | 1 | mode-decomposition (taper) |
| Near-to-Far / Radiation | 4 | antenna-radiation, cavity-farfield, metasurface_lens, zone_plate |
| Nonlinear Optics | 1 | 3rd-harm-1d |
| Optical Forces | 1 | parallel-wvgs-force |
| Perturbation Theory | 1 | perturbation_theory |
| Absorbed Power | 1 | absorbed_power_density |
| Stochastic / LED | 1 | stochastic_emitter |
| GDSII / Circuits | 1 | coupler |

### Tidy3D — 199 Examples (key categories)

| Category | Count | Notable Examples |
|----------|-------|-----------------|
| Core Tutorials | 21 | StartHere, BoundaryConditions, TFSF, Symmetry, AutoGrid |
| Materials | 9 | Dispersion, Fitting, FullyAnisotropic, Gyrotropic, TimeModulation |
| Geometry / Import | 8 | GDSImport, STLImport, TriangleMesh, PolySlab |
| Mode Solving | 8 | ModeSolver, ModeOverlap, BatchModeSolver |
| Waveguide Components | 21 | RingResonator, DirectionalCoupler, YJunction, MMI1x4, EdgeCoupler |
| Gratings | 5 | GratingCoupler, GratingEfficiency, BraggGratings |
| Photonic Crystals | 8 | Bandstructure, NanobeamCavity, OptimizedL3 |
| Metasurfaces | 11 | Metalens, VortexMetasurface, DielectricMetasurfaceAbsorber |
| Plasmonics | 7 | PlasmonicNanoparticle, YagiUdaNanoantenna |
| Scattering / Far-Field | 6 | Near2FarSphereRCS, PECSphereRCS, MultipoleExpansion |
| Resonators | 5 | CavityFOM, ResonanceFinder, HighQSi |
| Nonlinear | 2 | KerrSidebands, SiWaveguideTPA |
| RF / Microwave | 21 | PatchAntenna, LumpedElements, CoupledLineBandpassFilter |
| Inverse Design (autograd) | 26 | Autograd series (0–26), covering bends, metalenses, WDM, etc. |
| Classical Optimization | 8 | CMA-ES, Bayesian, Genetic Algorithm, PSO, DBS |
| Multi-Physics | 7 | HeatSolver, ChargeSolver, ThermallyTunedRingResonator |

### fdtdx — 5 Examples

| # | File | Physics | Key Feature |
|---|------|---------|-------------|
| 1 | `optimize_ceviche_corner.py` | Waveguide bend inverse design | Full differentiable optimization loop |
| 2 | `simulate_gaussian_source.py` | Gaussian beam + time reversal | Forward + backward FDTD verification |
| 3 | `simulate_gaussian_source_anisotropic.py` | Anisotropic slab interaction | Per-axis permittivity/permeability |
| 4 | `width_sweep_analysis.py` | Waveguide loss parametric sweep | Mode solver vs. FDTD validation |
| 5 | `check_waveguide_modes.py` | SOI waveguide mode check | Mode injection verification |

---

## Proposed Khronos Examples

Examples are organized by category. Each entry specifies:
- **Priority** (P0–P3) aligned with the roadmap
- **Validates**: what correctness/feature claim the example tests
- **Requires**: roadmap features needed before this example can work
- **Reference**: which solver examples it mirrors
- **Status**: `CAN BUILD NOW` vs. `BLOCKED ON §X.Y`

### Category 1: Foundational Validation

These examples validate core FDTD correctness against analytical solutions.
They form the regression test suite backbone.

---

#### E1. Dielectric Slab Transmission (Fabry-Perot) ✓ EXISTS

**File:** `examples/periodic_slab.jl`
**Priority:** P0
**Validates:** Broadband pulse propagation, DFT monitors, power extraction,
two-run normalization workflow
**Reference:** Meep `bend-flux` (normalization pattern), Tidy3D `StartHere`
**Status:** `EXISTS` — the only quantitatively validated example today.

**Improvements needed:**
- Add 2D version (`TwoD_TM`) to validate 2D code paths
- Add explicit flux monitor once §1.3 + §3.3 are implemented
- Add error metric (max |T_khronos - T_analytic|) printed to stdout

---

#### E2. Fresnel Reflectance (Normal Incidence)

**Priority:** P0
**Validates:** Material boundaries, field continuity at interfaces, DFT
monitor accuracy
**Reference:** Meep `refl-quartz` (normal-incidence reflectance vs. Fresnel)
**Computes:** R(λ) for a planar air → n=3.5 dielectric interface; compare
to Fresnel formula R = |(n-1)/(n+1)|²

**Setup:**
- 1D-equivalent (thin slab in 3D with tiny transverse extent)
- GaussianPulseSource, broadband (0.4–1.0 μm range)
- Two DFT flux monitors: reflected (behind source) and transmitted
- Two-run normalization (empty → with slab)

**Status:** `CAN BUILD NOW`
**New features exercised:** Two-run normalization workflow, reflected flux

---

#### E3. Angular Fresnel Reflectance (Oblique Incidence)

**Priority:** P0
**Validates:** Bloch-periodic boundaries, oblique plane wave injection,
angle-dependent reflectance
**Reference:** Meep `refl-angular` (Brewster's angle validation)
**Computes:** R(λ, θ) for P-polarized light at an air → n=3.5 interface;
identify Brewster's angle at arctan(3.5) ≈ 74°

**Setup:**
- Thin cell in y-z, Bloch-periodic in y
- PlaneWaveSource at angle θ via k_point
- DFT flux monitors for reflected/transmitted power
- Sweep over θ = 0°, 10°, 20°, ..., 80°

**Status:** `BLOCKED ON §1.1 + §3.1` (periodic/Bloch BCs)
**Validates roadmap:** §1.1 Boundary Spec, §3.1 Bloch Kernels

---

#### E4. Mie Scattering Cross Section

**Priority:** P0
**Validates:** Spherical geometry (Ball), flux monitors on a closed
surface, scattering cross-section extraction, two-run normalization
**Reference:** Meep `mie_scattering`, Tidy3D `Near2FarSphereRCS`
**Computes:** σ_scat(λ) / πr² for a dielectric sphere (n=2, r=1μm);
compare to analytical Mie series (PyMieScatt or Julia equivalent)

**Setup:**
- 3D, PML on all faces
- PlaneWaveSource (GaussianPulse, broadband)
- 6 DFT flux monitors forming a closed box around sphere
- Normalization run without sphere
- Compare to Mie series solution

**Status:** `BLOCKED ON §1.3 + §3.3` (flux monitors)
**Validates roadmap:** §1.3 FluxMonitor, §3.3 Flux Kernels

---

#### E5. 2D Waveguide Bend Transmission

**Priority:** P0
**Validates:** 2D simulation (`TwoD_TM`), curved geometry, broadband
transmission, flux normalization
**Reference:** Meep `bent-waveguide` / `bend-flux`
**Computes:** T(λ) through a 90° dielectric waveguide bend (ε=12);
compare straight-waveguide normalization vs. bend

**Setup:**
- 2D TM, PML on all faces
- Block geometry: straight waveguide + 90° bend section
- GaussianPulseSource
- DFT flux monitors at input and output ports

**Status:** `BLOCKED ON §1.3 + §3.3` (flux monitors)
**Validates roadmap:** 2D code paths, §1.3 FluxMonitor

---

### Category 2: Resonators and Spectral Analysis

---

#### E6. Ring Resonator (Q-factor Extraction)

**Priority:** P1
**Validates:** Curved geometry, resonance detection, quality factor
computation, convergence-based run termination
**Reference:** Meep `ring`, Tidy3D `RingResonator`
**Computes:** Resonant frequencies ωₙ and quality factors Qₙ of a
dielectric ring resonator

**Setup:**
- 2D, PML, Cylinder geometry (ring = outer cylinder - inner cylinder)
- GaussianPulseSource (broadband, inside ring)
- Time monitor at a point inside the ring
- Post-process: FFT or Harminv-style analysis to extract ω and Q

**Status:** `BLOCKED ON §3.12` (time monitor kernel) + resonance extraction
**Validates roadmap:** §3.12 TimeMonitor, §3.19 Harminv (optional)

---

#### E7. Photonic Crystal Waveguide Bands

**Priority:** P1
**Validates:** Periodic boundaries, Bloch k-sweep, band structure
extraction
**Reference:** Meep `holey-wvg-bands`
**Computes:** ω(k) band diagram of a 1D photonic crystal waveguide
(periodic holes in a dielectric strip)

**Setup:**
- 2D, periodic in x (propagation), PML in y
- Sweep k_x over the irreducible Brillouin zone
- At each k, run broadband pulse and extract resonances
- Plot ω vs. k band diagram

**Status:** `BLOCKED ON §1.1 + §3.1` (periodic/Bloch BCs)
**Validates roadmap:** §1.1 Boundary Spec, §3.1 Periodic Kernels

---

#### E8. Photonic Crystal Cavity Q

**Priority:** P2
**Validates:** Defect cavity in a periodic lattice, high-Q resonance
extraction
**Reference:** Meep `holey-wvg-cavity`, Tidy3D `NanobeamCavity`/`OptimizedL3`
**Computes:** Resonant frequency and Q-factor of a defect cavity in a 1D
photonic crystal waveguide; Q vs. number of mirror periods

**Status:** `BLOCKED ON §1.1 + §3.1` (periodic BCs) + §3.12 (time monitor)

---

### Category 3: Waveguide Mode Analysis

---

#### E9. Mode Source Validation (Existing Waveguide Example Enhancement)

**Priority:** P0
**Validates:** ModeSource accuracy, mode profile fidelity, guided power
**Reference:** fdtdx `check_waveguide_modes`, `width_sweep_analysis`
**Computes:** Inject fundamental mode into Si waveguide; measure mode
overlap at output to verify >99% coupling efficiency

**Setup:**
- Enhance existing `waveguide.jl` with:
  - DFT flux monitor at input and output planes
  - Mode overlap computation (once §1.4 + §3.4 are implemented)
  - Print coupling efficiency and insertion loss (dB)

**Status:** Partial `CAN BUILD NOW` (visual check); full validation
`BLOCKED ON §1.3 + §3.3` (flux) and §1.4 + §3.4 (mode overlap)

---

#### E10. Waveguide Taper S-Parameters

**Priority:** P1
**Validates:** Mode decomposition, S-parameter extraction, geometry
(tapered prism or interpolated cuboids)
**Reference:** Meep `mode-decomposition` (linear taper reflectance)
**Computes:** S₁₁ (reflectance) and S₂₁ (transmittance) of fundamental
mode through a linear waveguide taper; verify R scales as 1/L²

**Setup:**
- 2D or 3D, PML
- Waveguide width taper from w₁ to w₂ over length L
- ModeSource at input, mode monitors at input (reflected) and output
- Sweep taper length L

**Status:** `BLOCKED ON §1.4 + §3.4` (mode decomposition)

---

#### E11. Directional Coupler

**Priority:** P2
**Validates:** Evanescent coupling, two-waveguide geometry, power
splitting ratio
**Reference:** Meep `coupler`, Tidy3D `DirectionalCoupler`
**Computes:** Coupling ratio (bar/cross port power) vs. coupling length
for two parallel Si waveguides

**Status:** `BLOCKED ON §1.3 + §3.3` (flux monitors)

---

### Category 4: Gratings and Diffraction

---

#### E12. Binary Grating Diffraction Orders

**Priority:** P1
**Validates:** Periodic boundaries, diffraction order decomposition,
subwavelength optics
**Reference:** Meep `binary_grating`, Tidy3D `GratingEfficiency`
**Computes:** Transmittance into diffraction orders (0th through ±5th)
for a periodic binary phase grating at normal incidence

**Setup:**
- 2D, periodic in x, PML in y
- Block geometry: periodic grating ridges
- PlaneWaveSource (broadband)
- Flux monitors above and below grating

**Status:** `BLOCKED ON §1.1 + §3.1` (periodic BCs) + §1.9 + §3.16
(diffraction monitor)

---

#### E13. Grating Coupler Efficiency

**Priority:** P2
**Validates:** 3D grating simulation, fiber coupling, angle-dependent
efficiency
**Reference:** Tidy3D `GratingCoupler`, `FocusedApodGC`
**Computes:** Coupling efficiency from free-space Gaussian beam to
waveguide mode via grating coupler

**Status:** `BLOCKED ON §1.1` (periodic BCs) + §1.4 (mode monitor)

---

### Category 5: Scattering and Far-Field

---

#### E14. Antenna Radiation Pattern (2D Dipole Far-Field)

**Priority:** P1
**Validates:** Near-to-far field transformation, radiation pattern
computation
**Reference:** Meep `antenna-radiation`, Tidy3D `AntennaCharacteristics`
**Computes:** Far-field radiation pattern E(θ) of a point dipole in
vacuum; verify isotropic pattern and total radiated power via far-field
flux integration

**Setup:**
- 2D, PML
- Point source (CW or pulse)
- Near-field DFT surface (box around source)
- Near-to-far transformation to observation circle at r >> λ
- Compare total far-field power to near-field flux (Poynting theorem check)

**Status:** `BLOCKED ON §1.5 + §3.10` (near-to-far field)

---

#### E15. Differential Scattering Cross Section

**Priority:** P2
**Validates:** 3D scattering, near-to-far field, angular distribution
**Reference:** Meep `differential_cross_section`, Tidy3D `Near2FarSphereRCS`
**Computes:** dσ/dΩ(θ) for a dielectric sphere; integrate to get total σ
and compare to Mie theory

**Status:** `BLOCKED ON §1.5 + §3.10` (near-to-far)

---

### Category 6: Dispersive and Advanced Materials

---

#### E16. Dispersive Slab Reflectance (Lorentzian Material)

**Priority:** P0
**Validates:** Dispersive material implementation (ADE), frequency-dependent
reflectance
**Reference:** Meep `refl-quartz` (Sellmeier/Lorentzian fused quartz)
**Computes:** R(λ) for a slab of Lorentzian material; compare to
analytic Fresnel with ε(ω) = ε_∞ + σ/(ω₀² - ω² - iγω)

**Setup:**
- 1D-equivalent geometry
- Material with one Lorentzian pole
- Broadband GaussianPulseSource
- DFT flux monitors for R and T
- Analytic comparison using Fresnel + Lorentzian ε(ω)

**Status:** `BLOCKED ON §1.2 + §3.2` (dispersive materials) + §1.3 + §3.3
(flux monitors)
**Validates roadmap:** §1.2 LorentzianPole, §3.2 ADE Kernels

---

#### E17. Drude Metal Reflectance

**Priority:** P1
**Validates:** Drude pole implementation, metallic response, plasma
frequency behavior
**Reference:** Meep material library metals, Tidy3D `Dispersion`
**Computes:** R(λ) for a thick Drude metal slab (e.g., simplified gold);
verify R ≈ 1 below plasma frequency, transition to transparent above

**Status:** `BLOCKED ON §1.2 + §3.2` (dispersive materials)

---

#### E18. Nonlinear Third-Harmonic Generation

**Priority:** P2
**Validates:** χ³ nonlinear material, third-harmonic spectral peak,
power scaling
**Reference:** Meep `3rd-harm-1d`, Tidy3D `KerrSidebands`
**Computes:** Transmitted power spectrum showing 3ω peak; verify P(3ω) ∝
(χ³)² in the weak-nonlinearity regime

**Status:** `BLOCKED ON §1.6 + §3.13` (nonlinear materials)

---

#### E19. Anisotropic Material Slab

**Priority:** P2
**Validates:** Per-axis permittivity, polarization-dependent transmission
**Reference:** fdtdx `simulate_gaussian_source_anisotropic`
**Computes:** Transmission through a slab with ε = (2, 1, 2); verify
polarization-dependent wavelength shortening

**Setup:**
- 3D, PML
- Cuboid slab with `Material(epsilonx=2, epsilony=1, epsilonz=2)`
- PlaneWaveSource (two runs: x-polarized and y-polarized)
- DFT monitors to compare T for each polarization

**Status:** `CAN BUILD NOW` (anisotropic ε already supported) — needs
flux monitors for quantitative validation, otherwise visual check works

---

### Category 7: Boundary Conditions

---

#### E20. PEC and PMC Cavity Modes

**Priority:** P2
**Validates:** PEC and PMC boundary conditions, cavity resonance formula
**Reference:** Tidy3D `BoundaryConditions`
**Computes:** Resonant frequencies of a rectangular PEC cavity; compare
to analytical f_mnp = c/(2π) √((mπ/a)² + (nπ/b)² + (pπ/d)²)

**Status:** `BLOCKED ON §1.1 + §3.9` (PEC/PMC boundaries)

---

#### E21. Periodic Boundary Verification

**Priority:** P0
**Validates:** Periodic BC correctness, field continuity across boundary
**Reference:** Meep `oblique-planewave`, Tidy3D `BoundaryConditions`
**Computes:** Plane wave propagation in periodic cell; verify field values
at opposite boundaries are identical (periodic) or phase-shifted (Bloch)

**Status:** `BLOCKED ON §1.1 + §3.1` (periodic BCs)

---

### Category 8: Symmetry and Performance

---

#### E22. Mirror Symmetry Acceleration

**Priority:** P2
**Validates:** Symmetry exploitation, memory/time reduction, result
equivalence
**Reference:** Meep `bend-flux` (mirror symmetry), Tidy3D `Symmetry`
**Computes:** Run waveguide bend with and without mirror symmetry; verify
identical flux results with 2× memory reduction

**Status:** `BLOCKED ON §1.7 + §2.4` (symmetry)

---

#### E23. Non-Uniform Grid Convergence

**Priority:** P1
**Validates:** Non-uniform grid accuracy, convergence rate, memory savings
**Reference:** Tidy3D `AutoGrid` / `CustomGrid`
**Computes:** Slab transmission with uniform vs. non-uniform grid; verify
identical results with fewer voxels; measure memory reduction

**Status:** `BLOCKED ON §2.5 + §3.6` (non-uniform grid)

---

### Category 9: Source Types

---

#### E24. Gaussian Beam Waist Verification ✓ PARTIAL

**File:** `examples/gaussian_beam.jl` (exists but needs quantitative check)
**Priority:** P1
**Validates:** GaussianBeamSource profile accuracy, beam waist, divergence
**Reference:** Tidy3D `AdvancedGaussianSources`
**Computes:** Measure beam waist w(z) at several planes along propagation;
compare to analytical Gaussian beam formula
w(z) = w₀ √(1 + (z/z_R)²)

**Improvements to existing example:**
- Add multiple DFT monitors at different z positions
- Fit Gaussian to transverse field profile at each z
- Compare extracted w(z) to theory

**Status:** Enhancement of existing example. `CAN BUILD NOW` (partial)

---

#### E25. CW Source Steady-State Verification

**Priority:** P1
**Validates:** ContinuousWaveSource reaches true steady state, field
amplitude matches expected value
**Reference:** Meep `solve-cw`
**Computes:** CW point source in vacuum; verify E-field amplitude at
distance r matches analytical Green's function |E| ∝ 1/r (3D) or
1/√r (2D)

**Status:** `CAN BUILD NOW`

---

### Category 10: Monitors and Post-Processing

---

#### E26. Time Monitor Validation

**Priority:** P2
**Validates:** Time monitor kernel, temporal field recording, FFT
consistency with DFT monitor
**Reference:** General FDTD practice
**Computes:** Record E(t) at a point using TimeMonitor; take FFT and
compare to DFTMonitor at same location — spectra should match

**Status:** `BLOCKED ON §3.12` (time monitor kernel)

---

#### E27. LDOS (Local Density of States)

**Priority:** P2
**Validates:** LDOS computation, Purcell enhancement near structures
**Reference:** Meep `metal-cavity-ldos`
**Computes:** LDOS at the center of a 2D metal cavity; compare to
analytical Purcell factor Q/(4π²V_eff)

**Status:** `BLOCKED ON §1.10 + §3.17` (LDOS) + §1.2 + §3.2 (dispersive
materials for metal)

---

### Category 11: Scaling, Speed, and Performance

This category is critical for demonstrating Khronos's primary value
proposition: GPU-accelerated FDTD that scales. The benchmarks here should
be run regularly and results tracked over time.

---

#### E28. Single-GPU Throughput vs. Problem Size

**Priority:** P0
**Validates:** GPU kernel efficiency, memory bandwidth utilization, and
the regime where GPU acceleration pays off vs. CPU
**Computes:** Cells/second as a function of grid size N³, from N=32
(tiny, launch-overhead-dominated) to N=2048 (memory-limit)

**Setup:**
- 3D, PML, point dipole in vacuum (minimal physics — isolates raw
  FDTD kernel throughput)
- Sweep grid sizes: 32³, 64³, 128³, 256³, 512³, 768³, 1024³, 1536³, 2048³
- Run 100 timesteps (warm, after JIT compilation)
- Report per size: cells/second, wall time per step, GPU memory used,
  GPU occupancy (if available)
- Plot: cells/s vs. N³ (should plateau at memory bandwidth limit)

**Expected behavior:**
- Small grids (< 64³): GPU underutilized, CPU may be faster
- Medium grids (128³–512³): GPU ramps up, 10-100× over CPU
- Large grids (≥ 1024³): Memory-bandwidth-limited plateau, peak cells/s

**Status:** `CAN BUILD NOW` — extends existing `benchmark/dipole.jl`
which already sweeps resolutions but does not report cells/s vs. N³

---

#### E29. Single-GPU Throughput vs. Physics Complexity

**Priority:** P0
**Validates:** Kernel specialization overhead — how much does adding
physics features (PML, materials, monitors, dispersive poles) cost?
**Computes:** Cells/second at fixed grid size (512³) with progressively
more physics enabled

**Setup — incremental physics layers:**

| Config | Physics | Expected Overhead |
|--------|---------|-------------------|
| A | Vacuum, no PML, no monitors | Baseline (fastest) |
| B | Vacuum + PML | +10-20% (ψ field updates) |
| C | Dielectric geometry (2 materials) + PML | +5% (ε lookup) |
| D | Multi-material (6 layers) + PML | +5% (same kernel) |
| E | Config C + DFT monitor (10 freqs) | +5-10% |
| F | Config C + dispersive material (1 pole) | +30-50% (ADE fields) |
| G | Config C + dispersive material (3 poles) | +50-80% |
| H | Config C + nonlinear χ³ | +10-20% |
| I | Config C + non-uniform grid (vector Δx) | +5-10% |

**Report:** Bar chart of cells/s for each config. This directly
demonstrates the value of Khronos's `Nothing`-dispatch specialization:
configs without a feature should have zero overhead for that feature.

**Status:** Partially `CAN BUILD NOW` (configs A–E); configs F–I
`BLOCKED` on respective roadmap features

---

#### E30. Backend Comparison (CUDA vs. ROCm vs. Metal vs. CPU)

**Priority:** P0
**Validates:** KernelAbstractions.jl multi-backend portability and
relative performance across hardware
**Computes:** Cells/second for identical problems on each backend

**Setup:**
- Fixed problem: 3D vacuum dipole, 512³ grid, PML, 100 timesteps
- Run on each available backend: `CUDABackend()`, `ROCBackend()`,
  `MetalBackend()`, `CPU()`
- Also test with materials (sphere scattering) and monitors

**Report per backend:**

```
Backend          Device               cells/s      relative
─────────────────────────────────────────────────────────────
CUDA             NVIDIA A100          1.5e9        1.00×
CUDA             NVIDIA H100          2.1e9        1.40×
ROCm             AMD MI250X           1.2e9        0.80×
Metal            Apple M1 Pro         0.3e9        0.20×
CPU (threads=8)  AMD EPYC 7763        0.05e9       0.03×
CPU (threads=1)  AMD EPYC 7763        0.008e9      0.005×
```

**Status:** `CAN BUILD NOW` — existing `benchmark/*.yml` files already
have A100/H100/M1 baselines but don't report in a unified comparison

---

#### E31. Precision Comparison (Float32 vs. Float64)

**Priority:** P1
**Validates:** Float32 delivers ~2× speedup and ~2× memory savings with
acceptable accuracy loss for most applications
**Computes:** Cells/second and accuracy for Float32 vs. Float64 on the
same problem

**Setup:**
- Problem: dielectric slab transmission (periodic_slab example)
- Run both `backend_number = Float32` and `backend_number = Float64`
- Compare: (1) cells/s, (2) peak GPU memory, (3) transmission spectrum
  accuracy vs. analytical Fabry-Perot

**Expected results:**
- Float32: ~2× cells/s, ~2× less memory, max |T_err| < 0.01
- Float64: baseline cells/s, baseline memory, max |T_err| < 0.001

**Status:** `CAN BUILD NOW`

---

#### E32. JIT Compilation Overhead

**Priority:** P1
**Validates:** First-run JIT cost and warm-run performance, so users
know what to expect
**Computes:** Wall time breakdown: JIT compilation vs. actual simulation

**Setup:**
- Problem: 3D waveguide with ModeSource (exercises the most complex
  kernel path)
- Measure: (1) cold start (first `run` call), (2) warm start (second
  `run` call with same types), (3) re-warm (different grid size, same
  types — should be fast due to cached compilation)
- Report time spent in: Julia compilation, GPU kernel compilation,
  mode solving, field initialization, actual timestep loop

**Status:** `CAN BUILD NOW` — existing examples already print cold/warm
timing but don't break it down systematically

---

#### E33. Memory Capacity Test (Maximum Problem Size)

**Priority:** P1
**Validates:** Maximum achievable problem size on a given GPU, memory
efficiency of field storage
**Computes:** Largest N³ grid that fits in GPU memory for each
configuration

**Setup:**
- Binary search for max N on a given GPU (e.g., A100 80GB, H100 80GB)
- Test three configurations:
  1. Vacuum + PML (6 field arrays + ψ arrays): expect ~1800³ on 80GB
  2. With 2 materials + DFT monitor: expect ~1600³
  3. With dispersive material (1 pole, adds 6 P arrays): expect ~1200³
- Report: N³, total voxels, memory used, memory per voxel

**Theoretical minimum memory per voxel:**
- 6 field components × 4 bytes (Float32) = 24 bytes/voxel
- With PML ψ: +12 auxiliary fields = +48 bytes in PML region
- With 1 dispersive pole: +6 P arrays = +24 bytes/voxel
- With DFT monitor: +2 complex × N_freq per monitor voxel

**Status:** `CAN BUILD NOW`

---

#### E34. Cross-Solver Performance Comparison

**Priority:** P0
**Validates:** Khronos's GPU performance advantage is real and quantified
**Computes:** Identical physical problems across Khronos, Meep, and fdtdx

**Benchmark suite — three problems at three scales:**

| Problem | Small | Medium | Large |
|---------|-------|--------|-------|
| Vacuum dipole | 128³ | 512³ | 1024³ |
| Sphere scattering | 128³ | 512³ | 1024³ |
| Slab + DFT monitor | 128³ | 512³ | 1024³ |

**Per solver:**

| Solver | Run Method |
|--------|------------|
| **Khronos** (CUDA, Float64) | `Khronos.run_benchmark(sim, 100)` |
| **Khronos** (CUDA, Float32) | Same, `Float32` |
| **Meep** (CPU, 1 thread) | `sim.run(until=...)` |
| **Meep** (CPU, 8 threads) | `sim.run(until=...)` with OMP |
| **fdtdx** (CUDA, Float32) | `run_fdtd(...)` |

**Report:**

```
Problem: Sphere Scattering 512³ (100 timesteps)
─────────────────────────────────────────────────
Solver               cells/s      speedup vs Meep-1T
Khronos CUDA F32     2.1e9        420×
Khronos CUDA F64     1.2e9        240×
fdtdx CUDA F32       1.8e9        360×
Meep CPU 8T          4.0e7        8×
Meep CPU 1T          5.0e6        1×
```

**Setup scripts:** Provide matched simulation definitions in Julia
(Khronos), Python (Meep), and Python/JAX (fdtdx) that set up physically
identical problems with the same grid, materials, and timestep count.

**Status:** `CAN BUILD NOW` for Khronos vs. Meep; fdtdx needs JAX env

---

#### E35. Weak Scaling (Multi-GPU)

**Priority:** P1
**Validates:** Multi-GPU halo exchange overhead, near-linear scaling
**Reference:** Meep MPI scaling, fdtdx JAX sharding
**Computes:** Fixed problem size per GPU (512³ per device); measure
aggregate throughput for 1, 2, 4, 8 GPUs

**Setup:**
- 3D, PML, dipole source in vacuum
- MPI-based domain decomposition along longest axis
- 100 timesteps per measurement

**Report:**

```
GPUs    Total grid     cells/s        efficiency
────────────────────────────────────────────────
1       512³           1.5e9          100%
2       512×512×1024   2.8e9          93%
4       512×512×2048   5.4e9          90%
8       512×512×4096   10.2e9         85%
```

**Key metric:** Parallel efficiency = (N-GPU throughput) / (N × 1-GPU
throughput). Target: >85% at 8 GPUs for single-axis decomposition.

**Status:** `BLOCKED ON §2.1 + §3.7` (domain decomposition + halo exchange)

---

#### E36. Strong Scaling (Multi-GPU)

**Priority:** P2
**Validates:** Multi-GPU speedup for a fixed large problem
**Computes:** Fixed total problem (2048³); divide across 1, 2, 4, 8 GPUs

**Report:**

```
GPUs    Partition      wall time/step   speedup
────────────────────────────────────────────────
1       2048³          680 ms           1.0×
2       2048×2048×1024 355 ms           1.9×
4       2048×2048×512  185 ms           3.7×
8       2048×2048×256  100 ms           6.8×
```

**Key metric:** Strong scaling efficiency. Communication-to-computation
ratio grows as slabs get thinner, so efficiency will degrade. Target:
>80% at 4 GPUs, >65% at 8 GPUs.

**Status:** `BLOCKED ON §2.1 + §3.7`

---

#### E37. Halo Exchange Overhead Profiling

**Priority:** P2
**Validates:** Communication is not the bottleneck; compute/communicate
overlap works
**Computes:** Breakdown of time per timestep into compute vs.
communication at various GPU counts and problem sizes

**Setup:**
- Instrument halo exchange with timers
- Measure: (1) kernel execution time, (2) halo pack time, (3) MPI
  send/recv time, (4) halo unpack time, (5) synchronization overhead
- Test with CUDA-aware MPI (direct GPU-GPU) vs. staged (GPU→CPU→GPU)

**Report:**

```
2 GPUs, 512³ per GPU, CUDA-aware MPI:
  Kernel:      8.2 ms (92%)
  Halo pack:   0.1 ms (1%)
  MPI comm:    0.5 ms (6%)
  Halo unpack: 0.1 ms (1%)
  Total:       8.9 ms

2 GPUs, 512³ per GPU, Staged MPI:
  Kernel:      8.2 ms (78%)
  D2H copy:    0.8 ms (8%)
  MPI comm:    0.7 ms (7%)
  H2D copy:    0.8 ms (8%)
  Total:       10.5 ms
```

**Status:** `BLOCKED ON §2.1 + §3.7`

---

#### E38. Large-Scale Realistic Benchmarks

**Priority:** P1
**Validates:** Khronos handles real-world-sized problems, not just toy
examples. These are the benchmarks that matter for marketing and papers.

**Three flagship problems:**

**E38a. Metalens (large metasurface)**
- 3D: ~500 × 500 × 100 cells at resolution 30 → ~750M voxels
- Periodic unit cell with dielectric pillars on SiO₂ substrate
- PlaneWaveSource (broadband), DFT transmission monitor
- Tests: periodic BCs, multi-material geometry, large grid
- **Status:** `BLOCKED ON §1.1 + §3.1` (periodic BCs)

**E38b. Photonic integrated circuit (multi-component)**
- 3D: ~2000 × 500 × 100 cells → ~100M voxels
- Si waveguides, bends, directional coupler on SiO₂
- ModeSource input, DFT flux monitors at each port
- Tests: long propagation, multiple materials, many monitors
- **Status:** `BLOCKED ON §1.3 + §3.3` (flux monitors)

**E38c. Full 3D Mie scattering (large sphere)**
- 3D: sphere radius = 10λ → ~600³ grid at 30 cells/λ → ~216M voxels
- PlaneWaveSource, 6-face flux box, PML
- Tests: large 3D problem, many timesteps, PML performance
- Compare to Mie theory at 50+ frequencies
- **Status:** `BLOCKED ON §1.3 + §3.3` (flux monitors)

**E38d. Vacuum propagation at memory limit (single-GPU stress test)**
- 3D: largest grid that fits on target GPU (e.g., 1800³ on A100 80GB)
- Point source, PML, no monitors, 100 steps
- Tests: raw throughput at maximum memory utilization, no OOM
- **Status:** `CAN BUILD NOW`

---

#### E39. Kernel Roofline Analysis

**Priority:** P2
**Validates:** Khronos kernels are memory-bandwidth-bound (expected for
FDTD) and operating near the hardware roofline
**Computes:** Arithmetic intensity and achieved bandwidth for each
kernel variant

**Setup:**
- Use NVIDIA Nsight Compute (or `CUDA.@profile`) to measure per-kernel:
  - DRAM bytes read/written
  - FLOPs executed
  - Achieved memory bandwidth (GB/s)
  - Arithmetic intensity (FLOPs/byte)
- Compare to device peak bandwidth (e.g., A100 = 2039 GB/s HBM)
- Plot on roofline diagram

**Expected results:**
- Vacuum curl kernel: ~0.5 FLOPs/byte (memory-bound), achieving
  ~70-80% peak bandwidth
- PML curl kernel: ~0.8 FLOPs/byte (still memory-bound, more ops
  per byte due to ψ updates)
- ADE kernel: ~1.0 FLOPs/byte (approaching compute-bound)

**Status:** `CAN BUILD NOW` (needs CUDA profiling tools)

---

#### E40. Non-Uniform Grid Performance and Accuracy

**Priority:** P1
**Validates:** Non-uniform grid delivers equivalent accuracy at lower
memory cost, with acceptable throughput overhead
**Computes:** Accuracy and performance comparison: uniform grid vs.
non-uniform grid for a problem with disparate feature scales

**Setup:**
- Problem: thin metal film (10nm) on glass substrate, illuminated by
  broadband planewave. Film requires fine grid (Δz = 1nm), substrate
  needs coarse grid (Δz = 50nm).
- Uniform grid: Δ = 1nm everywhere → ~2000³ grid (~enormous)
- Non-uniform grid: Δz = 1nm near film, grading to 50nm away → ~200³
  equivalent
- Compare: (1) R(λ) spectra should match, (2) memory: 10-100× reduction,
  (3) throughput: slight penalty from vector Δx reads

**Status:** `BLOCKED ON §2.5 + §3.6` (non-uniform grid)

---

### Category 12: Inverse Design (Future)

---

#### E41. Waveguide Bend Topology Optimization

**Priority:** P2
**Validates:** Adjoint solver, Enzyme.jl AD through FDTD kernels,
convergence to known-good design
**Reference:** fdtdx `optimize_ceviche_corner`, Tidy3D `Autograd8WaveguideBend`
**Computes:** Optimize permittivity distribution in a 2D design region to
minimize insertion loss of a 90° waveguide bend

**Status:** `BLOCKED ON §2.7` (adjoint compilation)

---

#### E42. Metalens Phase Map Optimization

**Priority:** P3
**Validates:** Large-scale inverse design, metasurface parameterization
**Reference:** Tidy3D `Autograd7Metalens`, Meep `metasurface_lens`
**Computes:** Optimize pillar radii in a metalens unit-cell library to
maximize focal spot intensity

**Status:** `BLOCKED ON §2.7` (adjoint) + §1.1 (periodic BCs)

---

### Category 13: Cylindrical Coordinates

---

#### E43. Cylinder Scattering (Cylindrical FDTD)

**Priority:** P2
**Validates:** Cylindrical coordinate kernels, r=0 singularity handling,
~90× speedup vs. 3D for rotationally symmetric problems
**Reference:** Meep `cylinder_cross_section`
**Computes:** Scattering cross section of a finite dielectric cylinder
using cylindrical (r,z) FDTD with angular mode index m

**Status:** `BLOCKED ON §3.14` (cylindrical kernels)

---

#### E44. Fresnel Zone Plate Focusing

**Priority:** P3
**Validates:** Cylindrical coordinates, near-to-far field, large-scale
diffractive optics
**Reference:** Meep `zone_plate`
**Computes:** Focal spot intensity profile of a binary zone plate using
cylindrical FDTD

**Status:** `BLOCKED ON §3.14` (cylindrical) + §1.5 + §3.10 (near-to-far)

---

### Category 14: CFS-PML and Boundary Stability

---

#### E45. CFS-PML Evanescent Wave Absorption

**Priority:** P1
**Validates:** CFS-PML (κ, α parameters) stability for evanescent fields
**Reference:** Tidy3D `AbsorbingBoundaryReflection`
**Computes:** Compare PML reflectance for standard σ-only PML vs. CFS-PML
with κ and α; show improved absorption at grazing incidence and for
evanescent waves

**Setup:**
- Dipole source near PML boundary
- Measure spurious reflection amplitude with and without κ/α
- Long-time simulation to test stability

**Status:** `BLOCKED ON §3.8` (CFS-PML activation)

---

### Category 15: 2D Fundamentals

---

#### E46. 2D TM Dipole Radiation

**Priority:** P0
**Validates:** 2D simulation code path (`TwoD_TM`), field patterns match
analytical 2D Green's function
**Reference:** Foundational FDTD validation
**Computes:** Ez field from a 2D point source; compare amplitude decay
to 1/√r (2D cylindrical wave)

**Status:** `CAN BUILD NOW`

---

#### E47. 2D TE Waveguide Propagation

**Priority:** P0
**Validates:** 2D TE mode (`TwoD_TE`), correct field components (Hx, Hy, Ez)
**Computes:** TE mode propagation in a dielectric slab waveguide; verify
confinement and propagation constant

**Status:** `CAN BUILD NOW`

---

### Category 16: Subpixel Averaging

---

#### E48. Subpixel Averaging Convergence

**Priority:** P1
**Validates:** Second-order convergence at curved dielectric interfaces
**Reference:** Meep subpixel smoothing, Tidy3D `SubpixelSpec`
**Computes:** Ring resonator frequency vs. resolution with and without
subpixel averaging; verify O(Δx²) convergence with averaging vs.
O(Δx) without

**Status:** `BLOCKED ON §3.5` (subpixel averaging)

---

### Category 17: Checkpoint and I/O

---

#### E49. Checkpoint / Restart

**Priority:** P3
**Validates:** HDF5 save/load of simulation state, exact restart
**Reference:** Meep HDF5, fdtdx JAX checkpointing
**Computes:** Run simulation for N steps, save state, restart, run N more
steps; compare to continuous 2N-step run — results should be identical

**Status:** `BLOCKED ON §3.20` (HDF5 I/O)

---

## Cross-Solver Example Coverage Matrix

This matrix shows which examples exist in each solver and whether Khronos
can implement them today or needs roadmap features.

| Example | Meep | Tidy3D | fdtdx | Khronos | Status |
|---------|------|--------|-------|---------|--------|
| **Basics** | | | | | |
| Dielectric slab transmission | `bend-flux` | `StartHere` | — | `periodic_slab.jl` ✓ | EXISTS |
| 2D dipole radiation | `straight-waveguide` | — | — | E36 | CAN BUILD |
| 2D waveguide propagation | `straight-waveguide` | — | — | E37 | CAN BUILD |
| Planewave in vacuum | `oblique-planewave` | — | `simulate_gaussian_source` | `planewave.jl` ✓ | EXISTS |
| CW steady-state check | `solve-cw` | — | — | E25 | CAN BUILD |
| **Fresnel / Reflectance** | | | | | |
| Normal Fresnel | `refl-quartz` | — | — | E2 | CAN BUILD |
| Angular Fresnel | `refl-angular` | `BroadbandPlaneWave...` | — | E3 | BLOCKED (BCs) |
| Dispersive Fresnel | `refl-quartz` | `Dispersion` | — | E16 | BLOCKED (ADE) |
| **Scattering** | | | | | |
| Mie scattering (sphere) | `mie_scattering` | `Near2FarSphereRCS` | — | E4 | BLOCKED (flux) |
| Differential cross section | `differential_cross_section` | `PECSphereRCS` | — | E15 | BLOCKED (N2F) |
| Cylinder cross section | `cylinder_cross_section` | — | — | E43 | BLOCKED (cyl) |
| **Waveguides** | | | | | |
| Waveguide mode launch | — | `ModalSourcesMonitors` | `check_waveguide_modes` | `waveguide.jl` ✓ | EXISTS |
| Waveguide bend | `bent-waveguide` | `EulerWaveguideBend` | — | E5 | BLOCKED (flux) |
| Waveguide taper | `mode-decomposition` | `WaveguideSizeConverter` | — | E10 | BLOCKED (mode) |
| Directional coupler | `coupler` | `DirectionalCoupler` | — | E11 | BLOCKED (flux) |
| Width sweep (loss) | — | — | `width_sweep_analysis` | E9 | PARTIAL |
| **Resonators** | | | | | |
| Ring resonator Q | `ring` | `RingResonator` | — | E6 | BLOCKED (time mon) |
| PhC cavity Q | `holey-wvg-cavity` | `NanobeamCavity` | — | E8 | BLOCKED (BCs) |
| PhC band structure | `holey-wvg-bands` | `Bandstructure` | — | E7 | BLOCKED (BCs) |
| **Gratings** | | | | | |
| Binary grating orders | `binary_grating` | `GratingEfficiency` | — | E12 | BLOCKED (BCs) |
| Grating coupler | — | `GratingCoupler` | — | E13 | BLOCKED (BCs) |
| **Far-Field** | | | | | |
| Dipole radiation pattern | `antenna-radiation` | `AntennaCharacteristics` | — | E14 | BLOCKED (N2F) |
| Zone plate focusing | `zone_plate` | `ZonePlateFieldProjection` | — | E44 | BLOCKED (cyl+N2F) |
| **Materials** | | | | | |
| Lorentzian dispersion | `refl-quartz` | `Dispersion` | — | E16 | BLOCKED (ADE) |
| Drude metal | — | `PlasmonicNanoparticle` | — | E17 | BLOCKED (ADE) |
| χ³ nonlinearity | `3rd-harm-1d` | `KerrSidebands` | — | E18 | BLOCKED (NL) |
| Anisotropic dielectric | `polarization_grating` | `FullyAnisotropic` | `gaussian_anisotropic` | E19 | CAN BUILD |
| **Boundaries** | | | | | |
| PEC/PMC cavity | — | `BoundaryConditions` | — | E20 | BLOCKED (PEC) |
| Periodic verification | `oblique-planewave` | `BoundaryConditions` | — | E21 | BLOCKED (BCs) |
| CFS-PML stability | — | `AbsorbingBoundaryReflection` | — | E45 | BLOCKED (CFS) |
| **Advanced** | | | | | |
| Mirror symmetry | `bend-flux` | `Symmetry` | — | E22 | BLOCKED (sym) |
| Non-uniform grid | — | `AutoGrid` | — | E23 | BLOCKED (grid) |
| LDOS | `metal-cavity-ldos` | — | — | E27 | BLOCKED (LDOS) |
| Subpixel convergence | — | — | — | E48 | BLOCKED (subpx) |
| **Performance** | | | | | |
| Cross-solver benchmark | — | — | — | E34 | CAN BUILD |
| Single-GPU throughput | — | — | — | E28/E29 | CAN BUILD |
| Backend comparison | — | — | — | E30 | CAN BUILD |
| Precision (F32 vs F64) | — | — | — | E31 | CAN BUILD |
| JIT overhead | — | — | — | E32 | CAN BUILD |
| Memory capacity | — | — | — | E33 | CAN BUILD |
| Weak scaling (multi-GPU) | — | — | — | E35 | BLOCKED (MPI) |
| Strong scaling (multi-GPU) | — | — | — | E36 | BLOCKED (MPI) |
| Halo profiling | — | — | — | E37 | BLOCKED (MPI) |
| Large-scale realistic | — | — | — | E38a-d | PARTIAL |
| Roofline analysis | — | — | — | E39 | CAN BUILD |
| Non-uniform grid perf | — | — | — | E40 | BLOCKED (grid) |
| **Inverse Design** | | | | | |
| Waveguide bend opt | — | `Autograd8WaveguideBend` | `optimize_ceviche_corner` | E41 | BLOCKED (adj) |
| Metalens opt | — | `Autograd7Metalens` | — | E42 | BLOCKED (adj) |
| **Gaussian Beam** | | | | | |
| Beam waist validation | — | `AdvancedGaussianSources` | `simulate_gaussian_source` | E24 / existing ✓ | PARTIAL |

---

## Implementation Priority

### Phase 1: Build Now (no new features required)

These examples can be implemented immediately with the current Khronos
feature set and provide valuable coverage:

| Example | What it validates |
|---------|-------------------|
| **E2** Fresnel Reflectance (Normal) | Material interfaces, DFT accuracy, two-run workflow |
| **E25** CW Steady-State Verification | CW source, Green's function decay |
| **E46** 2D TM Dipole | 2D code path validation |
| **E47** 2D TE Waveguide | 2D TE code path validation |
| **E19** Anisotropic Material Slab | Per-axis permittivity |
| **E1** (enhance) Periodic Slab | Add 2D version, error metrics |
| **E24** (enhance) Gaussian Beam | Add waist measurement |
| | |
| **Scaling / Performance (CAN BUILD NOW):** | |
| **E28** Single-GPU Throughput vs. Size | GPU utilization curve, crossover vs. CPU |
| **E29** Throughput vs. Physics Complexity | Kernel specialization overhead |
| **E30** Backend Comparison | CUDA/ROCm/Metal/CPU relative speed |
| **E31** Precision Comparison (F32 vs F64) | Speed/accuracy tradeoff |
| **E32** JIT Compilation Overhead | Cold vs. warm start timing |
| **E33** Memory Capacity Test | Max problem size per GPU |
| **E34** Cross-Solver Performance | Khronos vs. Meep vs. fdtdx |
| **E38d** Vacuum Stress Test | Max grid at memory limit |
| **E39** Kernel Roofline Analysis | Memory bandwidth utilization |

### Phase 2: After P0 Features (§1.1–§1.3, §3.1–§3.3)

Once periodic BCs and flux monitors are implemented:

| Example | Unlocked by |
|---------|-------------|
| **E3** Angular Fresnel | §1.1 + §3.1 (Bloch BCs) |
| **E4** Mie Scattering | §1.3 + §3.3 (flux monitors) |
| **E5** 2D Waveguide Bend | §1.3 + §3.3 (flux monitors) |
| **E21** Periodic Verification | §1.1 + §3.1 (periodic BCs) |
| **E9** Mode Source (full) | §1.3 + §3.3 (flux monitors) |
| **E38b** PIC Benchmark | §1.3 + §3.3 (flux monitors) |
| **E38c** Large Mie Benchmark | §1.3 + §3.3 (flux monitors) |

### Phase 3: After P0–P1 Features (§1.2, §3.2, §1.4, §3.4, §3.8, §3.12)

Once dispersive materials, mode monitors, CFS-PML, and time monitors work:

| Example | Unlocked by |
|---------|-------------|
| **E16** Dispersive Fresnel | §1.2 + §3.2 (ADE) |
| **E17** Drude Metal | §1.2 + §3.2 (ADE) |
| **E6** Ring Resonator Q | §3.12 (time monitor) |
| **E7** PhC Bands | §1.1 + §3.1 (BCs) |
| **E10** Waveguide Taper | §1.4 + §3.4 (mode overlap) |
| **E45** CFS-PML Stability | §3.8 (CFS-PML) |
| **E12** Binary Grating | §1.1 + §1.9 + §3.16 |
| **E40** Non-Uniform Grid Perf | §2.5 + §3.6 (grid) |

### Phase 4: After P2 Features

| Example | Unlocked by |
|---------|-------------|
| **E14** Antenna Pattern | §1.5 + §3.10 (near-to-far) |
| **E18** THG Nonlinear | §1.6 + §3.13 (nonlinear) |
| **E22** Symmetry | §1.7 + §2.4 (symmetry) |
| **E23** Non-Uniform Grid Accuracy | §2.5 + §3.6 (grid) |
| **E43** Cylindrical | §3.14 (cylindrical) |
| **E35/E36/E37** Multi-GPU Scaling | §2.1 + §3.7 (MPI) |
| **E41** Inverse Design | §2.7 (adjoint) |
| **E38a** Metalens Benchmark | §1.1 (periodic BCs) |

---

## Quantitative Validation Standards

Every example with an analytical or cross-solver reference should report:

1. **Accuracy metric**: Max absolute error, RMS error, or relative error
   vs. analytical/reference solution.
2. **Convergence**: Error vs. resolution (cells/wavelength) should show
   expected order (first-order without subpixel, second-order with).
3. **Performance**: Cells/second, total wall time, peak GPU memory.
4. **Pass/fail threshold**: Each example should define an acceptable error
   bound (e.g., |T_khronos - T_analytic| < 0.01 for all frequencies).

Example output format:

```
========================================
Khronos.jl Validation: E2 Fresnel Reflectance
========================================
Grid:       200 x 1 x 1600 (resolution=40)
Backend:    CUDA (NVIDIA A100)
Precision:  Float64

Results:
  Max |R_err|:     0.0023
  RMS R_err:       0.0008
  Frequencies:     100 points (0.4 - 1.0 μm)

Performance:
  Wall time:       12.3 s (110 timesteps warm)
  Cells/second:    1.2e9
  GPU memory:      0.8 GB

Status: PASS (max error < 0.01 threshold)
========================================
```
