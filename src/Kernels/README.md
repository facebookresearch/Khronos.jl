# Kernels/

FDTD timestepping kernels for Khronos.jl. Each timestep advances the
electromagnetic fields by one Δt through a leapfrog update of the curl
equations (Maxwell's equations in discrete form).

## File Layout

```
Kernels.jl            Hub: includes sub-files, then dispatch + orchestration
Helpers.jl            @inline math primitives shared by all kernel tiers
ReferenceKernels.jl   KernelAbstractions @kernel functions (backend-portable)
CUDAKernels.jl        Raw CUDA.jl kernels (NVIDIA-only, highest throughput)
Dispersive.jl         ADE polarization (Drude/Lorentz) + χ3 Kerr nonlinearity
```

Include order matters: `Helpers` → `ReferenceKernels` → `CUDAKernels` →
`Dispersive`, then the hub code in `Kernels.jl` which depends on all four.

## FDTD Update Equations

A full timestep consists of two half-steps:

```
H half-step:   B_new = B_old + Δt · ∇×E       then   H = μ⁻¹ · B
E half-step:   D_new = D_old + Δt · ∇×H       then   E = ε⁻¹ · (D - P)
```

With PML (Perfectly Matched Layer) absorbing boundaries, the curl update
becomes a 3-stage cascade per component (C → U → T), where intermediate
fields absorb the PML conductivity σ. Without PML, T = T + Δt·curl directly.

## Composable Dispatch Architecture

The dispatch logic in `Kernels.jl` (`step_H_fused!` / `step_E_fused!`)
selects the optimal kernel for each chunk based on its physics at runtime:

```
For each chunk:
  ┌─ CUDA + uniform grid + no PML + no sources + scalar μ?
  │   → Raw CUDA fused kernel (B/D eliminated, ~9 arrays)
  │
  ├─ No PML (but has sources, per-voxel μ, or non-CUDA)?
  │   → KA fused curl+update kernel (B/D kept, sources handled)
  │
  ├─ CUDA + uniform grid + PML + no sources + no material σ?
  │   → Per-component raw CUDA PML kernels with σ-skipping
  │
  └─ PML fallback (complex fields, dispersive, sources, non-uniform)?
      → KA separate curl + update kernels (most general)
```

Julia's multiple dispatch compiles specialized versions for each combination
of `Nothing` vs concrete types. When a field (e.g., source arrays, PML
auxiliary arrays) is `nothing`, the compiler eliminates all code paths
touching it — zero runtime cost for unused features.

### Key Dispatch Signals

| Condition | What it controls |
|-----------|-----------------|
| `has_any_pml(chunk)` | PML cascade vs direct curl update |
| `_fields_are_complex(sim)` | Bloch BC (complex fields) — forces KA path |
| `_grid_is_uniform(sim)` | Scalar Δx/Δy/Δz — required for raw CUDA |
| `sim.sources_active` | Whether source arrays are passed (Nothing after cutoff) |
| `g.μ_inv isa Real` | Scalar μ — enables B-elimination in BH kernels |
| `g.ε_inv_x isa AbstractArray` | Per-voxel vs scalar ε — selects DE kernel variant |
| `f.fPDx isa Nothing` | No dispersive polarization — enables D-elimination |

## Kernel Tiers

### Tier 1: Raw CUDA Kernels (`CUDAKernels.jl`)

Hand-written CUDA kernels using `blockIdx()`/`threadIdx()` directly.
Bypass KernelAbstractions overhead (CartesianIndex construction, KA index
management) for ~2× higher throughput than KA equivalents.

**Interior (no PML) — B/D elimination:**

When there is no PML, no active sources, and no polarization, the
intermediate B/D fields can be algebraically eliminated:

```
H_new = H_old + μ⁻¹ · Δt · ∇×E     (instead of: B += Δt·∇×E; H = μ⁻¹·B)
E_new = E_old + ε⁻¹ · Δt · ∇×H     (instead of: D += Δt·∇×H; E = ε⁻¹·D)
```

This saves 6 array reads + 6 writes per voxel per step and reduces unique
arrays from 15 to 9, improving L2 cache and TLB utilization.

Three variants:
- `_cuda_fused_BH_kernel!` — scalar μ⁻¹, always used for H update
- `_cuda_fused_DE_kernel!` — per-voxel ε⁻¹ arrays
- `_cuda_fused_DE_scalar_kernel!` — scalar ε⁻¹ (vacuum/uniform medium)

**PML — per-component with σ-skipping:**

Each kernel processes ONE field component with a fused curl→U→B/D→W→H/E
pipeline. Register pressure drops from ~50-60 (3-component) to ~20-25.

σ-skipping: when all 3 σ values are zero (interior voxels within a PML
chunk, typically 35-90% of the chunk), the kernel takes a fast path that
eliminates B/D entirely and skips U/W auxiliary reads/writes.

Six kernels (3 BH + 3 DE), each following cyclic permutation of axes:
- `_cuda_pml_BH_{x,y,z}_kernel!`
- `_cuda_pml_DE_{x,y,z}_kernel!`

The DE variants accept both scalar and per-voxel ε⁻¹ via the `_get_m`
helper (dispatches on `Real` vs `AbstractArray`).

**Legacy 3-component PML** (kept for reference):
- `_cuda_pml_BH_kernel!`, `_cuda_pml_DE_kernel!`, `_cuda_pml_DE_scalar_kernel!`

### Tier 2: KernelAbstractions Kernels (`ReferenceKernels.jl`)

Backend-portable `@kernel` functions that run on CUDA, ROCm, Metal, or CPU
via KernelAbstractions.jl. Used as the fallback when raw CUDA conditions
aren't met (non-uniform grids, Bloch BC, active sources in PML chunks, etc.).

**3-component kernels:**
- `step_curl!` — PML curl cascade for all 3 components in one kernel
- `update_field!` — material update (E=ε⁻¹·D or H=μ⁻¹·B) for all 3 components
- `step_curl_and_update!` — fused curl+update for non-PML chunks (saves B/D re-read)
- `step_curl_and_update_pml!` — fused curl+update for PML chunks (T_new kept in register)

**Per-component kernels** (reduced register pressure, 3× more launches):
- `step_curl_comp!` — single-component curl, ~30 registers (vs ~64 for 3-component)
- `update_field_comp!` — single-component material update

**Launch wrappers** (iterate over chunks, pass correct field arrays):
- `step_B_from_E!` / `step_D_from_H!` — curl launchers (3-component)
- `update_H_from_B!` / `update_E_from_D!` — update launchers (3-component)
- `step_B_from_E_comp!` / `step_D_from_H_comp!` — curl launchers (per-component)
- `update_H_from_B_comp!` / `update_E_from_D_comp!` — update launchers (per-component)

### Helpers (`Helpers.jl`)

`@inline` functions shared across all kernel tiers:

| Category | Functions | Purpose |
|----------|-----------|---------|
| Finite differences | `d_dx!`, `d_dy!`, `d_dz!` | Forward/backward differences with uniform or non-uniform grid |
| Curl operators | `curl_x!`, `curl_y!`, `curl_z!` | Cross-product curl components |
| PML cascade | `generic_curl!` (8 variants) | Dispatches on `Nothing` to eliminate dead PML stages |
| PML σ access | `get_σ`, `get_σD` | Index into 1D PML σ arrays (or return `nothing`) |
| Field update | `update_field_generic` (4 variants) | A = m⁻¹·(T + S - P) with optional W-accumulation |
| Fused update | `update_field_from_T` (4 variants) | Same but takes T as register value (avoids re-read) |
| Material access | `get_m_inv`, `_get_m` | Dispatch on scalar vs per-voxel material arrays |
| Grid spacing | `get_inv_dx` | Dispatch on scalar Δ (uniform) vs vector Δ (non-uniform) |

### Dispersive Physics (`Dispersive.jl`)

**ADE polarization** (Drude/Lorentz susceptibilities):
- `update_polarization_kernel!` — advance one pole: P^{n+1} from P^n, P^{n-1}, E^n
- `accumulate_polarization_kernel!` — add pole's P to total fPD{x,y,z}
- `zero_polarization_kernel!` — clear total P before re-accumulation
- `step_polarization!` — orchestration: zero → update each pole → accumulate

**χ3 Kerr nonlinearity:**
- `_chi3_correction_kernel!` — E_corrected = E / (1 + χ3·|E|²)
- `step_chi3_correction!` — orchestration, no-op if no χ3 materials

## Orchestration (`Kernels.jl`)

### `step!(sim)` — One Full Timestep

```
1. Deactivate source arrays if t > last_source_time  (enables Nothing specialization)
2. CUDA Graph replay path (if captured)              → skip to monitors
3. Graph capture attempt (once sources off)           → replay subsequent steps
4. Normal path:
   a. update_magnetic_sources!
   b. step_H_fused!      → per-chunk kernel dispatch + halo exchange
   c. update_H_monitors!
   d. update_electric_sources!
   e. step_E_fused!      → per-chunk kernel dispatch + halo exchange
   f. step_chi3_correction!
   g. step_polarization!
   h. update_E_monitors!
5. increment_timestep!
```

### CUDA Graphs

After sources deactivate, the FDTD kernels become time-invariant (no
source-array arguments change between steps). `_try_capture_graphs!`
captures two sub-graphs (H-step and E-step) and replays them for all
subsequent timesteps, eliminating kernel launch overhead. Monitor updates
run outside the graph since they have time-varying DFT phase arguments.

Disabled for: complex fields (Bloch BC), multi-stream mode, distributed runs.

### Multi-Stream Dispatch

When `sim._use_multi_stream` is true, each chunk is launched on its own
CUDA stream for concurrent execution. A synchronization barrier runs
before the halo exchange. This helps when chunk sizes are small enough
that a single kernel doesn't saturate the GPU.

## Performance Notes

- FDTD is **memory-bandwidth bound**: the bottleneck is reading/writing
  field arrays from GPU global memory, not arithmetic.
- The raw CUDA interior kernel achieves ~85-90% of peak memory bandwidth
  on A100/GH200 by minimizing array count and using simple integer indexing.
- PML σ-skipping reduces effective memory traffic by 35-50% for chunks
  that mix interior and PML voxels.
- `KHRONOS_CUDA_WORKGROUP_SIZE` (default 256) and `KHRONOS_WORKGROUP_SIZE`
  (default 64) environment variables tune thread block sizes.
- `KHRONOS_CUDA_GRAPHS=0` disables graph capture for debugging.
