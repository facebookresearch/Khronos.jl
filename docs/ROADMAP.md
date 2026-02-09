# Khronos.jl Feature Roadmap

This document catalogs every feature gap between Khronos.jl and the three
reference FDTD solvers (Meep, Tidy3D, fdtdx), organized around a three-layer
architecture that cleanly separates concerns and enables long-term scalability.

Reference codebases:

| Solver | Language | GPU | Differentiable | Open-Source |
|--------|----------|-----|----------------|-------------|
| **Meep** | C++ / Python | No | Adjoint (autograd, black-box) | Yes (GPL-2) |
| **Tidy3D** | Python API / proprietary backend | Yes (cloud) | Adjoint (autograd) | API only |
| **fdtdx** | Python / JAX | Yes (JAX) | Yes (time-reversible custom VJP) | Yes (MIT) |
| **Khronos.jl** | Julia | Yes (KernelAbstractions) | Not yet (Enzyme.jl potential) | Yes (MIT) |

---

## Architecture Overview

Khronos.jl is structured as three distinct layers, each with a clear
responsibility boundary:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Frontend                                          │
│  User-facing API, geometry/material/source/monitor types,   │
│  simulation specification DSL, GUI potential                 │
│                                                             │
│  Emits: Simulation IR (declarative, language-agnostic)      │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Graph Compiler                                    │
│  Domain decomposition, kernel selection/specialization,     │
│  memory planning, launch configuration, schedule generation │
│                                                             │
│  Emits: Execution plan (kernel DAG + memory map)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Engine Backend                                    │
│  Optimized FDTD kernels, field updates, halo exchange,      │
│  multi-backend execution (CUDA/ROCm/Metal/CPU)              │
│                                                             │
│  Executes: The plan, producing simulation results           │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1 (Frontend)** is concerned with *what* the user wants to simulate.
It provides the API for defining geometries, materials, sources, monitors,
and boundary conditions. Long-term, it emits a declarative IR (leveraging
`GeometryPrimitives.jl`) that is language-agnostic, enabling non-Julia
frontends (Python bindings, GUI, config files) to target the same compiler.

**Layer 2 (Graph Compiler)** is concerned with *how* to execute the
simulation efficiently. It takes the declarative IR and produces an
execution plan: which kernels to launch, in what order, with what
specializations, on what devices, and with what memory layout. Initially
this layer handles domain decomposition and kernel selection; progressively
it evolves toward full type-specialized codegen.

**Layer 3 (Engine Backend)** is concerned with *executing* the plan. It owns
the actual FDTD kernels (`@kernel` functions), field array storage, time
stepping, PML updates, source injection, and monitor accumulation. It
receives a fully resolved execution plan and runs it.

---

## Priority Definitions

| Tier | Label | Meaning |
|------|-------|---------|
| **P0** | Critical | Blocks the majority of practical FDTD workflows |
| **P1** | High | Required for competitive feature parity or significant performance gains |
| **P2** | Medium | Important for specific application domains |
| **P3** | Low | Nice-to-have, niche, or complementary |

---

# Layer 1: Frontend

The Frontend layer defines the user-facing API and the declarative
simulation specification. Features here affect how users describe
simulations, what physical models they can express, and how the simulation
description is represented internally.

---

## 1.1 Boundary Specification API — P0

**Present in:** Meep, Tidy3D, fdtdx

**Why critical:** Users currently have no way to request anything other than
PML. The API must support per-axis, per-side boundary type selection.

**Current state:** `README.md:27` states "All simulations require PML." No
boundary type selection exists in the user API.

**Implementation guidance:**

Add a per-axis, per-side boundary type enum and specification struct:

```julia
@enum BoundaryType PML_BC Periodic_BC Bloch_BC PEC_BC PMC_BC

struct BoundarySpec
    x::Tuple{BoundaryType, BoundaryType}  # (minus, plus)
    y::Tuple{BoundaryType, BoundaryType}
    z::Tuple{BoundaryType, BoundaryType}
end
```

Tidy3D's `BoundarySpec` with per-face `Boundary(minus, plus)` is a good
model. Add a `boundary_spec` field to `Simulation`, defaulting to PML on
all faces for backward compatibility.

For Bloch-periodic BCs, add a `k_point::Union{SVector{3},Nothing}` field
on `Simulation` that the user sets to specify the Bloch wave vector.

**Estimated scope:** ~80 lines in `DataStructures.jl`, `Simulation.jl`.

---

## 1.2 Dispersive Material Types (Lorentz / Drude / Debye Poles) — P0

**Present in:** Meep, Tidy3D (both extensive)

**Why critical:** Metals (Au, Ag, Cu, Al), semiconductors at optical
frequencies, and any broadband simulation through dispersive dielectrics
require frequency-dependent permittivity. Users need to be able to specify
these material models.

**Current state:** `DataStructures.jl:240` has `#TODO: add info relevant for
polarizability`. The `Polarizability.jl` section (line 249-252) is empty.

**Implementation guidance:**

Define a `Susceptibility` abstract type with concrete subtypes:

```julia
abstract type Susceptibility end

struct LorentzianPole{N} <: Susceptibility
    σ::N          # oscillator strength (Δε * ω₀²)
    ω₀::N         # resonance frequency
    γ::N          # damping rate
end

struct DrudePole{N} <: Susceptibility
    σ::N          # plasma-frequency-squared term
    γ::N          # collision frequency
end
```

Add a `poles::Vector{Susceptibility}` field to `Material`.

For maximum generality, consider implementing the Pole-Residue
representation (as in Tidy3D's `PoleResidue` class). All Lorentz, Drude,
Debye, and Sellmeier models can be converted to pole-residue pairs `(aₙ, cₙ)`.

**Key references:**
- Taflove & Hagness, *Computational Electrodynamics*, Ch. 9 (ADE method)
- Meep `meep.hpp:88-278` (susceptibility class hierarchy)

**Estimated scope:** ~150 lines for types in new `Susceptibility.jl` and
modifications to `DataStructures.jl`.

---

## 1.3 Flux Monitor Type — P0

**Present in:** Meep (`dft_flux`), Tidy3D (`FluxMonitor`), fdtdx (`PoyntingFluxDetector`)

**Why critical:** Computing transmission and reflection spectra is the
single most common FDTD post-processing task. Users need to be able to
specify flux monitors in the simulation description.

**Current state:** Khronos has `DFTMonitor` (single-component
frequency-domain fields) but no flux monitor type.

**Implementation guidance:**

Add a `FluxMonitor` type that specifies a planar region and frequency list:

```julia
struct FluxMonitor
    center::SVector{3}
    size::SVector{3}        # one dimension should be 0 (planar)
    freqs::Vector{Float64}
    normal::Direction        # derived from size, or user-specified
end
```

The flux computation itself happens in the Engine (see §3.3). The Frontend
only needs the type definition and validation (ensure exactly one dimension
is zero).

**Estimated scope:** ~50 lines in `Monitors.jl` / `DataStructures.jl`.

---

## 1.4 Mode Decomposition Monitor — P1

**Present in:** Meep (MPB integration), Tidy3D (`ModeMonitor`), fdtdx (`ModeOverlapDetector`)

**Why it matters:** Determines how much power couples into each waveguide
mode. Essential for photonic circuit design (S-parameters, insertion loss,
crosstalk).

**Current state:** Khronos has a mode solver (`VectorModesolver.jl`) and
mode source injection, but no overlap monitor type.

**Implementation guidance:**

Add a `ModeMonitor` type specifying the cross-section plane, modes of
interest, and frequencies. The mode-solve and overlap computation happens
in the Engine (see §3.4).

**Estimated scope:** ~50 lines.

---

## 1.5 Near-to-Far Field Monitor — P1

**Present in:** Meep (full dyadic Green's function), Tidy3D (angle/Cartesian/k-space)

**Why it matters:** Computing radiation patterns, scattering cross-sections,
and far-field intensities without simulating the full propagation distance.

**Implementation guidance:**

Add monitor types for near-to-far field projection:

```julia
struct NearFieldSurface
    center::SVector{3}
    size::SVector{3}
    freqs::Vector{Float64}
end

struct FarFieldProjection
    surface::NearFieldSurface
    observation::Union{AngleGrid, CartesianGrid, KSpaceGrid}
end
```

Support observation in angle space (θ, φ), Cartesian space (x, y at
distance d), and k-space (kx/k₀, ky/k₀), following Tidy3D's model.

**Estimated scope:** ~100 lines for type definitions.

---

## 1.6 Nonlinear Material Types (χ² and χ³) — P2

**Present in:** Meep (χ² and χ³ with Padé approximant), Tidy3D (Kerr, TPA, FCA)

**Implementation guidance:**

Add `chi2` and `chi3` fields to `Material`:

```julia
struct Material{N}
    ...
    chi2::N   # second-order nonlinear susceptibility
    chi3::N   # third-order nonlinear susceptibility
end
```

The nonlinear update equations are handled by the Engine (see §3.6).

**Key references:**
- Meep `step_generic.cpp:535-547`
- Tidy3D `nonlinear.py` (`KerrNonlinearity`, `TwoPhotonAbsorption`)

**Estimated scope:** ~30 lines in `DataStructures.jl`.

---

## 1.7 Symmetry Specification — P2

**Present in:** Meep (mirror, rotate2, rotate4), Tidy3D (per-axis ±1)

**Implementation guidance:**

Add a `symmetry::SVector{3,Int}` field on `Simulation` (values: 0, +1, -1
per axis). The Graph Compiler uses this to halve grid dimensions and the
Engine applies symmetry-aware boundary conditions. The Frontend only needs
the type and validation.

**Key references:**
- Tidy3D `Simulation.symmetry` (per-axis 0/+1/-1)

**Estimated scope:** ~30 lines.

---

## 1.8 TFSF (Total-Field/Scattered-Field) Source — P2

**Present in:** Tidy3D (`TFSF`), fdtdx (`tfsf.py`)

**Implementation guidance:**

Add a `TFSFSource` type specifying the injection box, plane wave direction,
polarization, and time profile. TFSF injects a plane wave inside a
rectangular volume while maintaining zero incident field outside.

**Key references:**
- Taflove & Hagness, Ch. 5.6 (TFSF formulation)
- Tidy3D `source/field.py` (`TFSF` class)

**Estimated scope:** ~60 lines for the type definition.

---

## 1.9 Diffraction Monitor — P2

**Present in:** Tidy3D (`DiffractionMonitor`), fdtdx (`DiffractiveDetector`)

**Implementation guidance:**

Add a `DiffractionMonitor` type for periodic structures. Specifies the
monitor plane and diffraction orders of interest. The FFT-based computation
happens in the Engine.

**Estimated scope:** ~40 lines.

---

## 1.10 LDOS Monitor — P2

**Present in:** Meep (`dft_ldos`)

**Implementation guidance:**

Add an `LDOSMonitor` type that is co-located with a point dipole source.
Records the DFT of the E-field at the source location.

**Key references:**
- Meep `dft_ldos.cpp:60-80`

**Estimated scope:** ~30 lines.

---

## 1.11 Force / Stress Tensor Monitor — P2

**Present in:** Meep (`dft_force`, stress tensor integration)

**Implementation guidance:**

Add a `ForceMonitor` type specifying a closed surface around an object.
The Maxwell stress tensor integration happens in the Engine.

**Key references:**
- Meep `stress.cpp:90-114`

**Estimated scope:** ~40 lines.

---

## 1.12 Custom Sources (Current and Time Profile) — P3

**Current state:** `CustomCurrentSource` at `SpatialSources.jl:252-277` is
`# TODO`. Custom time source at `TimeSources.jl:138` is `# TODO`.

**Implementation:** Allow users to pass arbitrary arrays (spatial current
distribution) and functions (time profile) to the source infrastructure.

**Estimated scope:** ~100 lines.

---

## 1.13 GDS Import — P3

**Present in:** Meep (libGDSII), Tidy3D (gdstk)

**Implementation:** Use `GDSTK.jl` or parse GDSII directly to extract
polygon vertices. Convert polygons to `GeometryPrimitives.jl` shapes (or
a new `PolySlab` type).

**Estimated scope:** ~200 lines.

---

## 1.14 PolySlab Geometry (Extruded Polygon with Sidewall Angle) — P3

**Present in:** Tidy3D (`PolySlab` with `sidewall_angle`), fdtdx (extruded polygon)

**Implementation:** Define a `PolySlab` shape with 2D polygon vertices,
extrusion axis, slab bounds, and optional sidewall taper angle. Use
point-in-polygon testing for the cross-section.

**Estimated scope:** ~200 lines.

---

## 1.15 STL / Triangle Mesh Import — P3

**Present in:** Tidy3D (`TriangleMesh`)

**Implementation:** Parse STL files into triangle meshes. Use ray-casting
(Möller-Trumbore intersection) for point-in-mesh testing during geometry
initialization.

**Estimated scope:** ~300 lines.

---

## 1.16 Astigmatic Gaussian Beam Source — P3

**Present in:** Tidy3D (`AstigmaticGaussianBeam`)

**Implementation:** Extend `GaussianBeamSource` to support two independent
waist radii and waist distances (one per transverse axis).

**Estimated scope:** ~50 lines.

---

## 1.17 Time Modulation of Materials — P3

**Present in:** Tidy3D (`ModulationSpec`, `SpaceTimeModulation`)

**Implementation:** Allow ε and σ to vary sinusoidally in time. Add a
`ModulationSpec` field to `Material` specifying modulation amplitude,
frequency, and phase.

**Estimated scope:** ~60 lines for types.

---

## 1.18 Lumped Circuit Elements — P3

**Present in:** Tidy3D (`LumpedResistor`, `RLCNetwork`)

**Implementation:** Define R, L, C element types that embed as modified
update equations in localized grid regions.

**Estimated scope:** ~80 lines for types.

---

## 1.19 Simulation IR Specification — P1

**Not present in:** any reference solver (all are monolithic)

**Why it matters:** A declarative, language-agnostic intermediate
representation decouples the user API from the compiler and engine. This
enables multiple frontends (Julia API, Python bindings, GUI, config files)
to target the same simulation pipeline.

**Implementation guidance:**

The IR should be a serializable representation of a complete simulation
specification, leveraging `GeometryPrimitives.jl` for shape primitives.
It captures:

1. **Domain**: cell size, resolution (uniform or non-uniform), dimension.
2. **Geometry**: ordered list of geometric objects with material assignments.
3. **Materials**: permittivity, permeability, conductivity, susceptibility
   poles, nonlinear coefficients.
4. **Sources**: type, spatial extent, time profile, polarization.
5. **Monitors**: type, location, frequencies, components.
6. **Boundaries**: per-axis, per-side boundary types + Bloch k-vector.
7. **Symmetries**: per-axis symmetry phases.
8. **Run parameters**: total time, Courant factor, convergence criteria.

Initially this can be a Julia struct hierarchy (essentially a cleaned-up
version of `SimulationData`). The language-agnostic serialization format
(JSON, MessagePack, or a schema-defined binary format) comes later.

**Estimated scope:** ~300 lines for initial Julia struct IR.

---

## 1.20 Energy Monitor — P3

**Present in:** fdtdx (`EnergyDetector`)

**Implementation:** Add an `EnergyMonitor` type. Computes
`u = ½(ε|E|² + μ|H|²)` at monitor locations, either instantaneous or
DFT-based.

**Estimated scope:** ~30 lines for the type.

---

# Layer 2: Graph Compiler

The Graph Compiler takes a complete simulation specification (from Layer 1)
and produces an optimized execution plan for Layer 3. It is responsible for
all decisions about *how* to run the simulation efficiently: domain
decomposition, kernel specialization, memory layout, device assignment, and
scheduling.

This layer evolves progressively:

- **Phase A (near-term):** Domain decomposition + kernel selection based on
  active physics modules. Essentially formalizes the logic currently
  embedded in `prepare_simulation!`.
- **Phase B (mid-term):** Full type-specialized kernel DAG generation.
  Analyzes which physics modules are active (dispersive? nonlinear? PML?)
  and produces a minimal kernel schedule, eliminating branches at runtime.
- **Phase C (long-term):** Cost-model-driven optimization. Profile-guided
  kernel fusion, occupancy tuning, memory placement (shared vs global),
  and multi-device load balancing.

---

## 2.1 Domain Decomposition Planner — P1

**Present in:** Meep (MPI), fdtdx (JAX `NamedSharding`), Tidy3D (cloud)

**Why it matters:** Single-GPU memory limits simulation size to ~2000³
voxels (Float32). The compiler must decide how to partition the domain
across devices.

**Current state:** No multi-device logic exists. Ghost cells exist
(`OffsetArrays` with ±1 padding) and 8 `# TODO update halo` comments mark
insertion points.

**Implementation guidance:**

The domain decomposition planner takes the simulation grid dimensions and
available device count, then produces a partition map:

1. **Single-axis decomposition** — Split along the longest axis. Assign
   contiguous slab regions to each device/rank. Emit halo-exchange
   instructions in the execution plan between each field-update substep.

2. **Cost-balanced partitioning** — Weight partitions by computational cost
   (PML regions are more expensive than free space). Meep
   `structure.cpp:66-94` implements binary-tree cost-based partitioning.

3. **Multi-axis decomposition** — For large device counts, split along
   2-3 axes using a 3D process grid.

The planner emits a `PartitionPlan` struct consumed by the Engine:

```julia
struct PartitionPlan
    axis::Int                       # split axis
    slabs::Vector{UnitRange{Int}}   # per-device grid ranges
    neighbors::Vector{Tuple{Int,Int}}  # (left, right) rank per device
    halo_width::Int
end
```

**Key references:**
- Meep `structure.cpp:66-94` (cost-based partitioning)
- fdtdx `sharding.py` (`get_named_sharding_from_shape`)

**Estimated scope:** ~300 lines.

---

## 2.2 Kernel Selection and Specialization — P1

**Not present in:** any reference solver (unique to Khronos)

**Why it matters:** Khronos's `Nothing`-vs-`AbstractArray` dispatch pattern
in `Timestep.jl` generates ~32+ specialized GPU kernels from a single
source. The Graph Compiler should formalize this: given the set of active
physics (PML? dispersive? nonlinear? magnetic materials?), select the
minimal set of kernel specializations needed.

**Current state:** Kernel specialization happens implicitly through Julia's
multiple dispatch at JIT time. This works but is opaque — the user and
compiler cannot inspect or optimize the kernel schedule.

**Implementation guidance:**

1. **Physics feature flags** — Analyze the simulation IR to determine which
   features are active:
   - PML boundaries → PML-aware curl kernels
   - Dispersive materials → ADE update kernels
   - Nonlinear materials → nonlinear constitutive update
   - μ ≠ 1 → non-trivial H-from-B update
   - Non-uniform grid → vector Δx kernels

2. **Kernel DAG** — Produce a directed acyclic graph of kernel invocations
   per timestep. Each node specifies the kernel function, its type
   parameters (which dispatch variant), and its array arguments. The DAG
   encodes dependencies (e.g., ADE update must happen between D-update and
   E-update).

3. **Specialization table** — For each region of the grid (interior, PML,
   interface), record which kernel variant applies. Regions with different
   physics get different kernel launches.

**Estimated scope:** ~400 lines.

---

## 2.3 Memory Planning — P2

**Why it matters:** GPU memory is the primary bottleneck for 3D FDTD. The
compiler should compute the total memory footprint before launching and
warn (or auto-adjust resolution) if it exceeds device capacity.

**Implementation guidance:**

1. **Static analysis** — From the simulation IR, compute:
   - Field arrays: `6 × Nx × Ny × Nz × sizeof(T)` (E and H)
   - PML arrays: auxiliary ψ fields in PML regions
   - Dispersive arrays: P and P_prev per pole per component
   - Monitor arrays: DFT accumulation buffers
   - Material arrays: ε, μ, σ per component

2. **Memory budget** — Query device memory via KernelAbstractions and
   compare against the computed requirement.

3. **Auto-adjustment** — Optionally suggest resolution reduction, single-
   precision, or multi-GPU decomposition if the problem exceeds single-GPU
   memory.

**Estimated scope:** ~200 lines.

---

## 2.4 Symmetry Exploitation Planner — P2

**Present in:** Meep (mirror, rotate2, rotate4), Tidy3D (per-axis ±1)

**Implementation guidance:**

When the simulation IR specifies symmetries (see §1.7), the compiler:

1. Halves the grid dimensions along each symmetric axis.
2. Modifies boundary conditions at symmetry planes: PEC-like (symmetric E,
   antisymmetric H) for +1 symmetry, PMC-like for -1 symmetry.
3. Applies phase factors when monitors read fields across the symmetry
   plane.
4. Reduces memory and computation by 2× per axis (up to 8× for 3 axes).

The compiler emits a modified grid spec and symmetry boundary instructions
for the Engine.

**Key references:**
- Meep `vec.hpp:1181-1231` (symmetry class)

**Estimated scope:** ~200 lines.

---

## 2.5 Non-Uniform Grid Planning — P1

**Present in:** Tidy3D (per-axis `GridSpec`, `AutoGrid`, `CustomGrid`)
**Not in:** Meep (scalar `a`), fdtdx (scalar `resolution`)

**Why it matters:** Structures with disparate feature scales waste enormous
memory with uniform grids. Per-axis resolution alone can give 4-100×
memory savings. **No other open-source GPU FDTD solver has this.**

**Current state:** `DataStructures.jl:341-343` defines `Δx, Δy, Δz` as
`Union{Vector{N}, N, Nothing}` — designed from day one for non-uniform
grids. But all code paths treat them as scalars.

**Implementation guidance:**

The compiler resolves grid spacing from the simulation IR:

1. **Per-axis uniform (Δx ≠ Δy ≠ Δz)** — Allow `resolution` to be a
   `Tuple{Int,Int,Int}`. Compute `Δx = cell_size[1]/resolution[1]` etc.
   Update Courant: `Δt = C / √(1/Δx² + 1/Δy² + 1/Δz²)`.

2. **Per-cell non-uniform** — Produce `Δx::Vector{N}` arrays. The
   compiler selects the correct kernel variants (vector Δx dispatch).
   PML σ profiles become per-cell. Courant condition uses
   `Δt = C × min(Δx_min, Δy_min, Δz_min) / √3`.

3. **AutoGrid (future)** — Compute per-cell sizes based on local
   wavelength in material and feature proximity (Tidy3D model).

**Key references:**
- Tidy3D `grid_spec.py` (`AutoGrid`, `CustomGrid`, `MeshOverrideStructure`)

**Estimated scope:** ~200 lines for per-axis uniform; ~400 additional for
per-cell planning.

---

## 2.6 Launch Configuration Optimizer — P3

**Why it matters:** GPU performance is sensitive to thread block size,
shared memory usage, and register pressure. The compiler should auto-tune
launch configurations based on kernel type and grid dimensions.

**Implementation guidance:**

1. Query device compute capability and SM count.
2. For each kernel, compute occupancy at various block sizes.
3. Select the block size that maximizes occupancy (or use a heuristic
   based on kernel characteristics — memory-bound vs compute-bound).
4. KernelAbstractions.jl's `@kernel` already handles some of this, but
   explicit workgroup size selection can improve performance by 10-30%.

**Estimated scope:** ~150 lines.

---

## 2.7 Adjoint Compilation — P2

**Present in:** Meep (autograd), Tidy3D (autograd), fdtdx (time-reversible JAX AD)

**Implementation guidance:**

The compiler supports inverse design by producing both forward and adjoint
execution plans:

1. **Enzyme.jl path (recommended)** — The compiler marks kernels for
   Enzyme differentiation. Unlike fdtdx (which manually implements
   `update_E_reverse()` and `update_H_reverse()` in `backward.py`),
   Enzyme.jl differentiates directly through KernelAbstractions kernels.

2. **Manual adjoint path** — The compiler generates the time-reversed
   kernel DAG with adjoint sources from objective function gradients.

Supporting infrastructure in the compiler:
- `DesignRegion` mapping latent parameters to permittivity arrays.
- Filter + projection pipeline (Gaussian filter → tanh projection).
  fdtdx's `SubpixelSmoothedProjection` (`projection.py:215`, based on
  arxiv.org/2503.20189) is the state of the art.
- Minimum feature size constraints (fdtdx `BrushConstraint2D`).

**Key references:**
- fdtdx `fdtd.py:16-230` (reversible FDTD with custom VJP)
- fdtdx `projection.py:215` (subpixel smoothed projection)
- Meep `python/adjoint/` (OptimizationProblem, DesignRegion)
- Tidy3D `plugins/invdes/` (TopologyDesignRegion, FilterProject)

**Estimated scope:** ~1000+ lines for the full pipeline.

---

## 2.8 KD-Tree Geometry Acceleration — P3

**Current state:** `Geometry.jl:8` notes this as a future optimization.
Currently uses linear search through geometry objects.

**Implementation guidance:**

During compilation, build a KD-tree (or BVH) over bounding boxes for the
geometry list. The Engine uses tree traversal instead of linear `findfirst`
during voxelization. Matters for scenes with hundreds of objects.

**Estimated scope:** ~150 lines (or use `NearestNeighbors.jl`).

---

# Layer 3: Engine Backend

The Engine Backend executes the plan produced by the Graph Compiler. It
owns the FDTD kernels, field array storage, time stepping, boundary
condition enforcement, source injection, and monitor accumulation. All
kernels use `KernelAbstractions.jl` `@kernel` macros for multi-backend
portability (CUDA, ROCm, Metal, oneAPI, CPU).

---

## 3.1 Periodic and Bloch-Periodic Boundary Kernels — P0

**Present in:** Meep, Tidy3D, fdtdx (periodic only)

**Why critical:** Required for photonic crystals, gratings, metasurfaces,
waveguide supercells. Without periodic BCs every such simulation must use
PML, wasting compute.

**Current state:** No periodic boundary code exists. The ghost-cell
infrastructure exists (`Fields.jl:128-139`, `OffsetArrays` with one ghost
pixel per side). The 8 `# TODO update halo` sites in `Timestep.jl`
(lines 74, 108, 144, 178, 235, 243, 253, 264) are the exact insertion
points.

**Implementation guidance:**

1. **Periodic BCs** — For each axis flagged as periodic, copy field values
   from one side of the domain to the ghost cells on the opposite side
   after each field-update substep. Each halo-update site becomes:

   ```julia
   if boundary_is_periodic(sim, X())
       copy_periodic_ghost!(sim.fields.fEx, X())
   end
   ```

   On GPU, this is a simple kernel or `copyto!` between slices.

2. **Bloch-periodic BCs** — Same as periodic, but multiply copied values
   by a complex phase factor `exp(i k · L)` where `k` is the Bloch wave
   vector and `L` is the lattice vector. This requires:
   - Complex-valued fields (switch `backend_number` to complex when
     `k_point != nothing`).
   - Phase factors `eikna = exp(i * k[d] * cell_size[d])` precomputed
     once per axis.

   Reference: Meep `boundaries.cpp:90-105` implements exactly this via
   `eikna[d]`, `coskna[d]`, `sinkna[d]`. fdtdx implements periodic BCs
   via `jnp.pad(mode="wrap")` in `curl.py`.

**Estimated scope:** ~220 lines in `Boundaries.jl`, `Timestep.jl`.

---

## 3.2 Dispersive Material ADE Kernels — P0

**Why critical:** The material types from §1.2 need corresponding update
kernels.

**Current state:** Polarization fields `Px/Py/Pz` are wired through
`update_field!` (`Timestep.jl:475-477`) but always passed as `Nothing`.
The kernel dispatch infrastructure means adding real P arrays will
auto-generate the correct specialized kernel with no changes to
`update_field_generic`.

**Implementation guidance:**

The standard approach is the Auxiliary Differential Equation (ADE) method.
For each dispersive pole, auxiliary polarization fields `P` and `P_prev`
are stored and updated at each timestep.

1. **Auxiliary fields** — For each pole, allocate two arrays per component
   (`P_current`, `P_prev`) with the same shape as the E-field arrays.
   Modify `Fields.jl` to allocate these when
   `any(obj -> has_poles(obj.material), geometry)`.

2. **ADE update kernel** — Insert a new kernel call between
   `step_D_from_H!` and `update_E_from_D!` (between lines 30 and 32 of
   `step!`). The Lorentzian ADE update is (Meep
   `susceptibility.cpp:254`):

   ```
   P_next = γ₁⁻¹ * (P_cur * (2 - ω₀²Δt²) - γ₁ * P_prev + ω₀²Δt² * σ * E)
   ```

   where `γ₁ = 1 + γΔt/2`, `γ₁⁻¹ = 1 / γ₁`.

   The Drude case sets the denominator term differently (no `ω₀²` in the
   resonance term).

3. **Constitutive update** — In `update_E_from_D!`, the P fields are
   already wired: `update_field_generic` at `Timestep.jl:437` adds
   `P[idx]` to the net field. Once P arrays are non-`Nothing`, this path
   activates automatically via dispatch.

**Key references:**
- Taflove & Hagness, Ch. 9 (ADE method)
- Meep `susceptibility.cpp:188-261` (Lorentzian ADE update)

**Estimated scope:** ~350 lines in new `Susceptibility.jl` kernel code,
modifications to `Fields.jl`, `Timestep.jl`.

---

## 3.3 Flux Computation Kernels — P0

**Why critical:** The flux monitor type from §1.3 needs corresponding
DFT accumulation and post-processing kernels.

**Implementation guidance:**

1. **DFTFluxMonitor kernel** — Store DFT accumulations for 4 tangential
   components (e.g., for a z-normal plane: `Ex, Ey, Hx, Hy`). Reuse the
   existing `update_dft_monitor!` kernel (`Monitors.jl:120-137`) for each
   component independently.

2. **Flux computation** — After the simulation, compute:

   ```julia
   function flux(m::DFTFluxMonitorData, freq_idx)
       # For z-normal: S_z = Ex*Hy - Ey*Hx
       return real(sum(m.Ex[:,:,freq_idx] .* conj(m.Hy[:,:,freq_idx])
                     - m.Ey[:,:,freq_idx] .* conj(m.Hx[:,:,freq_idx])))
   end
   ```

   Scale by `Δx * Δy / (N_timesteps)²` for proper normalization.

3. **Normal direction handling** — Determine the 4 tangential components
   from the monitor plane normal using the existing `Direction` type.

**Key references:**
- Meep `dft.cpp` (DFT flux implementation)
- fdtdx `poynting_flux.py:11` (`PoyntingFluxDetector`)

**Estimated scope:** ~150 lines in `Monitors.jl`.

---

## 3.4 Mode Overlap Computation — P1

**Why it matters:** The mode monitor type from §1.4 needs the actual
overlap integral computation.

**Implementation guidance:**

1. Record DFT of all 4 tangential field components on a cross-section
   plane (reuse `DFTMonitor` infrastructure).

2. Compute the mode profile at each DFT frequency using
   `VectorModesolver.jl`.

3. Compute the overlap integral:
   ```
   α = (1/4) ∫ (E_mode × H*_sim + E*_sim × H_mode) · dA
   ```

   This is exactly fdtdx's `ModeOverlapDetector` formula (`mode.py:15`).

4. The complex coefficient `α` gives both amplitude and phase. Power in
   mode `n` is `|αₙ|²`.

**Key references:**
- Meep `meep.hpp:2145-2151` (`get_overlap`, `get_mode_flux_overlap`)
- fdtdx `detectors/mode.py`

**Estimated scope:** ~200 lines.

---

## 3.5 Subpixel Averaging — P1

**Present in:** Meep (anisotropic averaging), Tidy3D (multiple methods)

**Why it matters:** Without subpixel averaging, FDTD convergence at curved
dielectric interfaces is first-order instead of second-order. In practice
this means ~4× finer grids (and ~64× more voxels in 3D) for equivalent
accuracy. On GPU where memory is the bottleneck, this directly limits
problem size.

**Current state:** `Geometry.jl` uses point sampling — each voxel material
is determined by a single `findfirst(point, geometry)` call (lines 187-203).

**Implementation guidance:**

The standard method (Farjadpour et al. 2006) computes an effective
inverse-permittivity tensor at each interface voxel:

```
ε⁻¹_eff = P‖ * <ε⁻¹> + P⊥ * <ε>⁻¹
```

where `<ε>` and `<ε⁻¹>` are volume-averaged permittivity and its inverse,
`P‖` projects parallel to the interface, and `P⊥` projects perpendicular.

1. **Normal vector computation** — Sample ε at N quadrature points within
   each voxel. The gradient of ε gives the interface normal. Meep uses
   sphere quadrature (`anisotropic_averaging.cpp:33-56`). A simpler start
   is central differences on the ε grid.

2. **Volume averaging** — For each voxel straddling a material interface,
   compute the fill fraction `f` and averaged `<ε>`, `<ε⁻¹>`.

3. **Tensor storage** — The effective `ε⁻¹` is a 3x3 symmetric tensor.
   For diagonal anisotropy (the common case), store 3 components per
   voxel. Khronos already supports per-component `εx, εy, εz` arrays.

4. **Integration** — Modify `_write_geometry_3d!` in `Geometry.jl` to
   detect boundary voxels and apply the averaging procedure.

**Key references:**
- A. Farjadpour et al., "Improving accuracy by subpixel smoothing in the
  finite-difference time domain," *Optics Letters* 31(20), 2006.
- Meep `anisotropic_averaging.cpp:90-150`
- Tidy3D `SubpixelSpec`

**Estimated scope:** ~400 lines in `Geometry.jl`.

---

## 3.6 Non-Uniform Grid Stencils — P1

**Why it matters:** The grid plans from §2.5 need corresponding stencil
implementations in the Engine.

**Current state:** The finite difference stencils (`Timestep.jl:395-400`)
already parameterize on `Δx, Δy, Δz` independently:

```julia
d_dx!(A, Δx, idx_curl, ix, iy, iz) = inv(Δx) * (A[ix+idx_curl,...] - A[ix,...])
```

**Implementation guidance:**

1. **Per-axis uniform** — No stencil changes needed. The existing functions
   work with independent scalar `Δx, Δy, Δz`.

2. **Per-cell non-uniform** — Add method overloads:
   ```julia
   d_dx!(A, Δx::AbstractVector, idx_curl, ix, iy, iz) =
       inv(Δx[ix]) * (A[ix+idx_curl,...] - A[ix,...])
   ```

   Julia's dispatch will generate a specialized GPU kernel that reads
   `Δx[ix]` per cell, while the uniform case continues to use a
   compile-time constant.

   PML sigma profiles become per-cell (already arrays, so natural).

**Estimated scope:** ~200 lines in `Timestep.jl`.

---

## 3.7 Halo Exchange (Multi-GPU/MPI) — P1

**Why it matters:** The partition plan from §2.1 needs corresponding data
movement kernels.

**Current state:** 8 `# TODO update halo` sites in `Timestep.jl`.

**Implementation guidance:**

Two approaches:

1. **MPI-based (Meep model)** — Use `MPI.jl` for inter-process
   communication. After each field-update substep, exchange ghost-cell
   slices between neighboring ranks. Use CUDA-aware MPI to avoid
   device-to-host copies.

2. **KernelAbstractions multi-device (fdtdx model)** — Use Julia's
   `Distributed` or `CUDA.jl`'s multi-device API with explicit `copyto!`
   for halo exchange.

Each halo-update site needs:

```julia
if num_processes > 1
    exchange_halo!(field_array, axis, rank, comm)
end
```

where `exchange_halo!` does non-blocking `MPI.Isend` / `MPI.Irecv` of
the boundary slices.

**Key references:**
- Meep `boundaries.cpp:35-77` (communication sequences)
- fdtdx `sharding.py`

**Estimated scope:** ~600 lines.

---

## 3.8 CFS-PML Activation (κ and α Parameters) — P1

**Present in:** Meep, Tidy3D, fdtdx (all use full CPML)

**Why it matters:** Standard PML (σ only) can be unstable for evanescent
waves, long-time simulations, and dispersive media. CFS-PML adds κ
(coordinate stretching) and α (complex shift) for improved absorption.

**Current state:** `BoundaryData` already defines `κBx/κBy/κBz`,
`κDx/κDy/κDz`, `αBx/αBy/αBz`, `αDx/αDy/αDz`
(`DataStructures.jl:145-158`) but they are all commented
`# currently not used`.

**Implementation guidance:**

The CFS-PML stretching factor is:

```
s(ω) = κ + σ / (α + iω)
```

In the time domain, the CPML update becomes (fdtdx `curl.py`):

```
b = exp(-(σ/κ + α) * Δt)
a = σ * (b - 1) / (σ + κα)
ψ_new = b * ψ_old + a * (∂F/∂x)
```

1. Populate κ and α arrays in `init_boundaries` (`Boundaries.jl:57-110`)
   alongside existing σ arrays.

2. Modify the `generic_curl!` overloads in `Timestep.jl` to use CPML
   coefficients `a` and `b` (precomputed from σ, κ, α) instead of raw σ.

3. Since κ and α fields already exist in the struct, this is mostly
   computing them and threading them into existing PML update paths.

**Key references:**
- J.A. Roden and S.D. Gedney, "Convolutional PML (CPML)," *Microwave and
  Optical Technology Letters*, 27(5), 2000.
- fdtdx `perfectly_matched_layer.py:117`

**Estimated scope:** ~200 lines in `Boundaries.jl` and `Timestep.jl`.

---

## 3.9 PEC and PMC Boundary Walls — P2

**Present in:** Meep, Tidy3D (`PECBoundary`, `PMCBoundary`)

**Implementation guidance:**

- **PEC:** Zero the tangential E-field components at the boundary. For a
  z-boundary: set `Ex = Ey = 0` on the wall.
- **PMC:** Zero the tangential H-field components. For a z-boundary:
  set `Hx = Hy = 0` on the wall.

These are trivial Dirichlet conditions applied after each field update.
`Boundaries.jl:117` has a comment referencing Dirichlet zeroing but no code.

**Estimated scope:** ~50 lines.

---

## 3.10 Near-to-Far Field Transformation Kernels — P1

**Why it matters:** The monitor types from §1.5 need the actual Green's
function integration.

**Implementation guidance:**

1. Record DFT of tangential E and H on a closed/open surface. Compute
   equivalent currents: `J = n × H`, `M = -n × E`.

2. Integrate against the free-space Green's function:
   - **3D:** Dyadic Green's function with 1/r, 1/r², 1/r³ terms
     (Meep `near2far.cpp:133-187`).
   - **2D:** Hankel functions (`H₀⁽¹⁾(kr)`, `H₁⁽¹⁾(kr)`).

3. For far-field only, use the asymptotic form (plane-wave approximation),
   reducing to a Fourier transform of the surface currents.

**Key references:**
- Meep `near2far.cpp`
- Tidy3D `FieldProjectionAngleMonitor`, `FieldProjectionCartesianMonitor`,
  `FieldProjectionKSpaceMonitor`
- Taflove & Hagness, Ch. 8

**Estimated scope:** ~500 lines.

---

## 3.11 Custom Permeability (μ) Wiring — P2

**Present in:** Meep, Tidy3D, fdtdx

**Current state:** The `Material` struct has `μx, μy, μz` fields.
`Geometry.jl` allocates `μ_inv_x/y/z` arrays but hardcodes `μ_inv = 1.0`
at line 312 with `# todo add μ support`.

**Implementation guidance:**

Wire the allocated μ arrays into `GeometryData` and pass them to
`update_H_from_B!` as the `m_inv` parameter (currently receives scalar
`1.0`). The kernel dispatch already handles `m_inv::AbstractArray` vs
`m_inv::Real` via `get_m_inv()` (`Timestep.jl:459-463`), so the kernel
side needs no changes.

**Estimated scope:** ~30 lines in `Geometry.jl`.

---

## 3.12 Time Monitor Kernel — P2

**Current state:** `TimeMonitor` struct and initialization exist
(`Monitors.jl:28-53`) but `#TODO add update and kernel functions for time
monitor` at line 55.

**Implementation guidance:**

Write an `update_monitor` dispatch for `TimeMonitorData` that copies the
current field value at the monitor location into the preallocated storage
at the current time index:

```julia
@kernel function update_time_monitor!(monitor_fields, sim_fields,
                                       offset_x, offset_y, offset_z, time_idx)
    ix, iy, iz = @index(Global, NTuple)
    monitor_fields[time_idx, ix, iy, iz] = sim_fields[ix+offset_x, iy+offset_y, iz+offset_z]
end
```

**Estimated scope:** ~50 lines.

---

## 3.13 Nonlinear Material Update Kernels — P2

**Why it matters:** The material types from §1.6 need corresponding
kernels.

**Implementation guidance:**

The nonlinear update modifies the E = ε⁻¹D constitutive relation to
include intensity-dependent terms. Meep uses a Padé approximant for
stability (`step_generic.cpp:535-547`):

```
u = (1 + c₂ + 2c₃) / (1 + 2c₂ + 3c₃)
E = u * ε⁻¹ * D
```

where `c₂ = D * χ² * (ε⁻¹)²` and `c₃ = |D|² * χ³ * (ε⁻¹)³`.

Allocate per-voxel arrays in `Geometry.jl` for the nonlinear coefficients,
and add a multiplicative factor in `update_field_generic`.

**Estimated scope:** ~200 lines.

---

## 3.14 Cylindrical Coordinate Kernels — P2

**Present in:** Meep (`Dcyl`, angular mode index `m`)

**Current state:** `abstract type Cylindrical <: Dimension` declared at
`DataStructures.jl:55`, `#TODO add support for cylindrical coordinates` at
`Simulation.jl:52`.

**Implementation guidance:**

Cylindrical FDTD replaces Cartesian curl equations with their cylindrical
equivalents (r, φ, z). For structures with rotational symmetry, the
φ-dependence is `exp(imφ)`, reducing 3D to 2D (r, z) with mode index `m`.

1. New field arrays: `Er, Eφ, Ez, Hr, Hφ, Hz` on a (r, z) grid.
2. Modified curl equations with `1/r` metric factors and `im/r` terms.
3. Special handling at `r = 0` (axis singularity).

**Key references:**
- Meep cylindrical coordinates implementation
- Taflove & Hagness, Ch. 12 (body-of-revolution FDTD)

**Estimated scope:** ~800 lines.

---

## 3.15 TFSF Injection Kernels — P2

**Why it matters:** The source type from §1.8 needs corresponding
injection logic.

**Implementation guidance:**

TFSF injects a plane wave inside a rectangular volume while maintaining
zero incident field outside. On each face of the TFSF box:

- Inner side: add incident E and H to the update equations.
- Outer side: subtract them.

The incident field comes from a 1D auxiliary simulation (or analytical
formula for plane waves) propagated along the k-vector.

**Key references:**
- Taflove & Hagness, Ch. 5.6
- Tidy3D `source/field.py`

**Estimated scope:** ~250 lines.

---

## 3.16 Diffraction Order Computation — P2

**Why it matters:** The monitor type from §1.9 needs FFT-based
computation.

**Implementation guidance:**

For periodic structures, compute power in each diffraction order by taking
the 2D FFT of the tangential DFT fields on a plane:

```julia
E_kx_ky = fft2(E_xy)
```

Extract power at discrete diffraction order wave vectors
`kx = 2πn/Lx`, `ky = 2πm/Ly`.

**Estimated scope:** ~150 lines.

---

## 3.17 LDOS Computation — P2

**Why it matters:** The monitor type from §1.10 needs the actual
computation.

**Implementation guidance:**

```
LDOS(ω) = -(2/π) * Re[E(ω) · J*(ω)] / |J(ω)|²
```

Record the DFT of the E-field at the source location and the DFT of the
source current.

**Key references:**
- Meep `dft_ldos.cpp:60-80`

**Estimated scope:** ~100 lines.

---

## 3.18 Stress Tensor Integration — P2

**Why it matters:** The monitor type from §1.11 needs the actual tensor
computation.

**Implementation guidance:**

Compute the Maxwell stress tensor from DFT fields on a closed surface:

```
T_ij = ε₀(E_iE_j - ½δ_ij|E|²) + μ₀⁻¹(H_iH_j - ½δ_ij|H|²)
```

Integrate over the surface to get the net force.

**Estimated scope:** ~200 lines.

---

## 3.19 Harminv (Resonance Extraction) — P2

**Present in:** Meep (via external harminv library)

**Implementation guidance:**

Use the filter diagonalization method (FDM) or harmonic inversion to
extract resonant frequencies `ωₙ`, decay rates `γₙ`, amplitudes, and
phases from time-domain field data. Implement directly in Julia or wrap
the C harminv library via `ccall`.

**Estimated scope:** ~200 lines.

---

## 3.20 HDF5 / Checkpoint I/O — P3

**Present in:** Meep (HDF5), fdtdx (JAX checkpointing)

**Implementation:** Use `HDF5.jl` to save/load field arrays, simulation
parameters, and monitor data. Enable checkpoint/restart for long
simulations.

**Estimated scope:** ~200 lines.

---

## 3.21 1D Simulations — P3

**Implementation:** When two cell dimensions are zero, set `ndims = 1`
and allocate only 1D field arrays. Add a `OneD` dimension type. Simplify
the curl to a single finite difference per component.

**Estimated scope:** ~100 lines.

---

## 3.22 Energy Computation Kernel — P3

**Implementation:** Compute `u = ½(ε|E|² + μ|H|²)` at monitor locations.
Either time-domain (instantaneous) or DFT-based.

**Estimated scope:** ~80 lines.

---

## 3.23 Lumped Element Update Equations — P3

**Why it matters:** The types from §1.18 need corresponding modified
update equations.

**Implementation:** A resistor is equivalent to a conductivity in a single
voxel; inductors and capacitors require auxiliary state variables with
their own update equations embedded in the E-field update step.

**Estimated scope:** ~150 lines.

---

## 3.24 Time Modulation Kernels — P3

**Why it matters:** The types from §1.17 need runtime modulation of ε and
σ.

**Implementation:** Add a time-dependent multiplicative factor to the
constitutive update. At each timestep, evaluate
`ε_eff(t) = ε₀ * (1 + Δε * sin(ωt + φ))` and update the permittivity
array or pass the modulation factor into the kernel.

**Estimated scope:** ~100 lines.

---

# Cross-Layer Features

Some features span multiple layers. They are listed here with references
to their per-layer components.

| Feature | Frontend | Compiler | Engine |
|---------|----------|----------|--------|
| **Periodic/Bloch BCs** | §1.1 (boundary spec) | — | §3.1 (kernels) |
| **Dispersive Materials** | §1.2 (material types) | §2.2 (kernel selection) | §3.2 (ADE kernels) |
| **Flux Monitors** | §1.3 (monitor type) | — | §3.3 (computation) |
| **Mode Decomposition** | §1.4 (monitor type) | — | §3.4 (overlap) |
| **Near-to-Far Field** | §1.5 (monitor types) | — | §3.10 (Green's fn) |
| **Non-Uniform Grid** | — | §2.5 (planning) | §3.6 (stencils) |
| **Multi-GPU** | — | §2.1 (decomposition) | §3.7 (halo exchange) |
| **Symmetry** | §1.7 (spec) | §2.4 (planning) | — |
| **Adjoint/Inverse Design** | — | §2.7 (compilation) | — |
| **Nonlinear Materials** | §1.6 (types) | §2.2 (kernel selection) | §3.13 (kernels) |
| **TFSF Source** | §1.8 (source type) | — | §3.15 (injection) |
| **Diffraction Monitor** | §1.9 (monitor type) | — | §3.16 (FFT) |
| **LDOS** | §1.10 (monitor type) | — | §3.17 (computation) |
| **Force/Stress** | §1.11 (monitor type) | — | §3.18 (tensor) |
| **Subpixel Averaging** | — | — | §3.5 (geometry init) |
| **CFS-PML** | — | — | §3.8 (coefficients) |
| **Time Modulation** | §1.17 (types) | — | §3.24 (kernels) |
| **Lumped Elements** | §1.18 (types) | — | §3.23 (equations) |

---

# Khronos-Unique Advantages

These are capabilities that Khronos already has or is uniquely positioned
to implement, which the reference solvers lack:

### A. Non-Uniform GPU FDTD (Open-Source First)

No open-source GPU FDTD solver supports non-uniform grids. Khronos's type
system (`Union{Vector{N}, N, Nothing}` for Δx/Δy/Δz) and dispatch-based
kernel specialization make this achievable with minimal code changes.
**Spans:** §2.5 (compiler planning) + §3.6 (engine stencils).

### B. Composable Kernel Specialization

The `Nothing`-vs-`AbstractArray` dispatch pattern in `Timestep.jl`
generates ~32+ specialized GPU kernels from a single source. This is
unique and should be preserved as new features are added — each new
physics module should follow the same pattern. The Graph Compiler (§2.2)
formalizes this into an explicit kernel selection step.

### C. Multi-Backend from Single Source

KernelAbstractions.jl targets CUDA, Metal, ROCm, oneAPI, and CPU from one
codebase. Ensure all new kernels use `@kernel` and `backend_array()`.

### D. Source-Level AD via Enzyme.jl

Unlike fdtdx's manual backward pass, Enzyme.jl can differentiate directly
through GPU kernels. This should be explored once the core feature set
stabilizes. The Graph Compiler (§2.7) manages the forward/adjoint plan
generation.

### E. User-Extensible Physics

Julia's open type system means users can define new material models,
source types, and monitors without modifying Khronos source code. The
three-layer architecture reinforces this: users can extend the Frontend
types and the Engine will dispatch to user-defined kernels.

### F. Runtime Precision Selection

`Float32` / `Float64` selectable per-simulation. Extend to mixed precision
(e.g., Float32 fields + Float64 monitor accumulation) for further GPU
memory savings without sacrificing output accuracy.

### G. Language-Agnostic IR (Future)

The declarative IR (§1.19) enables non-Julia frontends to target the same
compiler and engine. This is unique among FDTD solvers and opens the door
to Python bindings, GUI tools, and config-file-driven workflows.

---

# Performance Optimization

This section tracks GPU kernel performance improvements and remaining
optimization opportunities in the Engine Backend.

---

## Current Performance (A100 80 GB, Float64)

After the Union type annotation fix (see §P.1), measured throughput on
an NVIDIA A100 80 GB SXM (2039 GB/s peak memory bandwidth):

| Benchmark | Throughput | Bandwidth | % Peak |
|-----------|-----------|-----------|--------|
| `step_B_from_E!` (256³, PML) | — | 1661 GB/s | 81% |
| `step_D_from_H!` (256³, PML) | — | 1656 GB/s | 81% |
| `update_H_from_B!` (256³, vacuum) | — | 1365 GB/s | 67% |
| `update_E_from_D!` (256³, with source) | — | 728 GB/s | 36% |
| Full `step!` (256³, PML) | 1747 MCells/s | 1258 GB/s | 62% |

Example throughput (all 3D with PML unless noted):

| Example | Grid | MCells/s | Notes |
|---------|------|----------|-------|
| `throughput_vs_size` | 256³ | 1828 | Peak throughput benchmark |
| `bandwidth_analysis` | 256³ | 1845 | CUDA-event-timed, consistent |
| `throughput_vs_complexity` A (vacuum, no PML) | 128³ | 2965 | Baseline — fastest |
| `throughput_vs_complexity` B (vacuum + PML) | 128³ | 1617 | +σ field updates |
| `throughput_vs_complexity` C (dielectric + PML) | 128³ | 1609 | +ε lookup |
| `throughput_vs_complexity` D (6-layer + PML) | 128³ | 1611 | Same kernel as C |
| `throughput_vs_complexity` E (dielectric + PML + DFT) | 128³ | 1585 | +DFT accumulation |
| `sphere` | 256³ | 2432 | Large vacuum region |
| `planewave` | — | 1690 | Warm steady-state |
| `fresnel_reflectance` | — | 1530 | Physics validation PASS |
| `waveguide` | — | 1333 | Mode source + DFT |
| `gaussian_beam` | — | 1304 | Gaussian beam source |
| `periodic_slab` | — | 1151 | Periodic + DFT |
| `precision_comparison` (f64) | — | 1000 | — |
| `precision_comparison` (f32) | — | 1455 | 1.45× over f64 |
| `dipole` | 32³ | 174 | Tiny grid, launch-limited |
| `anisotropic_slab` | — | — | Physics validation PASS |

---

## P.1 Union Type Annotation Removal — Completed

**Impact:** 4.8× throughput improvement (363 → 1747 MCells/s at 256³).
Memory bandwidth utilization from 13% to 62% of A100 peak.

**Root cause:** The `@kernel` functions `step_curl!` and `update_field!`
in `Timestep.jl` had `::Union{AbstractArray,Nothing}` type annotations
on all parameters. This told GPUCompiler to compile a single generic
kernel handling all Union variants with runtime dispatch, instead of
specializing a concrete kernel per call-site type signature.

**Fix:** Removed all Union type annotations from `step_curl!` and
`update_field!` `@kernel` signatures (parameters are now untyped). Added
`@inline` to all helper functions (`generic_curl!`, `update_field_generic`,
`scale_by_half`, `get_σ`, `get_σD`, `get_m_inv`) to ensure the GPU
compiler inlines through the dispatch chain.

**Composability preserved:** The `Fields` struct still uses
`Union{T,Nothing}` for conditional field allocation — only the fields
required by the active physics are allocated. KernelAbstractions
specializes a separate GPU kernel for each unique combination of concrete
argument types at the call site. The `Nothing`-dispatch composability
pattern works *because of* type removal, not despite it.

---

## P.2 Cache Kernel Objects — Open

**Priority:** P1
**Estimated impact:** ~4% overhead reduction per timestep.

**Problem:** `step_curl!(backend_engine)` and `update_field!(backend_engine)`
create a new KernelAbstractions kernel wrapper object on every call. This
involves looking up the non-`const` module-level global `backend_engine`,
dynamic dispatch on its type, and constructing the kernel closure.

**Fix:** Cache the kernel objects after `prepare_simulation!`. Store
pre-built kernel closures (one per unique type signature) in a field on
`SimulationData` or a module-level cache keyed by type. Alternatively,
declare `backend_engine` as `const` and let the compiler constant-fold
the kernel construction.

---

## P.3 Reset Source Arrays After Shutoff — Open

**Priority:** P1
**Estimated impact:** Up to 1.88× speedup for `update_field!` kernels
after pulsed sources terminate.

**Problem:** When a pulsed source (e.g., `GaussianPulseSource`) turns off,
its source contribution array (`fSDz`, etc.) remains allocated as an
`OffsetArray` filled with zeros. The `update_field!` kernel continues
executing the slow path with asymmetric types (`Sx=Nothing, Sy=Nothing,
Sz=OffsetArray`), increasing register pressure and reducing bandwidth
utilization from 67% to 36% of peak.

**Root cause:** The `Fields` struct is immutable (`@with_kw struct`), so
`fSDz` cannot be set to `nothing` after source termination.

**Fix options:**
1. Make `Fields` a `mutable struct` (simplest, minor allocation overhead).
2. Reconstruct the `Fields` struct with source arrays set to `nothing`
   after `last_source_time(sim)`.
3. Add a `source_active::Bool` flag and pass `nothing` to the kernel
   when sources are inactive (cleanest, no struct change).

---

## P.4 Float64 Contamination in Time Sources — Open

**Priority:** P2
**Estimated impact:** Float32 speedup from 1.45× to ~2× over Float64.

**Problem:** `eval_time_source` for `GaussianPulseData` uses Julia
literals `pi`, `im`, and `exp` which produce `ComplexF64` regardless of
the simulation precision type. When the simulation runs in Float32, every
source evaluation promotes to Float64 and back.

**Fix:** Parameterize the time source evaluation on the simulation's
number type `N`:
```julia
eval_time_source(src::GaussianPulseData, t, N::Type) = ...
    T = real(N)
    ω = T(2) * T(π) * T(src.fcen)
    ...
```

---

## P.5 OffsetArray Indexing Overhead — Open

**Priority:** P2
**Estimated impact:** ~7% overhead per array access on GPU.

**Problem:** PML auxiliary fields use `OffsetArrays` with offset indices.
Each `A[i, j, k]` access on GPU incurs 3 extra integer subtractions
(`i - offset_i`, etc.) that the GPU compiler does not fold away.

**Fix options:**
1. Use raw `CuArray` with manual index arithmetic in the kernel.
2. Pre-subtract offsets and store as a kernel parameter.
3. Replace `OffsetArray` with a thin wrapper that stores the offset as a
   compile-time constant (`Val{N}`) so the GPU compiler can fold it.

---

## P.6 128³ L2 Cache Performance Dip — Open

**Priority:** P3
**Estimated impact:** Localized to 128³ grid size.

**Problem:** At 128³ (130³ with ghost cells), the working set falls in a
pathological middle ground between L2 cache reuse and full GPU memory
bandwidth saturation. Power-of-2 alignment may also cause cache bank
conflicts.

**Fix:** Investigate workgroup size tuning and grid padding for this
specific size. Low priority since production workloads are typically
larger.

---

## P.7 Cold-Start JIT Compilation — Open

**Priority:** P3
**Estimated impact:** `prepare_simulation!` takes ~25s on first call.

**Problem:** The `Fields{T}` constructor with 30+ `Union{T,Nothing}`
fields triggers extensive JIT compilation. Subsequent calls are fast due
to method caching.

**Fix options:**
1. `PrecompileTools.jl` workload that exercises the common field
   configurations during package precompilation.
2. Reduce the number of Union fields by grouping auxiliary arrays into
   sub-structs (e.g., `PMLFields`, `SourceFields`).
