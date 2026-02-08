# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Kernel-level profiling: measure per-kernel timing, launch overhead,
# workgroup size, and achieved memory bandwidth.
#
# Goal: identify why DRAM bandwidth utilization is ~7% of A100 peak.

import Khronos
using CUDA
using KernelAbstractions

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

N = 256
cell_val = Float64(N) / 10.0
dpml = 1.0

sources = [
    Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0],
        size   = [0.0, 0.0, 0.0],
    ),
]

sim = Khronos.Simulation(
    cell_size   = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution  = 10,
    sources     = sources,
    boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
)

Khronos.prepare_simulation!(sim)

# ── 1. Count field arrays and compute bytes per voxel ─────────────────────
println("=" ^ 80)
println("SECTION 1: Field Array Inventory")
println("=" ^ 80)

n_arrays = 0
total_bytes = 0
for fname in fieldnames(typeof(sim.fields))
    f = getfield(sim.fields, fname)
    if f !== nothing && isa(f, AbstractArray)
        global n_arrays += 1
        nbytes = length(f) * sizeof(eltype(f))
        global total_bytes += nbytes
        println("  $(rpad(fname, 12))  $(size(f))  $(round(nbytes / 1024^2, digits=1)) MB")
    else
        println("  $(rpad(fname, 12))  nothing")
    end
end

println()
println("Total arrays: $n_arrays")
println("Total GPU memory: $(round(total_bytes / 1024^2, digits=1)) MB")
println("Voxels: $(sim.Nx) x $(sim.Ny) x $(sim.Nz) = $(sim.Nx * sim.Ny * sim.Nz)")
println()

# ── 2. Warmup ─────────────────────────────────────────────────────────────
println("Warming up...")
for _ in 1:30
    Khronos.step!(sim)
end
CUDA.synchronize()

# ── 3. Per-kernel timing with CUDA events ─────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 2: Per-Kernel Timing (CUDA Events)")
println("=" ^ 80)

n_measure = 100

# Helper: time a function with CUDA events
function cuda_time(f, n_reps)
    CUDA.synchronize()
    start_ev = CUDA.CuEvent()
    stop_ev = CUDA.CuEvent()
    CUDA.record(start_ev)
    for _ in 1:n_reps
        f()
    end
    CUDA.record(stop_ev)
    CUDA.synchronize()
    return CUDA.elapsed(start_ev, stop_ev) / n_reps
end

# Time the full step
t_full_step = cuda_time(() -> Khronos.step!(sim), n_measure)
println("Full step!():          $(round(t_full_step * 1e6, digits=1)) μs")
println()

# Time individual components by monkey-patching
# We can't directly instrument Khronos internals, so we'll call the
# individual functions directly.

t = Khronos.round_time(sim)

# Magnetic sources
t_mag_src = cuda_time(() -> Khronos.update_magnetic_sources!(sim, t), n_measure)

# step_B_from_E
t_step_B = cuda_time(() -> Khronos.step_B_from_E!(sim), n_measure)

# update_H_from_B
t_update_H = cuda_time(() -> Khronos.update_H_from_B!(sim), n_measure)

# H monitors
t_H_mon = cuda_time(() -> Khronos.update_H_monitors!(sim, t), n_measure)

# Electric sources
t_elec_src = cuda_time(() -> Khronos.update_electric_sources!(sim, t + sim.Δt/2), n_measure)

# step_D_from_H
t_step_D = cuda_time(() -> Khronos.step_D_from_H!(sim), n_measure)

# update_E_from_D
t_update_E = cuda_time(() -> Khronos.update_E_from_D!(sim), n_measure)

# E monitors
t_E_mon = cuda_time(() -> Khronos.update_E_monitors!(sim, t + sim.Δt/2), n_measure)

# increment_timestep (CPU only)
t_inc = @elapsed for _ in 1:n_measure; Khronos.increment_timestep!(sim); end
t_inc /= n_measure

kernel_times = [
    ("update_magnetic_sources!", t_mag_src),
    ("step_B_from_E!",          t_step_B),
    ("update_H_from_B!",       t_update_H),
    ("update_H_monitors!",     t_H_mon),
    ("update_electric_sources!", t_elec_src),
    ("step_D_from_H!",         t_step_D),
    ("update_E_from_D!",       t_update_E),
    ("update_E_monitors!",     t_E_mon),
    ("increment_timestep!",    t_inc),
]

total_parts = sum(t for (_, t) in kernel_times)
println("Per-kernel breakdown:")
println("  $(rpad("Kernel", 30))  $(rpad("Time (μs)", 12))  $(rpad("% of step", 10))")
println("  " * "-" ^ 55)
for (name, t) in kernel_times
    pct = t / t_full_step * 100
    println("  $(rpad(name, 30))  $(rpad(round(t * 1e6, digits=1), 12))  $(rpad(round(pct, digits=1), 10))%")
end
println("  " * "-" ^ 55)
println("  $(rpad("Sum of parts", 30))  $(rpad(round(total_parts * 1e6, digits=1), 12))  $(round(total_parts / t_full_step * 100, digits=1))%")
println("  $(rpad("Full step! (measured)", 30))  $(rpad(round(t_full_step * 1e6, digits=1), 12))  100.0%")
overhead_pct = (t_full_step - total_parts) / t_full_step * 100
println("\n  Unaccounted overhead: $(round(overhead_pct, digits=1))% (Julia dispatch, kernel object creation, etc.)")

# ── 4. Bandwidth analysis ────────────────────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 3: Achieved Memory Bandwidth")
println("=" ^ 80)

voxels = sim.Nx * sim.Ny * sim.Nz

# Compute bytes read/written per voxel for each major kernel.
# step_curl! (PML case, per component):
#   Reads:  3 source field values (stencil neighbors) = 3 * 8 = 24 bytes
#   Reads:  C[idx], U[idx], T[idx] = 3 * 8 = 24 bytes
#   Reads:  σD[idx] = 8 bytes, σ_next/σ_prev = ~0 (1D, cached)
#   Writes: C[idx], U[idx], T[idx] = 3 * 8 = 24 bytes
#   Per component: 56 bytes read + 24 bytes write = 80 bytes
#   × 3 components = 240 bytes/voxel
#
# But for the stencil, each component reads 4 source values (2 per derivative):
#   curl_x: Ay[ix,iy,iz], Ay[ix,iy,iz+1], Az[ix,iy,iz], Az[ix,iy+1,iz] = 4 reads
#   curl_y: Az[ix,iy,iz], Az[ix+1,iy,iz], Ax[ix,iy,iz], Ax[ix,iy,iz+1] = 2 new
#   curl_z: Ax[ix,iy,iz], Ax[ix,iy+1,iz], Ay[ix,iy,iz], Ay[ix+1,iy,iz] = 2 new
#   Total unique source reads: ~9 values × 8 bytes = 72 bytes
#   (cache reuse expected for repeated A[ix,iy,iz] accesses)
#
# Plus per-component PML aux: 3 × (read C + read U + read T + write C + write U + write T)
#   = 3 × (3×8 read + 3×8 write) = 3 × 48 = 144 bytes
# Plus σD: 3 × 8 = 24 bytes
# Total step_curl!: 72 + 144 + 24 = 240 bytes/voxel

bytes_step_curl = 240  # bytes per voxel

# update_field! (PML case, per component):
#   Read:  A[idx], T[idx], W[idx] = 3 × 8 = 24 bytes
#   Read:  m_inv (scalar, cached) + σ (1D, cached) ≈ 0
#   Write: W[idx], A[idx] = 2 × 8 = 16 bytes
#   Per component: 40 bytes, × 3 components = 120 bytes/voxel
# (with S and P being Nothing, those are skipped)

bytes_update_field = 120  # bytes per voxel

bytes_per_step = 2 * bytes_step_curl + 2 * bytes_update_field  # 2 curl + 2 update

println("Estimated bytes per voxel per step: $bytes_per_step")
println("  step_curl!:     $(bytes_step_curl) bytes/voxel × 2 = $(2 * bytes_step_curl)")
println("  update_field!:  $(bytes_update_field) bytes/voxel × 2 = $(2 * bytes_update_field)")
println()

# Per-kernel achieved bandwidth
bw_step_B = voxels * bytes_step_curl / t_step_B / 1e9
bw_update_H = voxels * bytes_update_field / t_update_H / 1e9
bw_step_D = voxels * bytes_step_curl / t_step_D / 1e9
bw_update_E = voxels * bytes_update_field / t_update_E / 1e9
bw_total = voxels * bytes_per_step / t_full_step / 1e9

gpu_name = CUDA.name(CUDA.device())
println("GPU: $gpu_name")
# A100 SXM4 80GB peak BW = 2039 GB/s
# A100 PCIe 80GB peak BW = 1935 GB/s
peak_bw = 2039.0
println("A100 SXM4 peak DRAM BW: $(peak_bw) GB/s")
println()

println("Per-kernel achieved bandwidth:")
println("  $(rpad("Kernel", 25))  $(rpad("GB/s", 10))  $(rpad("% of peak", 10))")
println("  " * "-" ^ 50)
println("  $(rpad("step_B_from_E!", 25))  $(rpad(round(bw_step_B, digits=1), 10))  $(round(bw_step_B/peak_bw*100, digits=1))%")
println("  $(rpad("update_H_from_B!", 25))  $(rpad(round(bw_update_H, digits=1), 10))  $(round(bw_update_H/peak_bw*100, digits=1))%")
println("  $(rpad("step_D_from_H!", 25))  $(rpad(round(bw_step_D, digits=1), 10))  $(round(bw_step_D/peak_bw*100, digits=1))%")
println("  $(rpad("update_E_from_D!", 25))  $(rpad(round(bw_update_E, digits=1), 10))  $(round(bw_update_E/peak_bw*100, digits=1))%")
println("  " * "-" ^ 50)
println("  $(rpad("Full step (aggregate)", 25))  $(rpad(round(bw_total, digits=1), 10))  $(round(bw_total/peak_bw*100, digits=1))%")

# ── 5. Launch overhead analysis ───────────────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 4: Kernel Launch Overhead Analysis")
println("=" ^ 80)

# Time an empty CUDA kernel to measure pure launch overhead
@kernel function empty_kernel!()
    ix, iy, iz = @index(Global, NTuple)
end

empty_k = empty_kernel!(CUDABackend())
t_empty = cuda_time(() -> empty_k(ndrange=(N, N, N)), n_measure)
println("Empty kernel launch: $(round(t_empty * 1e6, digits=1)) μs")

# Count total kernel launches per step
# Main kernels: 4 (step_B, update_H, step_D, update_E)
# Source kernels: depends on sources. With 1 Ez source:
#   update_magnetic_sources! calls step_sources! for Hx, Hy, Hz → 0 launches (no H sources)
#   update_electric_sources! calls step_sources! for Ex, Ey, Ez → 1 launch (Ez source)
# Monitor kernels: depends on monitors. For benchmark with no DFT monitors: 0
n_main_kernels = 4
n_source_kernels = 1  # 1 Ez source
n_monitor_kernels = length(sim.monitor_data)
n_total_kernels = n_main_kernels + n_source_kernels + n_monitor_kernels

println("Kernel launches per step: $n_total_kernels")
println("  Main FDTD kernels: $n_main_kernels")
println("  Source kernels:    $n_source_kernels")
println("  Monitor kernels:  $n_monitor_kernels")
println("Est. launch overhead per step: $(round(n_total_kernels * t_empty * 1e6, digits=1)) μs ($(round(n_total_kernels * t_empty / t_full_step * 100, digits=1))%)")

# ── 6. Workgroup size check ───────────────────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 5: Workgroup Size & Occupancy")
println("=" ^ 80)

# KernelAbstractions doesn't expose workgroup size easily, but we can
# check what CUDA sees. Let's use CUDA's occupancy API.
# First, let's see what KA picks by probing the compiled kernel.

# We can check the CUDA kernel's attributes after launching
println("Grid dimensions (ndrange): $(sim.Nx) × $(sim.Ny) × $(sim.Nz)")
println()

# Try to get kernel info from CUDA's perspective
# Launch a kernel and then inspect CUDA state
println("Checking CUDA kernel attributes...")
try
    # Get the actual CUDA function from a KA kernel
    curl_kernel = Khronos.step_curl!(CUDABackend())

    # The KA kernel doesn't directly expose the CuFunction, but we can
    # check occupancy via CUDA's runtime API by timing with different
    # explicit workgroup sizes
    println("\nWorkgroup size scan (step_curl! kernel):")
    println("  $(rpad("Workgroup", 12))  $(rpad("Time (μs)", 12))  $(rpad("Rel. perf", 10))")
    println("  " * "-" ^ 40)

    # Time the default (KA-chosen) workgroup size
    t_default = cuda_time(() -> Khronos.step_B_from_E!(sim), 50)
    println("  $(rpad("default", 12))  $(rpad(round(t_default * 1e6, digits=1), 12))  1.00x")

    # Try explicit workgroup sizes by directly creating kernels
    for ws in [64, 128, 256, 512, 1024]
        try
            kern = Khronos.step_curl!(CUDABackend(), workgroupsize=ws)
            t_ws = cuda_time(() -> kern(
                sim.fields.fEx, sim.fields.fEy, sim.fields.fEz,
                sim.fields.fBx, sim.fields.fBy, sim.fields.fBz,
                sim.fields.fCBx, sim.fields.fCBy, sim.fields.fCBz,
                sim.fields.fUBx, sim.fields.fUBy, sim.fields.fUBz,
                sim.geometry_data.σBx, sim.geometry_data.σBy, sim.geometry_data.σBz,
                sim.boundary_data.σBx, sim.boundary_data.σBy, sim.boundary_data.σBz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, 1,
                ndrange = (sim.Nx, sim.Ny, sim.Nz),
            ), 50)
            speedup = t_default / t_ws
            println("  $(rpad(ws, 12))  $(rpad(round(t_ws * 1e6, digits=1), 12))  $(round(speedup, digits=2))x")
        catch e
            println("  $(rpad(ws, 12))  FAILED: $(typeof(e))")
        end
    end

    # Also try 3D workgroup sizes
    println("\n3D Workgroup size scan:")
    println("  $(rpad("Workgroup", 20))  $(rpad("Time (μs)", 12))  $(rpad("Rel. perf", 10))")
    println("  " * "-" ^ 50)
    for (wx, wy, wz) in [(8,8,4), (8,8,8), (4,4,16), (16,16,1), (32,8,1), (8,4,8), (4,8,8)]
        try
            kern = Khronos.step_curl!(CUDABackend(), workgroupsize=(wx, wy, wz))
            t_ws = cuda_time(() -> kern(
                sim.fields.fEx, sim.fields.fEy, sim.fields.fEz,
                sim.fields.fBx, sim.fields.fBy, sim.fields.fBz,
                sim.fields.fCBx, sim.fields.fCBy, sim.fields.fCBz,
                sim.fields.fUBx, sim.fields.fUBy, sim.fields.fUBz,
                sim.geometry_data.σBx, sim.geometry_data.σBy, sim.geometry_data.σBz,
                sim.boundary_data.σBx, sim.boundary_data.σBy, sim.boundary_data.σBz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, 1,
                ndrange = (sim.Nx, sim.Ny, sim.Nz),
            ), 50)
            speedup = t_default / t_ws
            println("  $(rpad("($wx,$wy,$wz)", 20))  $(rpad(round(t_ws * 1e6, digits=1), 12))  $(round(speedup, digits=2))x")
        catch e
            println("  $(rpad("($wx,$wy,$wz)", 20))  FAILED: $(typeof(e))")
        end
    end
catch e
    println("Workgroup scan failed: $e")
end

# ── 7. Memory access pattern analysis ─────────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 6: Memory Access Pattern Analysis")
println("=" ^ 80)

println("""
Key observations from code review:

1. STENCIL ACCESS PATTERN (step_curl!):
   - curl_x reads: Ay[ix,iy,iz+1]-Ay[ix,iy,iz], Az[ix,iy+1,iz]-Az[ix,iy,iz]
   - curl_y reads: Az[ix+1,iy,iz]-Az[ix,iy,iz], Ax[ix,iy,iz+1]-Ax[ix,iy,iz]
   - curl_z reads: Ax[ix,iy+1,iz]-Ax[ix,iy,iz], Ay[ix+1,iy,iz]-Ay[ix,iy,iz]

   With column-major (Julia/Fortran) layout and (Nx+2, Ny+2, Nz+2) arrays:
   - A[ix+1, iy, iz] is STRIDE-1 neighbor → COALESCED (good)
   - A[ix, iy+1, iz] is STRIDE-(Nx+2) neighbor → STRIDED (bad for L1)
   - A[ix, iy, iz+1] is STRIDE-(Nx+2)*(Ny+2) neighbor → VERY STRIDED (bad)

   For N=256: stride-y = 258, stride-z = 258*258 = 66564
   Reading A[ix,iy,iz+1] jumps 66564*8 ≈ 520 KB — far exceeds L1 cache.

2. PML σ ACCESS (get_σ):
   - σ[2*idx-1]: stride-2 access on 1D arrays → wastes 50% of cache lines
   - But these are small 1D arrays (N elements) → likely fit in L2 cache

3. OFFSETARRAY OVERHEAD:
   - Every array access goes through: OffsetArray → parent GPU array
   - The offset computation (index - offset) happens for every access
   - With Julia's GPU compiler, this SHOULD be optimized away, but verify

4. CARTESIANINDEX USAGE:
   - CartesianIndex(ix,iy,iz) created per thread, used for all auxiliary accesses
   - This means auxiliary arrays (C, U, T, W, P, S) use 3D indexing
   - The stencil reads use explicit (ix, iy+1, iz) indexing

5. REGISTER PRESSURE:
   - step_curl! takes 22 arguments + computes 3 components with temporaries
   - Each component of generic_curl! uses C_old, U_old temporaries
   - Total per thread: ~22 argument registers + ~12 working registers
   - If >32 registers used, occupancy drops; if >64, severe spilling likely

6. KERNEL FUSION OPPORTUNITY:
   - Currently 4 separate kernel launches per step
   - Each launch reads/writes overlapping arrays (E shared between curl and update)
   - Fusing step_curl! + update_field! could halve the DRAM traffic
   - But would require careful dependency management (B→H, then D→E)
""")

# ── 8. Array size analysis ────────────────────────────────────────────────
println("=" ^ 80)
println("SECTION 7: Working Set vs Cache Hierarchy")
println("=" ^ 80)

array_size_bytes = (N + 2)^3 * 8
println("Single array size: $(round(array_size_bytes / 1024^2, digits=1)) MB  (($(N+2))^3 × 8 bytes)")
println("Total arrays: $n_arrays")
println("Total working set: $(round(n_arrays * array_size_bytes / 1024^2, digits=0)) MB")
println()

# A100 cache hierarchy:
# L1: 192 KB per SM (shared with shared memory), 108 SMs
# L2: 40 MB total
# HBM2e: 80 GB, 2039 GB/s
println("A100 cache hierarchy:")
println("  L1 per SM:   192 KB (108 SMs)")
println("  L2 total:    40 MB")
println("  HBM2e:       80 GB @ 2039 GB/s")
println()

# For step_curl!, how many arrays are touched?
# Source reads: Ex, Ey, Ez (or Hx, Hy, Hz) = 3 arrays
# Aux reads+writes: Cx, Cy, Cz, Ux, Uy, Uz = 6 arrays
# Target writes: Bx, By, Bz (or Dx, Dy, Dz) = 3 arrays (also read)
# σD: 3 arrays, σ boundary: 3 arrays (1D, small)
# Total: 15 3D arrays × $(round(array_size_bytes / 1024^2, digits=1)) MB = $(round(15 * array_size_bytes / 1024^2, digits=0)) MB
println("step_curl! touches ~15 3D arrays = $(round(15 * array_size_bytes / 1024^2, digits=0)) MB")
println("  → $(round(15 * array_size_bytes / 1024^2 / 40, digits=1))x the L2 cache")
println()
println("update_field! touches ~9 3D arrays = $(round(9 * array_size_bytes / 1024^2, digits=0)) MB")
println("  → $(round(9 * array_size_bytes / 1024^2 / 40, digits=1))x the L2 cache")

# ── 9. Throughput summary ────────────────────────────────────────────────
println()
println("=" ^ 80)
println("SECTION 8: Summary & Root Cause Hypothesis")
println("=" ^ 80)

mcells_s = voxels / t_full_step / 1e6
println("Achieved throughput: $(round(mcells_s, digits=0)) MCells/s")
println("Achieved aggregate bandwidth: $(round(bw_total, digits=1)) GB/s")
println("Peak DRAM bandwidth: $(peak_bw) GB/s")
println("Bandwidth utilization: $(round(bw_total / peak_bw * 100, digits=1))%")
println()
println("If we achieved 80% of peak BW ($(round(0.8 * peak_bw, digits=0)) GB/s):")
expected_mcells = 0.8 * peak_bw * 1e9 / bytes_per_step / 1e6
println("  → Expected throughput: $(round(expected_mcells, digits=0)) MCells/s")
println("  → Current: $(round(mcells_s, digits=0)) MCells/s")
println("  → Potential speedup: $(round(expected_mcells / mcells_s, digits=1))x")
