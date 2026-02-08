# Deep-dive: why is update_field! 4x slower in bandwidth than step_curl!?
#
# Tests:
# 1. Measure raw DRAM bandwidth with a simple copy kernel
# 2. Compare KernelAbstractions overhead vs raw CUDA kernels
# 3. 3D indexing and OffsetArray overhead
# 4. Stencil access patterns (stride analysis)
# 5. update_field! pattern isolation
# 6. Register pressure / occupancy

import Khronos
using CUDA
using KernelAbstractions
using OffsetArrays

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

const N = 256
const Nx = N; const Ny = N; const Nz = N
const n_elements = (N + 2)^3
const dpml = 1.0

function cuda_time_events(f, n_reps; n_warmup=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    start_ev = CUDA.CuEvent()
    stop_ev = CUDA.CuEvent()
    CUDA.record(start_ev)
    for _ in 1:n_reps; f(); end
    CUDA.record(stop_ev)
    CUDA.synchronize()
    return CUDA.elapsed(start_ev, stop_ev) / n_reps
end

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Raw DRAM Bandwidth Baseline (using CUDA.jl @cuda)
# ═══════════════════════════════════════════════════════════════════════════
println("=" ^ 80)
println("SECTION 1: Raw DRAM Bandwidth Baseline")
println("=" ^ 80)

# Use CUDA.jl's built-in copyto! and broadcasting for baseline
A1d = CUDA.rand(Float64, n_elements)
B1d = CUDA.rand(Float64, n_elements)
C1d = CUDA.rand(Float64, n_elements)

# Copy via copyto! (uses optimized CUDA memcpy)
t_memcpy = cuda_time_events(() -> copyto!(B1d, A1d), 200)
bw_memcpy = 2 * n_elements * 8 / t_memcpy / 1e9
println("cudaMemcpy (1R+1W):   $(round(bw_memcpy, digits=1)) GB/s ($(round(t_memcpy*1e6, digits=1)) μs)")

# SAXPY via broadcasting
t_bcast = cuda_time_events(() -> (C1d .= 2.0 .* A1d .+ B1d), 200)
bw_bcast = 3 * n_elements * 8 / t_bcast / 1e9
println("Broadcast SAXPY:      $(round(bw_bcast, digits=1)) GB/s ($(round(t_bcast*1e6, digits=1)) μs)")

# 5-array sum via broadcasting
D1d = CUDA.rand(Float64, n_elements)
E1d = CUDA.rand(Float64, n_elements)
F1d = CUDA.rand(Float64, n_elements)
t_5sum = cuda_time_events(() -> (F1d .= A1d .+ B1d .+ C1d .+ D1d .+ E1d), 200)
bw_5sum = 6 * n_elements * 8 / t_5sum / 1e9
println("Broadcast 5-sum:      $(round(bw_5sum, digits=1)) GB/s ($(round(t_5sum*1e6, digits=1)) μs)")

# 9R+6W pattern via broadcasting (matches update_field! data volume)
G1d = CUDA.rand(Float64, n_elements)
H1d = CUDA.rand(Float64, n_elements)
I1d = CUDA.rand(Float64, n_elements)
J1d = CUDA.rand(Float64, n_elements)
K1d = CUDA.rand(Float64, n_elements)
L1d = CUDA.rand(Float64, n_elements)
M1d = CUDA.rand(Float64, n_elements)
N1d_ = CUDA.rand(Float64, n_elements)
O1d = CUDA.rand(Float64, n_elements)

function bcast_15arr!()
    # 9 reads + 6 writes: simulate 3-component update
    # Comp X: read Gx,Hx,Ix → write Gx,Ix
    I1d .= G1d .+ H1d .+ I1d
    G1d .= G1d .+ I1d
    # Comp Y: read Jy,Ky,Ly → write Jy,Ly
    L1d .= J1d .+ K1d .+ L1d
    J1d .= J1d .+ L1d
    # Comp Z: read Mz,Nz,Oz → write Mz,Oz
    O1d .= M1d .+ N1d_ .+ O1d
    M1d .= M1d .+ O1d
    return
end
t_15arr = cuda_time_events(bcast_15arr!, 200)
bw_15arr = 15 * n_elements * 8 / t_15arr / 1e9  # Actually 18 ops: 12R + 6W per element
println("Broadcast 15-array:   $(round(bw_15arr, digits=1)) GB/s ($(round(t_15arr*1e6, digits=1)) μs)")

println()
println("This shows achievable peak bandwidth on this GPU.")
println("GPU: $(CUDA.name(CUDA.device()))")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: KernelAbstractions Overhead
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 2: KernelAbstractions vs CUDA Broadcasting")
println("=" ^ 80)

@kernel function ka_copy!(dst, src)
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i]
end

@kernel function ka_saxpy!(dst, src1, src2, alpha::Float64)
    i = @index(Global, Linear)
    @inbounds dst[i] = alpha * src1[i] + src2[i]
end

A_ka = CUDA.rand(Float64, n_elements)
B_ka = CUDA.rand(Float64, n_elements)
C_ka = CUDA.rand(Float64, n_elements)

ka_copy_k = ka_copy!(CUDABackend())
ka_saxpy_k = ka_saxpy!(CUDABackend())

t_ka_copy = cuda_time_events(() -> ka_copy_k(B_ka, A_ka, ndrange=n_elements), 200)
bw_ka_copy = 2 * n_elements * 8 / t_ka_copy / 1e9
println("KA Copy (1R+1W):      $(round(bw_ka_copy, digits=1)) GB/s ($(round(t_ka_copy*1e6, digits=1)) μs)")

t_ka_saxpy = cuda_time_events(() -> ka_saxpy_k(C_ka, A_ka, B_ka, 2.0, ndrange=n_elements), 200)
bw_ka_saxpy = 3 * n_elements * 8 / t_ka_saxpy / 1e9
println("KA SAXPY (2R+1W):     $(round(bw_ka_saxpy, digits=1)) GB/s ($(round(t_ka_saxpy*1e6, digits=1)) μs)")

println()
println("KA copy overhead vs memcpy: $(round((t_ka_copy - t_memcpy) / t_memcpy * 100, digits=1))%")
println("KA copy overhead vs broadcast: $(round((t_ka_copy - t_bcast) / t_bcast * 100, digits=1))% (different op, just timing ref)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: 3D Indexing & OffsetArray Overhead
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 3: 3D Indexing & OffsetArray Overhead")
println("=" ^ 80)

A_3d = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)
B_3d = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)

# KA 3D copy with plain arrays
@kernel function ka_3d_copy_plain!(dst, src)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds dst[ix+1, iy+1, iz+1] = src[ix+1, iy+1, iz+1]
end

ka_3d_plain = ka_3d_copy_plain!(CUDABackend())
t_3d_plain = cuda_time_events(() -> ka_3d_plain(B_3d, A_3d, ndrange=(Nx, Ny, Nz)), 200)
bw_3d_plain = 2 * Nx * Ny * Nz * 8 / t_3d_plain / 1e9
println("KA 3D plain (ix+1):  $(round(bw_3d_plain, digits=1)) GB/s ($(round(t_3d_plain*1e6, digits=1)) μs)")

# KA 3D copy with OffsetArray + CartesianIndex (Khronos pattern)
A_off = OffsetArray(CUDA.rand(Float64, Nx+2, Ny+2, Nz+2), -1, -1, -1)
B_off = OffsetArray(CUDA.rand(Float64, Nx+2, Ny+2, Nz+2), -1, -1, -1)

@kernel function ka_3d_copy_offset!(dst, src)
    ix, iy, iz = @index(Global, NTuple)
    idx = CartesianIndex(ix, iy, iz)
    @inbounds dst[idx] = src[idx]
end

ka_3d_off = ka_3d_copy_offset!(CUDABackend())
t_3d_off = cuda_time_events(() -> ka_3d_off(B_off, A_off, ndrange=(Nx, Ny, Nz)), 200)
bw_3d_off = 2 * Nx * Ny * Nz * 8 / t_3d_off / 1e9
println("KA 3D OffsetArray+CI: $(round(bw_3d_off, digits=1)) GB/s ($(round(t_3d_off*1e6, digits=1)) μs)")

println("\nOffsetArray+CI overhead: $(round((t_3d_off - t_3d_plain) / t_3d_plain * 100, digits=1))%")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Stencil Access Patterns (using KA)
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 4: Stencil Access Patterns")
println("=" ^ 80)

src_3d = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)
dst_3d = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)
src2_3d = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)

# Stride-1 stencil (d/dx — coalesced in Julia column-major)
@kernel function ka_stencil_x!(dst, src)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds dst[ix+1, iy+1, iz+1] = src[ix+2, iy+1, iz+1] - src[ix+1, iy+1, iz+1]
end

# Stride-Nx stencil (d/dy)
@kernel function ka_stencil_y!(dst, src)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds dst[ix+1, iy+1, iz+1] = src[ix+1, iy+2, iz+1] - src[ix+1, iy+1, iz+1]
end

# Stride-Nx*Ny stencil (d/dz — worst case)
@kernel function ka_stencil_z!(dst, src)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds dst[ix+1, iy+1, iz+1] = src[ix+1, iy+1, iz+2] - src[ix+1, iy+1, iz+1]
end

# Full curl: 4 reads from 2 arrays + 1 write
@kernel function ka_full_curl!(dst, src1, src2)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        Kx = (src1[ix+1, iy+1, iz+2] - src1[ix+1, iy+1, iz+1]) -
             (src2[ix+1, iy+2, iz+1] - src2[ix+1, iy+1, iz+1])
        dst[ix+1, iy+1, iz+1] = Kx
    end
end

bytes_stencil = 3 * Nx * Ny * Nz * 8  # 2R + 1W

k_sx = ka_stencil_x!(CUDABackend())
k_sy = ka_stencil_y!(CUDABackend())
k_sz = ka_stencil_z!(CUDABackend())
k_curl = ka_full_curl!(CUDABackend())

t_sx = cuda_time_events(() -> k_sx(dst_3d, src_3d, ndrange=(Nx,Ny,Nz)), 200)
t_sy = cuda_time_events(() -> k_sy(dst_3d, src_3d, ndrange=(Nx,Ny,Nz)), 200)
t_sz = cuda_time_events(() -> k_sz(dst_3d, src_3d, ndrange=(Nx,Ny,Nz)), 200)

println("Stencil d/dx (stride-1):        $(round(bytes_stencil/t_sx/1e9, digits=1)) GB/s  ($(round(t_sx*1e6, digits=1)) μs)")
println("Stencil d/dy (stride-$(Nx+2)):     $(round(bytes_stencil/t_sy/1e9, digits=1)) GB/s  ($(round(t_sy*1e6, digits=1)) μs)")
println("Stencil d/dz (stride-$((Nx+2)*(Ny+2))): $(round(bytes_stencil/t_sz/1e9, digits=1)) GB/s  ($(round(t_sz*1e6, digits=1)) μs)")

bytes_curl = 5 * Nx * Ny * Nz * 8  # 4R + 1W (with cache reuse of src[ix+1,iy+1,iz+1])
t_curl_v = cuda_time_events(() -> k_curl(dst_3d, src_3d, src2_3d, ndrange=(Nx,Ny,Nz)), 200)
println("Full curl (4R+1W, 2 arrays):    $(round(bytes_curl/t_curl_v/1e9, digits=1)) GB/s  ($(round(t_curl_v*1e6, digits=1)) μs)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: update_field! Pattern Isolation (using KA)
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 5: update_field! Pattern Isolation")
println("=" ^ 80)

# 1-component PML update via KA
@kernel function ka_update_1comp!(A, T, W, sigma::Float64, m_inv::Float64)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        W_old = W[ix+1, iy+1, iz+1]
        net_field = T[ix+1, iy+1, iz+1]
        W[ix+1, iy+1, iz+1] = m_inv * net_field
        A[ix+1, iy+1, iz+1] = A[ix+1, iy+1, iz+1] +
            (1.0 + sigma) * W[ix+1, iy+1, iz+1] - (1.0 - sigma) * W_old
    end
end

# 3-component PML update via KA (Khronos pattern)
@kernel function ka_update_3comp!(
    Ax, Ay, Az, Tx, Ty, Tz, Wx, Wy, Wz,
    sigma::Float64, m_inv::Float64)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        # X component
        W_old = Wx[ix+1, iy+1, iz+1]
        Wx[ix+1, iy+1, iz+1] = m_inv * Tx[ix+1, iy+1, iz+1]
        Ax[ix+1, iy+1, iz+1] = Ax[ix+1, iy+1, iz+1] +
            (1.0 + sigma) * Wx[ix+1, iy+1, iz+1] - (1.0 - sigma) * W_old

        # Y component
        W_old = Wy[ix+1, iy+1, iz+1]
        Wy[ix+1, iy+1, iz+1] = m_inv * Ty[ix+1, iy+1, iz+1]
        Ay[ix+1, iy+1, iz+1] = Ay[ix+1, iy+1, iz+1] +
            (1.0 + sigma) * Wy[ix+1, iy+1, iz+1] - (1.0 - sigma) * W_old

        # Z component
        W_old = Wz[ix+1, iy+1, iz+1]
        Wz[ix+1, iy+1, iz+1] = m_inv * Tz[ix+1, iy+1, iz+1]
        Az[ix+1, iy+1, iz+1] = Az[ix+1, iy+1, iz+1] +
            (1.0 + sigma) * Wz[ix+1, iy+1, iz+1] - (1.0 - sigma) * W_old
    end
end

# Full step_curl!-like kernel via KA (all 3 components, no PML for simplicity)
@kernel function ka_curl_3comp!(Tx, Ty, Tz, Ax, Ay, Az, dt::Float64, dx::Float64, dy::Float64, dz::Float64)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        inv_dx = 1.0 / dx; inv_dy = 1.0 / dy; inv_dz = 1.0 / dz

        # curl_x = dAy/dz - dAz/dy
        Kx = dt * (inv_dz * (Ay[ix+1, iy+1, iz+2] - Ay[ix+1, iy+1, iz+1]) -
                    inv_dy * (Az[ix+1, iy+2, iz+1] - Az[ix+1, iy+1, iz+1]))
        Tx[ix+1, iy+1, iz+1] = Tx[ix+1, iy+1, iz+1] + Kx

        # curl_y = dAz/dx - dAx/dz
        Ky = dt * (inv_dx * (Az[ix+2, iy+1, iz+1] - Az[ix+1, iy+1, iz+1]) -
                    inv_dz * (Ax[ix+1, iy+1, iz+2] - Ax[ix+1, iy+1, iz+1]))
        Ty[ix+1, iy+1, iz+1] = Ty[ix+1, iy+1, iz+1] + Ky

        # curl_z = dAx/dy - dAy/dx
        Kz = dt * (inv_dy * (Ax[ix+1, iy+2, iz+1] - Ax[ix+1, iy+1, iz+1]) -
                    inv_dx * (Ay[ix+2, iy+1, iz+1] - Ay[ix+1, iy+1, iz+1]))
        Tz[ix+1, iy+1, iz+1] = Tz[ix+1, iy+1, iz+1] + Kz
    end
end

# Allocate test arrays
test_arrays = [CUDA.rand(Float64, Nx+2, Ny+2, Nz+2) for _ in 1:12]

bytes_1c = 5 * Nx * Ny * Nz * 8   # 3R + 2W
bytes_3c = 15 * Nx * Ny * Nz * 8  # 9R + 6W

k_upd1 = ka_update_1comp!(CUDABackend())
k_upd3 = ka_update_3comp!(CUDABackend())
k_curl3 = ka_curl_3comp!(CUDABackend())

t_upd1 = cuda_time_events(() -> k_upd1(test_arrays[1], test_arrays[2], test_arrays[3],
    0.1, 1.0, ndrange=(Nx,Ny,Nz)), 200)
println("KA 1-comp update (3R+2W):   $(round(bytes_1c/t_upd1/1e9, digits=1)) GB/s  ($(round(t_upd1*1e6, digits=1)) μs)")

t_upd3 = cuda_time_events(() -> k_upd3(
    test_arrays[1], test_arrays[2], test_arrays[3],
    test_arrays[4], test_arrays[5], test_arrays[6],
    test_arrays[7], test_arrays[8], test_arrays[9],
    0.1, 1.0, ndrange=(Nx,Ny,Nz)), 200)
println("KA 3-comp update (9R+6W):   $(round(bytes_3c/t_upd3/1e9, digits=1)) GB/s  ($(round(t_upd3*1e6, digits=1)) μs)")

# Curl: 9 unique reads + 3 read-modify-writes = 12R + 3W = 15 arrays × voxel × 8
bytes_curl3 = 15 * Nx * Ny * Nz * 8
t_curl3 = cuda_time_events(() -> k_curl3(
    test_arrays[1], test_arrays[2], test_arrays[3],
    test_arrays[4], test_arrays[5], test_arrays[6],
    0.5, 0.1, 0.1, 0.1, ndrange=(Nx,Ny,Nz)), 200)
println("KA 3-comp curl (9R+3RMW):   $(round(bytes_curl3/t_curl3/1e9, digits=1)) GB/s  ($(round(t_curl3*1e6, digits=1)) μs)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Compare with Actual Khronos Kernels
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 6: Actual Khronos Kernels")
println("=" ^ 80)

cell_val = Float64(N) / 10.0
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

# Warmup
for _ in 1:30; Khronos.step!(sim); end
CUDA.synchronize()

# Time each kernel
n_rep = 100
t_step_B = cuda_time_events(() -> Khronos.step_B_from_E!(sim), n_rep)
t_upd_H = cuda_time_events(() -> Khronos.update_H_from_B!(sim), n_rep)
t_step_D = cuda_time_events(() -> Khronos.step_D_from_H!(sim), n_rep)
t_upd_E = cuda_time_events(() -> Khronos.update_E_from_D!(sim), n_rep)
t_full = cuda_time_events(() -> Khronos.step!(sim), n_rep)

voxels = Nx * Ny * Nz

# Print results with BW estimates
# step_curl! estimated at 240 bytes/voxel, update_field! at 120 bytes/voxel
println("Khronos kernel timing:")
println("  $(rpad("Kernel", 25))  $(rpad("Time (μs)", 12))  $(rpad("BW @240B/vx", 14))  $(rpad("BW @120B/vx", 14))")
println("  " * "-" ^ 70)
for (name, t, bytes_vx) in [
    ("step_B_from_E!", t_step_B, 240),
    ("update_H_from_B!", t_upd_H, 120),
    ("step_D_from_H!", t_step_D, 240),
    ("update_E_from_D!", t_upd_E, 120),
]
    bw = voxels * bytes_vx / t / 1e9
    println("  $(rpad(name, 25))  $(rpad(round(t*1e6, digits=1), 12))  $(bytes_vx==240 ? rpad(string(round(bw, digits=1))*" GB/s", 14) : rpad("-", 14))  $(bytes_vx==120 ? rpad(string(round(bw, digits=1))*" GB/s", 14) : rpad("-", 14))")
end
println("  $(rpad("full step!", 25))  $(rpad(round(t_full*1e6, digits=1), 12))")

println()
println("Comparison table:")
println("  $(rpad("Pattern", 30))  $(rpad("GB/s", 12))")
println("  " * "-" ^ 45)
println("  $(rpad("Broadcast SAXPY (peak ref)", 30))  $(round(bw_bcast, digits=1))")
println("  $(rpad("KA 3D copy", 30))  $(round(bw_3d_plain, digits=1))")
println("  $(rpad("KA 3D OffsetArray+CI", 30))  $(round(bw_3d_off, digits=1))")
println("  $(rpad("KA stencil d/dx", 30))  $(round(bytes_stencil/t_sx/1e9, digits=1))")
println("  $(rpad("KA stencil d/dz", 30))  $(round(bytes_stencil/t_sz/1e9, digits=1))")
println("  $(rpad("KA 3-comp curl (no PML)", 30))  $(round(bytes_curl3/t_curl3/1e9, digits=1))")
println("  $(rpad("KA 3-comp update (no PML)", 30))  $(round(bytes_3c/t_upd3/1e9, digits=1))")
println("  $(rpad("Khronos step_curl!", 30))  $(round(voxels*240/t_step_B/1e9, digits=1))")
println("  $(rpad("Khronos update_field!", 30))  $(round(voxels*120/t_upd_H/1e9, digits=1))")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Check if PML arrays are the cause of update_field! slowdown
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 7: PML Array Analysis for update_field!")
println("=" ^ 80)

# Check what arrays are actually non-nothing in the update calls
println("update_H_from_B! array analysis:")
for (name, val) in [
    ("fHx", sim.fields.fHx), ("fHy", sim.fields.fHy), ("fHz", sim.fields.fHz),
    ("fBx", sim.fields.fBx), ("fBy", sim.fields.fBy), ("fBz", sim.fields.fBz),
    ("fWBx", sim.fields.fWBx), ("fWBy", sim.fields.fWBy), ("fWBz", sim.fields.fWBz),
    ("fPBx", sim.fields.fPBx), ("fPBy", sim.fields.fPBy), ("fPBz", sim.fields.fPBz),
    ("fSBx", sim.fields.fSBx), ("fSBy", sim.fields.fSBy), ("fSBz", sim.fields.fSBz),
    ("μ_inv", sim.geometry_data.μ_inv),
    ("μ_inv_x", sim.geometry_data.μ_inv_x), ("μ_inv_y", sim.geometry_data.μ_inv_y), ("μ_inv_z", sim.geometry_data.μ_inv_z),
    ("σBx", sim.boundary_data.σBx), ("σBy", sim.boundary_data.σBy), ("σBz", sim.boundary_data.σBz),
]
    if val === nothing
        println("  $(rpad(name, 10)) = nothing")
    elseif isa(val, Number)
        println("  $(rpad(name, 10)) = $(val) (scalar)")
    elseif isa(val, AbstractArray)
        println("  $(rpad(name, 10)) = Array $(size(val))")
    else
        println("  $(rpad(name, 10)) = $(typeof(val))")
    end
end

println()
println("update_E_from_D! array analysis:")
for (name, val) in [
    ("fEx", sim.fields.fEx), ("fEy", sim.fields.fEy), ("fEz", sim.fields.fEz),
    ("fDx", sim.fields.fDx), ("fDy", sim.fields.fDy), ("fDz", sim.fields.fDz),
    ("fWDx", sim.fields.fWDx), ("fWDy", sim.fields.fWDy), ("fWDz", sim.fields.fWDz),
    ("fPDx", sim.fields.fPDx), ("fPDy", sim.fields.fPDy), ("fPDz", sim.fields.fPDz),
    ("fSDx", sim.fields.fSDx), ("fSDy", sim.fields.fSDy), ("fSDz", sim.fields.fSDz),
    ("ε_inv", sim.geometry_data.ε_inv),
    ("ε_inv_x", sim.geometry_data.ε_inv_x), ("ε_inv_y", sim.geometry_data.ε_inv_y), ("ε_inv_z", sim.geometry_data.ε_inv_z),
    ("σDx", sim.boundary_data.σDx), ("σDy", sim.boundary_data.σDy), ("σDz", sim.boundary_data.σDz),
]
    if val === nothing
        println("  $(rpad(name, 10)) = nothing")
    elseif isa(val, Number)
        println("  $(rpad(name, 10)) = $(val) (scalar)")
    elseif isa(val, AbstractArray)
        println("  $(rpad(name, 10)) = Array $(size(val))")
    else
        println("  $(rpad(name, 10)) = $(typeof(val))")
    end
end

println()
println("step_B_from_E! array analysis:")
for (name, val) in [
    ("fCBx", sim.fields.fCBx), ("fCBy", sim.fields.fCBy), ("fCBz", sim.fields.fCBz),
    ("fUBx", sim.fields.fUBx), ("fUBy", sim.fields.fUBy), ("fUBz", sim.fields.fUBz),
    ("geo σBx", sim.geometry_data.σBx), ("geo σBy", sim.geometry_data.σBy), ("geo σBz", sim.geometry_data.σBz),
    ("bnd σBx", sim.boundary_data.σBx), ("bnd σBy", sim.boundary_data.σBy), ("bnd σBz", sim.boundary_data.σBz),
]
    if val === nothing
        println("  $(rpad(name, 12)) = nothing")
    elseif isa(val, Number)
        println("  $(rpad(name, 12)) = $(val) (scalar)")
    elseif isa(val, AbstractArray)
        println("  $(rpad(name, 12)) = Array $(size(val))  ndims=$(ndims(val))")
    else
        println("  $(rpad(name, 12)) = $(typeof(val))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Final Analysis
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("SECTION 8: Root Cause Analysis")
println("=" ^ 80)

println("""
Based on measurements, the root causes of low bandwidth are:

1. BASELINE: What bandwidth does this GPU actually achieve?
   - cudaMemcpy:        $(round(bw_memcpy, digits=0)) GB/s
   - Broadcast SAXPY:   $(round(bw_bcast, digits=0)) GB/s
   - KA copy:           $(round(bw_ka_copy, digits=0)) GB/s

2. 3D INDEXING COST:
   - KA 1D copy:        $(round(bw_ka_copy, digits=0)) GB/s
   - KA 3D plain copy:  $(round(bw_3d_plain, digits=0)) GB/s
   → 3D indexing overhead: $(round((1 - bw_3d_plain/bw_ka_copy) * 100, digits=1))%

3. OFFSETARRAY COST:
   - KA 3D plain:       $(round(bw_3d_plain, digits=0)) GB/s
   - KA 3D OffsetArray: $(round(bw_3d_off, digits=0)) GB/s
   → OffsetArray overhead: $(round((1 - bw_3d_off/bw_3d_plain) * 100, digits=1))%

4. STENCIL STRIDING COST:
   - d/dx (stride-1):   $(round(bytes_stencil/t_sx/1e9, digits=0)) GB/s
   - d/dy (stride-$(Nx+2)): $(round(bytes_stencil/t_sy/1e9, digits=0)) GB/s
   - d/dz (stride-$((Nx+2)*(Ny+2))): $(round(bytes_stencil/t_sz/1e9, digits=0)) GB/s

5. MULTI-COMPONENT COST:
   - KA 3-comp update:  $(round(bytes_3c/t_upd3/1e9, digits=0)) GB/s
   - Khronos update:    $(round(voxels*120/t_upd_H/1e9, digits=0)) GB/s
   → Khronos overhead vs KA equivalent: $(round((1 - (voxels*120/t_upd_H)/(bytes_3c/t_upd3)) * 100, digits=1))%

6. SUMMARY:
   Khronos step_curl!:  $(round(voxels*240/t_step_B/1e9, digits=0)) GB/s
   Khronos update_field!: $(round(voxels*120/t_upd_H/1e9, digits=0)) GB/s
   Peak achievable:     $(round(bw_memcpy, digits=0)) GB/s
""")
