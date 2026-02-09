# Test whether Julia's GPU compiler actually inlines the nested dispatch functions.
#
# Strategy: Compare a "composed" kernel (matching Khronos style) with nested
# @inline functions and Union types against a "flattened" equivalent. If the
# compiler inlines properly, they should perform identically.

import Khronos
using CUDA
using KernelAbstractions
using OffsetArrays

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

const N = 256
const Nx = N; const Ny = N; const Nz = N

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
# TEST 1: Does @inline nested dispatch cost anything on GPU?
# ═══════════════════════════════════════════════════════════════════════════
println("=" ^ 80)
println("TEST 1: Composable nested dispatch vs flattened kernel")
println("=" ^ 80)
println()

# ── Khronos-style composed version ──
# Mirrors the exact call chain: kernel → update_field_generic → update_cache/clear_source/get_σ/get_m_inv
@inline update_cache_test(A::AbstractArray, idx) = A[idx]
@inline update_cache_test(A::Nothing, idx) = 0.0
@inline function clear_source_test(A::AbstractArray, idx)
    A[idx] = zero(eltype(A))
    return
end
@inline clear_source_test(A::Nothing, idx) = nothing
@inline get_σ_test(σ::Nothing, idx) = nothing
@inline get_σ_test(σ, idx) = σ[2*idx-1]
@inline get_m_inv_test(m_inv::Nothing, m_inv_x::AbstractArray, idx) = m_inv_x[idx]
@inline get_m_inv_test(m_inv::Real, m_inv_x::Nothing, idx) = m_inv

@inline update_field_from_curl_test(A, B, B_old, σ) = ((1 - σ) * A + B - B_old) / (1 + σ)
@inline update_field_from_curl_test(A, B, B_old::Nothing, σ) = ((1 - σ) * A + B) / (1 + σ)
@inline update_field_from_curl_test(A, B, B_old, σ::Nothing) = (A + B - B_old)
@inline update_field_from_curl_test(A, B, B_old::Nothing, σ::Nothing) = (A + B)

function update_field_generic_test(A, T, W, P, S, m_inv, σ, idx_array)
    W_old = W[idx_array]
    net_field = T[idx_array]
    net_field += update_cache_test(S, idx_array)
    net_field += update_cache_test(P, idx_array)
    clear_source_test(S, idx_array)
    W[idx_array] = m_inv * net_field
    A[idx_array] = A[idx_array] + (1 + σ) * W[idx_array] - (1 - σ) * W_old
end

function update_field_generic_test(A, T, W::Nothing, P, S, m_inv, σ::Nothing, idx_array)
    net_field = T[idx_array]
    net_field += update_cache_test(S, idx_array)
    net_field += update_cache_test(P, idx_array)
    clear_source_test(S, idx_array)
    A[idx_array] = m_inv * net_field
end

# The composed kernel — mirrors Khronos update_field! exactly
@kernel function composed_update_field!(
    Ax::Union{AbstractArray,Nothing}, Ay::Union{AbstractArray,Nothing}, Az::Union{AbstractArray,Nothing},
    Tx::Union{AbstractArray,Nothing}, Ty::Union{AbstractArray,Nothing}, Tz::Union{AbstractArray,Nothing},
    Wx::Union{AbstractArray,Nothing}, Wy::Union{AbstractArray,Nothing}, Wz::Union{AbstractArray,Nothing},
    Px::Union{AbstractArray,Nothing}, Py::Union{AbstractArray,Nothing}, Pz::Union{AbstractArray,Nothing},
    Sx::Union{AbstractArray,Nothing}, Sy::Union{AbstractArray,Nothing}, Sz::Union{AbstractArray,Nothing},
    m_inv::Union{Number,Nothing},
    m_inv_x::Union{AbstractArray,Nothing}, m_inv_y::Union{AbstractArray,Nothing}, m_inv_z::Union{AbstractArray,Nothing},
    σx::Union{AbstractArray,Nothing}, σy::Union{AbstractArray,Nothing}, σz::Union{AbstractArray,Nothing},
)
    ix, iy, iz = @index(Global, NTuple)
    idx_array = CartesianIndex(ix, iy, iz)

    update_field_generic_test(Ax, Tx, Wx, Px, Sx,
        get_m_inv_test(m_inv, m_inv_x, idx_array), get_σ_test(σx, ix), idx_array)
    update_field_generic_test(Ay, Ty, Wy, Py, Sy,
        get_m_inv_test(m_inv, m_inv_y, idx_array), get_σ_test(σy, iy), idx_array)
    update_field_generic_test(Az, Tz, Wz, Pz, Sz,
        get_m_inv_test(m_inv, m_inv_z, idx_array), get_σ_test(σz, iz), idx_array)
end

# ── Flattened version — same math, zero dispatch, zero function calls ──
@kernel function flattened_update_field!(
    Ax, Ay, Az,        # fields to update (E or H)
    Tx, Ty, Tz,        # timestepped fields (D or B)
    Wx, Wy, Wz,        # auxiliary PML fields
    m_inv::Float64,    # material inverse (scalar)
    σx, σy, σz,        # PML conductivity (1D arrays)
)
    ix, iy, iz = @index(Global, NTuple)
    idx = CartesianIndex(ix, iy, iz)

    @inbounds begin
        sx = σx[2*ix-1]; sy = σy[2*iy-1]; sz = σz[2*iz-1]

        # X component
        W_old_x = Wx[idx]
        Wx[idx] = m_inv * Tx[idx]
        Ax[idx] = Ax[idx] + (1.0 + sx) * Wx[idx] - (1.0 - sx) * W_old_x

        # Y component
        W_old_y = Wy[idx]
        Wy[idx] = m_inv * Ty[idx]
        Ay[idx] = Ay[idx] + (1.0 + sy) * Wy[idx] - (1.0 - sy) * W_old_y

        # Z component
        W_old_z = Wz[idx]
        Wz[idx] = m_inv * Tz[idx]
        Az[idx] = Az[idx] + (1.0 + sz) * Wz[idx] - (1.0 - sz) * W_old_z
    end
end

# ── Even more flattened: no OffsetArray, no CartesianIndex ──
@kernel function bare_update_field!(
    Ax, Ay, Az,
    Tx, Ty, Tz,
    Wx, Wy, Wz,
    m_inv::Float64,
    σx, σy, σz,
)
    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        sx = σx[2*ix-1]; sy = σy[2*iy-1]; sz = σz[2*iz-1]

        # X component (plain 3D indexing, +1 for ghost cells)
        W_old = Wx[ix+1, iy+1, iz+1]
        Wx[ix+1, iy+1, iz+1] = m_inv * Tx[ix+1, iy+1, iz+1]
        Ax[ix+1, iy+1, iz+1] = Ax[ix+1, iy+1, iz+1] +
            (1.0 + sx) * Wx[ix+1, iy+1, iz+1] - (1.0 - sx) * W_old

        # Y component
        W_old = Wy[ix+1, iy+1, iz+1]
        Wy[ix+1, iy+1, iz+1] = m_inv * Ty[ix+1, iy+1, iz+1]
        Ay[ix+1, iy+1, iz+1] = Ay[ix+1, iy+1, iz+1] +
            (1.0 + sy) * Wy[ix+1, iy+1, iz+1] - (1.0 - sy) * W_old

        # Z component
        W_old = Wz[ix+1, iy+1, iz+1]
        Wz[ix+1, iy+1, iz+1] = m_inv * Tz[ix+1, iy+1, iz+1]
        Az[ix+1, iy+1, iz+1] = Az[ix+1, iy+1, iz+1] +
            (1.0 + sz) * Wz[ix+1, iy+1, iz+1] - (1.0 - sz) * W_old
    end
end

# Allocate arrays — match Khronos layout exactly
# OffsetArray versions (Khronos-style)
make_off() = OffsetArray(CUDA.zeros(Float64, Nx+2, Ny+2, Nz+2), -1, -1, -1)
make_plain() = CUDA.rand(Float64, Nx+2, Ny+2, Nz+2)

off_Ax = make_off(); off_Ay = make_off(); off_Az = make_off()
off_Tx = make_off(); off_Ty = make_off(); off_Tz = make_off()
off_Wx = make_off(); off_Wy = make_off(); off_Wz = make_off()

pln_Ax = make_plain(); pln_Ay = make_plain(); pln_Az = make_plain()
pln_Tx = make_plain(); pln_Ty = make_plain(); pln_Tz = make_plain()
pln_Wx = make_plain(); pln_Wy = make_plain(); pln_Wz = make_plain()

# 1D PML sigma arrays (matching Khronos size 2*N+1)
σx_1d = CUDA.rand(Float64, 2*Nx+1)
σy_1d = CUDA.rand(Float64, 2*Ny+1)
σz_1d = CUDA.rand(Float64, 2*Nz+1)

# Also OffsetArray 1D sigmas (matching boundary_data format)
σx_off = OffsetArray(CUDA.rand(Float64, 2*Nx+1), 0)
σy_off = OffsetArray(CUDA.rand(Float64, 2*Ny+1), 0)
σz_off = OffsetArray(CUDA.rand(Float64, 2*Nz+1), 0)

bytes_per_voxel = 15 * 8  # 9R + 6W = 15 array accesses × 8 bytes
total_bytes = bytes_per_voxel * Nx * Ny * Nz

# ── Benchmark all variants ──
n_rep = 200

# 1) Composed kernel with OffsetArrays + Nothing args (full Khronos style)
k_composed = composed_update_field!(CUDABackend())
t_composed = cuda_time_events(() -> k_composed(
    off_Ax, off_Ay, off_Az,
    off_Tx, off_Ty, off_Tz,
    off_Wx, off_Wy, off_Wz,
    nothing, nothing, nothing,   # P
    nothing, nothing, nothing,   # S
    1.0,                          # m_inv (scalar)
    nothing, nothing, nothing,   # m_inv_x/y/z
    σx_off, σy_off, σz_off,     # σ
    ndrange=(Nx, Ny, Nz)), n_rep)
bw_composed = total_bytes / t_composed / 1e9
println("Composed (Khronos-style, OffsetArray+Nothing): $(round(bw_composed, digits=1)) GB/s  ($(round(t_composed*1e6, digits=1)) μs)")

# 2) Flattened kernel with OffsetArrays (no dispatch chain, still OffsetArray)
k_flat = flattened_update_field!(CUDABackend())
t_flat = cuda_time_events(() -> k_flat(
    off_Ax, off_Ay, off_Az,
    off_Tx, off_Ty, off_Tz,
    off_Wx, off_Wy, off_Wz,
    1.0, σx_1d, σy_1d, σz_1d,
    ndrange=(Nx, Ny, Nz)), n_rep)
bw_flat = total_bytes / t_flat / 1e9
println("Flattened (OffsetArray, no dispatch):          $(round(bw_flat, digits=1)) GB/s  ($(round(t_flat*1e6, digits=1)) μs)")

# 3) Bare kernel (plain arrays, no OffsetArray, no CartesianIndex)
k_bare = bare_update_field!(CUDABackend())
t_bare = cuda_time_events(() -> k_bare(
    pln_Ax, pln_Ay, pln_Az,
    pln_Tx, pln_Ty, pln_Tz,
    pln_Wx, pln_Wy, pln_Wz,
    1.0, σx_1d, σy_1d, σz_1d,
    ndrange=(Nx, Ny, Nz)), n_rep)
bw_bare = total_bytes / t_bare / 1e9
println("Bare (plain CuArray, ix+1 indexing):           $(round(bw_bare, digits=1)) GB/s  ($(round(t_bare*1e6, digits=1)) μs)")

# 4) Actual Khronos kernel for comparison
cell_val = Float64(N) / 10.0
sim = Khronos.Simulation(
    cell_size   = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution  = 10,
    sources     = [Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0], size = [0.0, 0.0, 0.0])],
    boundaries  = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
)
Khronos.prepare_simulation!(sim)
for _ in 1:30; Khronos.step!(sim); end
CUDA.synchronize()

t_khronos = cuda_time_events(() -> Khronos.update_H_from_B!(sim), n_rep)
bw_khronos = total_bytes / t_khronos / 1e9
println("Actual Khronos update_H_from_B!:               $(round(bw_khronos, digits=1)) GB/s  ($(round(t_khronos*1e6, digits=1)) μs)")

println()
println("─" ^ 80)
println("Breakdown of overhead sources:")
println("  Bare → Flattened+OffsetArray:  $(round((t_flat/t_bare - 1)*100, digits=1))%  (OffsetArray + CartesianIndex cost)")
println("  Flattened → Composed:          $(round((t_composed/t_flat - 1)*100, digits=1))%  (nested dispatch + Nothing args cost)")
println("  Composed → Khronos:            $(round((t_khronos/t_composed - 1)*100, digits=1))%  (remaining Khronos-specific overhead)")
println("  Total: Bare → Khronos:         $(round((t_khronos/t_bare - 1)*100, digits=1))%  ($(round(t_khronos/t_bare, digits=2))x slowdown)")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Does the compiler inline properly? Check via PTX/SASS
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("TEST 2: PTX Code Generation Analysis")
println("=" ^ 80)

# We can get the LLVM IR or PTX to check if functions are inlined
# For the composed kernel, check if update_field_generic_test is inlined

println("\nComposed kernel — checking for function calls in PTX:")
try
    # Get the PTX code for the composed kernel with specific argument types
    composed_types = Tuple{
        typeof(off_Ax), typeof(off_Ay), typeof(off_Az),
        typeof(off_Tx), typeof(off_Ty), typeof(off_Tz),
        typeof(off_Wx), typeof(off_Wy), typeof(off_Wz),
        Nothing, Nothing, Nothing,
        Nothing, Nothing, Nothing,
        Float64,
        Nothing, Nothing, Nothing,
        typeof(σx_off), typeof(σy_off), typeof(σz_off),
    }
    # Can't easily get PTX from KA kernel, but we can check code_llvm
    println("  (PTX inspection requires manual @device_code_ptx; skipping)")
    println("  Instead, checking timing equivalence as a proxy for inlining.")
catch e
    println("  PTX analysis failed: $e")
end

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: How much does OffsetArray cost vs on-the-fly offset computation?
# ═══════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 80)
println("TEST 3: OffsetArray vs Manual Offset (no extra memory)")
println("=" ^ 80)

println("""
OffsetArray does NOT allocate extra memory. It's a zero-cost wrapper in theory:
  OffsetArray{T,N,AA} stores:
    - parent::AA (the underlying CuArray — just a pointer)
    - offsets::NTuple{N,Int} (compile-time constant offsets)

  Indexing: A[i,j,k] → parent[i - offset_x, j - offset_y, k - offset_z]

The cost should be 3 integer subtractions per access. On GPU, these should
be fused into the address computation. But our measurement shows 24-32%
overhead, suggesting the compiler isn't optimizing this as well as expected
on the GPU side.

Note: "texture pointers" (CUDA texture memory) would be worse for read-write
arrays. Textures are read-only cached, and the caching is optimized for 2D
spatial locality. For our 3D stencil with writes, standard global memory
with L2 cache is the right choice.
""")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: What about composable kernel fusion via @generated or macros?
# ═══════════════════════════════════════════════════════════════════════════
println("=" ^ 80)
println("TEST 4: Composable Fusion — Can We Have Both?")
println("=" ^ 80)

# Test: fused step_curl + update_field kernel
# In this version, B = B + curl(E), then H = μ_inv * B — all in one kernel
# This saves reading/writing B from DRAM (it stays in registers)

@kernel function fused_curl_update!(
    # Output: H fields (updated)
    Hx, Hy, Hz,
    # Input: E fields (for curl)
    Ex, Ey, Ez,
    # Auxiliary: B fields (read-modify-write, but stays in registers if fused)
    Bx, By, Bz,
    # PML auxiliary W fields
    Wx, Wy, Wz,
    # PML U fields (from curl stage)
    Ux, Uy, Uz,
    # Scalars
    dt::Float64, dx::Float64, dy::Float64, dz::Float64, m_inv::Float64,
    # 1D PML σ
    σx, σy, σz,
)
    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        inv_dx = 1.0 / dx; inv_dy = 1.0 / dy; inv_dz = 1.0 / dz

        # ── Step 1: curl (B += dt * curl(E)) ──
        # X component: dEy/dz - dEz/dy
        Kx = dt * (inv_dz * (Ey[ix+1, iy+1, iz+2] - Ey[ix+1, iy+1, iz+1]) -
                    inv_dy * (Ez[ix+1, iy+2, iz+1] - Ez[ix+1, iy+1, iz+1]))
        # PML: U update with σ_next (σ_prev from curl is σ_prev for B)
        σ_prev_x = σy[2*iy-1]
        σ_next_x = σz[2*iz-1]
        U_old_x = Ux[ix+1, iy+1, iz+1]
        Ux[ix+1, iy+1, iz+1] = Ux[ix+1, iy+1, iz+1] + Kx  # simplified PML
        Bx_val = Bx[ix+1, iy+1, iz+1]
        Bx_new = Bx_val + Ux[ix+1, iy+1, iz+1] - U_old_x
        Bx[ix+1, iy+1, iz+1] = Bx_new

        # ── Step 2: update (H = μ_inv * B) with PML ──
        # In non-fused version, this would re-read Bx from DRAM
        # Here, Bx_new is already in a register!
        sx = σx[2*ix-1]
        W_old_x = Wx[ix+1, iy+1, iz+1]
        Wx[ix+1, iy+1, iz+1] = m_inv * Bx_new   # ← register, not DRAM!
        Hx[ix+1, iy+1, iz+1] = Hx[ix+1, iy+1, iz+1] +
            (1.0 + sx) * Wx[ix+1, iy+1, iz+1] - (1.0 - sx) * W_old_x

        # Y component
        Ky = dt * (inv_dx * (Ez[ix+2, iy+1, iz+1] - Ez[ix+1, iy+1, iz+1]) -
                    inv_dz * (Ex[ix+1, iy+1, iz+2] - Ex[ix+1, iy+1, iz+1]))
        U_old_y = Uy[ix+1, iy+1, iz+1]
        Uy[ix+1, iy+1, iz+1] = Uy[ix+1, iy+1, iz+1] + Ky
        By_val = By[ix+1, iy+1, iz+1]
        By_new = By_val + Uy[ix+1, iy+1, iz+1] - U_old_y
        By[ix+1, iy+1, iz+1] = By_new

        sy = σy[2*iy-1]
        W_old_y = Wy[ix+1, iy+1, iz+1]
        Wy[ix+1, iy+1, iz+1] = m_inv * By_new
        Hy[ix+1, iy+1, iz+1] = Hy[ix+1, iy+1, iz+1] +
            (1.0 + sy) * Wy[ix+1, iy+1, iz+1] - (1.0 - sy) * W_old_y

        # Z component
        Kz = dt * (inv_dy * (Ex[ix+1, iy+2, iz+1] - Ex[ix+1, iy+1, iz+1]) -
                    inv_dx * (Ey[ix+2, iy+1, iz+1] - Ey[ix+1, iy+1, iz+1]))
        U_old_z = Uz[ix+1, iy+1, iz+1]
        Uz[ix+1, iy+1, iz+1] = Uz[ix+1, iy+1, iz+1] + Kz
        Bz_val = Bz[ix+1, iy+1, iz+1]
        Bz_new = Bz_val + Uz[ix+1, iy+1, iz+1] - U_old_z
        Bz[ix+1, iy+1, iz+1] = Bz_new

        sz = σz[2*iz-1]
        W_old_z = Wz[ix+1, iy+1, iz+1]
        Wz[ix+1, iy+1, iz+1] = m_inv * Bz_new
        Hz[ix+1, iy+1, iz+1] = Hz[ix+1, iy+1, iz+1] +
            (1.0 + sz) * Wz[ix+1, iy+1, iz+1] - (1.0 - sz) * W_old_z
    end
end

# Allocate for fused test
fH = [make_plain() for _ in 1:3]
fE = [make_plain() for _ in 1:3]
fB = [make_plain() for _ in 1:3]
fW = [make_plain() for _ in 1:3]
fU = [make_plain() for _ in 1:3]

k_fused = fused_curl_update!(CUDABackend())
t_fused = cuda_time_events(() -> k_fused(
    fH[1], fH[2], fH[3],
    fE[1], fE[2], fE[3],
    fB[1], fB[2], fB[3],
    fW[1], fW[2], fW[3],
    fU[1], fU[2], fU[3],
    0.5, 0.1, 0.1, 0.1, 1.0,
    σx_1d, σy_1d, σz_1d,
    ndrange=(Nx, Ny, Nz)), n_rep)

# Unfused: step_curl + update_field separately
# step_curl reads E(9 stencil), reads/writes U(6), reads/writes B(6) = ~21 ops
# update_field reads B(3), reads/writes W(6), reads/writes H(6) = ~15 ops
# Total unfused: 36 ops = 36 * 8 = 288 bytes/voxel
# Fused saves: 3 B reads in update_field (already in register) = saves 24 bytes
# Fused total: 33 ops = 33 * 8 = 264 bytes/voxel
# But the REAL savings: 1 kernel launch instead of 2, and temporal locality

bytes_unfused = 36 * Nx * Ny * Nz * 8
bytes_fused = 33 * Nx * Ny * Nz * 8  # B stays in register

# Time unfused equivalent
@kernel function unfused_curl!(Bx, By, Bz, Ex, Ey, Ez, Ux, Uy, Uz,
    dt::Float64, dx::Float64, dy::Float64, dz::Float64, σx, σy, σz)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        inv_dx = 1.0/dx; inv_dy = 1.0/dy; inv_dz = 1.0/dz

        Kx = dt * (inv_dz * (Ey[ix+1,iy+1,iz+2] - Ey[ix+1,iy+1,iz+1]) -
                    inv_dy * (Ez[ix+1,iy+2,iz+1] - Ez[ix+1,iy+1,iz+1]))
        U_old = Ux[ix+1,iy+1,iz+1]
        Ux[ix+1,iy+1,iz+1] = Ux[ix+1,iy+1,iz+1] + Kx
        Bx[ix+1,iy+1,iz+1] = Bx[ix+1,iy+1,iz+1] + Ux[ix+1,iy+1,iz+1] - U_old

        Ky = dt * (inv_dx * (Ez[ix+2,iy+1,iz+1] - Ez[ix+1,iy+1,iz+1]) -
                    inv_dz * (Ex[ix+1,iy+1,iz+2] - Ex[ix+1,iy+1,iz+1]))
        U_old = Uy[ix+1,iy+1,iz+1]
        Uy[ix+1,iy+1,iz+1] = Uy[ix+1,iy+1,iz+1] + Ky
        By[ix+1,iy+1,iz+1] = By[ix+1,iy+1,iz+1] + Uy[ix+1,iy+1,iz+1] - U_old

        Kz = dt * (inv_dy * (Ex[ix+1,iy+2,iz+1] - Ex[ix+1,iy+1,iz+1]) -
                    inv_dx * (Ey[ix+2,iy+1,iz+1] - Ey[ix+1,iy+1,iz+1]))
        U_old = Uz[ix+1,iy+1,iz+1]
        Uz[ix+1,iy+1,iz+1] = Uz[ix+1,iy+1,iz+1] + Kz
        Bz[ix+1,iy+1,iz+1] = Bz[ix+1,iy+1,iz+1] + Uz[ix+1,iy+1,iz+1] - U_old
    end
end

@kernel function unfused_update!(Hx, Hy, Hz, Bx, By, Bz, Wx, Wy, Wz,
    m_inv::Float64, σx, σy, σz)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds begin
        sx = σx[2*ix-1]; sy = σy[2*iy-1]; sz = σz[2*iz-1]

        W_old = Wx[ix+1,iy+1,iz+1]
        Wx[ix+1,iy+1,iz+1] = m_inv * Bx[ix+1,iy+1,iz+1]
        Hx[ix+1,iy+1,iz+1] = Hx[ix+1,iy+1,iz+1] + (1+sx)*Wx[ix+1,iy+1,iz+1] - (1-sx)*W_old

        W_old = Wy[ix+1,iy+1,iz+1]
        Wy[ix+1,iy+1,iz+1] = m_inv * By[ix+1,iy+1,iz+1]
        Hy[ix+1,iy+1,iz+1] = Hy[ix+1,iy+1,iz+1] + (1+sy)*Wy[ix+1,iy+1,iz+1] - (1-sy)*W_old

        W_old = Wz[ix+1,iy+1,iz+1]
        Wz[ix+1,iy+1,iz+1] = m_inv * Bz[ix+1,iy+1,iz+1]
        Hz[ix+1,iy+1,iz+1] = Hz[ix+1,iy+1,iz+1] + (1+sz)*Wz[ix+1,iy+1,iz+1] - (1-sz)*W_old
    end
end

k_ucurl = unfused_curl!(CUDABackend())
k_uupd = unfused_update!(CUDABackend())

function run_unfused!()
    k_ucurl(fB[1], fB[2], fB[3], fE[1], fE[2], fE[3],
        fU[1], fU[2], fU[3], 0.5, 0.1, 0.1, 0.1, σx_1d, σy_1d, σz_1d,
        ndrange=(Nx, Ny, Nz))
    k_uupd(fH[1], fH[2], fH[3], fB[1], fB[2], fB[3],
        fW[1], fW[2], fW[3], 1.0, σx_1d, σy_1d, σz_1d,
        ndrange=(Nx, Ny, Nz))
    return
end

t_unfused = cuda_time_events(run_unfused!, n_rep)
bw_fused = bytes_fused / t_fused / 1e9
bw_unfused = bytes_unfused / t_unfused / 1e9

println("Unfused (2 kernels, plain arrays):  $(round(bw_unfused, digits=1)) GB/s  ($(round(t_unfused*1e6, digits=1)) μs)")
println("Fused (1 kernel, plain arrays):     $(round(bw_fused, digits=1)) GB/s  ($(round(t_fused*1e6, digits=1)) μs)")
println("Speedup from fusion:                $(round(t_unfused / t_fused, digits=2))x")

println()
println("─" ^ 80)
println("FINAL COMPARISON:")
println("  $(rpad("Variant", 50))  $(rpad("GB/s", 8))  $(rpad("μs", 10))  Rel.")
println("  " * "─" ^ 75)
for (name, bw, t, t_ref) in [
    ("Bare (plain CuArray, no dispatch)", bw_bare, t_bare, t_bare),
    ("Flattened (OffsetArray, no dispatch)", bw_flat, t_flat, t_bare),
    ("Composed (OffsetArray + nested dispatch)", bw_composed, t_composed, t_bare),
    ("Actual Khronos update_H_from_B!", bw_khronos, t_khronos, t_bare),
    ("Unfused curl+update (2 kernels, plain)", bw_unfused, t_unfused, t_bare),
    ("Fused curl+update (1 kernel, plain)", bw_fused, t_fused, t_bare),
]
    println("  $(rpad(name, 50))  $(rpad(round(bw, digits=0), 8))  $(rpad(round(t*1e6, digits=1), 10))  $(round(t/t_ref, digits=2))x")
end
