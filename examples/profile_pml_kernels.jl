# Profile per-component PML kernels + interior kernels
# Run with:
#   ncu --set full --launch-skip 20 --launch-count 12 julia --project=. examples/profile_pml_kernels.jl
#   nsys profile --trace=cuda julia --project=. examples/profile_pml_kernels.jl

using Khronos
using CUDA

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

Nx, Ny, Nz = 256, 256, 256
T = Float32

# Field arrays (padded +2 for stencil halo)
Ex = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Ey = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Ez = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
Hx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Hy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Hz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
Bx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); By = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Bz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
Dx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Dy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); Dz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)

# PML auxiliary arrays
UBx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WBx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
UBy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WBy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
UBz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WBz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
UDx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WDx = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
UDy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WDy = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)
UDz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2); WDz = CUDA.zeros(T, Nx+2, Ny+2, Nz+2)

# Per-voxel epsilon
eps_inv_x = CUDA.ones(T, Nx, Ny, Nz); eps_inv_y = CUDA.ones(T, Nx, Ny, Nz); eps_inv_z = CUDA.ones(T, Nx, Ny, Nz)

# PML sigma arrays (1D, zero-filled = interior behavior)
σ_zero = CUDA.zeros(T, 2*Nx+1)
# Sigma with actual PML values (edge behavior)
σ_pml = CUDA.zeros(T, 2*Nx+1)
σ_pml_cpu = zeros(T, 2*Nx+1)
for i in 1:20  # PML in first 20 cells
    σ_pml_cpu[2*i-1] = T(0.1 * (i/20)^3)
end
copyto!(σ_pml, σ_pml_cpu)

for arr in (Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx, Dy, Dz)
    fill!(arr, T(0.001))
end

m_inv = T(1.0)
dt_dx = T(0.5); dt_dy = T(0.5); dt_dz = T(0.5)
m_dt_dx = m_inv * dt_dx; m_dt_dy = m_inv * dt_dy; m_dt_dz = m_inv * dt_dz
iNx = Int32(Nx)
cuda_wg = 256
nblocks_x = cld(Int(iNx), cuda_wg)

println("Grid: $(Nx)×$(Ny)×$(Nz) = $(Nx*Ny*Nz) voxels")
println("Blocks: $(nblocks_x)×$(Ny)×$(Nz), threads: $(cuda_wg)")
println()

# === Warmup (20 launches, skipped by ncu) ===
for _ in 1:2
    # Interior BH (1 launch)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_fused_BH_kernel!(
        Ex,Ey,Ez, Hx,Hy,Hz, m_dt_dx,m_dt_dy,m_dt_dz, iNx)
    # Interior DE per-voxel (1 launch)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_fused_DE_kernel!(
        Hx,Hy,Hz, Ex,Ey,Ez, eps_inv_x,eps_inv_y,eps_inv_z, dt_dx,dt_dy,dt_dz, iNx)
    # PML BH x,y,z with σ=0 (3 launches)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_x_kernel!(
        Ey,Ez, Bx,Hx, UBx,WBx, σ_zero,σ_zero,σ_zero, m_inv, dt_dy,dt_dz, iNx)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_y_kernel!(
        Ez,Ex, By,Hy, UBy,WBy, σ_zero,σ_zero,σ_zero, m_inv, dt_dz,dt_dx, iNx)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_z_kernel!(
        Ex,Ey, Bz,Hz, UBz,WBz, σ_zero,σ_zero,σ_zero, m_inv, dt_dx,dt_dy, iNx)
    # PML DE x,y,z with σ=0 (3 launches)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_x_kernel!(
        Hy,Hz, Dx,Ex, UDx,WDx, σ_zero,σ_zero,σ_zero, eps_inv_x, dt_dy,dt_dz, iNx)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_y_kernel!(
        Hz,Hx, Dy,Ey, UDy,WDy, σ_zero,σ_zero,σ_zero, eps_inv_y, dt_dz,dt_dx, iNx)
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_z_kernel!(
        Hx,Hy, Dz,Ez, UDz,WDz, σ_zero,σ_zero,σ_zero, eps_inv_z, dt_dx,dt_dy, iNx)
    # Old 3-component PML BH (1 launch) — for comparison
    @cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_kernel!(
        Ex,Ey,Ez, Bx,By,Bz, Hx,Hy,Hz, UBx,UBy,UBz, WBx,WBy,WBz,
        σ_zero,σ_zero,σ_zero, m_inv, dt_dx,dt_dy,dt_dz, iNx, Int32(0),Int32(0),Int32(0))
end
CUDA.synchronize()

# === Profiled launches (12 total) ===
println("Launching profiled kernels...")

# 1. Interior BH fused
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_fused_BH_kernel!(
    Ex,Ey,Ez, Hx,Hy,Hz, m_dt_dx,m_dt_dy,m_dt_dz, iNx)

# 2. Interior DE fused (per-voxel ε)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_fused_DE_kernel!(
    Hx,Hy,Hz, Ex,Ey,Ez, eps_inv_x,eps_inv_y,eps_inv_z, dt_dx,dt_dy,dt_dz, iNx)

# 3-5. Per-component PML BH (σ=0, fast path)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_x_kernel!(
    Ey,Ez, Bx,Hx, UBx,WBx, σ_zero,σ_zero,σ_zero, m_inv, dt_dy,dt_dz, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_y_kernel!(
    Ez,Ex, By,Hy, UBy,WBy, σ_zero,σ_zero,σ_zero, m_inv, dt_dz,dt_dx, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_z_kernel!(
    Ex,Ey, Bz,Hz, UBz,WBz, σ_zero,σ_zero,σ_zero, m_inv, dt_dx,dt_dy, iNx)

# 6-8. Per-component PML DE (σ=0, fast path, per-voxel ε)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_x_kernel!(
    Hy,Hz, Dx,Ex, UDx,WDx, σ_zero,σ_zero,σ_zero, eps_inv_x, dt_dy,dt_dz, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_y_kernel!(
    Hz,Hx, Dy,Ey, UDy,WDy, σ_zero,σ_zero,σ_zero, eps_inv_y, dt_dz,dt_dx, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_DE_z_kernel!(
    Hx,Hy, Dz,Ez, UDz,WDz, σ_zero,σ_zero,σ_zero, eps_inv_z, dt_dx,dt_dy, iNx)

# 9-11. Per-component PML BH with active PML σ (slow path)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_x_kernel!(
    Ey,Ez, Bx,Hx, UBx,WBx, σ_pml,σ_pml,σ_pml, m_inv, dt_dy,dt_dz, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_y_kernel!(
    Ez,Ex, By,Hy, UBy,WBy, σ_pml,σ_pml,σ_pml, m_inv, dt_dz,dt_dx, iNx)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_z_kernel!(
    Ex,Ey, Bz,Hz, UBz,WBz, σ_pml,σ_pml,σ_pml, m_inv, dt_dx,dt_dy, iNx)

# 12. Old 3-component PML BH (baseline comparison)
@cuda blocks=(nblocks_x, Ny, Nz) threads=(cuda_wg,1,1) Khronos._cuda_pml_BH_kernel!(
    Ex,Ey,Ez, Bx,By,Bz, Hx,Hy,Hz, UBx,UBy,UBz, WBx,WBy,WBz,
    σ_zero,σ_zero,σ_zero, m_inv, dt_dx,dt_dy,dt_dz, iNx, Int32(0),Int32(0),Int32(0))

CUDA.synchronize()
println("Done — 12 profiled launches.")
