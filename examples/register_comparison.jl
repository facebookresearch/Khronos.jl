# Quick register count comparison: new per-component vs old 3-component kernels
using Khronos, CUDA
Khronos.choose_backend(Khronos.CUDADevice(), Float32)

n = 32; T = Float32
arr = CUDA.zeros(T, n+2, n+2, n+2)
σ = CUDA.zeros(T, 2*n+1)
eps = CUDA.ones(T, n, n, n)
s = T(0.5); m = T(1.0); iNx = Int32(n)

println("=" ^ 70)
println("Kernel Register Counts: New Per-Component vs Old 3-Component")
println("=" ^ 70)

function report(name, k)
    regs = CUDA.registers(k)
    local_mem = CUDA.memory(k).local
    blks = min(div(65536, regs * 256), 32)
    occ = min(blks * 256, 2048) / 2048 * 100
    spill = local_mem > 0 ? " *** SPILL $(local_mem)B ***" : ""
    println("  $(rpad(name, 32)) $(lpad(regs,3)) regs | $(lpad("$(occ)%",7)) occ @256$(spill)")
end

println("\n--- NEW per-component PML kernels ---")
report("PML BH X", @cuda launch=false Khronos._cuda_pml_BH_x_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,iNx))
report("PML BH Y", @cuda launch=false Khronos._cuda_pml_BH_y_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,iNx))
report("PML BH Z", @cuda launch=false Khronos._cuda_pml_BH_z_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,iNx))
report("PML DE X (per-voxel ε)", @cuda launch=false Khronos._cuda_pml_DE_x_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,eps,s,s,iNx))
report("PML DE Y (per-voxel ε)", @cuda launch=false Khronos._cuda_pml_DE_y_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,eps,s,s,iNx))
report("PML DE Z (per-voxel ε)", @cuda launch=false Khronos._cuda_pml_DE_z_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,eps,s,s,iNx))
report("PML DE X (scalar ε)", @cuda launch=false Khronos._cuda_pml_DE_x_kernel!(arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,iNx))

println("\n--- OLD 3-component PML kernels ---")
report("Old PML BH (3-comp)", @cuda launch=false Khronos._cuda_pml_BH_kernel!(arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,s,iNx,Int32(0),Int32(0),Int32(0)))
report("Old PML DE (3-comp,arr)", @cuda launch=false Khronos._cuda_pml_DE_kernel!(arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,σ,σ,σ,eps,eps,eps,s,s,s,iNx,Int32(0),Int32(0),Int32(0)))
report("Old PML DE (3-comp,scl)", @cuda launch=false Khronos._cuda_pml_DE_scalar_kernel!(arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,σ,σ,σ,m,s,s,s,iNx,Int32(0),Int32(0),Int32(0)))

println("\n--- Interior (no-PML) kernels ---")
report("Interior BH (fused)", @cuda launch=false Khronos._cuda_fused_BH_kernel!(arr,arr,arr,arr,arr,arr,s,s,s,iNx))
report("Interior DE (fused,arr)", @cuda launch=false Khronos._cuda_fused_DE_kernel!(arr,arr,arr,arr,arr,arr,eps,eps,eps,s,s,s,iNx))
report("Interior DE (fused,scl)", @cuda launch=false Khronos._cuda_fused_DE_scalar_kernel!(arr,arr,arr,arr,arr,arr,m,s,s,s,iNx))

println("\n" * "=" ^ 70)
