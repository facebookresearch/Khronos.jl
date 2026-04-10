# Test: Bypass adj_src_scale entirely.
# Use a unit-amplitude point source at the monitor location for adjoint.
# Then manually compute what scale factor makes the gradient match FD.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random
using Statistics

Khronos.choose_backend(Khronos.CPUDevice(), Float64)

ε_hi = 4.0; ε_lo = 1.0; design_Lx = 2.0; pml = 1.0; pad = 0.5
fcen = 0.5; fwidth = 0.15; cell_y = 3.0; cell_x = design_Lx + 2*pad + 2*pml
output_mon_x = design_Lx/2 + pad/2
Nx_d, Ny_d = 7, 10; n_params = Nx_d * Ny_d
fixed_runtime = 50.0

geometry = [Khronos.Object(Cuboid([0,0,0],[cell_x+1,cell_y+1,0.0]),Khronos.Material(ε=ε_lo))]
sources = [Khronos.UniformSource(time_profile=Khronos.GaussianPulseSource(fcen=fcen,fwidth=fwidth),
    component=Khronos.Ez(),center=[-(design_Lx/2+pad),0,0],
    size=[0,cell_y-2*pml,0],amplitude=1.0)]

dr = Khronos.DesignRegion(volume=Khronos.Volume(center=[0,0,0],
    size=[design_Lx,cell_y-2*pml,0]),
    design_parameters=fill(0.5,n_params),grid_size=(Nx_d,Ny_d),ε_min=ε_lo,ε_max=ε_hi)

sim = Khronos.Simulation(cell_size=[cell_x,cell_y,0],cell_center=[0,0,0],resolution=20,
    geometry=geometry,sources=sources,boundaries=[[pml,pml],[pml,pml],[0,0]],
    monitors=Khronos.Monitor[])
Khronos.prepare_simulation!(sim)
Khronos.init_design_region!(sim, dr)

# ── Step 1: Forward run ─────────────────────────────────────────
Random.seed!(42)
rho0 = 0.3 .+ 0.4 .* rand(n_params)

Khronos.update_design!(sim, dr, rho0)
Khronos.reset_fields!(sim)
empty!(sim.monitor_data)

# Install objective monitor + design region monitors
obj_mon = Khronos.DFTMonitor(component=Khronos.Ez(),
    center=[output_mon_x,0,0],size=[0,cell_y-2*pml,0],frequencies=[fcen])
push!(sim.monitor_data, Khronos.init_monitors(sim, obj_mon))

fwd_design_mons = Khronos.install_design_region_monitors!(sim, dr, [fcen])
for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end

Khronos.run(sim; until_after_sources=Khronos.stop_when_dft_decayed(
    tolerance=1e-4, maximum_runtime=fixed_runtime))

# Save forward data
fwd_ez_design = Array(fwd_design_mons[3].monitor_data.fields)  # Ez component
obj_dft = Array(obj_mon.monitor_data.fields)
f0 = real(sum(abs2.(obj_dft)))
println("Forward: f0 = $f0")
println("  Obj DFT shape: $(size(obj_dft)), max|val|=$(maximum(abs.(obj_dft)))")
println("  Fwd Ez design shape: $(size(fwd_ez_design))")

# ── Step 2: Adjoint run with UNIT source ─────────────────────────
# Place a single-point unit-amplitude source at center of monitor
# (instead of per-voxel dJ·scale source)
Khronos.reset_fields!(sim)
empty!(sim.monitor_data)

adj_design_mons = Khronos.install_design_region_monitors!(sim, dr, [fcen])
for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end

# Replace sources with dJ-weighted Ez SourceData at the monitor
# (correct dJ amplitudes but NO adj_src_scale — just raw dJ)
tp = Khronos.GaussianPulseSource(fcen=fcen, fwidth=fwidth)

# dJ = ∂f/∂z* = z for f = |z|²
dJ = vec(obj_dft[:, :, :, 1])  # per-voxel Wirtinger derivative
dJ_3d = reshape(ComplexF64.(dJ), size(obj_dft)[1:3]...)
println("  dJ shape: $(size(dJ_3d)), max|dJ|=$(maximum(abs.(dJ_3d)))")

dJ_gpu = Khronos.complex_backend_array(Khronos.complex_backend_number.(dJ_3d))
sd_adj = Khronos.SourceData{Khronos.complex_backend_array}(
    amplitude_data = dJ_gpu,
    time_src = tp,
    gv = obj_mon.monitor_data.gv,
    component = Khronos.Ez())

# Keep original forward sources for last_source_time
sim.source_data = [sd_adj]
for chunk in sim.chunk_data; chunk.source_data = sim.source_data; end

Khronos.run(sim; until_after_sources=Khronos.stop_when_dft_decayed(
    tolerance=1e-4, maximum_runtime=fixed_runtime))

adj_ez_design = Array(adj_design_mons[3].monitor_data.fields)
println("\nAdjoint (unit source): Ez design shape: $(size(adj_ez_design))")
println("  max|adj_ez| = $(maximum(abs.(adj_ez_design)))")

# ── Step 3: Compute gradient from raw overlap ────────────────────
# gradient_raw_k = dε · Σ_voxels w_ki · Re[conj(adj_ez_i) · fwd_ez_i]
dε = ε_hi - ε_lo
weights = dr.interp_weights_Ez
nx = size(fwd_ez_design, 1)
ny = size(fwd_ez_design, 2)
nz = size(fwd_ez_design, 3)

grad_raw = zeros(n_params)
for (yee_idx, design_idx, w) in weights
    iy = div(yee_idx - 1, nx) + 1
    ix = mod(yee_idx - 1, nx) + 1
    ix > nx && continue; iy > ny && continue

    overlap = zero(ComplexF64)
    for iz in 1:nz
        overlap += conj(adj_ez_design[ix, iy, iz, 1]) * fwd_ez_design[ix, iy, iz, 1]
    end
    if dr.volume.size[3] == 0.0; overlap /= max(1, nz); end
    grad_raw[design_idx] += real(dε * w * overlap)
end

# ── Step 4: FD gradient for comparison ───────────────────────────
function eval_objective(rho_vec)
    Khronos.update_design!(sim, dr, rho_vec)
    Khronos.reset_fields!(sim)
    empty!(sim.monitor_data)
    mon = Khronos.DFTMonitor(component=Khronos.Ez(),
        center=[output_mon_x,0,0],size=[0,cell_y-2*pml,0],frequencies=[fcen])
    push!(sim.monitor_data, Khronos.init_monitors(sim, mon))
    for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end
    sim.sources = sources  # restore forward sources
    Khronos.add_sources(sim, sim.sources)
    for chunk in sim.chunk_data; chunk.source_data = sim.source_data; end
    Khronos.run(sim; until_after_sources=Khronos.stop_when_dft_decayed(
        tolerance=1e-4, maximum_runtime=fixed_runtime))
    return real(sum(abs2.(Array(mon.monitor_data.fields))))
end

db = 5e-3
Random.seed!(99)
test_idx = sort(randperm(n_params)[1:5])

println("\n" * "=" ^ 70)
println("Raw overlap gradient vs FD (unit adjoint source, no scaling)")
println("=" ^ 70)
println(@sprintf("  %-4s  %-12s  %-12s  %-8s", "idx", "grad_raw", "fd_grad", "ratio"))
println("  " * "-" ^ 45)

ratios = Float64[]
for k in test_idx
    rho_p = copy(rho0); rho_p[k] = min(rho0[k]+db, 1.0)
    rho_m = copy(rho0); rho_m[k] = max(rho0[k]-db, 0.0)
    f_p = eval_objective(rho_p)
    f_m = eval_objective(rho_m)
    fd = (f_p - f_m) / (rho_p[k] - rho_m[k])
    ratio = abs(fd) > 1e-10 ? grad_raw[k] / fd : NaN
    push!(ratios, ratio)
    @printf("  %-4d  %+10.4e  %+10.4e  %+8.4f\n", k, grad_raw[k], fd, ratio)
end

valid_ratios = filter(!isnan, ratios)
if length(valid_ratios) > 0
    println("\n  Mean ratio: $(mean(valid_ratios))")
    println("  Std ratio:  $(std(valid_ratios))")
    println("  This ratio is the adj_src_scale factor needed to correct the gradient")
end
