# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Staged validation of the adjoint gradient pipeline.
# Each stage tests one component in isolation.

import Khronos
using GeometryPrimitives
using LinearAlgebra
using Printf
using Random
using Statistics

Khronos.choose_backend(Khronos.CPUDevice(), Float64)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 1: Wirtinger derivative (dJ computation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("=" ^ 70)
println("STAGE 1: Wirtinger derivative validation")
println("=" ^ 70)

# For f(z) = |z|², the conjugate Wirtinger ∂f/∂z* = z
z1 = [3.0 + 4.0im, -1.0 + 2.0im, 0.5 - 0.3im]
f_abs2(x...) = real(sum(abs2.(x[1])))
dJ1 = Khronos._compute_jacobian(f_abs2, Any[z1], 1)
stage1_pass = isapprox(dJ1, z1, rtol=1e-10)
println("  f = |z|²:  dJ = ∂f/∂z* should equal z")
println("  z  = $z1")
println("  dJ = $dJ1")
println("  PASS: $stage1_pass")

# For f(z) = Re(z), ∂f/∂z* = 1/2
z2 = [2.0 + 3.0im]
f_re(x...) = real(x[1][1])
dJ2 = Khronos._compute_jacobian(f_re, Any[z2], 1)
stage1b_pass = isapprox(dJ2, [0.5 + 0.0im], rtol=1e-10)
println("\n  f = Re(z): dJ = ∂f/∂z* should equal 0.5")
println("  dJ = $dJ2")
println("  PASS: $stage1b_pass")

# For f(z) = |z|⁴ = (z·z*)², ∂f/∂z* = 2·z·|z|²
z3 = [1.0 + 1.0im]
f_abs4(x...) = real(sum(abs2.(x[1])))^2
dJ3 = Khronos._compute_jacobian(f_abs4, Any[z3], 1)
expected3 = 2 .* z3 .* abs2.(z3)
stage1c_pass = isapprox(dJ3, expected3, rtol=1e-8)
println("\n  f = |z|⁴: dJ = ∂f/∂z* should equal 2z|z|²")
println("  dJ       = $dJ3")
println("  expected = $expected3")
println("  PASS: $stage1c_pass")

stage1_all = stage1_pass && stage1b_pass && stage1c_pass
println("\n  STAGE 1 OVERALL: $(stage1_all ? "PASS ✓" : "FAIL ✗")")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2: Adjoint source scale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 2: Adjoint source scale validation")
println("=" ^ 70)

# Verify the iomega correction
dt = 0.025
fcen = 0.5
ω = 2π * fcen
iomega_computed = (1.0 - exp(-im * ω * dt)) / dt
iomega_expected = im * ω  # should be close for small dt
println("  iomega computed: $(abs(iomega_computed))")
println("  iomega expected (jω): $(abs(iomega_expected))")
println("  relative diff: $(@sprintf("%.2e", abs(iomega_computed - iomega_expected) / abs(iomega_expected)))")
println("  (should be O(Δt) ≈ $(dt))")

# Verify fwd_dtft is nonzero and has reasonable magnitude
# Use the actual test setup
println("\n  [Checking fwd_dtft with actual simulation setup...]")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 3: Forward simulation DFT fields at design region
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 3: Forward DFT fields at design region")
println("=" ^ 70)

ε_hi = 4.0; ε_lo = 1.0; design_Lx = 2.0
pml = 1.0; pad = 0.5; fwidth = 0.15
cell_y = 3.0; cell_x = design_Lx + 2 * pad + 2 * pml

geometry = [Khronos.Object(Cuboid([0, 0, 0], [cell_x + 1, cell_y + 1, 0.0]),
    Khronos.Material(ε = ε_lo))]
sources = [Khronos.UniformSource(
    time_profile = Khronos.GaussianPulseSource(fcen = fcen, fwidth = fwidth),
    component = Khronos.Ez(),
    center = [-(design_Lx / 2 + pad), 0.0, 0.0],
    size = [0.0, cell_y - 2 * pml, 0.0],
    amplitude = 1.0)]

Nx_d, Ny_d = 7, 10
n_params = Nx_d * Ny_d

dr = Khronos.DesignRegion(
    volume = Khronos.Volume(center = [0.0, 0.0, 0.0],
        size = [design_Lx, cell_y - 2 * pml, 0.0]),
    design_parameters = fill(0.5, n_params),
    grid_size = (Nx_d, Ny_d), ε_min = ε_lo, ε_max = ε_hi)

output_mon_x = design_Lx / 2 + pad / 2
output_obj = Khronos.FourierFieldsObjective(
    volume = Khronos.Volume(center = [output_mon_x, 0.0, 0.0],
        size = [0.0, cell_y - 2 * pml, 0.0]),
    component = Khronos.Ez())

objective = ez_fields -> real(sum(abs2.(ez_fields)))

sim = Khronos.Simulation(
    cell_size = [cell_x, cell_y, 0.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 20,
    geometry = geometry,
    sources = sources,
    boundaries = [[pml, pml], [pml, pml], [0.0, 0.0]],
    monitors = Khronos.Monitor[])

Khronos.prepare_simulation!(sim)
Khronos.init_design_region!(sim, dr)

fixed_runtime = 50.0
opt = Khronos.OptimizationProblem(
    sim = sim,
    objective_functions = [objective],
    objective_arguments = Khronos.ObjectiveQuantity[output_obj],
    design_regions = [dr],
    frequencies = [fcen],
    decay_by = 1e-4,
    maximum_run_time = fixed_runtime)

# Use uniform ρ=0.5 for predictable results
rho0 = fill(0.5, n_params)

Khronos.update_design!(sim, dr, rho0)
opt.current_state = :INIT
f0 = Khronos.forward_run!(opt)
println("  f0 = $f0")

# Check forward DFT at design region
for (i, comp) in enumerate(["Ex", "Ey", "Ez"])
    fwd_f = Array(opt.forward_design_monitors[i].monitor_data.fields)
    println("  Fwd DFT $comp: shape=$(size(fwd_f)), max|val|=$(maximum(abs.(fwd_f)))")
end

# For 2D TE, only Ez should be nonzero
fwd_ex_max = maximum(abs.(Array(opt.forward_design_monitors[1].monitor_data.fields)))
fwd_ey_max = maximum(abs.(Array(opt.forward_design_monitors[2].monitor_data.fields)))
fwd_ez_max = maximum(abs.(Array(opt.forward_design_monitors[3].monitor_data.fields)))
stage3_pass = fwd_ex_max < 1e-10 && fwd_ey_max < 1e-10 && fwd_ez_max > 0.1
println("  Ex≈0, Ey≈0, Ez>0: $stage3_pass")

# Check y-symmetry of Ez DFT (uniform design should be symmetric)
fwd_ez = Array(opt.forward_design_monitors[3].monitor_data.fields)[:, :, 1, 1]
ny_ez = size(fwd_ez, 2)
sym_err = maximum(abs.(fwd_ez[:, 1:ny_ez÷2] .- fwd_ez[:, ny_ez:-1:ny_ez÷2+2]))
println("  y-symmetry error of forward Ez DFT: $(@sprintf("%.2e", sym_err))")
println("  (should be ~0 for uniform design with symmetric BCs)")

println("\n  STAGE 3: $(stage3_pass ? "PASS ✓" : "FAIL ✗")")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 4: Adjoint source and adjoint DFT fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 4: Adjoint source and adjoint DFT fields")
println("=" ^ 70)

# Compute dJ
dJ = Khronos._compute_jacobian(opt.objective_functions[1], opt.results_list, 1)
eval_data = vec(opt.results_list[1])
println("  dJ matches eval_data (∂f/∂z* = z for |z|²): $(isapprox(dJ, eval_data, rtol=1e-8))")

# Run adjoint
Khronos.adjoint_run!(opt)

# Check adjoint DFT
for (i, comp) in enumerate(["Ex", "Ey", "Ez"])
    adj_f = Array(opt.adjoint_design_monitors[i].monitor_data.fields)
    println("  Adj DFT $comp: shape=$(size(adj_f)), max|val|=$(maximum(abs.(adj_f)))")
end

adj_ez = Array(opt.adjoint_design_monitors[3].monitor_data.fields)[:, :, 1, 1]
println("\n  Adjoint Ez DFT spatial pattern:")
println("    mean |adj_ez| = $(mean(abs.(adj_ez)))")
println("    std  |adj_ez| = $(std(abs.(adj_ez)))")
println("    coefficient of variation = $(@sprintf("%.2f", std(abs.(adj_ez)) / mean(abs.(adj_ez))))")
println("    (>0.1 means spatially varying, which is expected for random design)")

# Check forward DFT was preserved through adjoint run
fwd_ez_after = Array(opt.forward_design_monitors[3].monitor_data.fields)[:, :, 1, 1]
fwd_preserved = isapprox(fwd_ez, fwd_ez_after, rtol=1e-10)
println("\n  Forward DFT preserved after adjoint: $fwd_preserved")

stage4_pass = fwd_preserved
println("\n  STAGE 4: $(stage4_pass ? "PASS ✓" : "NEEDS REVIEW")")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 5: Gradient kernel (overlap integral + restriction)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 5: Gradient kernel validation")
println("=" ^ 70)

Khronos.calculate_gradient!(opt)
adj_grad = vec(opt.gradient)
println("  |gradient| = $(norm(adj_grad))")
println("  gradient range: [$(minimum(adj_grad)), $(maximum(adj_grad))]")

# Check: are ALL gradient components the same sign?
n_pos = count(x -> x > 0, adj_grad)
n_neg = count(x -> x < 0, adj_grad)
println("  Positive components: $n_pos / $(length(adj_grad))")
println("  Negative components: $n_neg / $(length(adj_grad))")
println("  (mixed signs expected for spatially-varying sensitivity)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 6: Direct overlap integral validation
# Manually compute Re[E_adj · E_fwd] at each Yee voxel and
# check if it varies spatially (as it should).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 6: Direct overlap integral analysis")
println("=" ^ 70)

fwd_ez_full = Array(opt.forward_design_monitors[3].monitor_data.fields)
adj_ez_full = Array(opt.adjoint_design_monitors[3].monitor_data.fields)

# Compute per-voxel overlap (no conj, as in the paper)
overlap_no_conj = real.(adj_ez_full[:, :, 1, 1] .* fwd_ez_full[:, :, 1, 1])
overlap_with_conj = real.(conj.(adj_ez_full[:, :, 1, 1]) .* fwd_ez_full[:, :, 1, 1])

println("  Per-voxel Re[E_adj · E_fwd] (no conj):")
println("    min = $(minimum(overlap_no_conj))")
println("    max = $(maximum(overlap_no_conj))")
println("    mean = $(mean(overlap_no_conj))")
n_pos_nc = count(x -> x > 0, overlap_no_conj)
println("    positive voxels: $n_pos_nc / $(length(overlap_no_conj))")
println("    sign variation: $(n_pos_nc > 0 && n_pos_nc < length(overlap_no_conj) ? "YES (good)" : "NO (suspicious)")")

println("\n  Per-voxel Re[conj(E_adj) · E_fwd] (with conj):")
println("    min = $(minimum(overlap_with_conj))")
println("    max = $(maximum(overlap_with_conj))")
println("    mean = $(mean(overlap_with_conj))")
n_pos_wc = count(x -> x > 0, overlap_with_conj)
println("    positive voxels: $n_pos_wc / $(length(overlap_with_conj))")
println("    sign variation: $(n_pos_wc > 0 && n_pos_wc < length(overlap_with_conj) ? "YES (good)" : "NO (suspicious)")")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 7: FD gradient for comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
println("\n" * "=" ^ 70)
println("STAGE 7: Finite difference comparison (5 random parameters)")
println("=" ^ 70)

function eval_objective(rho_vec)
    Khronos.update_design!(sim, dr, rho_vec)
    Khronos.reset_fields!(sim)
    empty!(sim.monitor_data)
    mon = Khronos.DFTMonitor(
        component = Khronos.Ez(),
        center = [output_mon_x, 0.0, 0.0],
        size = [0.0, cell_y - 2 * pml, 0.0],
        frequencies = [fcen])
    push!(sim.monitor_data, Khronos.init_monitors(sim, mon))
    for chunk in sim.chunk_data; chunk.monitor_data = sim.monitor_data; end
    Khronos.run(sim; until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-4, maximum_runtime = fixed_runtime))
    return real(sum(abs2.(Array(mon.monitor_data.fields))))
end

Random.seed!(99)
test_indices = sort(randperm(n_params)[1:5])
db = 5e-3

println(@sprintf("  %-4s  %-12s  %-12s  %-8s", "idx", "adj_grad", "fd_grad", "ratio"))
println("  " * "-" ^ 45)

for k in test_indices
    rho_p = copy(rho0); rho_p[k] = min(rho0[k] + db, 1.0)
    rho_m = copy(rho0); rho_m[k] = max(rho0[k] - db, 0.0)
    f_p = eval_objective(rho_p)
    f_m = eval_objective(rho_m)
    fd = (f_p - f_m) / (rho_p[k] - rho_m[k])
    ratio = abs(fd) > 1e-10 ? adj_grad[k] / fd : NaN
    @printf("  %-4d  %+10.4e  %+10.4e  %+8.4f\n", k, adj_grad[k], fd, ratio)
end

println("\n" * "=" ^ 70)
println("DONE")
println("=" ^ 70)
