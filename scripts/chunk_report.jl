# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Diagnostic: print the chunk layout for a simulation without running it.
# Usage: include("scripts/chunk_report.jl")

import Khronos
using GeometryPrimitives

"""
    chunk_report(sim; prepare=true)

Print a detailed report of the chunk layout for a simulation.
If `prepare=true`, calls `init_geometry` and `init_boundaries` (but not full prepare).
"""
function chunk_report(sim::Khronos.SimulationData; prepare=true)
    if prepare
        Khronos.init_geometry(sim, sim.geometry)
        Khronos.init_boundaries(sim, sim.boundaries)
        Khronos.add_sources(sim, sim.sources)
    end

    plan = Khronos.plan_chunks(sim)

    total_voxels = sim.Nx * sim.Ny * max(sim.Nz, 1)

    println("=" ^ 78)
    println("CHUNK LAYOUT REPORT")
    println("=" ^ 78)
    println("  Domain:        $(sim.Nx) × $(sim.Ny) × $(max(sim.Nz,1)) = $(total_voxels) voxels")
    println("  Cell size:     $(sim.cell_size)")
    println("  Resolution:    $(sim.resolution)")
    println("  num_chunks:    $(sim.num_chunks)")
    println("  Planned chunks: $(plan.total_chunks)")
    println("  Adjacencies:   $(length(plan.adjacency))")

    if !isnothing(sim.boundaries)
        for (axis, label) in enumerate(["x", "y", "z"])
            if length(sim.boundaries) >= axis
                pml_l = sim.boundaries[axis][1]
                pml_r = sim.boundaries[axis][2]
                Δ = [sim.Δx, sim.Δy, sim.Δz][axis]
                N = [sim.Nx, sim.Ny, max(sim.Nz, 1)][axis]
                pml_l_cells = pml_l > 0 ? ceil(Int, pml_l / Δ) : 0
                pml_r_cells = pml_r > 0 ? ceil(Int, pml_r / Δ) : 0
                println("  PML $label:        left=$(pml_l)μm ($(pml_l_cells) cells), right=$(pml_r)μm ($(pml_r_cells) cells)")
            end
        end
    else
        println("  PML:           none")
    end

    println()
    println("-" ^ 78)
    println("  #  │ Grid range                    │ Voxels   │ Physics flags")
    println("-" ^ 78)

    for spec in plan.chunks
        gv = spec.grid_volume
        pf = spec.physics
        nvox = gv.Nx * gv.Ny * max(1, gv.Nz)
        pct = round(100 * nvox / total_voxels, digits=1)

        # Build flags string
        flags = String[]
        pf.has_epsilon && push!(flags, "ε")
        pf.has_mu && push!(flags, "μ")
        pf.has_sigma_D && push!(flags, "σD")
        pf.has_sigma_B && push!(flags, "σB")
        pf.has_pml_x && push!(flags, "PML_x")
        pf.has_pml_y && push!(flags, "PML_y")
        pf.has_pml_z && push!(flags, "PML_z")
        pf.has_sources && push!(flags, "src")
        pf.has_monitors && push!(flags, "mon")
        isempty(flags) && push!(flags, "vacuum")

        range_str = "[$(gv.start_idx[1]):$(gv.end_idx[1]), $(gv.start_idx[2]):$(gv.end_idx[2]), $(gv.start_idx[3]):$(gv.end_idx[3])]"

        println("  $(lpad(spec.id, 2)) │ $(rpad(range_str, 29)) │ $(lpad(nvox, 8)) │ $(join(flags, ", "))")
    end
    println("-" ^ 78)

    # Kernel variant analysis
    println()
    println("KERNEL SPECIALIZATION ANALYSIS")
    println("-" ^ 78)

    # Group chunks by unique physics signature
    physics_groups = Dict{Khronos.PhysicsFlags, Vector{Int}}()
    for spec in plan.chunks
        if !haskey(physics_groups, spec.physics)
            physics_groups[spec.physics] = Int[]
        end
        push!(physics_groups[spec.physics], spec.id)
    end

    println("  Unique kernel variants: $(length(physics_groups))")
    for (pf, ids) in physics_groups
        flags = String[]
        pf.has_epsilon && push!(flags, "ε")
        pf.has_mu && push!(flags, "μ")
        pf.has_sigma_D && push!(flags, "σD")
        pf.has_sigma_B && push!(flags, "σB")
        pf.has_pml_x && push!(flags, "PML_x")
        pf.has_pml_y && push!(flags, "PML_y")
        pf.has_pml_z && push!(flags, "PML_z")
        isempty(flags) && push!(flags, "vacuum-only")

        # Count Nothing args this variant would produce
        nothing_count = 0
        !pf.has_epsilon && (nothing_count += 3)  # ε_inv_x/y/z
        !pf.has_mu && (nothing_count += 3)       # μ_inv_x/y/z
        !pf.has_sigma_D && (nothing_count += 3)  # σDx/y/z
        !pf.has_sigma_B && (nothing_count += 3)  # σBx/y/z
        !pf.has_pml_x && !pf.has_pml_y && !pf.has_pml_z && (nothing_count += 18)  # C/U/W for B and D

        total_voxels_in_group = sum(
            plan.chunks[id].grid_volume.Nx * plan.chunks[id].grid_volume.Ny * max(1, plan.chunks[id].grid_volume.Nz)
            for id in ids
        )

        println("  Variant: $(rpad(join(flags, "+"), 30)) chunks=$(ids)  voxels=$(total_voxels_in_group)  nothing_args≈$(nothing_count)")
    end

    # Ideal decomposition analysis
    println()
    println("IDEAL DECOMPOSITION ANALYSIS")
    println("-" ^ 78)
    if !isnothing(sim.boundaries) && all(length(b) >= 2 for b in sim.boundaries)
        ndims = sim.ndims
        ideal_regions = 1  # interior
        if ndims == 3
            ideal_regions += 6  # faces (1D PML)
            ideal_regions += 12 # edges (2D PML)
            ideal_regions += 8  # corners (3D PML)
            println("  3D with PML on all faces: ideal = 27 regions (1 interior + 6 faces + 12 edges + 8 corners)")
        elseif ndims == 2
            ideal_regions += 4  # sides (1D PML)
            ideal_regions += 4  # corners (2D PML)
            println("  2D with PML on all sides: ideal = 9 regions (1 interior + 4 sides + 4 corners)")
        end
        if !isnothing(sim.geometry) && !isempty(sim.geometry)
            n_objs = length(sim.geometry)
            println("  Plus $(n_objs) geometry object(s): interior may split further")
        end
        println("  Current plan: $(plan.total_chunks) chunk(s)")
        if plan.total_chunks == 1
            println("  ⚠ ALL voxels use the SAME kernel variant — no specialization benefit")
        elseif plan.total_chunks < ideal_regions
            println("  ⚠ Fewer chunks than ideal — some chunks have mixed physics")
        end
    end

    println("=" ^ 78)
    return plan
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run reports on all examples
# ═══════════════════════════════════════════════════════════════════════════════

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

# --- 1. Dipole ---
println("\n\n", "▶" ^ 30, " DIPOLE ", "◀" ^ 30)
sim_dipole = Khronos.Simulation(
    cell_size = [4.0, 4.0, 4.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 10,
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
            amplitude = 1im,
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
)
chunk_report(sim_dipole)

# Same with num_chunks=8
println("\n  >> Now with num_chunks=8:")
sim_dipole_8 = Khronos.Simulation(
    cell_size = [4.0, 4.0, 4.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 10,
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
            amplitude = 1im,
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    num_chunks = 8,
)
chunk_report(sim_dipole_8)

# --- 2. Sphere ---
println("\n\n", "▶" ^ 30, " SPHERE ", "◀" ^ 30)
s_xyz = 2.0 + 1.0 + 2 * 1.0
sim_sphere = Khronos.Simulation(
    cell_size = [s_xyz, s_xyz, s_xyz],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 20,
    sources = [
        Khronos.PlaneWaveSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            center = [0.0, 0.0, -s_xyz / 2.0 + 1.0],
            size = [Inf, Inf, 0.0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
            amplitude = 1im,
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 1.0), Khronos.Material(ε = 3, σD = 5))],
)
chunk_report(sim_sphere)

# Same with num_chunks=8
println("\n  >> Now with num_chunks=8:")
sim_sphere_8 = Khronos.Simulation(
    cell_size = [s_xyz, s_xyz, s_xyz],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 20,
    sources = [
        Khronos.PlaneWaveSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0),
            center = [0.0, 0.0, -s_xyz / 2.0 + 1.0],
            size = [Inf, Inf, 0.0],
            k_vector = [0.0, 0.0, 1.0],
            polarization_angle = 0.0,
            amplitude = 1im,
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    geometry = [Khronos.Object(Ball([0.0, 0.0, 0.0], 1.0), Khronos.Material(ε = 3, σD = 5))],
    num_chunks = 8,
)
chunk_report(sim_sphere_8)

# --- 3. Waveguide ---
println("\n\n", "▶" ^ 30, " WAVEGUIDE ", "◀" ^ 30)
sim_wg = Khronos.Simulation(
    cell_size = [4.0, 4.0, 6.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 25,
    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [100.0, 0.5, 0.22]), Khronos.Material(ε = 3.4^2)),
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]), Khronos.Material(ε = 1.44^2)),
    ],
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / 1.55),
            component = Khronos.Ez(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
)
chunk_report(sim_wg)

# Same with num_chunks=8
println("\n  >> Now with num_chunks=8:")
sim_wg_8 = Khronos.Simulation(
    cell_size = [4.0, 4.0, 6.0],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 25,
    geometry = [
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [100.0, 0.5, 0.22]), Khronos.Material(ε = 3.4^2)),
        Khronos.Object(Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]), Khronos.Material(ε = 1.44^2)),
    ],
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen = 1.0 / 1.55),
            component = Khronos.Ez(),
            center = [0.0, 0.0, 0.0],
            size = [0.0, 0.0, 0.0],
        ),
    ],
    boundaries = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    num_chunks = 8,
)
chunk_report(sim_wg_8)

# --- 4. Throughput configs ---
println("\n\n", "▶" ^ 30, " THROUGHPUT CONFIGS ", "◀" ^ 30)
cell_val = 128.0 / 10.0
dpml = 1.0
base_source = Khronos.UniformSource(
    time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
    component = Khronos.Ez(),
    center = [0.0, 0.0, 0.0],
    size = [0.0, 0.0, 0.0],
)

# Config A: vacuum no PML
println("\n  Config A: Vacuum (no PML)")
sim_a = Khronos.Simulation(
    cell_size = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 10,
    sources = [base_source],
)
chunk_report(sim_a)

# Config B: vacuum + PML
println("\n  Config B: Vacuum + PML")
sim_b = Khronos.Simulation(
    cell_size = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 10,
    sources = [base_source],
    boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
)
chunk_report(sim_b)

# Config B with num_chunks=27
println("\n  Config B: Vacuum + PML (num_chunks=27 -- ideal for 3D PML)")
sim_b27 = Khronos.Simulation(
    cell_size = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution = 10,
    sources = [base_source],
    boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
    num_chunks = 27,
)
chunk_report(sim_b27)

println("\n\nDone.")
