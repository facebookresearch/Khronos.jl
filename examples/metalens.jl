# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Pancharatnam-Berry Phase Metalens — Khronos.jl
#
# 3D FDTD simulation of a visible-wavelength metalens: rotated rectangular
# TiO2 pillars on SiO2 substrate, circularly polarized illumination at 660 nm.
# Based on the design from Khorasaninejad et al. (Science 2016).
#
# Usage:
#   julia --project=. examples/metalens.jl --cells 75 --float32   # 75×75 lens, single GPU
#   julia --project=. examples/metalens.jl --benchmark --float32   # benchmark (no monitors)
#
#   # Multi-GPU:
#   ~/.julia/bin/mpiexecjl -np 2 julia -t auto --project=. examples/metalens.jl \
#       --cells 153 --float32

import Khronos
using GeometryPrimitives
using StaticArrays
using LinearAlgebra
using CUDA

# ── Parse command-line options ────────────────────────────────────────────────

const USE_SMALL = "--small" in ARGS
const USE_MEDIUM = "--medium" in ARGS
const BENCHMARK_MODE = "--benchmark" in ARGS
const PROFILE_MODE = "--profile" in ARGS
const USE_FLOAT32 = "--float32" in ARGS
const ENABLE_VIZ = "--viz" in ARGS
const CUSTOM_CELLS = let idx = findfirst(==("--cells"), ARGS)
    idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx+1]) : nothing
end

# ── MPI initialization (before backend, so is_distributed() is available) ────

const USE_MPI = haskey(ENV, "OMPI_COMM_WORLD_SIZE") || haskey(ENV, "PMI_RANK") || haskey(ENV, "SLURM_PROCID")
if USE_MPI
    Khronos.init_mpi!()
end

# ── Backend selection ─────────────────────────────────────────────────────────

precision = USE_FLOAT32 ? Float32 : Float64
Khronos.choose_backend(Khronos.CUDADevice(), precision)

if USE_MPI
    Khronos.select_device_for_rank!()
end

const RANK = USE_MPI ? Khronos.mpi_rank() : 0
const NRANKS = USE_MPI ? Khronos.mpi_size() : 1

# ── Physical parameters ──────────────────────────────────────────────────────

nm = 1e-3                                # Khronos uses μm

wavelength   = 660 * nm                  # 0.660 μm
NA           = 0.8
n_TiO2       = 2.40
n_SiO2       = 1.46
ε_TiO2       = n_TiO2^2                  # 5.76
ε_SiO2       = n_SiO2^2                  # 2.1316

rect_width   = 85  * nm                  # 0.085 μm (narrow pillar dimension)
rect_length  = 410 * nm                  # 0.410 μm (long pillar dimension)
lens_thick   = 600 * nm                  # 0.600 μm (pillar height)
cell_length  = 430 * nm                  # 0.430 μm (unit cell pitch)
spacing      = 1.0 * wavelength          # 0.660 μm (buffer above/below lens)

# ── Domain size ───────────────────────────────────────────────────────────────

if CUSTOM_CELLS !== nothing
    side_length = CUSTOM_CELLS * cell_length
elseif USE_SMALL
    side_length = 10 / 1.5 * wavelength  # ~4.4 μm
elseif USE_MEDIUM
    side_length = 50 * cell_length       # ~21.5 μm
else
    side_length = 100 * wavelength       # 66.0 μm, full benchmark
end

N_cells = floor(Int, side_length / cell_length)
length_xy = N_cells * cell_length

focal_length = length_xy / (2 * NA) * sqrt(1 - NA^2)

length_z = spacing + lens_thick + 1.1 * focal_length + spacing

# ── Chunking ──────────────────────────────────────────────────────────────────

const NUM_CHUNKS = let idx = findfirst(==("--num-chunks"), ARGS)
    idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx+1]) : :auto
end

if RANK == 0
    println("="^60)
    println("Metalens Khronos.jl Simulation")
    println("="^60)
    println("  Lens diameter:    $(round(length_xy, digits=3)) μm ($(N_cells)×$(N_cells) cells)")
    println("  Focal length:     $(round(focal_length, digits=3)) μm")
    println("  Domain size:      $(round(length_xy, digits=3)) × $(round(length_xy, digits=3)) × $(round(length_z, digits=3)) μm")
    println("  Total pillars:    $(N_cells * N_cells)")
    println("  Precision:        $(precision)")
    println("  GPUs:             $(NRANKS)")
    println("  Chunk strategy:   $(NUM_CHUNKS == :auto ? ":auto (PML-grid)" : "BSP $(NUM_CHUNKS) chunks")")
end

# ── Grid resolution ───────────────────────────────────────────────────────────

grids_per_wavelength = 18
dl = wavelength / grids_per_wavelength    # ~0.0367 μm
resolution = 1.0 / dl                     # ~27.27 pixels/μm

Nx_est = floor(Int, length_xy * resolution)
Ny_est = floor(Int, length_xy * resolution)
Nz_est = floor(Int, length_z * resolution)
if RANK == 0
    println("  Grid:             $(Nx_est) × $(Ny_est) × $(Nz_est) = $(Nx_est * Ny_est * Nz_est) voxels")
    println("  Resolution:       $(round(resolution, digits=2)) px/μm (dl=$(round(dl*1000, digits=2)) nm)")
    println("="^60)
end

# ── Materials ─────────────────────────────────────────────────────────────────

TiO2_mat = Khronos.Material(ε = ε_TiO2)
SiO2_mat = Khronos.Material(ε = ε_SiO2)

# ── Geometry construction ─────────────────────────────────────────────────────

# Pillar center z-coordinate
center_z = -length_z / 2 + spacing + lens_thick / 2

# Substrate: large Cuboid filling lower domain
# (Don't use Inf per ~/Khronos.jl/src/Geometry.jl:6)
substrate = Khronos.Object(
    Cuboid(
        [0.0, 0.0, -length_z + spacing],
        [2 * length_xy, 2 * length_xy, length_z],
    ),
    SiO2_mat,
)

# Phase formula (hyperbolic lens profile for Pancharatnam-Berry phase)
function pb_theta(x, y)
    return π / wavelength * (focal_length - sqrt(x^2 + y^2 + focal_length^2))
end

# Create a rotated pillar at position (x, y) with rotation angle theta about z
function make_pillar(x, y, theta)
    # Cuboid axes: columns are the rotated coordinate axes
    axes = SMatrix{3,3}(
        cos(theta), sin(theta), 0.0,
       -sin(theta), cos(theta), 0.0,
        0.0,        0.0,        1.0,
    )
    Cuboid(
        SVector(x, y, center_z),
        SVector(rect_width, rect_length, lens_thick),
        axes,
    )
end

# Pillar center coordinates
centers = cell_length .* (0:N_cells-1) .- length_xy / 2 .+ cell_length / 2

# Assemble geometry: substrate + all pillars
RANK == 0 && println("Building geometry ($(N_cells * N_cells) pillars)...")
t_geom = time()

geometry = Khronos.Object[substrate]
sizehint!(geometry, N_cells * N_cells + 1)

for cx in centers, cy in centers
    push!(geometry, Khronos.Object(make_pillar(cx, cy, pb_theta(cx, cy)), TiO2_mat))
end

RANK == 0 && println("  Geometry built in $(round(time() - t_geom, digits=3))s")

# ── Sources (circular polarization: Ex + i*Ey) ───────────────────────────────

fcen = 1.0 / wavelength                   # ~1.5152 /μm
fwidth = fcen / 10.0
src_z = -length_z / 2 + 2 * dl            # 2 cells above bottom edge

source_x = Khronos.PlaneWaveSource(
    time_profile = Khronos.GaussianPulseSource(fcen=fcen, fwidth=fwidth),
    center = [0.0, 0.0, src_z],
    size = [Inf, Inf, 0.0],
    polarization_angle = 0.0,              # x-polarized
    k_vector = [0.0, 0.0, 1.0],
    amplitude = 1.0,
)

source_y = Khronos.PlaneWaveSource(
    time_profile = Khronos.GaussianPulseSource(fcen=fcen, fwidth=fwidth),
    center = [0.0, 0.0, src_z],
    size = [Inf, Inf, 0.0],
    polarization_angle = π / 2,            # y-polarized
    k_vector = [0.0, 0.0, 1.0],
    amplitude = -1im,                      # -π/2 phase offset for circular pol
)

sources = [source_x, source_y]

# ── Monitors ──────────────────────────────────────────────────────────────────

z_focal = -length_z / 2 + spacing + lens_thick + focal_length

monitors = Khronos.Monitor[]

if !BENCHMARK_MODE
    # Focal plane monitors (Ex and Ey for intensity |Ex|^2 + |Ey|^2)
    mon_focal_ex = Khronos.DFTMonitor(
        component = Khronos.Ex(),
        center = [0.0, 0.0, z_focal],
        size = [length_xy, length_xy, 0.0],
        frequencies = [fcen],
    )

    mon_focal_ey = Khronos.DFTMonitor(
        component = Khronos.Ey(),
        center = [0.0, 0.0, z_focal],
        size = [length_xy, length_xy, 0.0],
        frequencies = [fcen],
    )

    # Focal plane H-field monitors for Poynting flux computation
    mon_focal_hx = Khronos.DFTMonitor(
        component = Khronos.Hx(),
        center = [0.0, 0.0, z_focal],
        size = [length_xy, length_xy, 0.0],
        frequencies = [fcen],
    )

    mon_focal_hy = Khronos.DFTMonitor(
        component = Khronos.Hy(),
        center = [0.0, 0.0, z_focal],
        size = [length_xy, length_xy, 0.0],
        frequencies = [fcen],
    )

    monitors = [mon_focal_ex, mon_focal_ey, mon_focal_hx, mon_focal_hy]

    # Additional monitors for single-GPU runs (XZ cross-section + near-field)
    if NRANKS == 1
        mon_xz_ex = Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0.0, 0.0, 0.0],
            size = [length_xy, 0.0, length_z],
            frequencies = [fcen],
        )

        mon_near_ex = Khronos.DFTMonitor(
            component = Khronos.Ex(),
            center = [0.0, 0.0, -length_z / 2 + spacing + lens_thick + spacing / 2],
            size = [length_xy, length_xy, 0.0],
            frequencies = [fcen],
        )

        push!(monitors, mon_xz_ex, mon_near_ex)
    end
end

# ── Boundaries (PML) ─────────────────────────────────────────────────────────

dpml = 15 * dl    # ~0.55 μm (15 PML cells, matching Tidy3D's npml=15)
boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

# ── Simulation construction ───────────────────────────────────────────────────

sim = Khronos.Simulation(
    cell_size   = [length_xy, length_xy, length_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution  = resolution,
    Courant     = 0.55,
    geometry    = geometry,
    sources     = sources,
    boundaries  = boundaries,
    monitors    = monitors,
    num_chunks  = NUM_CHUNKS,
)

# ── Run ───────────────────────────────────────────────────────────────────────

if BENCHMARK_MODE || PROFILE_MODE
    RANK == 0 && println("\nRunning benchmark (110 timesteps)...")
    timestep_rate = Khronos.run_benchmark(sim, 110)
    if RANK == 0
        println("\nBenchmark result: $(round(timestep_rate, digits=1)) MVoxels/s")
        println("                  $(round(timestep_rate / 1000, digits=3)) GVoxels/s")
        if NRANKS > 1
            println("  Per-GPU:        $(round(timestep_rate / NRANKS, digits=1)) MVoxels/s")
            println("  Scaling eff:    (compare to single-GPU baseline)")
        end
    end

    if PROFILE_MODE
        # Print chunk decomposition
        if RANK == 0
            println("\n" * "="^60)
            println("CHUNK DECOMPOSITION")
            println("="^60)
        end
        for (i, chunk) in enumerate(sim.chunk_data)
            pf = chunk.spec.physics
            gv = chunk.spec.grid_volume
            nvox = gv.Nx * gv.Ny * max(1, gv.Nz)
            pml_str = join(filter(!isempty, [
                pf.has_pml_x ? "PML-x" : "",
                pf.has_pml_y ? "PML-y" : "",
                pf.has_pml_z ? "PML-z" : "",
            ]), "+")
            if isempty(pml_str); pml_str = "interior"; end
            g = chunk.geometry_data
            eps_type = (g.ε_inv_x isa AbstractArray) ? "per-voxel-ε($(size(g.ε_inv_x)))" : "scalar-ε=$(g.ε_inv)"
            mu_type = g.μ_inv isa Real ? "scalar-μ" : "per-voxel-μ"
            println("  Chunk $i: $(gv.Nx)×$(gv.Ny)×$(gv.Nz) = $(nvox) voxels | $pml_str | $eps_type | $mu_type | src=$(pf.has_sources)")
        end

        # Profile individual timestep components
        CUDA.synchronize()
        println("\n" * "="^60)
        println("PER-STEP PROFILING (4 steps, GPU-synchronized)")
        println("="^60)

        for step_i in 1:4
            t = Khronos.round_time(sim)

            # Source step H
            t0 = time_ns()
            if sim.sources_active
                Khronos.update_magnetic_sources!(sim, t)
            end
            CUDA.synchronize()
            t_srcH = (time_ns() - t0) / 1e6

            # Step H (curl + update for all chunks + halo)
            t0 = time_ns()
            Khronos.step_H_fused!(sim)
            CUDA.synchronize()
            t_stepH = (time_ns() - t0) / 1e6

            # Monitor H
            t0 = time_ns()
            Khronos.update_H_monitors!(sim, t)
            CUDA.synchronize()
            t_monH = (time_ns() - t0) / 1e6

            # Source step E
            t0 = time_ns()
            if sim.sources_active
                Khronos.update_electric_sources!(sim, t + sim.Δt / 2)
            end
            CUDA.synchronize()
            t_srcE = (time_ns() - t0) / 1e6

            # Step E
            t0 = time_ns()
            Khronos.step_E_fused!(sim)
            CUDA.synchronize()
            t_stepE = (time_ns() - t0) / 1e6

            # Monitor E
            t0 = time_ns()
            Khronos.update_E_monitors!(sim, t + sim.Δt / 2)
            CUDA.synchronize()
            t_monE = (time_ns() - t0) / 1e6

            t0 = time_ns()
            Khronos.increment_timestep!(sim)
            t_inc = (time_ns() - t0) / 1e6

            total = t_srcH + t_stepH + t_monH + t_srcE + t_stepE + t_monE + t_inc
            println("  Step $step_i: total=$(round(total, digits=2))ms | srcH=$(round(t_srcH, digits=2))ms | stepH=$(round(t_stepH, digits=2))ms | monH=$(round(t_monH, digits=2))ms | srcE=$(round(t_srcE, digits=2))ms | stepE=$(round(t_stepE, digits=2))ms | monE=$(round(t_monE, digits=2))ms")
        end

        # Print chunk voxel budget
        total_vox = sum(c.ndrange[1] * c.ndrange[2] * c.ndrange[3] for c in sim.chunk_data)
        pml_vox = sum(c.ndrange[1] * c.ndrange[2] * c.ndrange[3] for c in sim.chunk_data if Khronos.has_any_pml(c.spec.physics))
        int_vox = total_vox - pml_vox
        println("\n  Voxel budget (this GPU): total=$(total_vox) | interior=$(int_vox) ($(round(100*int_vox/total_vox, digits=1))%) | PML=$(pml_vox) ($(round(100*pml_vox/total_vox, digits=1))%)")

        # Count chunks by type
        n_int = count(c -> !Khronos.has_any_pml(c.spec.physics), sim.chunk_data)
        n_pml_scalar = count(c -> Khronos.has_any_pml(c.spec.physics) && !(c.geometry_data.ε_inv_x isa AbstractArray), sim.chunk_data)
        n_pml_array = count(c -> Khronos.has_any_pml(c.spec.physics) && (c.geometry_data.ε_inv_x isa AbstractArray), sim.chunk_data)
        println("  Chunks: $(length(sim.chunk_data)) total | $n_int interior | $n_pml_array PML(per-voxel-ε) | $n_pml_scalar PML(scalar-ε)")

        # ── Detailed per-chunk kernel timing ──
        # Manually replicates step_H_fused!/step_E_fused! dispatch to time each chunk
        println("\n" * "="^60)
        println("PER-CHUNK KERNEL TIMING (4 steps, CUDA events)")
        println("="^60)

        backend_engine = Khronos.backend_engine
        cuda_wg = parse(Int, get(ENV, "KHRONOS_CUDA_WORKGROUP_SIZE", "256"))
        use_raw_pml = get(ENV, "KHRONOS_RAW_PML", "0") == "1"
        dt_dx = sim.Δt / sim.Δx; dt_dy = sim.Δt / sim.Δy; dt_dz = sim.Δt / sim.Δz
        bn = Khronos.backend_number

        for step_i in 1:4
            t = Khronos.round_time(sim)
            sa = sim.sources_active

            # Source H timing
            CUDA.synchronize()
            t0 = time_ns()
            if sa; Khronos.update_magnetic_sources!(sim, t); end
            CUDA.synchronize()
            t_srcH = (time_ns() - t0) / 1e6

            # Per-chunk H timing
            t_H_int = 0.0; t_H_pml = 0.0
            for chunk in sim.chunk_data
                CUDA.synchronize()
                t0 = time_ns()
                f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data; nr = chunk.ndrange
                is_pml = Khronos.has_any_pml(chunk.spec.physics)

                if backend_engine isa Khronos.CUDABackend && !is_pml && !chunk.spec.physics.has_sources && g.μ_inv isa Real
                    iNx = Int32(nr[1]); nblocks_x = cld(Int(iNx), cuda_wg)
                    @cuda blocks=(nblocks_x, Int(nr[2]), Int(nr[3])) threads=(cuda_wg,1,1) Khronos._cuda_fused_BH_kernel!(
                        f.fEx, f.fEy, f.fEz, f.fHx, f.fHy, f.fHz,
                        bn(g.μ_inv) * bn(dt_dx), bn(g.μ_inv) * bn(dt_dy), bn(g.μ_inv) * bn(dt_dz), iNx)
                elseif !is_pml
                    sim._cached_fused_kernel(
                        f.fEx, f.fEy, f.fEz, f.fBx, f.fBy, f.fBz, f.fHx, f.fHy, f.fHz,
                        sa ? f.fSBx : nothing, sa ? f.fSBy : nothing, sa ? f.fSBz : nothing,
                        g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                        sim.Δt, sim.Δx, sim.Δy, sim.Δz, 1, ndrange=nr)
                else
                    sim._cached_curl_kernel(
                        f.fEx, f.fEy, f.fEz, f.fBx, f.fBy, f.fBz, f.fCBx, f.fCBy, f.fCBz,
                        f.fUBx, f.fUBy, f.fUBz, g.σBx, g.σBy, g.σBz, b.σBx, b.σBy, b.σBz,
                        sim.Δt, sim.Δx, sim.Δy, sim.Δz, 1, ndrange=nr)
                    sim._cached_update_kernel(
                        f.fHx, f.fHy, f.fHz, f.fBx, f.fBy, f.fBz,
                        f.fWBx, f.fWBy, f.fWBz, f.fPBx, f.fPBy, f.fPBz,
                        sa ? f.fSBx : nothing, sa ? f.fSBy : nothing, sa ? f.fSBz : nothing,
                        g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                        b.σBx, b.σBy, b.σBz, ndrange=nr)
                end
                CUDA.synchronize()
                dt = (time_ns() - t0) / 1e6
                if is_pml; t_H_pml += dt; else; t_H_int += dt; end
            end

            # Halo exchange
            CUDA.synchronize()
            t0 = time_ns()
            Khronos.exchange_halos!(sim, :H)
            CUDA.synchronize()
            t_halo_H = (time_ns() - t0) / 1e6

            # Monitor H
            t0 = time_ns()
            Khronos.update_H_monitors!(sim, t)
            CUDA.synchronize()
            t_monH = (time_ns() - t0) / 1e6

            # Source E timing
            t0 = time_ns()
            if sa; Khronos.update_electric_sources!(sim, t + sim.Δt / 2); end
            CUDA.synchronize()
            t_srcE = (time_ns() - t0) / 1e6

            # Per-chunk E timing
            t_E_int = 0.0; t_E_pml = 0.0
            for chunk in sim.chunk_data
                CUDA.synchronize()
                t0 = time_ns()
                f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data; nr = chunk.ndrange
                is_pml = Khronos.has_any_pml(chunk.spec.physics)

                if backend_engine isa Khronos.CUDABackend && !is_pml && !chunk.spec.physics.has_sources && g.ε_inv_x isa AbstractArray
                    iNx = Int32(nr[1]); nblocks_x = cld(Int(iNx), cuda_wg)
                    @cuda blocks=(nblocks_x, Int(nr[2]), Int(nr[3])) threads=(cuda_wg,1,1) Khronos._cuda_fused_DE_kernel!(
                        f.fHx, f.fHy, f.fHz, f.fEx, f.fEy, f.fEz,
                        g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                        bn(dt_dx), bn(dt_dy), bn(dt_dz), iNx)
                elseif backend_engine isa Khronos.CUDABackend && !is_pml && !chunk.spec.physics.has_sources && g.ε_inv isa Real
                    iNx = Int32(nr[1]); nblocks_x = cld(Int(iNx), cuda_wg)
                    @cuda blocks=(nblocks_x, Int(nr[2]), Int(nr[3])) threads=(cuda_wg,1,1) Khronos._cuda_fused_DE_scalar_kernel!(
                        f.fHx, f.fHy, f.fHz, f.fEx, f.fEy, f.fEz,
                        bn(g.ε_inv), bn(dt_dx), bn(dt_dy), bn(dt_dz), iNx)
                elseif !is_pml
                    sim._cached_fused_kernel(
                        f.fHx, f.fHy, f.fHz, f.fDx, f.fDy, f.fDz, f.fEx, f.fEy, f.fEz,
                        sa ? f.fSDx : nothing, sa ? f.fSDy : nothing, sa ? f.fSDz : nothing,
                        g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                        sim.Δt, sim.Δx, sim.Δy, sim.Δz, -1, ndrange=nr)
                else
                    sim._cached_curl_kernel(
                        f.fHx, f.fHy, f.fHz, f.fDx, f.fDy, f.fDz, f.fCDx, f.fCDy, f.fCDz,
                        f.fUDx, f.fUDy, f.fUDz, g.σDx, g.σDy, g.σDz, b.σDx, b.σDy, b.σDz,
                        sim.Δt, sim.Δx, sim.Δy, sim.Δz, -1, ndrange=nr)
                    sim._cached_update_kernel(
                        f.fEx, f.fEy, f.fEz, f.fDx, f.fDy, f.fDz,
                        f.fWDx, f.fWDy, f.fWDz, f.fPDx, f.fPDy, f.fPDz,
                        sa ? f.fSDx : nothing, sa ? f.fSDy : nothing, sa ? f.fSDz : nothing,
                        g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                        b.σDx, b.σDy, b.σDz, ndrange=nr)
                end
                CUDA.synchronize()
                dt = (time_ns() - t0) / 1e6
                if is_pml; t_E_pml += dt; else; t_E_int += dt; end
            end

            # Halo exchange E
            CUDA.synchronize()
            t0 = time_ns()
            Khronos.exchange_halos!(sim, :E)
            CUDA.synchronize()
            t_halo_E = (time_ns() - t0) / 1e6

            # Monitor E
            t0 = time_ns()
            Khronos.update_E_monitors!(sim, t + sim.Δt / 2)
            CUDA.synchronize()
            t_monE = (time_ns() - t0) / 1e6

            Khronos.increment_timestep!(sim)

            total = t_srcH + t_H_int + t_H_pml + t_halo_H + t_monH + t_srcE + t_E_int + t_E_pml + t_halo_E + t_monE
            println("  Step $step_i: total=$(round(total, digits=1))ms")
            println("    H: interior=$(round(t_H_int, digits=2))ms | PML=$(round(t_H_pml, digits=2))ms | halo=$(round(t_halo_H, digits=2))ms | src=$(round(t_srcH, digits=2))ms | mon=$(round(t_monH, digits=2))ms")
            println("    E: interior=$(round(t_E_int, digits=2))ms | PML=$(round(t_E_pml, digits=2))ms | halo=$(round(t_halo_E, digits=2))ms | src=$(round(t_srcE, digits=2))ms | mon=$(round(t_monE, digits=2))ms")

            # Bandwidth analysis for interior
            int_voxels = sum(c.ndrange[1] * c.ndrange[2] * c.ndrange[3] for c in sim.chunk_data if !Khronos.has_any_pml(c.spec.physics))
            bh_bw = int_voxels * 36 / (t_H_int * 1e-3) / 1e9
            de_bw = int_voxels * 48 / (t_E_int * 1e-3) / 1e9
            println("    BW: BH=$(round(bh_bw, digits=0)) GB/s ($(round(bh_bw/3350*100, digits=1))%) | DE=$(round(de_bw, digits=0)) GB/s ($(round(de_bw/3350*100, digits=1))%)")
        end
    end
else
    RANK == 0 && println("\nRunning simulation...")
    Khronos.run(sim,
        until_after_sources = Khronos.stop_when_dft_decayed(
            tolerance = 1e-8,
            minimum_runtime = 20.0 / fwidth,
            maximum_runtime = 40.0 / fwidth,
        )
    )

    # ── Post-processing ───────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("Post-processing")
    println("="^60)

    # Extract focal plane fields.
    # Yee stagger means different components have different grid sizes:
    #   Ex: (Nx, Ny+1)    Ey: (Nx+1, Ny)
    #   Hx: (Nx+1, Ny)    Hy: (Nx, Ny+1)
    # Interpolate each to the cell-center grid (Nx, Ny) by averaging neighbors.
    Ex_raw = Array(sim.monitors[1].monitor_data.fields)[:,:,1,1]  # (Nx, Ny+1)
    Ey_raw = Array(sim.monitors[2].monitor_data.fields)[:,:,1,1]  # (Nx+1, Ny)

    # Average along the staggered dimension to get cell-center values
    Ex_focal = (Ex_raw[:, 1:end-1] .+ Ex_raw[:, 2:end]) ./ 2
    Ey_focal = (Ey_raw[1:end-1, :] .+ Ey_raw[2:end, :]) ./ 2

    # Trim to common size (Yee grid rounding may cause ±1 difference)
    nx_common = min(size(Ex_focal, 1), size(Ey_focal, 1))
    ny_common = min(size(Ex_focal, 2), size(Ey_focal, 2))
    Ex_focal = Ex_focal[1:nx_common, 1:ny_common]
    Ey_focal = Ey_focal[1:nx_common, 1:ny_common]

    # Focal plane intensity
    I_focal = abs2.(Ex_focal) .+ abs2.(Ey_focal)
    I_max = maximum(I_focal)
    println("  Peak focal intensity: $(round(I_max, sigdigits=4))")

    # FWHM estimation (along x through center)
    ny_center = size(I_focal, 2) ÷ 2 + 1
    line_cut = I_focal[:, ny_center]
    half_max = I_max / 2
    above_half = line_cut .>= half_max
    if any(above_half)
        idx_above = findall(above_half)
        fwhm_pixels = idx_above[end] - idx_above[1] + 1
        fwhm_um = fwhm_pixels * dl
        println("  FWHM:                 $(round(fwhm_um * 1000, digits=1)) nm ($(fwhm_pixels) pixels)")
        println("  Diffraction limit:    $(round(0.5 * wavelength / NA * 1000, digits=1)) nm (0.5λ/NA)")
    end

    # Poynting flux at focal plane (Sz = Re(Ex * Hy_conj - Ey * Hx_conj))
    # Hx has same stagger as Ey: (Nx+1, Ny), Hy same as Ex: (Nx, Ny+1)
    # Monitors: [1]=Ex, [2]=Ey, [3]=Hx, [4]=Hy at focal plane
    if length(sim.monitors) >= 4
        Hx_raw = Array(sim.monitors[3].monitor_data.fields)[:,:,1,1]  # (Nx+1, Ny)
        Hy_raw = Array(sim.monitors[4].monitor_data.fields)[:,:,1,1]  # (Nx, Ny+1)
        Hx_focal = (Hx_raw[1:end-1, :] .+ Hx_raw[2:end, :]) ./ 2    # (Nx, Ny)
        Hy_focal = (Hy_raw[:, 1:end-1] .+ Hy_raw[:, 2:end]) ./ 2    # (Nx, Ny)
        Hx_focal = Hx_focal[1:nx_common, 1:ny_common]
        Hy_focal = Hy_focal[1:nx_common, 1:ny_common]
        Sz = real.(Ex_focal .* conj.(Hy_focal) .- Ey_focal .* conj.(Hx_focal))
        total_flux = sum(Sz) * dl * dl
        println("  Total focal flux:     $(round(total_flux, sigdigits=4))")
    end

    println("="^60)

    # ── Visualization ──────────────────────────────────────────────────────────
    if ENABLE_VIZ
        using CairoMakie

        outdir = joinpath(@__DIR__, "metalens_output")
        mkpath(outdir)
        println("\nGenerating plots in $(outdir)/...")

        # Physical coordinate axes for focal plane
        nx, ny = size(I_focal)
        xs_focal = range(-length_xy/2, length_xy/2, length=nx)
        ys_focal = range(-length_xy/2, length_xy/2, length=ny)

        # 1. Focal plane intensity heatmap
        fig1 = Figure(size=(700, 600))
        ax1 = Axis(fig1[1, 1],
            title = "Focal Plane Intensity |Ex|² + |Ey|²",
            xlabel = "x (μm)", ylabel = "y (μm)", aspect = DataAspect())
        hm1 = heatmap!(ax1, collect(xs_focal), collect(ys_focal), I_focal, colormap = :inferno)
        Colorbar(fig1[1, 2], hm1, label = "Intensity (a.u.)")
        save(joinpath(outdir, "focal_intensity.png"), fig1)
        println("  Saved focal_intensity.png")

        # 2. Focal spot line cuts (x and y through center)
        fig2 = Figure(size=(800, 400))
        ax2a = Axis(fig2[1, 1],
            title = "Focal Spot Line Cut (y=0)",
            xlabel = "x (μm)", ylabel = "Intensity (a.u.)")
        ny_c = ny ÷ 2 + 1
        lines!(ax2a, collect(xs_focal), I_focal[:, ny_c], color = :red, linewidth = 2)
        if any(above_half)
            hlines!(ax2a, [half_max], color = :gray, linestyle = :dash, label = "FWHM")
        end
        ax2b = Axis(fig2[1, 2],
            title = "Focal Spot Line Cut (x=0)",
            xlabel = "y (μm)", ylabel = "Intensity (a.u.)")
        nx_c = nx ÷ 2 + 1
        lines!(ax2b, collect(ys_focal), I_focal[nx_c, :], color = :blue, linewidth = 2)
        if any(above_half)
            hlines!(ax2b, [half_max], color = :gray, linestyle = :dash)
        end
        save(joinpath(outdir, "focal_linecuts.png"), fig2)
        println("  Saved focal_linecuts.png")

        # 3. XZ cross-section (monitor index 5, single-GPU only)
        if length(sim.monitors) >= 5
            xz_raw = Array(sim.monitors[5].monitor_data.fields)[:,:,1,1]
            xz_intensity = abs2.(xz_raw)
            nxz_x, nxz_z = size(xz_intensity)
            xs_xz = range(-length_xy/2, length_xy/2, length=nxz_x)
            zs_xz = range(-length_z/2, length_z/2, length=nxz_z)

            fig3 = Figure(size=(900, 400))
            ax3 = Axis(fig3[1, 1],
                title = "XZ Cross-Section |Ex|² (y=0)",
                xlabel = "x (μm)", ylabel = "z (μm)", aspect = DataAspect())
            hm3 = heatmap!(ax3, collect(xs_xz), collect(zs_xz), xz_intensity, colormap = :inferno)
            hlines!(ax3, [z_focal], color = :cyan, linestyle = :dash, linewidth = 1)
            hlines!(ax3, [center_z - lens_thick/2, center_z + lens_thick/2],
                    color = :white, linestyle = :dot, linewidth = 1)
            Colorbar(fig3[1, 2], hm3, label = "|Ex|² (a.u.)")
            save(joinpath(outdir, "xz_cross_section.png"), fig3)
            println("  Saved xz_cross_section.png")
        end

        # 4. Poynting flux heatmap at focal plane
        if @isdefined(Sz)
            fig4 = Figure(size=(700, 600))
            ax4 = Axis(fig4[1, 1],
                title = "Poynting Flux Sz at Focal Plane",
                xlabel = "x (μm)", ylabel = "y (μm)", aspect = DataAspect())
            hm4 = heatmap!(ax4, collect(xs_focal), collect(ys_focal), Sz, colormap = :viridis)
            Colorbar(fig4[1, 2], hm4, label = "Sz (a.u.)")
            save(joinpath(outdir, "focal_poynting.png"), fig4)
            println("  Saved focal_poynting.png")
        end

        println("  All plots saved to $(outdir)/")
    end
end

# ── MPI cleanup ──────────────────────────────────────────────────────────────
if USE_MPI
    using MPI
    MPI.Finalize()
end
