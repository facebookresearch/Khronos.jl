# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Metalens Simulation using Khronos.jl
#
# Reproduces the Tidy3D Pancharatnam-Berry phase metalens benchmark:
# 153×153 rotated rectangular TiO2 pillars on SiO2 substrate,
# illuminated with circular polarization at 660 nm, NA=0.8.
#
# Usage:
#   julia --project=. examples/metalens.jl                    # full 153×153 lens
#   julia --project=. examples/metalens.jl --small            # small 10×10 test lens
#   julia --project=. examples/metalens.jl --medium           # medium 50×50 lens
#   julia --project=. examples/metalens.jl --benchmark        # benchmark mode (no DFT)
#   julia --project=. examples/metalens.jl --float32          # use Float32 precision
#   julia --project=. examples/metalens.jl --small --viz      # run small lens with plots
#
# Multi-GPU:
#   mpirun -np N julia --project=. examples/metalens.jl

import Khronos
using GeometryPrimitives
using StaticArrays
using LinearAlgebra

# ── Parse command-line options ────────────────────────────────────────────────

const USE_SMALL = "--small" in ARGS
const USE_MEDIUM = "--medium" in ARGS
const BENCHMARK_MODE = "--benchmark" in ARGS
const USE_FLOAT32 = "--float32" in ARGS
const ENABLE_VIZ = "--viz" in ARGS

# ── Backend selection ─────────────────────────────────────────────────────────

precision = USE_FLOAT32 ? Float32 : Float64
Khronos.choose_backend(Khronos.CUDADevice(), precision)

# Initialize MPI if running distributed
if haskey(ENV, "OMPI_COMM_WORLD_SIZE") || haskey(ENV, "PMI_RANK") || haskey(ENV, "SLURM_PROCID")
    Khronos.init_mpi!()
    Khronos.select_device_for_rank!()
end

# ── Physical parameters ──────────────────────────────────────────────────────
# All values from ~/metalens/Metalens_Optimize.py

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

if USE_SMALL
    side_length = 10 / 1.5 * wavelength  # ~4.4 μm, matching optimize script
elseif USE_MEDIUM
    side_length = 50 * cell_length       # ~21.5 μm, 50 unit cells
else
    side_length = 100 * wavelength       # 66.0 μm, full benchmark
end

N_cells = floor(Int, side_length / cell_length)
length_xy = N_cells * cell_length

focal_length = length_xy / (2 * NA) * sqrt(1 - NA^2)

length_z = spacing + lens_thick + 1.1 * focal_length + spacing

println("="^60)
println("Metalens Khronos.jl Simulation")
println("="^60)
println("  Lens diameter:    $(round(length_xy, digits=3)) μm ($(N_cells)×$(N_cells) cells)")
println("  Focal length:     $(round(focal_length, digits=3)) μm")
println("  Domain size:      $(round(length_xy, digits=3)) × $(round(length_xy, digits=3)) × $(round(length_z, digits=3)) μm")
println("  Total pillars:    $(N_cells * N_cells)")
println("  Precision:        $(precision)")

# ── Grid resolution ───────────────────────────────────────────────────────────

grids_per_wavelength = 18
dl = wavelength / grids_per_wavelength    # ~0.0367 μm
resolution = 1.0 / dl                     # ~27.27 pixels/μm

Nx_est = floor(Int, length_xy * resolution)
Ny_est = floor(Int, length_xy * resolution)
Nz_est = floor(Int, length_z * resolution)
println("  Grid:             $(Nx_est) × $(Ny_est) × $(Nz_est) = $(Nx_est * Ny_est * Nz_est) voxels")
println("  Resolution:       $(round(resolution, digits=2)) px/μm (dl=$(round(dl*1000, digits=2)) nm)")
println("="^60)

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
println("Building geometry ($(N_cells * N_cells) pillars)...")
t_geom = time()

geometry = Khronos.Object[substrate]
sizehint!(geometry, N_cells * N_cells + 1)

for cx in centers, cy in centers
    push!(geometry, Khronos.Object(make_pillar(cx, cy, pb_theta(cx, cy)), TiO2_mat))
end

println("  Geometry built in $(round(time() - t_geom, digits=3))s")

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
z_near  = -length_z / 2 + spacing + lens_thick + spacing / 2

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

    # XZ cross-section monitor (for field visualization through lens center)
    mon_xz_ex = Khronos.DFTMonitor(
        component = Khronos.Ex(),
        center = [0.0, 0.0, 0.0],
        size = [length_xy, 0.0, length_z],
        frequencies = [fcen],
    )

    # Near-field plane (just above pillars)
    mon_near_ex = Khronos.DFTMonitor(
        component = Khronos.Ex(),
        center = [0.0, 0.0, z_near],
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

    monitors = [mon_focal_ex, mon_focal_ey, mon_xz_ex, mon_near_ex,
                mon_focal_hx, mon_focal_hy]
end

# ── Boundaries (PML) ─────────────────────────────────────────────────────────

dpml = 15 * dl    # ~0.55 μm (15 PML cells, matching Tidy3D's npml=15)
boundaries = [[dpml, dpml], [dpml, dpml], [dpml, dpml]]

# ── Simulation construction ───────────────────────────────────────────────────

sim = Khronos.Simulation(
    cell_size   = [length_xy, length_xy, length_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution  = resolution,
    geometry    = geometry,
    sources     = sources,
    boundaries  = boundaries,
    monitors    = monitors,
    num_chunks  = :auto,
)

# ── Run ───────────────────────────────────────────────────────────────────────

if BENCHMARK_MODE
    println("\nRunning benchmark (110 timesteps)...")
    timestep_rate = Khronos.run_benchmark(sim, 110)
    println("\nBenchmark result: $(round(timestep_rate, digits=1)) MVoxels/s")
    println("                  $(round(timestep_rate / 1000, digits=3)) GVoxels/s")
else
    println("\nRunning simulation...")
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
    if length(sim.monitors) >= 6
        Hx_raw = Array(sim.monitors[5].monitor_data.fields)[:,:,1,1]  # (Nx+1, Ny)
        Hy_raw = Array(sim.monitors[6].monitor_data.fields)[:,:,1,1]  # (Nx, Ny+1)
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

        # 3. XZ cross-section from DFT monitor (monitor index 3)
        xz_raw = Array(sim.monitors[3].monitor_data.fields)[:,:,1,1]
        xz_intensity = abs2.(xz_raw)
        nxz_x, nxz_z = size(xz_intensity)
        xs_xz = range(-length_xy/2, length_xy/2, length=nxz_x)
        zs_xz = range(-length_z/2, length_z/2, length=nxz_z)

        fig3 = Figure(size=(900, 400))
        ax3 = Axis(fig3[1, 1],
            title = "XZ Cross-Section |Ex|² (y=0)",
            xlabel = "x (μm)", ylabel = "z (μm)", aspect = DataAspect())
        hm3 = heatmap!(ax3, collect(xs_xz), collect(zs_xz), xz_intensity, colormap = :inferno)
        # Mark focal plane and pillar region
        hlines!(ax3, [z_focal], color = :cyan, linestyle = :dash, linewidth = 1)
        hlines!(ax3, [center_z - lens_thick/2, center_z + lens_thick/2],
                color = :white, linestyle = :dot, linewidth = 1)
        Colorbar(fig3[1, 2], hm3, label = "|Ex|² (a.u.)")
        save(joinpath(outdir, "xz_cross_section.png"), fig3)
        println("  Saved xz_cross_section.png")

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

        # 5. Geometry cross-section through pillar layer
        fig5 = Khronos.plot2D(sim, nothing,
            Khronos.Volume(center=[0.0, 0.0, center_z], size=[length_xy, length_xy, 0.0]),
            plot_geometry=true)
        save(joinpath(outdir, "geometry_xy.png"), fig5)
        println("  Saved geometry_xy.png")

        println("  All plots saved to $(outdir)/")
    end
end
