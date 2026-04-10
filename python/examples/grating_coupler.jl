import Khronos as K
using GeometryPrimitives, CUDA, Printf, DelimitedFiles
K.choose_backend(K.CUDADevice(), Float32)
ENV["KHRONOS_CUDA_GRAPHS"] = "0"

# ================================================================== #
# Parameters
# ================================================================== #
n_si = 3.48; n_sio2 = 1.44
si = K.Material(ε=n_si^2); sio2 = K.Material(ε=n_sio2^2)
si_thickness = 0.22; box_thickness = 3.0; clad_height = 2.5

fcen = 1/1.55; fmin = 1/1.60; fmax = 1/1.50
fwidth = fmax - fmin; nfreq = 101
freqs = collect(range(fmin, fmax, length=nfreq))

# Fiber: 15° from normal (standard SiPh convention)
fiber_angle_deg = 15.0; fiber_waist = 5.2
fiber_angle_rad = deg2rad(fiber_angle_deg)
# k-vector: mostly -z (downward), slight +x (toward waveguide)
beam_kdir = [sin(fiber_angle_rad), 0.0, -cos(fiber_angle_rad)]

# Domain
dpml = 1.0; resolution = 30; margin = 2.0
gc_x_min = -32.5; gc_x_max = 0.05; gc_y_half = 10.5
gc_center_x = (gc_x_min + gc_x_max) / 2
grating_teeth_center_x = -24.0
wg_width = 0.5       # 500 nm strip waveguide
wg_extension = 6.0   # extend waveguide 6 µm beyond GDS edge (into PML)
sx = abs(gc_x_min) + gc_x_max + wg_extension + 2margin
sy = 2gc_y_half + 2margin
sz = box_thickness + si_thickness + clad_height
center_z = (clad_height + si_thickness - box_thickness) / 2

# Source position — must be inside domain, outside PML
pml_top = center_z + sz/2 - dpml
source_z = min(si_thickness + 1.5, pml_top - 0.5)
source_x = grating_teeth_center_x - (source_z - si_thickness) * tand(fiber_angle_deg)

println("=" ^ 60)
println("UBC ebeam_gc_te1550 — Grating Coupler 3D FDTD")
println("=" ^ 60)
@printf("  Wavelength:   1.50 – 1.60 μm (%d pts)\n", nfreq)
@printf("  Fiber:        %.0f° from normal, waist=%.1f μm\n", fiber_angle_deg, fiber_waist)
@printf("  Resolution:   %d px/μm\n", resolution)
@printf("  Domain:       %.1f × %.1f × %.1f μm\n", sx, sy, sz)
@printf("  Source:       x=%.2f z=%.2f k=(%.3f, %.3f, %.3f)\n",
    source_x, source_z, beam_kdir...)
println(); flush(stdout)

# ================================================================== #
# Geometry
# ================================================================== #
println("Importing GDS..."); flush(stdout)
lib = K.read_gds(expanduser("~/ubc/ubcpdk/gds/EBeam/ebeam_gc_te1550.gds"))
gc = K.gds_to_objects(lib, "ebeam_gc_te1550", Dict((1,0)=>(0.0, si_thickness, si)); axis=3)
@printf("  %d objects\n", length(gc)); flush(stdout)

box = K.Object(Cuboid([gc_center_x,0.0,-box_thickness/2],[1e6,1e6,box_thickness]), sio2)
bg  = K.Object(Cuboid([gc_center_x,0.0,center_z],[1e6,1e6,1e6]), K.Material(ε=1.0))

# Waveguide extension: extends from GDS edge (x≈0) through to the +x PML
# This ensures the guided mode is absorbed by PML instead of reflecting
wg_ext = K.Object(
    Cuboid([gc_x_max + wg_extension/2, 0.0, si_thickness/2],
           [wg_extension + 2*dpml, wg_width, si_thickness]),
    si)

geometry = vcat(gc, [wg_ext, box, bg])

# Domain center accounts for waveguide extension
cell_center_x = (gc_x_min + gc_x_max + wg_extension) / 2

# ================================================================== #
# Source and monitors
# ================================================================== #
source = K.GaussianBeamSource(
    time_profile = K.GaussianPulseSource(fcen=fcen, fwidth=2π*fwidth),
    center = [source_x, 0.0, source_z],
    size = [15.0, 15.0, 0.0],
    beam_waist = fiber_waist,
    beam_center = [grating_teeth_center_x, 0.0, Float64(si_thickness)],
    k_vector = beam_kdir,
    polarization = [0.0, 1.0, 0.0],
)

# Monitor on the waveguide, 1 µm before PML starts
# PML +x boundary: cell_center_x + sx/2 - dpml
pml_right = cell_center_x + sx/2 - dpml
mon_x = gc_x_max + 2.0  # 2 µm past GDS edge, well before PML
@printf("  Monitor x:    %.2f (PML at %.2f)\n", mon_x, pml_right)

output_monitor = K.FluxMonitor(center=[mon_x,0.0,si_thickness/2],
    size=[0.0,wg_width*3,si_thickness*4], frequencies=freqs)
dft_xz = K.DFTMonitor(center=[cell_center_x,0.0,center_z],
    size=[sx-2dpml,0.0,sz-2dpml], frequencies=[fcen], component=K.Ey())
dft_xy = K.DFTMonitor(center=[cell_center_x,0.0,si_thickness/2],
    size=[sx-2dpml,sy-2dpml,0.0], frequencies=[fcen], component=K.Ey())
# YZ cross-section at the monitor plane
dft_yz = K.DFTMonitor(center=[mon_x,0.0,center_z],
    size=[0.0,wg_width*6,sz-2dpml], frequencies=[fcen], component=K.Ey())

# ================================================================== #
# Build and run
# ================================================================== #
sim = K.Simulation(
    cell_size=[sx,sy,sz], cell_center=[cell_center_x,0.0,center_z], resolution=resolution,
    geometry=geometry, sources=[source], monitors=[output_monitor, dft_xz, dft_xy, dft_yz],
    boundaries=[[dpml,dpml],[dpml,dpml],[dpml,dpml]],
    boundary_conditions=[[K.PML(),K.PML()],[K.PML(),K.PML()],[K.PML(),K.PML()]],
    Courant=0.5, subpixel_smoothing=K.NoSmoothing())

mem = K.estimate_memory(sim)
@printf("  GPU memory: %.0f MB\n\n", mem.total_bytes/1e6); flush(stdout)

println("Running 3D FDTD..."); flush(stdout)
K.run(sim, until_after_sources=K.stop_when_dft_decayed(
    tolerance=1e-4, minimum_runtime=50.0, maximum_runtime=500.0))
println("Complete.\n")

# ================================================================== #
# Extract results
# ================================================================== #
flux = Array(K.get_flux(output_monitor))
wavelengths = 1.0 ./ freqs
peak_flux = maximum(abs.(flux))
eff = peak_flux > 0 ? abs.(flux) ./ peak_flux : zeros(nfreq)
eff_db = 10 .* log10.(max.(eff, 1e-10))

# DFT fields — squeeze to 2D
function squeeze_field(f)
    intensity = abs.(f).^2
    while ndims(intensity) > 2
        dim = findfirst(d -> size(intensity, d) <= 2, 1:ndims(intensity))
        isnothing(dim) && break
        intensity = dropdims(sum(intensity, dims=dim), dims=dim)
    end
    return intensity
end

field_xz = squeeze_field(Array(K.get_dft_fields(dft_xz)))
field_xy = squeeze_field(Array(K.get_dft_fields(dft_xy)))
field_yz = squeeze_field(Array(K.get_dft_fields(dft_yz)))

# Epsilon cross-sections — sample analytically from geometry shapes
eps_res = 1000  # high resolution for smooth geometry rendering
xrange = (cell_center_x - (sx-2dpml)/2, cell_center_x + (sx-2dpml)/2)
(eps_xz, eps_xz_xs, eps_xz_zs) = K.sample_geometry_slice(
    geometry, :y, 0.0;
    x_range=xrange,
    y_range=(center_z - (sz-2dpml)/2, center_z + (sz-2dpml)/2),
    resolution=eps_res)
(eps_xy, eps_xy_xs, eps_xy_ys) = K.sample_geometry_slice(
    geometry, :z, si_thickness/2;
    x_range=xrange,
    y_range=(-(sy-2dpml)/2, (sy-2dpml)/2),
    resolution=eps_res)
(eps_yz, eps_yz_ys, eps_yz_zs) = K.sample_geometry_slice(
    geometry, :x, mon_x;
    x_range=(-wg_width*3, wg_width*3),
    y_range=(center_z - (sz-2dpml)/2, center_z + (sz-2dpml)/2),
    resolution=eps_res)

@printf("  Peak flux: %.4e  NaN: %s\n", peak_flux, any(isnan, flux))
@printf("  XZ field: %s  eps_xz: %s\n", size(field_xz), size(eps_xz))
@printf("  XY field: %s  eps_xy: %s\n", size(field_xy), size(eps_xy))
flush(stdout)

# Results table
peak_i = argmax(abs.(flux))
half = peak_flux / 2; inband = wavelengths[abs.(flux) .>= half]
bw = length(inband) >= 2 ? abs(inband[end]-inband[1])*1000 : 0.0

println("\n" * "=" ^ 60)
println("Coupling Efficiency")
println("=" ^ 60)
step = max(1, nfreq ÷ 20)
for i in 1:step:nfreq
    @printf("  %8.1f nm    %8.4f    %8.2f dB\n", wavelengths[i]*1000, eff[i], eff_db[i])
end
@printf("\n  Peak at %.1f nm, 3dB BW: %.1f nm\n", wavelengths[peak_i]*1000, bw)

# ================================================================== #
# Save data
# ================================================================== #
outdir = joinpath(@__DIR__, "grating_coupler_output")
mkpath(outdir)
open(joinpath(outdir, "gc_te1550_results.csv"), "w") do io
    println(io, "wavelength_um,coupled_flux,relative_efficiency,relative_efficiency_dB")
    for i in 1:nfreq
        @printf(io, "%.6f,%.6e,%.6f,%.2f\n", wavelengths[i], flux[i], eff[i], eff_db[i])
    end
end

# ================================================================== #
# Plots — meep-style: epsilon base layer + field overlay
# ================================================================== #
try
    using CairoMakie

    # Monitor extent coordinates
    xmin = cell_center_x - (sx - 2dpml)/2
    xmax = cell_center_x + (sx - 2dpml)/2
    zmin = center_z - (sz - 2dpml)/2
    zmax = center_z + (sz - 2dpml)/2
    ymin = -(sy - 2dpml)/2
    ymax = (sy - 2dpml)/2

    # ---- Figure 0: Epsilon sanity check ----
    fig0 = Figure(size=(1400, 600))

    ax01 = Axis(fig0[1, 1], xlabel="x (μm)", ylabel="z (μm)",
        title="ε(x,z) at y=0", aspect=DataAspect())
    heatmap!(ax01, eps_xz_xs, eps_xz_zs, eps_xz', colormap=:binary, colorrange=(1, 13))
    Colorbar(fig0[1, 2], colormap=:binary, limits=(1, 13), label="ε")

    ax02 = Axis(fig0[2, 1], xlabel="x (μm)", ylabel="y (μm)",
        title="ε(x,y) at z≈Si/2", aspect=DataAspect())
    heatmap!(ax02, eps_xy_xs, eps_xy_ys, eps_xy', colormap=:binary, colorrange=(1, 13))

    save(joinpath(outdir, "fig0_epsilon.png"), fig0, px_per_unit=2)
    println("  Figure 0 (epsilon): fig0_epsilon.png")

    # ---- Figure 0b: Epsilon YZ at monitor location ----
    fig0b = Figure(size=(500, 500))
    ax0b = Axis(fig0b[1, 1], xlabel="y (μm)", ylabel="z (μm)",
        title="ε(y,z) at x=$(round(mon_x, digits=1)) (monitor plane)", aspect=DataAspect())
    heatmap!(ax0b, eps_yz_ys, eps_yz_zs, eps_yz', colormap=:binary, colorrange=(1, 13))
    save(joinpath(outdir, "fig0b_epsilon_yz.png"), fig0b, px_per_unit=2)
    println("  Figure 0b (epsilon YZ): fig0b_epsilon_yz.png")

    # ---- Figure 1: XZ cross-section — fields over geometry ----
    using Statistics
    fig1 = Figure(size=(1400, 500))
    ax1 = Axis(fig1[1, 1], xlabel="x (μm)", ylabel="z (μm)",
        title="|Ey|² over ε — XZ (y=0) at λ=1550 nm", aspect=DataAspect())

    # Field intensity as base layer (full opacity)
    fld_xs = range(xmin, xmax, length=size(field_xz, 1))
    fld_zs = range(zmin, zmax, length=size(field_xz, 2))
    fmax_xz = quantile(vec(field_xz), 0.95)
    heatmap!(ax1, fld_xs, fld_zs, field_xz', colormap=:inferno,
        colorrange=(0, max(fmax_xz, 1e-10)))
    # Geometry contours overlaid on top
    contour!(ax1, eps_xz_xs, eps_xz_zs, eps_xz', levels=[2.0, 6.0],
        color=:white, linewidth=0.8)

    save(joinpath(outdir, "fig1_field_xz.png"), fig1, px_per_unit=2)
    println("  Figure 1 (XZ fields): fig1_field_xz.png")

    # ---- Figure 2: XY cross-section — fields over geometry ----
    fig2 = Figure(size=(1400, 600))
    ax2 = Axis(fig2[1, 1], xlabel="x (μm)", ylabel="y (μm)",
        title="|Ey|² over ε — XY (z=Si/2) at λ=1550 nm", aspect=DataAspect())

    fld_xs2 = range(xmin, xmax, length=size(field_xy, 1))
    fld_ys2 = range(ymin, ymax, length=size(field_xy, 2))
    fmax_xy = quantile(vec(field_xy), 0.95)
    heatmap!(ax2, fld_xs2, fld_ys2, field_xy', colormap=:inferno,
        colorrange=(0, max(fmax_xy, 1e-10)))
    contour!(ax2, eps_xy_xs, eps_xy_ys, eps_xy', levels=[2.0, 6.0],
        color=:white, linewidth=0.8)

    save(joinpath(outdir, "fig2_field_xy.png"), fig2, px_per_unit=2)
    println("  Figure 2 (XY fields): fig2_field_xy.png")

    # ---- Figure 2b: YZ field at monitor location ----
    fig2b = Figure(size=(500, 500))
    ax2b = Axis(fig2b[1, 1], xlabel="y (μm)", ylabel="z (μm)",
        title="|Ey|² — YZ (x=$(round(mon_x, digits=1))) at λ=1550 nm", aspect=DataAspect())
    # Epsilon base
    heatmap!(ax2b, eps_yz_ys, eps_yz_zs, eps_yz', colormap=:binary, colorrange=(1, 13))
    # Field overlay
    yz_ys = range(-wg_width*3, wg_width*3, length=size(field_yz, 1))
    yz_zs = range(zmin, zmax, length=size(field_yz, 2))
    fmax_yz = length(field_yz) > 0 ? quantile(vec(field_yz), 0.95) : 1.0
    heatmap!(ax2b, yz_ys, yz_zs, field_yz', colormap=:inferno,
        colorrange=(0, max(fmax_yz, 1e-10)), alpha=0.6)
    save(joinpath(outdir, "fig2b_field_yz.png"), fig2b, px_per_unit=2)
    println("  Figure 2b (YZ fields): fig2b_field_yz.png")

    # ---- Figure 3: Coupling efficiency ----
    fig3 = Figure(size=(1100, 500))
    ax3 = Axis(fig3[1, 1], xlabel="Wavelength (nm)",
        ylabel="Relative Coupling Efficiency", title="Coupling (linear)")
    lines!(ax3, wavelengths.*1000, eff, color=:blue, linewidth=2)
    band!(ax3, wavelengths.*1000, zeros(nfreq), eff, color=(:blue, 0.15))
    hlines!(ax3, [0.5], color=:red, linestyle=:dash, linewidth=0.8, label="-3 dB")
    xlims!(ax3, 1500, 1600); ylims!(ax3, 0, 1.05)
    axislegend(ax3, position=:rt)

    ax4 = Axis(fig3[1, 2], xlabel="Wavelength (nm)",
        ylabel="Coupling (dB, rel. to peak)", title="Coupling (dB)")
    lines!(ax4, wavelengths.*1000, eff_db, color=:red, linewidth=2)
    hlines!(ax4, [-3], color=:blue, linestyle=:dash, linewidth=0.8, label="-3 dB")
    xlims!(ax4, 1500, 1600); ylims!(ax4, -15, 1)
    axislegend(ax4, position=:rt)

    Label(fig3[0, :], @sprintf(
        "ebeam_gc_te1550 — Peak: %.1f nm, 3dB BW: %.1f nm — Fiber %d° from normal, %d px/μm",
        wavelengths[peak_i]*1000, bw, Int(fiber_angle_deg), resolution), fontsize=13)

    save(joinpath(outdir, "fig3_coupling_efficiency.png"), fig3, px_per_unit=2)
    println("  Figure 3 (coupling): fig3_coupling_efficiency.png")

catch e
    println("  Plot error: $e")
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt); println()
    end
end

println("\nDone. Output: $outdir")
