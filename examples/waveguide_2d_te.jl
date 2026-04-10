# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# E47: 2D TE Waveguide Propagation (quasi-2D via thin 3D slice)
#
# Validates TE-polarized waveguide confinement by launching an Ez source inside
# a dielectric slab waveguide in a thin 3D domain. Verifies that the guided
# mode is confined within the high-index core, with evanescent tails decaying
# in the cladding.
#
# Reference: Meep straight-waveguide, Foundational FDTD validation

import Khronos
using CairoMakie
using GeometryPrimitives

# ── Scalable parameters ──────────────────────────────────────────────────────
resolution = 25        # pixels/μm
n_core     = 3.4       # core refractive index (e.g. Si)
n_clad     = 1.44      # cladding refractive index (e.g. SiO₂)
wg_width   = 0.5       # waveguide width (μm)
cell_x     = 8.0       # propagation length (μm)
cell_y     = 4.0       # transverse extent (μm)
dpml       = 0.5       # PML thickness
fcen       = 0.65      # source frequency (1/μm) → λ ≈ 1.55 μm
t_end      = 30.0      # simulation time
# ─────────────────────────────────────────────────────────────────────────────

function main(; resolution=resolution, n_core=n_core, n_clad=n_clad,
                wg_width=wg_width, cell_x=cell_x, cell_y=cell_y,
                dpml=dpml, fcen=fcen, t_end=t_end)

    Khronos.choose_backend(Khronos.CUDADevice(), Float64)

    # Thin z-extent: 1 voxel (quasi-2D in a 3D solver)
    dz = 1.0 / resolution

    geometry = [
        # Waveguide core (centered at y=0, extends across full x and z)
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [cell_x, wg_width, dz]),
            Khronos.Material(ε = n_core^2),
        ),
        # Cladding fills entire domain
        Khronos.Object(
            Cuboid([0.0, 0.0, 0.0], [cell_x, cell_y, dz]),
            Khronos.Material(ε = n_clad^2),
        ),
    ]

    # Ez source → TE polarization pattern (Ez, Hx, Hy)
    src_x = -cell_x/2 + dpml + 0.5
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.ContinuousWaveSource(fcen=fcen),
            component = Khronos.Ez(),
            center = [src_x, 0.0, 0.0],
            size   = [0.0, wg_width * 0.8, 0.0],
            amplitude = 1im,
        ),
    ]

    # DFT monitors at two x-positions to verify propagation
    mon_x1 = src_x + 1.0
    mon_x2 = cell_x/2 - dpml - 0.5
    mon_y_size = cell_y - 2*dpml
    monitors = [
        Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [mon_x1, 0.0, 0.0],
            size   = [0.0, mon_y_size, 0.0],
            frequencies = [fcen],
        ),
        Khronos.DFTMonitor(
            component = Khronos.Ez(),
            center = [mon_x2, 0.0, 0.0],
            size   = [0.0, mon_y_size, 0.0],
            frequencies = [fcen],
        ),
    ]

    sim = Khronos.Simulation(
        cell_size   = [cell_x, cell_y, dz],
        cell_center = [0.0, 0.0, 0.0],
        resolution  = resolution,
        geometry    = geometry,
        sources     = sources,
        boundaries  = [[dpml, dpml], [dpml, dpml], [0.0, 0.0]],
        monitors    = monitors,
    )

    Khronos.run(sim, until=t_end)

    # ── Extract fields ───────────────────────────────────────────────────────
    Khronos.prepare_simulation!(sim)
    kc = max(div(sim.Nz, 2), 1)
    ez = Array(collect(sim.fields.fEz[1:sim.Nx, 1:sim.Ny, kc]))

    # DFT profiles at the two monitor planes (monitor varies along y-axis)
    profile_near = Array(abs.(sim.monitors[1].monitor_data.fields[1, :, 1, 1]))
    profile_far  = Array(abs.(sim.monitors[2].monitor_data.fields[1, :, 1, 1]))

    # ── Confinement check ────────────────────────────────────────────────────
    # Verify that most energy is within the waveguide core region
    ny_mon = length(profile_far)
    ys_mon = range(-mon_y_size/2, mon_y_size/2, length=ny_mon)
    core_mask = abs.(collect(ys_mon)) .< wg_width/2
    power_in_core = sum(profile_far[core_mask].^2)
    power_total   = sum(profile_far.^2)
    confinement   = power_total > 0 ? power_in_core / power_total : 0.0

    println("\n", "="^60)
    println("2D TE Waveguide Propagation Validation (quasi-2D)")
    println("="^60)
    println("  Polarization:    TE (Ez source)")
    println("  Core index:      $n_core, Clad index: $n_clad")
    println("  Waveguide width: $wg_width μm")
    println("  Confinement:     $(round(confinement*100, digits=1))% in core")
    println("  Status:          $(confinement > 0.3 ? "PASS" : "FAIL") (>30% in core)")
    println("="^60)

    # ── Visualization ────────────────────────────────────────────────────────
    f = Figure(size=(900, 500))

    # 2D field pattern
    ax1 = Axis(f[1, 1:2], xlabel="x (μm)", ylabel="y (μm)",
               title="Ez field (TE waveguide, XY plane)", aspect=DataAspect())
    xs_2d = range(-cell_x/2, cell_x/2, length=size(ez,1))
    ys_2d = range(-cell_y/2, cell_y/2, length=size(ez,2))
    vmax = maximum(abs.(ez))
    hm = heatmap!(ax1, collect(xs_2d), collect(ys_2d), ez,
                  colormap=:bluesreds, colorrange=(-vmax, vmax))
    # Draw waveguide boundaries
    hlines!(ax1, [-wg_width/2, wg_width/2], color=:white, linestyle=:dash, linewidth=1)

    # Transverse profiles
    ax2 = Axis(f[2, 1], xlabel="y (μm)", ylabel="|Ez| (a.u.)",
               title="Near field (x=$(round(mon_x1,digits=1)))")
    if maximum(profile_near) > 0
        lines!(ax2, collect(ys_mon), profile_near ./ maximum(profile_near), color=:blue)
    end
    vlines!(ax2, [-wg_width/2, wg_width/2], color=:gray, linestyle=:dash)

    ax3 = Axis(f[2, 2], xlabel="y (μm)", ylabel="|Ez| (a.u.)",
               title="Far field (x=$(round(mon_x2,digits=1)))")
    if maximum(profile_far) > 0
        lines!(ax3, collect(ys_mon), profile_far ./ maximum(profile_far), color=:blue)
    end
    vlines!(ax3, [-wg_width/2, wg_width/2], color=:gray, linestyle=:dash)

    save("waveguide_2d_te.png", f)
    println("Saved: waveguide_2d_te.png")

    return confinement
end

main()
