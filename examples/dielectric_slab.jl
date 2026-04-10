# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Periodic BC + FluxMonitor validation: dielectric slab transmission.
#
# A plane wave transmits through a dielectric slab (ε=4, n=2) in a
# periodic domain (xy). The transmitted flux is measured and compared
# against the analytical Fresnel transmission coefficient.
#
# This validates:
#   - Periodic boundary conditions (xy)
#   - PlaneWave source (normal incidence)
#   - FluxMonitor (Poynting flux integration)
#   - PML (z boundaries)

t_script_start = time()

import Khronos
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

# Dielectric slab
n_slab = 2.0
ε_slab = n_slab^2  # 4.0
slab_thickness = 0.5  # μm

# Wavelength range
λ_center = 1.0  # μm
freq_center = 1.0 / λ_center
freq_min = 1.0 / 1.5
freq_max = 1.0 / 0.6
fwidth = 2π * 0.5 * (freq_max - freq_min)

# Monitor frequencies
n_freqs = 51
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

# Periodic in x,y — just one cell wide (plane wave is uniform)
cell_xy = 0.1  # μm (small, since fields are uniform in xy)

# z-direction: slab + buffer + PML
buffer = 1.5 * λ_center
pml_thickness = 1.0
domain_z = slab_thickness + 2 * buffer
cell_z = domain_z + 2 * pml_thickness

resolution = 40  # pts/μm

# ------------------------------------------------------------------- #
# Geometry
# ------------------------------------------------------------------- #

geometry = [
    # Dielectric slab centered at origin
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [cell_xy + 1.0, cell_xy + 1.0, slab_thickness]),
        Khronos.Material(ε = ε_slab),
    ),
]

# ------------------------------------------------------------------- #
# Source: plane wave propagating in +z
# ------------------------------------------------------------------- #

source_z = -slab_thickness / 2 - buffer / 2

function make_sources()
    [Khronos.PlaneWaveSource(
        time_profile = Khronos.GaussianPulseSource(
            fcen = freq_center,
            fwidth = fwidth,
        ),
        center = [0.0, 0.0, source_z],
        size = [cell_xy, cell_xy, 0.0],
        k_vector = [0.0, 0.0, 1.0],
        polarization_angle = 0.0,
        amplitude = 1.0,
    )]
end

# ------------------------------------------------------------------- #
# Monitors: flux monitor after slab (same position in both runs)
# ------------------------------------------------------------------- #

flux_z_monitor = slab_thickness / 2 + buffer / 2

monitors_slab = Khronos.Monitor[
    Khronos.FluxMonitor(
        center = [0.0, 0.0, flux_z_monitor],
        size = [cell_xy, cell_xy, 0.0],
        frequencies = monitor_freqs,
    ),
]

monitors_ref = Khronos.Monitor[
    Khronos.FluxMonitor(
        center = [0.0, 0.0, flux_z_monitor],
        size = [cell_xy, cell_xy, 0.0],
        frequencies = monitor_freqs,
    ),
]

# ------------------------------------------------------------------- #
# Boundaries: periodic in x,y, PML in z
# ------------------------------------------------------------------- #

boundaries = [
    [0.0, 0.0],            # x: no PML (periodic)
    [0.0, 0.0],            # y: no PML (periodic)
    [pml_thickness, pml_thickness],  # z: PML both sides
]

boundary_conditions = [
    [Khronos.Periodic(), Khronos.Periodic()],  # x: periodic
    [Khronos.Periodic(), Khronos.Periodic()],  # y: periodic
    [Khronos.PML(), Khronos.PML()],            # z: PML
]

# ------------------------------------------------------------------- #
# Simulation 1: Reference (empty — no slab)
# ------------------------------------------------------------------- #

sim_ref = Khronos.Simulation(
    cell_size = [cell_xy, cell_xy, cell_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution = resolution,
    geometry = nothing,
    sources = make_sources(),
    monitors = monitors_ref,
    boundaries = boundaries,
    boundary_conditions = boundary_conditions,
)

# ------------------------------------------------------------------- #
# Simulation 2: With dielectric slab
# ------------------------------------------------------------------- #

sim_slab = Khronos.Simulation(
    cell_size = [cell_xy, cell_xy, cell_z],
    cell_center = [0.0, 0.0, 0.0],
    resolution = resolution,
    geometry = geometry,
    sources = make_sources(),
    monitors = monitors_slab,
    boundaries = boundaries,
    boundary_conditions = boundary_conditions,
)

# ------------------------------------------------------------------- #
# Run both simulations
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("Periodic BC + FluxMonitor Validation")
println("Dielectric slab (n=$n_slab, t=$(slab_thickness) μm)")
println("Wavelength range: 0.6–1.5 μm")
println("Cell: $(cell_xy) × $(cell_xy) × $(round(cell_z, digits=2)) μm")
println("Resolution: $resolution pts/μm")
println("=" ^ 60)

stop_cond = Khronos.stop_when_dft_decayed(
    tolerance = 1e-6,
    minimum_runtime = 50.0,
    maximum_runtime = 500.0,
)

println("\n--- Reference simulation (empty) ---")
Khronos.run(sim_ref, until_after_sources = stop_cond)

println("\n--- Slab simulation ---")
Khronos.run(sim_slab, until_after_sources = Khronos.stop_when_dft_decayed(
    tolerance = 1e-6,
    minimum_runtime = 50.0,
    maximum_runtime = 500.0,
))

# ------------------------------------------------------------------- #
# Post-processing: compute transmission
# ------------------------------------------------------------------- #

flux_trans = Khronos.get_flux(monitors_slab[1])
flux_inc = Khronos.get_flux(monitors_ref[1])

println("\nRaw flux values (sampled):")
for i in 1:10:n_freqs
    println("  f=$(round(monitor_freqs[i], digits=4)): flux_ref=$(round(flux_inc[i], sigdigits=4)), flux_slab=$(round(flux_trans[i], sigdigits=4))")
end

# Normalize: T(f) = flux_slab / flux_empty
T_sim = flux_trans ./ flux_inc

# ------------------------------------------------------------------- #
# Analytical reference: Fresnel transmission through a slab
# ------------------------------------------------------------------- #

# For a dielectric slab of thickness d, refractive index n, at normal
# incidence in vacuum (n1 = 1), the transmission coefficient is:
#
#   T = |t12 * t21 * exp(i*n*k*d) / (1 + r12*r21*exp(2i*n*k*d))|²
#
# where r12 = (1 - n) / (1 + n), t12 = 2/(1+n), r21 = -r12, t21 = 2n/(1+n)
# and k = 2π*freq (since c=1 in Khronos units).

function fresnel_slab_transmission(freq, n_slab, thickness)
    r12 = (1.0 - n_slab) / (1.0 + n_slab)
    t12 = 2.0 / (1.0 + n_slab)
    r21 = -r12
    t21 = 2.0 * n_slab / (1.0 + n_slab)
    k = 2π * freq
    phase = n_slab * k * thickness
    T = abs2(t12 * t21 * exp(im * phase) / (1.0 + r12 * r21 * exp(2im * phase)))
    return T
end

T_analytical = [fresnel_slab_transmission(f, n_slab, slab_thickness) for f in monitor_freqs]
wavelengths = 1.0 ./ monitor_freqs

# ------------------------------------------------------------------- #
# Results comparison
# ------------------------------------------------------------------- #

println("\n" * "=" ^ 60)
println("RESULTS: Transmission comparison")
println("=" ^ 60)

max_error = 0.0
for i in 1:n_freqs
    err = abs(T_sim[i] - T_analytical[i])
    global max_error = max(max_error, err)
    if i % 10 == 1
        println("  λ = $(round(wavelengths[i], digits=4)) μm: T_sim = $(round(T_sim[i], digits=4)), T_analytical = $(round(T_analytical[i], digits=4)), error = $(round(err, digits=6))")
    end
end

println("\nMax absolute error: $(round(max_error, digits=6))")
if max_error < 0.08
    println("✓ PASS: Simulation matches Fresnel theory within 8%")
else
    println("✗ FAIL: Error exceeds 8% threshold")
end

t_script_total = time() - t_script_start
println("\nScript wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
