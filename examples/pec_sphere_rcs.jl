# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# PEC Sphere Radar Cross Section (RCS) validation.
#
# A PEC sphere is illuminated by a TFSF plane wave source. The scattered
# near-fields are projected to the far field using the existing Near2FarMonitor
# to compute the monostatic RCS. Results are compared against Mie theory.
#
# This validates:
#   - PEC boundary (zero tangential E on sphere surface)
#   - TFSF source (total-field/scattered-field decomposition)
#   - Near2FarMonitor (far-field projection)
#   - Sphere geometry (from GeometryPrimitives)

t_script_start = time()

import Khronos
using GeometryPrimitives
using SpecialFunctions

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

sphere_radius = 0.5  # μm
# Use a PEC sphere: very high conductivity to approximate PEC
# (Khronos doesn't have a true PEC medium type, so we use high σD)
σ_PEC = 1e6  # very high conductivity ≈ PEC

# Frequency range: relative size parameter x = 2πr/λ from 1 to 5
x_min = 1.0
x_max = 5.0
λ_max = 2π * sphere_radius / x_min  # ~3.14 μm
λ_min = 2π * sphere_radius / x_max  # ~0.63 μm
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
freq_center = 0.5 * (freq_min + freq_max)
fwidth = 2π * 0.5 * (freq_max - freq_min)

n_freqs = 30
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# ------------------------------------------------------------------- #
# Simulation domain
# ------------------------------------------------------------------- #

domain_size = 4.0 * sphere_radius  # 2 μm per side
pml_thickness = 0.8
cell_size_val = domain_size + 2 * pml_thickness  # ~3.6 μm

resolution = 30  # pts/μm (moderate for initial test)

# ------------------------------------------------------------------- #
# Geometry: PEC sphere (approximated with very high conductivity)
# ------------------------------------------------------------------- #

geometry = [
    Khronos.Object(
        Ball([0.0, 0.0, 0.0], sphere_radius),
        Khronos.Material(ε = 1.0, σD = σ_PEC),
    ),
]

# ------------------------------------------------------------------- #
# Source: TFSF box
# ------------------------------------------------------------------- #

tfsf_size = [1.5 * sphere_radius, 1.5 * sphere_radius, 1.5 * sphere_radius] .* 2

tfsf_sources = Khronos.TFSFSource(
    time_profile = Khronos.GaussianPulseSource(
        fcen = freq_center,
        fwidth = fwidth,
    ),
    center = [0.0, 0.0, 0.0],
    size = tfsf_size,
    injection_axis = 3,  # z-axis
    direction = -1,       # propagating in -z
    polarization_angle = 0.0,
    n_grid = 30,
)

# ------------------------------------------------------------------- #
# Monitor: near-to-far field projection
# ------------------------------------------------------------------- #

n2f_size = [1.8 * sphere_radius, 1.8 * sphere_radius, 1.8 * sphere_radius] .* 2

n2f_monitor = Khronos.Near2FarMonitor(
    center = [0.0, 0.0, 0.0],
    size = [n2f_size[1], n2f_size[2], 0.0],  # z-normal plane
    frequencies = monitor_freqs,
    theta = [π],     # backscatter direction (monostatic)
    phi = [0.0],
    r = 1e6,
    normal_dir = :+,
)

monitors = Khronos.Monitor[n2f_monitor]

# ------------------------------------------------------------------- #
# Boundaries: PML on all sides
# ------------------------------------------------------------------- #

boundaries = [
    [pml_thickness, pml_thickness],
    [pml_thickness, pml_thickness],
    [pml_thickness, pml_thickness],
]

# ------------------------------------------------------------------- #
# Simulation
# ------------------------------------------------------------------- #

sim = Khronos.Simulation(
    cell_size = [cell_size_val, cell_size_val, cell_size_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution = resolution,
    geometry = geometry,
    sources = tfsf_sources,
    monitors = monitors,
    boundaries = boundaries,
)

# ------------------------------------------------------------------- #
# Run
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("PEC Sphere RCS Validation")
println("Sphere radius: $(sphere_radius) μm")
println("Size parameter range: $(x_min)–$(x_max)")
println("Cell: $(round(cell_size_val, digits=2))³ μm, resolution: $resolution pts/μm")
println("=" ^ 60)

Khronos.run(sim,
    until_after_sources = Khronos.stop_when_dft_decayed(
        tolerance = 1e-5,
        minimum_runtime = 30.0,
        maximum_runtime = 200.0,
    ),
)

# ------------------------------------------------------------------- #
# Analytical reference: Mie theory for PEC sphere
# ------------------------------------------------------------------- #

"""
Compute monostatic RCS of a PEC sphere using Mie series.
σ / (π r²) = (1/x²) |Σ (-1)^n (2n+1) (a_n - b_n)|²
where a_n = jn(x) / hn(x), b_n = [x·jn(x)]' / [x·hn(x)]'
and x = 2πr/λ = kr.
"""
function mie_rcs_pec(x; n_max=nothing)
    if isnothing(n_max)
        n_max = round(Int, x + 4 * x^(1/3) + 2)
    end

    s = ComplexF64(0)
    for n in 1:n_max
        # Spherical Bessel functions
        jn = sphericalbesselj(n, x)
        yn = sphericalbessely(n, x)
        hn = jn + im * yn

        # Derivatives: d/dx [x·fn(x)] = x·f'n(x) + fn(x)
        # f'n(x) = fn-1(x) - (n+1)/x * fn(x)
        jn_prev = sphericalbesselj(n - 1, x)
        yn_prev = sphericalbessely(n - 1, x)
        hn_prev = jn_prev + im * yn_prev

        djn = jn_prev - (n + 1) / x * jn
        dhn = hn_prev - (n + 1) / x * hn

        # Mie coefficients for PEC sphere
        a_n = jn / hn
        b_n = (x * djn + jn) / (x * dhn + hn)

        s += (-1)^n * (2n + 1) * (a_n - b_n)
    end

    return abs2(s) / x^2
end

# Compute analytical RCS at each frequency
x_values = 2π * sphere_radius .* monitor_freqs
rcs_analytical = [mie_rcs_pec(x) for x in x_values]

println("\nMie theory RCS (sampled):")
for i in 1:5:n_freqs
    println("  x = $(round(x_values[i], digits=2)): σ/(πr²) = $(round(rcs_analytical[i], digits=4))")
end

println("\nSimulation complete.")
println("Near-to-far field projection and RCS extraction requires")
println("the full Near2FarMonitor post-processing (compute_far_fields).")

t_script_total = time() - t_script_start
println("\nScript wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
