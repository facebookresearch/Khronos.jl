# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Photonic crystal slab band structure using Bloch boundary conditions.
#
# A 2D square-lattice photonic crystal (air holes in dielectric slab)
# is simulated with Bloch-periodic x,y boundaries and PML in z.
# Point dipole sources excite resonances. DFT monitors at many frequencies
# identify resonant modes as peaks in the spectral response.
#
# This validates:
#   - Bloch boundary conditions (complex fields, phase-shifted halo)
#   - DFT monitor spectral response
#   - Point dipole source (UniformSource with size=0)

t_script_start = time()

import Khronos
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

# ------------------------------------------------------------------- #
# Physical parameters
# ------------------------------------------------------------------- #

a = 1.0           # lattice constant (μm)
r_hole = 0.2 * a  # hole radius
t_slab = 0.5 * a  # slab thickness
ε_slab = 12.0     # slab permittivity (silicon-like)

# Frequency range of interest (in units of c/a, but since a=1, these are just freqs)
freq_min = 0.1
freq_max = 0.5
n_freqs = 200  # dense frequency sampling to resolve resonances
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

# Bloch k-path: Γ → X → M → Γ (in units of 2π/a)
Nk = 4  # points per segment
k_points_Gamma_X = [[kx, 0.0] for kx in range(0, 0.5, length=Nk)]
k_points_X_M = [[0.5, ky] for ky in range(0, 0.5, length=Nk)[2:end]]
k_points_M_Gamma = [[k, k] for k in range(0.5, 0, length=Nk)[2:end]]
k_path = vcat(k_points_Gamma_X, k_points_X_M, k_points_M_Gamma)

# ------------------------------------------------------------------- #
# Simulation domain: single unit cell
# ------------------------------------------------------------------- #

pml_z = 1.0
buffer_z = 1.0
cell_z = t_slab + 2 * buffer_z + 2 * pml_z
resolution = 20

# ------------------------------------------------------------------- #
# Geometry
# ------------------------------------------------------------------- #

geometry = [
    # Air hole (higher priority)
    Khronos.Object(
        Cylinder([0.0, 0.0, 0.0], r_hole, t_slab, [0.0, 0.0, 1.0]),
        Khronos.Material(ε = 1.0),
    ),
    # Dielectric slab
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [a + 1.0, a + 1.0, t_slab]),
        Khronos.Material(ε = ε_slab),
    ),
]

# ------------------------------------------------------------------- #
# Run one simulation per k-point
# ------------------------------------------------------------------- #

println("=" ^ 60)
println("Photonic Crystal Band Structure")
println("Square lattice, a=$a μm, r=$(r_hole) μm, t=$(t_slab) μm")
println("ε = $ε_slab, resolution = $resolution pts/μm")
println("k-path: $(length(k_path)) points along Γ-X-M-Γ")
println("=" ^ 60)

all_peaks = Vector{Vector{Float64}}()

for (ik, kp) in enumerate(k_path)
    kx = kp[1] * 2π / a
    ky = kp[2] * 2π / a

    # Source: Hz point dipole off-center to excite all modes
    source_pos = [0.13 * a, 0.27 * a, 0.0]

    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(
                fcen = 0.5 * (freq_min + freq_max),
                fwidth = 2π * 0.5 * (freq_max - freq_min),
            ),
            component = Khronos.Hz(),
            center = source_pos,
            size = [0.0, 0.0, 0.0],
        ),
    ]

    # DFT monitor: record Hz at a point across all frequencies
    monitor_pos = [0.37 * a, 0.11 * a, 0.0]

    monitors = Khronos.Monitor[
        Khronos.DFTMonitor(
            component = Khronos.Hz(),
            center = monitor_pos,
            size = [0.0, 0.0, 0.0],
            frequencies = monitor_freqs,
        ),
    ]

    # Boundaries: Bloch in x,y, PML in z
    boundaries = [
        [0.0, 0.0],
        [0.0, 0.0],
        [pml_z, pml_z],
    ]

    boundary_conditions = [
        [Khronos.Bloch(k = kx), Khronos.Bloch(k = kx)],
        [Khronos.Bloch(k = ky), Khronos.Bloch(k = ky)],
        [Khronos.PML(), Khronos.PML()],
    ]

    sim = Khronos.Simulation(
        cell_size = [a, a, cell_z],
        cell_center = [0.0, 0.0, 0.0],
        resolution = resolution,
        geometry = geometry,
        sources = sources,
        monitors = monitors,
        boundaries = boundaries,
        boundary_conditions = boundary_conditions,
    )

    # Run long enough for resonances to ring up
    run_time = 200.0  # in simulation time units
    Khronos.run(sim, until_after_sources = run_time)

    # Extract DFT magnitude spectrum
    dft_fields = Khronos.get_dft_fields(monitors[1])
    spectrum = vec(abs.(Array(dft_fields)))

    # The spectrum length may differ from n_freqs due to DFT array shape
    n_spec = min(length(spectrum), n_freqs)

    # Find peaks: frequencies where spectrum exceeds threshold
    threshold = 0.2 * maximum(spectrum[1:n_spec])
    peaks = Float64[]
    for i in 2:n_spec-1
        if spectrum[i] > threshold &&
           spectrum[i] > spectrum[i-1] &&
           spectrum[i] > spectrum[i+1]
            push!(peaks, monitor_freqs[i])
        end
    end

    push!(all_peaks, peaks)
    println("  k=($(kp[1]), $(kp[2])): $(length(peaks)) modes at f = $(round.(peaks, digits=3))")
end

# ------------------------------------------------------------------- #
# Summary
# ------------------------------------------------------------------- #

println("\n" * "=" ^ 60)
println("BAND STRUCTURE RESULTS")
println("=" ^ 60)
println("Expected: photonic band gap around f ≈ 0.3-0.4 c/a")
println("(comparable to Fan & Joannopoulos, PRB 65, 235112, 2002)")
println()
for (ik, kp) in enumerate(k_path)
    freqs = all_peaks[ik]
    println("  k=($(round(kp[1], digits=2)), $(round(kp[2], digits=2))): f = $(round.(freqs, digits=3))")
end

t_script_total = time() - t_script_start
println("\nScript wall time: $(round(t_script_total, digits=2))s")
println("=" ^ 60)
