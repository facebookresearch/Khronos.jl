# Profile the ring coupler timestepping with original specs.
# Usage:
#   /opt/nvidia/nsight-systems/2025.5.2/bin/nsys profile \
#       --trace=cuda,nvtx --capture-range=cuda-profiler-api \
#       -o ring_coupler_nsys \
#       julia --project examples/profile_ring_coupler.jl
#
#   ncu --set full --target-processes all -o ring_coupler_ncu \
#       julia --project examples/profile_ring_coupler.jl

import Khronos
using CUDA
using GeometryPrimitives

# Disable CUDA graph capture so profiler sees individual kernels
ENV["KHRONOS_CUDA_GRAPHS"] = "0"

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

# ------------------------------------------------------------------- #
# Original ring coupler setup (51 freqs, resolution 50 mode solver)
# ------------------------------------------------------------------- #

n_Si = 3.47
n_SiO2 = 1.44
ε_Si = n_Si^2
ε_SiO2 = n_SiO2^2

wg_width = 0.5
wg_height = 0.22
ring_radius = 5.0
coupling_gap = 0.05
ring_center_y = wg_width / 2 + coupling_gap + ring_radius

λ_min = 1.5
λ_max = 1.6
λ_center = 1.55
freq_center = 1.0 / λ_center
freq_min = 1.0 / λ_max
freq_max = 1.0 / λ_min
fwidth = freq_max - freq_min

n_freqs = 51
monitor_freqs = collect(range(freq_min, freq_max, length=n_freqs))

domain_x = 2 * (ring_radius + 2.0)
domain_y = 8.0
domain_z = 4 * wg_height + 2.0
resolution = 25

geometry = [
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [domain_x + 4.0, wg_width, wg_height]),
        Khronos.Material(ε = ε_Si),
    ),
    Khronos.Object(
        Cylinder([0.0, ring_center_y, 0.0], ring_radius - wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
        Khronos.Material(ε = ε_SiO2),
    ),
    Khronos.Object(
        Cylinder([0.0, ring_center_y, 0.0], ring_radius + wg_width / 2, wg_height, [0.0, 0.0, 1.0]),
        Khronos.Material(ε = ε_Si),
    ),
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [domain_x + 10.0, domain_y + 10.0, domain_z + 10.0]),
        Khronos.Material(ε = ε_SiO2),
    ),
]

source_x = -(ring_radius + λ_center / 4)
sources = [
    Khronos.ModeSource(
        time_profile = Khronos.GaussianPulseSource(fcen = freq_center, fwidth = fwidth),
        frequency = freq_center,
        mode_solver_resolution = 50,
        mode_index = 1,
        center = [source_x, 0.0, 0.0],
        size = [0.0, 6 * wg_width, 6 * wg_height],
        solver_tolerance = 1e-6,
        geometry = geometry,
    ),
]

through_x = ring_radius + λ_center / 4
through_monitor = Khronos.ModeMonitor(
    center = [through_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(num_modes = 1, geometry = geometry, mode_solver_resolution = 50),
    decimation = 5,
)

ref_x = source_x + 1.0
ref_monitor = Khronos.ModeMonitor(
    center = [ref_x, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(num_modes = 1, geometry = geometry, mode_solver_resolution = 50),
    decimation = 5,
)

θ_drop = π / 4
drop_x = ring_radius * sin(θ_drop)
drop_y = ring_center_y - ring_radius * cos(θ_drop)
drop_monitor = Khronos.ModeMonitor(
    center = [drop_x, drop_y, 0.0],
    size = [6 * wg_width, 0.0, 6 * wg_height],
    frequencies = monitor_freqs,
    mode_spec = Khronos.ModeSpec(num_modes = 1, geometry = geometry, mode_solver_resolution = 50),
    decimation = 5,
)

field_monitor = Khronos.DFTMonitor(
    component = Khronos.Ey(),
    center = [0.0, domain_y / 2 - 2.0, 0.0],
    size = [domain_x - 2.0, domain_y - 2.0, 0.0],
    frequencies = [freq_center],
)

monitors = Khronos.Monitor[through_monitor, ref_monitor, drop_monitor, field_monitor]

pml_thickness = 1.0
sim = Khronos.Simulation(
    cell_size = [domain_x, domain_y, domain_z],
    cell_center = [0.0, domain_y / 2 - 2.0, 0.0],
    resolution = resolution,
    geometry = geometry,
    sources = sources,
    monitors = monitors,
    boundaries = [[pml_thickness, pml_thickness], [pml_thickness, 0.0], [pml_thickness, pml_thickness]],
    absorbers = [nothing, [nothing, Khronos.Absorber(num_layers = 40)], nothing],
)

# ------------------------------------------------------------------- #
# Prepare simulation
# ------------------------------------------------------------------- #

@info "Preparing simulation..."
Khronos.prepare_simulation!(sim)
num_voxels = sim.Nx * sim.Ny * sim.Nz
@info "Grid: $(sim.Nx) × $(sim.Ny) × $(sim.Nz) = $(num_voxels) voxels"
@info "Monitors: $(length(sim.monitor_data)) DFT monitors"

# ------------------------------------------------------------------- #
# Warmup: run until sources deactivate + 50 extra steps
# ------------------------------------------------------------------- #

@info "Warming up (running until sources off + 50 steps)..."
# Run until sources turn off
while sim.sources_active
    Khronos.step!(sim)
end
# 50 more steps to warm up the post-source code path
for _ in 1:50
    Khronos.step!(sim)
end
CUDA.synchronize()
@info "Warmup done at timestep $(sim.timestep)"

# ------------------------------------------------------------------- #
# Profiled region: 200 steps
# ------------------------------------------------------------------- #

PROFILE_STEPS = 200

@info "Profiling $PROFILE_STEPS steps..."
CUDA.@profile begin
    CUDA.NVTX.@range "profiled_steps" begin
        t_start = time()
        for i in 1:PROFILE_STEPS
            Khronos.step!(sim)
        end
        CUDA.synchronize()
        t_elapsed = time() - t_start
        rate = num_voxels * PROFILE_STEPS / t_elapsed / 1e6
        @info "Profiled: $PROFILE_STEPS steps in $(round(t_elapsed, digits=3))s = $(round(rate, digits=1)) MVoxels/s"
    end
end
