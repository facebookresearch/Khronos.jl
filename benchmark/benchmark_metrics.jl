# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# GPU profiling metrics collection for Khronos benchmarks.
#
# Provides phase-level timing, kernel register/occupancy analysis,
# achieved bandwidth computation, and memory footprint reporting.
# Uses only CUDA.jl APIs — no NCU/Nsys binaries required.

module BenchmarkMetrics

import Khronos
using CUDA

export collect_phase_timing, collect_kernel_metrics, compute_bandwidth,
       collect_memory_info, print_metrics_summary, run_metrics,
       collect_and_store_metrics, kernel_metrics_to_dict

# ── GPU peak bandwidth detection ────────────────────────────────────────────

function get_peak_bandwidth_gbps()
    if !CUDA.functional()
        return 0.0
    end
    dev = CUDA.device()
    clock_khz = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    bus_width_bits = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    # clock_khz is in kHz; DDR: effective rate = 2 × clock rate
    # peak (bytes/s) = 2 * clock_rate_Hz * bus_width_bytes
    # peak (GB/s) = peak (bytes/s) / 1e9
    peak_gbps = 2.0 * (clock_khz * 1e3) * (bus_width_bits / 8) / 1e9
    return peak_gbps
end

function get_gpu_sm_specs()
    if !CUDA.functional()
        return (sms=0, max_threads_per_sm=0, max_regs_per_block=0, max_blocks_per_sm=0)
    end
    dev = CUDA.device()
    return (
        sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        max_threads_per_sm = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        max_regs_per_block = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),
        max_blocks_per_sm = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR),
    )
end

# ── Phase-level timing ──────────────────────────────────────────────────────

"""
    collect_phase_timing(sim, n_steps=50)

Measure per-phase timing of FDTD timesteps with GPU synchronization barriers.
Returns a NamedTuple with average ms/step for each phase.
"""
function collect_phase_timing(sim, n_steps::Int=50)
    if !sim.is_prepared
        Khronos.prepare_simulation!(sim)
    end

    # Warmup (ensure kernels are JIT-compiled)
    for _ in 1:10
        Khronos.step!(sim)
    end
    CUDA.synchronize()

    t_src_H = 0.0
    t_step_H = 0.0
    t_mon_H = 0.0
    t_src_E = 0.0
    t_step_E = 0.0
    t_mon_E = 0.0

    for _ in 1:n_steps
        t = Khronos.round_time(sim)

        # Magnetic sources
        CUDA.synchronize()
        t0 = time()
        if sim.sources_active
            Khronos.update_magnetic_sources!(sim, t)
        end
        CUDA.synchronize()
        t_src_H += time() - t0

        # H-field update
        t0 = time()
        Khronos.step_H_fused!(sim)
        CUDA.synchronize()
        t_step_H += time() - t0

        # H monitors
        t0 = time()
        Khronos.update_H_monitors!(sim, t)
        CUDA.synchronize()
        t_mon_H += time() - t0

        # Electric sources
        t0 = time()
        if sim.sources_active
            Khronos.update_electric_sources!(sim, t + sim.Δt / 2)
        end
        CUDA.synchronize()
        t_src_E += time() - t0

        # E-field update
        t0 = time()
        Khronos.step_E_fused!(sim)
        CUDA.synchronize()
        t_step_E += time() - t0

        # E monitors
        t0 = time()
        Khronos.update_E_monitors!(sim, t + sim.Δt / 2)
        CUDA.synchronize()
        t_mon_E += time() - t0

        Khronos.increment_timestep!(sim)
    end

    # Convert to ms/step averages
    to_ms(t) = t / n_steps * 1000

    total = t_src_H + t_step_H + t_mon_H + t_src_E + t_step_E + t_mon_E

    return (
        step_H_ms   = to_ms(t_step_H),
        step_E_ms   = to_ms(t_step_E),
        src_H_ms    = to_ms(t_src_H),
        src_E_ms    = to_ms(t_src_E),
        mon_H_ms    = to_ms(t_mon_H),
        mon_E_ms    = to_ms(t_mon_E),
        total_ms    = to_ms(total),
        n_steps     = n_steps,
    )
end

# ── Kernel register/occupancy analysis ──────────────────────────────────────

struct KernelInfo
    name::String
    registers::Int
    max_threads::Int
    shared_mem::Int
    local_mem::Int
    occupancy_pct::Float64
end

"""
    collect_kernel_metrics(precision_type)

Compile each CUDA kernel variant and extract register counts, memory usage,
and theoretical occupancy. Returns a Vector{KernelInfo}.
"""
function collect_kernel_metrics(precision_type::Type{T}) where {T}
    if !CUDA.functional()
        return KernelInfo[]
    end

    specs = get_gpu_sm_specs()
    n = 32
    arr = CUDA.zeros(T, n + 2, n + 2, n + 2)
    sigma_arr = CUDA.zeros(T, n + 2)
    iNx = Int32(n)
    s = T(0.01)
    wg = 256  # default workgroup size

    results = KernelInfo[]

    kernel_configs = [
        ("BH interior", () -> @cuda(launch=false, Khronos._cuda_fused_BH_kernel!(
            arr, arr, arr, arr, arr, arr, s, s, s, iNx))),
        ("DE per-voxel eps", () -> @cuda(launch=false, Khronos._cuda_fused_DE_kernel!(
            arr, arr, arr, arr, arr, arr, arr, arr, arr, s, s, s, iNx))),
        ("DE scalar eps", () -> @cuda(launch=false, Khronos._cuda_fused_DE_scalar_kernel!(
            arr, arr, arr, arr, arr, arr, s, s, s, s, iNx))),
        ("PML BH", () -> @cuda(launch=false, Khronos._cuda_pml_BH_kernel!(
            arr, arr, arr, arr, arr, arr, arr, arr, arr,
            arr, arr, arr, arr, arr, arr, sigma_arr, sigma_arr, sigma_arr,
            s, s, s, s, iNx, Int32(1), Int32(1), Int32(1)))),
        ("PML DE per-voxel", () -> @cuda(launch=false, Khronos._cuda_pml_DE_kernel!(
            arr, arr, arr, arr, arr, arr, arr, arr, arr,
            arr, arr, arr, arr, arr, arr, sigma_arr, sigma_arr, sigma_arr,
            arr, arr, arr, s, s, s, iNx, Int32(1), Int32(1), Int32(1)))),
        ("PML DE scalar", () -> @cuda(launch=false, Khronos._cuda_pml_DE_scalar_kernel!(
            arr, arr, arr, arr, arr, arr, arr, arr, arr,
            arr, arr, arr, arr, arr, arr, sigma_arr, sigma_arr, sigma_arr,
            s, s, s, s, iNx, Int32(1), Int32(1), Int32(1)))),
    ]

    for (name, compile_fn) in kernel_configs
        try
            k = compile_fn()
            regs = CUDA.registers(k)
            max_t = CUDA.maxthreads(k)
            shared = CUDA.memory(k).shared
            local_m = CUDA.memory(k).local

            # Theoretical occupancy at default workgroup size
            regs_per_block = regs * wg
            blocks_per_sm = min(
                div(specs.max_regs_per_block, regs_per_block),
                specs.max_blocks_per_sm,
                div(specs.max_threads_per_sm, wg)
            )
            threads_per_sm = blocks_per_sm * wg
            occ = threads_per_sm / specs.max_threads_per_sm * 100

            push!(results, KernelInfo(name, regs, max_t, shared, local_m, occ))
        catch e
            push!(results, KernelInfo(name, -1, 0, 0, 0, 0.0))
        end
    end

    # Free synthetic arrays
    CUDA.unsafe_free!(arr)
    CUDA.unsafe_free!(sigma_arr)

    return results
end

# ── Achieved bandwidth ──────────────────────────────────────────────────────

"""
    compute_bandwidth(sim, phase_timing, precision_type)

Compute achieved memory bandwidth from phase timing and grid size.
Returns NamedTuple with GB/s and % of peak for H and E update phases.
"""
function compute_bandwidth(sim, phase_timing, precision_type::Type{T}) where {T}
    num_voxels = sim.Nx * sim.Ny * sim.Nz
    bytes_per_elem = sizeof(T)

    # Interior BH kernel: reads 6 E-fields, writes 3 H-fields
    # Each field: read + write neighbor = ~2 reads + 1 write per component
    # Minimal model: 6 reads + 3 writes = 9 arrays × bytes_per_elem
    bh_bytes_per_voxel = 9 * bytes_per_elem  # conservative: no counting stencil reuse

    # Interior DE kernel: reads 6 H-fields + 3 eps arrays, writes 3 E-fields
    # Minimal model: 9 reads + 3 writes = 12 arrays × bytes_per_elem
    de_bytes_per_voxel = 12 * bytes_per_elem

    peak_gbps = get_peak_bandwidth_gbps()

    # Convert ms/step to seconds, compute bandwidth
    bh_time_s = phase_timing.step_H_ms / 1000
    de_time_s = phase_timing.step_E_ms / 1000

    bh_gbps = bh_time_s > 0 ? num_voxels * bh_bytes_per_voxel / bh_time_s / 1e9 : 0.0
    de_gbps = de_time_s > 0 ? num_voxels * de_bytes_per_voxel / de_time_s / 1e9 : 0.0

    return (
        bh_gbps = bh_gbps,
        de_gbps = de_gbps,
        bh_pct_peak = peak_gbps > 0 ? bh_gbps / peak_gbps * 100 : 0.0,
        de_pct_peak = peak_gbps > 0 ? de_gbps / peak_gbps * 100 : 0.0,
        bh_bytes_per_voxel = bh_bytes_per_voxel,
        de_bytes_per_voxel = de_bytes_per_voxel,
        peak_gbps = peak_gbps,
    )
end

# ── Memory footprint ────────────────────────────────────────────────────────

"""
    collect_memory_info()

Snapshot GPU memory allocation.
"""
function collect_memory_info()
    if !CUDA.functional()
        return (used_gb=0.0, total_gb=0.0, free_gb=0.0)
    end
    total = CUDA.total_memory()
    free = CUDA.available_memory()
    used = total - free
    return (
        used_gb = used / 1e9,
        total_gb = total / 1e9,
        free_gb = free / 1e9,
    )
end

# ── Pretty-print summary ────────────────────────────────────────────────────

"""
    print_metrics_summary(sim, phase_timing, kernel_metrics, bandwidth, memory_info; label="")

Print a formatted metrics summary table.
"""
function print_metrics_summary(sim, phase_timing, kernel_metrics, bandwidth, memory_info; label="")
    num_voxels = sim.Nx * sim.Ny * sim.Nz
    rate_mcells = num_voxels / (phase_timing.total_ms / 1000) / 1e6

    header = isempty(label) ? "METRICS" : "METRICS: $label"
    println("\n", "=" ^ 70)
    println(header)
    println("=" ^ 70)

    # Grid info
    println("Grid: $(sim.Nx) x $(sim.Ny) x $(sim.Nz) = $(num_voxels) voxels ($(round(num_voxels/1e6, digits=1))M)")

    # Phase timing
    println("\nPhase Timing ($(phase_timing.n_steps) steps avg):")
    total = phase_timing.total_ms
    for (name, val) in [
        ("step_H_fused", phase_timing.step_H_ms),
        ("step_E_fused", phase_timing.step_E_ms),
        ("sources (H+E)", phase_timing.src_H_ms + phase_timing.src_E_ms),
        ("monitors (H+E)", phase_timing.mon_H_ms + phase_timing.mon_E_ms),
    ]
        pct = total > 0 ? val / total * 100 : 0.0
        println("  $(rpad(name * ":", 20)) $(lpad(string(round(val, digits=3)), 8)) ms/step  ($(lpad(string(round(pct, digits=1)), 5))%)")
    end
    println("  $(rpad("total:", 20)) $(lpad(string(round(total, digits=3)), 8)) ms/step")
    println("  $(rpad("rate:", 20)) $(lpad(string(round(rate_mcells, digits=1)), 8)) MCells/s  ($(round(rate_mcells/1000, digits=2)) GVoxels/s)")

    # Kernel metrics
    if length(kernel_metrics) > 0
        println("\nKernel Metrics:")
        for ki in kernel_metrics
            if ki.registers < 0
                println("  $(rpad(ki.name, 22)) compile error")
                continue
            end
            spill_str = ki.local_mem > 0 ? "$(ki.local_mem)B spill" : "0 spill"
            println("  $(rpad(ki.name, 22)) $(lpad(ki.registers, 3)) regs, $(rpad(spill_str, 10)), $(lpad(string(round(ki.occupancy_pct, digits=0)), 4))% occ @ wg=256")
        end
    end

    # Bandwidth
    println("\nAchieved Bandwidth:")
    println("  BH (H-update):  $(lpad(string(round(bandwidth.bh_gbps, digits=0)), 5)) GB/s  ($(lpad(string(round(bandwidth.bh_pct_peak, digits=1)), 5))% of $(round(bandwidth.peak_gbps, digits=0)) GB/s peak)")
    println("  DE (E-update):  $(lpad(string(round(bandwidth.de_gbps, digits=0)), 5)) GB/s  ($(lpad(string(round(bandwidth.de_pct_peak, digits=1)), 5))% of $(round(bandwidth.peak_gbps, digits=0)) GB/s peak)")

    # Memory
    println("\nGPU Memory:")
    println("  Used:  $(round(memory_info.used_gb, digits=1)) GB / $(round(memory_info.total_gb, digits=1)) GB")

    println("=" ^ 70)
end

# ── Convenience function ────────────────────────────────────────────────────

"""
    run_metrics(sim, precision_type; label="", n_steps=50)

Collect and print all metrics for a simulation.
"""
function run_metrics(sim, precision_type::Type{T}; label::String="", n_steps::Int=50) where {T}
    if !CUDA.functional()
        println("Metrics collection requires CUDA backend. Skipping.")
        return nothing
    end

    phase_timing = collect_phase_timing(sim, n_steps)
    kernel_metrics = collect_kernel_metrics(precision_type)
    bandwidth = compute_bandwidth(sim, phase_timing, precision_type)
    memory_info = collect_memory_info()

    print_metrics_summary(sim, phase_timing, kernel_metrics, bandwidth, memory_info; label=label)
    return (phase_timing=phase_timing, kernel_metrics=kernel_metrics,
            bandwidth=bandwidth, memory_info=memory_info)
end

# ── YAML storage helpers ────────────────────────────────────────────────────

"""
    kernel_metrics_to_dict(kernel_metrics)

Convert kernel metrics to a Dict suitable for YAML serialization.
Stored once per hardware/backend/precision (not per-config — registers don't change with grid size).
"""
function kernel_metrics_to_dict(kernel_metrics::Vector{KernelInfo})
    d = Dict{String,Any}()
    for ki in kernel_metrics
        ki.registers < 0 && continue
        d[ki.name] = Dict{String,Any}(
            "registers"     => ki.registers,
            "max_threads"   => ki.max_threads,
            "shared_mem_bytes" => ki.shared_mem,
            "local_mem_bytes"  => ki.local_mem,
            "occupancy_pct" => ki.occupancy_pct,
        )
    end
    return d
end

"""
    collect_and_store_metrics(sim, precision_type, benchmark_dict; n_steps=50, label="")

Collect metrics for a simulation config and store them into the benchmark_dict
(the per-config YAML dict that already holds timestep_rate, resolution, etc.).

Stored fields (units noted in key names):
  - phase_step_H_ms, phase_step_E_ms, phase_sources_ms, phase_monitors_ms, phase_total_ms
  - bandwidth_H_update_GBps, bandwidth_E_update_GBps, bandwidth_peak_GBps
  - bandwidth_H_pct_peak, bandwidth_E_pct_peak
  - gpu_memory_used_GB
  - grid_Nx, grid_Ny, grid_Nz, grid_total_voxels
"""
function collect_and_store_metrics(sim, precision_type::Type{T}, benchmark_dict::Dict;
                                   n_steps::Int=50, label::String="") where {T}
    if !CUDA.functional()
        println("Metrics collection requires CUDA backend. Skipping.")
        return
    end

    phase_timing = collect_phase_timing(sim, n_steps)
    bandwidth = compute_bandwidth(sim, phase_timing, precision_type)
    memory_info = collect_memory_info()

    # Store phase timing (ms/step)
    benchmark_dict["phase_step_H_ms"] = round(phase_timing.step_H_ms, digits=3)
    benchmark_dict["phase_step_E_ms"] = round(phase_timing.step_E_ms, digits=3)
    benchmark_dict["phase_sources_ms"] = round(phase_timing.src_H_ms + phase_timing.src_E_ms, digits=3)
    benchmark_dict["phase_monitors_ms"] = round(phase_timing.mon_H_ms + phase_timing.mon_E_ms, digits=3)
    benchmark_dict["phase_total_ms"] = round(phase_timing.total_ms, digits=3)

    # Store achieved bandwidth (GB/s)
    benchmark_dict["bandwidth_H_update_GBps"] = round(bandwidth.bh_gbps, digits=1)
    benchmark_dict["bandwidth_E_update_GBps"] = round(bandwidth.de_gbps, digits=1)
    benchmark_dict["bandwidth_peak_GBps"] = round(bandwidth.peak_gbps, digits=1)
    benchmark_dict["bandwidth_H_pct_peak"] = round(bandwidth.bh_pct_peak, digits=1)
    benchmark_dict["bandwidth_E_pct_peak"] = round(bandwidth.de_pct_peak, digits=1)

    # Store memory (GB)
    benchmark_dict["gpu_memory_used_GB"] = round(memory_info.used_gb, digits=1)

    # Store grid dimensions
    benchmark_dict["grid_Nx"] = sim.Nx
    benchmark_dict["grid_Ny"] = sim.Ny
    benchmark_dict["grid_Nz"] = sim.Nz
    benchmark_dict["grid_total_voxels"] = sim.Nx * sim.Ny * sim.Nz

    # Print summary
    kernel_metrics = collect_kernel_metrics(precision_type)
    print_metrics_summary(sim, phase_timing, kernel_metrics, bandwidth, memory_info; label=label)
end

end
