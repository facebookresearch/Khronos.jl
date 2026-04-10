# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Here lie the core timestepping kernels for the FDTD algorithm. Thanks to
# multiple dispatch, we can simply focus on the fundamental (most complicated)
# cases. We often need to use "function barriers" to ensure type stability.

include("./Helpers.jl")
include("./ReferenceKernels.jl")
include("./CUDAKernels.jl")
include("./Dispersive.jl")

"""
    step!(sim::SimulationData)

Primary timestepping routine.

Includes all the relevent steps to complete one full timestep, such as evolving
all of the fields and updating all of the sources and monitors.
"""
function step!(sim::SimulationData)

    t = round_time(sim)

    # P.3: Deactivate source arrays after all sources have shut off
    # This lets the update_field! kernel specialize to the Nothing path
    # Skip for CW sources (which have no cutoff)
    if sim.sources_active
        try
            if t > last_source_time(sim)
                sim.sources_active = false
            end
        catch
            # CW sources don't have a cutoff — sources stay active forever
        end
    end

    # CUDA Graph replay path (post-source only): the FDTD kernels
    # (curl + update + halo) are captured into two sub-graphs.
    # Monitor updates run outside the graph since they have time-varying arguments.
    if !isnothing(sim._cuda_graph_exec_H)
        CUDA.launch(sim._cuda_graph_exec_H)
        update_H_monitors!(sim, t)
        CUDA.launch(sim._cuda_graph_exec_E)
        step_chi3_correction!(sim)
        step_polarization!(sim)
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Attempt graph capture once sources have deactivated (local only)
    if !sim.sources_active && !is_distributed() && _try_capture_graphs!(sim)
        CUDA.launch(sim._cuda_graph_exec_H)
        update_H_monitors!(sim, t)
        CUDA.launch(sim._cuda_graph_exec_E)
        step_chi3_correction!(sim)
        step_polarization!(sim)
        update_E_monitors!(sim, t + sim.Δt / 2)
        increment_timestep!(sim)
        return
    end

    # Normal path (source-active, JIT warmup, non-CUDA backend, or distributed)
    if sim.sources_active
        update_magnetic_sources!(sim, t)
    end

    step_H_fused!(sim)

    update_H_monitors!(sim, t)

    if sim.sources_active
        update_electric_sources!(sim, t + sim.Δt / 2)
    end

    step_E_fused!(sim)

    # χ3 Kerr nonlinear correction (applied after E update, before polarization)
    step_chi3_correction!(sim)
    # E^{n+1} uses old P^n (with chi1 correction in ε_inv_eff), then P^{n+1}
    # is computed using the new E^{n+1}. This matches meep/src/step.cpp where
    # susceptibility P is updated after E, creating the correct semi-implicit coupling.
    step_polarization!(sim)

    update_E_monitors!(sim, t + sim.Δt / 2)

    increment_timestep!(sim)
end

"""
    _try_capture_graphs!(sim::SimulationData) -> Bool

Attempt to capture two CUDA graphs for the steady-state FDTD step:
  - Graph H: step_B_from_E! + update_H_from_B! (curl B kernels + update H kernels + halo exchange)
  - Graph E: step_D_from_H! + update_E_from_D! (curl D kernels + update E kernels + halo exchange)

Returns true if capture succeeded, false otherwise.
"""
function _try_capture_graphs!(sim::SimulationData)
    # Allow disabling via environment variable
    get(ENV, "KHRONOS_CUDA_GRAPHS", "1") == "0" && return false
    # Disable for complex fields (Bloch BC) — graph capture assumes fixed types
    _fields_are_complex(sim) && return false
    # Disable graph capture when multi-stream is active — standard capture
    # doesn't support cross-stream operations without STREAM_CAPTURE_MODE_RELAXED.
    sim._use_multi_stream && return false

    # Pre-compile all kernel variants by running one step outside capture.
    # This ensures JIT compilation (which allocates GPU memory) doesn't
    # invalidate the subsequent graph capture recording pass.
    step_H_fused!(sim)
    exchange_halos!(sim, :H)
    step_E_fused!(sim)
    exchange_halos!(sim, :E)
    CUDA.synchronize()

    try
        graph_H = CUDA.capture(; throw_error=false) do
            step_H_fused!(sim)
        end
        if isnothing(graph_H)
            @warn("CUDA Graph capture failed for step_H_fused!")
            return false
        end

        graph_E = CUDA.capture(; throw_error=false) do
            step_E_fused!(sim)
        end
        if isnothing(graph_E)
            @warn("CUDA Graph capture failed for step_E_fused!")
            return false
        end

        sim._cuda_graph_exec_H = CUDA.instantiate(graph_H)
        sim._cuda_graph_exec_E = CUDA.instantiate(graph_E)

        if !is_distributed() || is_root()
            @info("CUDA Graph capture successful — using graph replay for subsequent steps")
        end
        return true
    catch e
        @warn("CUDA Graph capture failed: $e\n$(sprint(showerror, e, catch_backtrace()))")
        return false
    end
end

# FIXME for non PML
get_step_boundaries(sim) = (sim.Nx, sim.Ny, sim.Nz)

# Check if fields are complex-valued (Bloch BC)
_fields_are_complex(sim) = !isnothing(sim.chunk_data) && !isempty(sim.chunk_data) &&
    !isnothing(sim.chunk_data[1].fields.fEx) && eltype(sim.chunk_data[1].fields.fEx) <: Complex

# Check if grid is uniform (scalar spacing) — raw CUDA kernels require this
_grid_is_uniform(sim) = sim.Δx isa Real && sim.Δy isa Real && (sim.ndims <= 2 || sim.Δz isa Real)
# ------------------------------------------------------------------- #
# Fused curl+update dispatch functions
#
# For chunks WITHOUT PML, use the fused step_curl_and_update! kernel
# to eliminate the B/D re-read. For PML chunks, fall back to the
# separate step_curl! + update_field! kernels.
# ------------------------------------------------------------------- #

function step_H_fused!(sim::SimulationData)
    curl_B_kernel = sim._cached_curl_kernel
    update_H_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    idx_curl = 1

    cuda_wg_x = sim._cached_cuda_wg_x
    cuda_wg_y = sim._cached_cuda_wg_y
    dt_dx = sim._cached_dt_dx
    dt_dy = sim._cached_dt_dy
    dt_dz = sim._cached_dt_dz
    use_ms = sim._use_multi_stream

    for (ci, chunk) in enumerate(sim.chunk_data)
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange

        if backend_engine isa CUDABackend && !_fields_are_complex(sim) && sim._cached_grid_is_uniform && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.μ_inv isa Real
            # Raw CUDA path: scalar μ, B eliminated — H_new = H_old + μ⁻¹·Δt·curl(E)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), Int(cuda_wg_x) * Int(cuda_wg_y))
            m_inv = backend_number(g.μ_inv)
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fHx, f.fHy, f.fHz,
                    m_inv * backend_number(dt_dx), m_inv * backend_number(dt_dy), m_inv * backend_number(dt_dz),
                    iNx, iNy)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) _cuda_fused_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fHx, f.fHy, f.fHz,
                    m_inv * backend_number(dt_dx), m_inv * backend_number(dt_dy), m_inv * backend_number(dt_dz),
                    iNx, iNy)
            end
        elseif !has_any_pml(chunk.spec.physics)
            # KA fused path: interior chunks with sources active or per-voxel μ
            fused_kernel = sim._cached_fused_kernel
            fused_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fBx, f.fBy, f.fBz,
                f.fHx, f.fHy, f.fHz,
                sa ? f.fSBx : nothing,
                sa ? f.fSBy : nothing,
                sa ? f.fSBz : nothing,
                nothing, nothing, nothing,  # No P for H update
                g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) &&
               sim._cached_grid_is_uniform && has_any_pml(chunk.spec.physics) &&
               (!chunk.spec.physics.has_sources || !sa) &&
               !chunk.spec.physics.has_sigma_B && g.μ_inv isa Real && f.fPBx isa Nothing
            # Fused 3-component raw CUDA PML: single kernel launch for all B→H
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), Int(cuda_wg_x) * Int(cuda_wg_y))
            dummy3d = f.fBx
            _σ(σ) = isnothing(σ) ? sim._cached_dummy_sigma : σ
            # PML flags: 1 if PML active on that axis, 0 otherwise
            _pml_flag(σ) = isnothing(σ) ? Int32(0) : Int32(1)

            if use_ms
                _s = sim._chunk_streams[ci]
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) stream=_s _cuda_pml_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fBx, f.fBy, f.fBz,
                    f.fHx, f.fHy, f.fHz,
                    isnothing(f.fUBx) ? dummy3d : f.fUBx,
                    isnothing(f.fUBy) ? dummy3d : f.fUBy,
                    isnothing(f.fUBz) ? dummy3d : f.fUBz,
                    isnothing(f.fWBx) ? dummy3d : f.fWBx,
                    isnothing(f.fWBy) ? dummy3d : f.fWBy,
                    isnothing(f.fWBz) ? dummy3d : f.fWBz,
                    _σ(b.σBx), _σ(b.σBy), _σ(b.σBz),
                    backend_number(g.μ_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy,
                    _pml_flag(b.σBx), _pml_flag(b.σBy), _pml_flag(b.σBz))
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) _cuda_pml_BH_kernel!(
                    f.fEx, f.fEy, f.fEz,
                    f.fBx, f.fBy, f.fBz,
                    f.fHx, f.fHy, f.fHz,
                    isnothing(f.fUBx) ? dummy3d : f.fUBx,
                    isnothing(f.fUBy) ? dummy3d : f.fUBy,
                    isnothing(f.fUBz) ? dummy3d : f.fUBz,
                    isnothing(f.fWBx) ? dummy3d : f.fWBx,
                    isnothing(f.fWBy) ? dummy3d : f.fWBy,
                    isnothing(f.fWBz) ? dummy3d : f.fWBz,
                    _σ(b.σBx), _σ(b.σBy), _σ(b.σBz),
                    backend_number(g.μ_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy,
                    _pml_flag(b.σBx), _pml_flag(b.σBy), _pml_flag(b.σBz))
            end
        else
            # PML KA path: separate curl + update
            curl_B_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fBx, f.fBy, f.fBz,
                f.fCBx, f.fCBy, f.fCBz,
                f.fUBx, f.fUBy, f.fUBz,
                g.σBx, g.σBy, g.σBz,
                b.σBx, b.σBy, b.σBz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
            update_H_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fBx, f.fBy, f.fBz,
                f.fWBx, f.fWBy, f.fWBz,
                f.fPBx, f.fPBy, f.fPBz,
                sa ? f.fSBx : nothing,
                sa ? f.fSBy : nothing,
                sa ? f.fSBz : nothing,
                g.μ_inv, g.μ_inv_x, g.μ_inv_y, g.μ_inv_z,
                b.σBx, b.σBy, b.σBz,
                ndrange = nr,
            )
        end
    end

    if use_ms
        # GPU-side sync: make default stream wait for all chunk streams
        for i in eachindex(sim._chunk_streams)
            CUDA.synchronize(sim._chunk_streams[i])
        end
    end

    exchange_halos!(sim, :H)
    return
end

function step_E_fused!(sim::SimulationData)
    curl_D_kernel = sim._cached_curl_kernel
    update_E_kernel = sim._cached_update_kernel
    sa = sim.sources_active
    idx_curl = -1

    cuda_wg_x = sim._cached_cuda_wg_x
    cuda_wg_y = sim._cached_cuda_wg_y
    dt_dx = sim._cached_dt_dx
    dt_dy = sim._cached_dt_dy
    dt_dz = sim._cached_dt_dz
    use_ms = sim._use_multi_stream

    for (ci, chunk) in enumerate(sim.chunk_data)
        f = chunk.fields; g = chunk.geometry_data; b = chunk.boundary_data
        nr = chunk.ndrange

        if backend_engine isa CUDABackend && !_fields_are_complex(sim) && sim._cached_grid_is_uniform && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.ε_inv_x isa AbstractArray && f.fPDx isa Nothing
            # Raw CUDA path: per-voxel ε, D eliminated — E_new = E_old + ε⁻¹·Δt·curl(H)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), Int(cuda_wg_x) * Int(cuda_wg_y))
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) _cuda_fused_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy)
            end
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) && sim._cached_grid_is_uniform && !has_any_pml(chunk.spec.physics) && (!chunk.spec.physics.has_sources || !sa) && g.ε_inv isa Real && f.fPDx isa Nothing
            # Raw CUDA path: scalar ε, D eliminated — E_new = E_old + ε⁻¹·Δt·curl(H)
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), Int(cuda_wg_x) * Int(cuda_wg_y))
            if use_ms
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) stream=sim._chunk_streams[ci] _cuda_fused_DE_scalar_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    backend_number(g.ε_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy)
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) _cuda_fused_DE_scalar_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fEx, f.fEy, f.fEz,
                    backend_number(g.ε_inv),
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy)
            end
        elseif !has_any_pml(chunk.spec.physics)
            # KA fused path: interior chunks with sources active
            fused_kernel = sim._cached_fused_kernel
            fused_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fDx, f.fDy, f.fDz,
                f.fEx, f.fEy, f.fEz,
                sa ? f.fSDx : nothing,
                sa ? f.fSDy : nothing,
                sa ? f.fSDz : nothing,
                f.fPDx, f.fPDy, f.fPDz,
                g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
        elseif backend_engine isa CUDABackend && !_fields_are_complex(sim) &&
               sim._cached_grid_is_uniform && has_any_pml(chunk.spec.physics) &&
               (!chunk.spec.physics.has_sources || !sa) &&
               !chunk.spec.physics.has_sigma_D && f.fPDx isa Nothing
            # Fused 3-component raw CUDA PML: single kernel launch for all D→E
            iNx = Int32(nr[1]); iNy = Int32(nr[2]); iNz = Int32(nr[3])
            nblocks_x = cld(Int(iNx), Int(cuda_wg_x) * Int(cuda_wg_y))
            dummy3d = f.fDx
            _σ(σ) = isnothing(σ) ? sim._cached_dummy_sigma : σ
            _pml_flag(σ) = isnothing(σ) ? Int32(0) : Int32(1)

            # Resolve per-voxel or scalar ε⁻¹ for each component
            eps_x = g.ε_inv_x isa AbstractArray ? g.ε_inv_x : backend_number(g.ε_inv)
            eps_y = g.ε_inv_y isa AbstractArray ? g.ε_inv_y : backend_number(g.ε_inv)
            eps_z = g.ε_inv_z isa AbstractArray ? g.ε_inv_z : backend_number(g.ε_inv)

            if use_ms
                _s = sim._chunk_streams[ci]
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) stream=_s _cuda_pml_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fDx, f.fDy, f.fDz,
                    f.fEx, f.fEy, f.fEz,
                    isnothing(f.fUDx) ? dummy3d : f.fUDx,
                    isnothing(f.fUDy) ? dummy3d : f.fUDy,
                    isnothing(f.fUDz) ? dummy3d : f.fUDz,
                    isnothing(f.fWDx) ? dummy3d : f.fWDx,
                    isnothing(f.fWDy) ? dummy3d : f.fWDy,
                    isnothing(f.fWDz) ? dummy3d : f.fWDz,
                    _σ(b.σDx), _σ(b.σDy), _σ(b.σDz),
                    eps_x, eps_y, eps_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy,
                    _pml_flag(b.σDx), _pml_flag(b.σDy), _pml_flag(b.σDz))
            else
                @cuda blocks=(nblocks_x, Int(iNy), Int(iNz)) threads=(Int(cuda_wg_x) * Int(cuda_wg_y), 1, 1) _cuda_pml_DE_kernel!(
                    f.fHx, f.fHy, f.fHz,
                    f.fDx, f.fDy, f.fDz,
                    f.fEx, f.fEy, f.fEz,
                    isnothing(f.fUDx) ? dummy3d : f.fUDx,
                    isnothing(f.fUDy) ? dummy3d : f.fUDy,
                    isnothing(f.fUDz) ? dummy3d : f.fUDz,
                    isnothing(f.fWDx) ? dummy3d : f.fWDx,
                    isnothing(f.fWDy) ? dummy3d : f.fWDy,
                    isnothing(f.fWDz) ? dummy3d : f.fWDz,
                    _σ(b.σDx), _σ(b.σDy), _σ(b.σDz),
                    eps_x, eps_y, eps_z,
                    backend_number(dt_dx), backend_number(dt_dy), backend_number(dt_dz),
                    iNx, iNy,
                    _pml_flag(b.σDx), _pml_flag(b.σDy), _pml_flag(b.σDz))
            end
        else
            # PML KA path: separate curl + update
            curl_D_kernel(
                f.fHx, f.fHy, f.fHz,
                f.fDx, f.fDy, f.fDz,
                f.fCDx, f.fCDy, f.fCDz,
                f.fUDx, f.fUDy, f.fUDz,
                g.σDx, g.σDy, g.σDz,
                b.σDx, b.σDy, b.σDz,
                sim.Δt, sim.Δx, sim.Δy, sim.Δz, idx_curl,
                ndrange = nr,
            )
            update_E_kernel(
                f.fEx, f.fEy, f.fEz,
                f.fDx, f.fDy, f.fDz,
                f.fWDx, f.fWDy, f.fWDz,
                f.fPDx, f.fPDy, f.fPDz,
                sa ? f.fSDx : nothing,
                sa ? f.fSDy : nothing,
                sa ? f.fSDz : nothing,
                g.ε_inv, g.ε_inv_x, g.ε_inv_y, g.ε_inv_z,
                b.σDx, b.σDy, b.σDz,
                ndrange = nr,
            )
        end
    end

    if use_ms
        # GPU-side sync: make default stream wait for all chunk streams
        for i in eachindex(sim._chunk_streams)
            CUDA.synchronize(sim._chunk_streams[i])
        end
    end

    exchange_halos!(sim, :E)
    return
end
function update_magnetic_sources!(sim::SimulationData, t::Real)
    map((c) -> step_sources!(sim, c, t), (Hx(), Hy(), Hz()))
    return
end

function update_electric_sources!(sim::SimulationData, t::Real)
    map((c) -> step_sources!(sim, c, t), (Ex(), Ey(), Ez()))
    return
end

function update_H_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        # Skip monitor data that don't have .component (e.g., Near2FarMonitorData)
        hasfield(typeof(m), :component) || continue
        # Decimation: skip DFT updates on non-decimation steps
        if hasfield(typeof(m), :decimation) && m.decimation > 1 && sim.timestep % m.decimation != 0
            continue
        end
        if is_magnetic(m.component)
            update_monitor(sim, m, time)
        end
    end

    # Monitors read local chunk data; no halo exchange needed.
    return
end

function update_E_monitors!(sim::SimulationData, time)
    for m in sim.monitor_data
        # Skip monitor data that don't have .component (e.g., Near2FarMonitorData)
        hasfield(typeof(m), :component) || continue
        # Decimation: skip DFT updates on non-decimation steps
        if hasfield(typeof(m), :decimation) && m.decimation > 1 && sim.timestep % m.decimation != 0
            continue
        end
        if is_electric(m.component)
            update_monitor(sim, m, time)
        end
    end

    # Monitors read local chunk data; no halo exchange needed.
    return
end
