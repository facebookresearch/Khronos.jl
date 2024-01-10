# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

abstract type Boundary end

function sigma_helper(idx,N,Δx,Δt,length_left,length_right)
    u0(pml_length) = -log(1e-15) / (4 * pml_length * 1/3) * (0.5 * Δt)
    u(x) = x^2  * 0.5 * (sign(x) + 1)

    total_length = N*Δx/2 # remember Δx is voxel width, but we have 2 σ's/voxel...
    real_idx = idx*Δx/2 # transform idx to position

    if real_idx < length_left
        return u0(length_left) * u((length_left - real_idx) / length_left)
    elseif (total_length-real_idx) < length_right
        return u0(length_right)* u((length_right-(total_length-real_idx))/length_right)
    else
        return 0
    end
end

function compute_sigma(sim,N,Δx,Δt,length_left,length_right)
    σ = zeros(N)
    if ((length_left != 0.0) || (length_right != 0.0))
        for idx in 1:N
            σ[idx] = sigma_helper(idx,N,Δx,Δt,length_left,length_right)
        end
    end
    return backend_array(σ)
end

function init_PML_in_direction(sim::SimulationData,dir::Direction)

end

function init_PML_in_direction(sim::SimulationData,dir::X)

end

function init_PML_in_direction(sim::SimulationData,dir::Y)

end

function init_PML_in_direction(sim::SimulationData,dir::Z)

end

function init_PML(sim::SimulationData, dims::TwoD)

end

function init_boundaries(sim::SimulationData, boundaries::Nothing)
    # No PML
    sim.boundary_data = BoundaryData{backend_array}()
end

function init_boundaries(sim::SimulationData, boundaries::Vector{Vector{T}}) where {T<:Real}
    sim.boundary_data = BoundaryData{backend_array}(
        σBx = compute_sigma(sim,2*sim.Nx+1,sim.Δx,sim.Δt,sim.boundaries[1][1],sim.boundaries[1][2]),
        σBy = compute_sigma(sim,2*sim.Ny+1,sim.Δy,sim.Δt,sim.boundaries[2][1],sim.boundaries[2][2]),
        σBz = (sim.ndims > 2) ? compute_sigma(sim,2*sim.Nz+1,sim.Δz,sim.Δt,sim.boundaries[3][1],sim.boundaries[3][2]) : nothing,
        σDx = compute_sigma(sim,2*sim.Nx+1,sim.Δx,sim.Δt,sim.boundaries[1][1],sim.boundaries[1][2]),
        σDy = compute_sigma(sim,2*sim.Ny+1,sim.Δy,sim.Δt,sim.boundaries[2][1],sim.boundaries[2][2]),
        σDz = (sim.ndims > 2) ? compute_sigma(sim,2*sim.Nz+1,sim.Δz,sim.Δt,sim.boundaries[3][1],sim.boundaries[3][2]) : nothing,
    )
end

"""
When we update the boundaries of a fields array, we have two options:
* update it from the appropriate curl/step equations
* zero it out (a Dirichlet condition)

"""

"""
We sometimes need to pull the corresponding PML conductivity just from the
standard field component (e.g. Bx, Ey). But sometimes, we want to pull either
the _next_ or _previous_ field array. To do this, we just add an additional
argument for the direction we truly care about.
"""
get_pml_conductivity_from_field(sim::SimulationData, ::Magnetic, ::X) = sim.boundary_data.σBx
get_pml_conductivity_from_field(sim::SimulationData, ::Magnetic, ::Y) = sim.boundary_data.σBy
get_pml_conductivity_from_field(sim::SimulationData, ::Magnetic, ::Z) = sim.boundary_data.σBz
get_pml_conductivity_from_field(sim::SimulationData, ::Electric, ::X) = sim.boundary_data.σDx
get_pml_conductivity_from_field(sim::SimulationData, ::Electric, ::Y) = sim.boundary_data.σDy
get_pml_conductivity_from_field(sim::SimulationData, ::Electric, ::Z) = sim.boundary_data.σDz
