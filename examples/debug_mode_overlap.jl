# Diagnostic: verify mode overlap normalization
#
# For a mode with itself, the overlap integral should give:
#   a+ = ∫∫ (E × H* + E* × H) · n̂ dA / (4 P_mode) = 1.0
#
# This tests the ModeMonitor overlap computation independent of the simulation.

import Khronos
using Khronos: VectorModesolver
using GeometryPrimitives

Khronos.choose_backend(Khronos.CPUDevice(), Float64)

# Simple waveguide geometry
n_Si = 3.47
n_SiO2 = 1.44
wg_width = 0.5
wg_height = 0.22

geometry = [
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [10.0, wg_width, wg_height]),
        Khronos.Material(ε = n_Si^2),
    ),
    Khronos.Object(
        Cuboid([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        Khronos.Material(ε = n_SiO2^2),
    ),
]

freq = 1.0 / 1.55  # center frequency
mode_res = 50

println("=" ^ 60)
println("Mode overlap self-consistency test")
println("=" ^ 60)

# Solve for mode on YZ plane (x-normal)
mode = Khronos.get_mode_profiles(;
    frequency = freq,
    mode_solver_resolution = mode_res,
    mode_index = 1,
    center = [0.0, 0.0, 0.0],
    size = [0.0, 6 * wg_width, 6 * wg_height],
    solver_tolerance = 1e-6,
    geometry = geometry,
)

println("Mode neff = $(mode.neff)")
println("Mode x range: $(first(mode.x)) to $(last(mode.x)), N = $(length(mode.x))")
println("Mode y range: $(first(mode.y)) to $(last(mode.y)), N = $(length(mode.y))")
println("Mode field shapes: Ex=$(size(mode.Ex)), Ey=$(size(mode.Ey)), Ez=$(size(mode.Ez))")
println("  Hx=$(size(mode.Hx)), Hy=$(size(mode.Hy)), Hz=$(size(mode.Hz))")

# For x-normal (YZ plane):
# tangential E: Ey, Ez
# tangential H: Hy, Hz
# Cross product: (E × H*) · x̂ = Ey*Hz* - Ez*Hy*
normal_axis = 1

# Squeeze mode fields to 2D
function squeeze_mode(f, na)
    if ndims(f) == 3
        s = if na == 1; f[1,:,:] elseif na == 2; f[:,1,:] else f[:,:,1] end
        return permutedims(s)
    elseif ndims(f) == 2
        return permutedims(f)
    else
        return reshape(f, 1, length(f))
    end
end

mode_Ey = squeeze_mode(mode.Ey, normal_axis)
mode_Ez = squeeze_mode(mode.Ez, normal_axis)
mode_Hy = squeeze_mode(mode.Hy, normal_axis)
mode_Hz = squeeze_mode(mode.Hz, normal_axis)

println("\nSqueezed mode field shapes:")
println("  Ey: $(size(mode_Ey)), Ez: $(size(mode_Ez))")
println("  Hy: $(size(mode_Hy)), Hz: $(size(mode_Hz))")

# Grid spacings
dx_mode = mode.x[2] - mode.x[1]
dy_mode = mode.y[2] - mode.y[1]
dA = dx_mode * dy_mode

println("  dx = $dx_mode, dy = $dy_mode, dA = $dA")

n1 = length(mode.x)
n2 = length(mode.y)

println("  n1 = $n1 (along mode.x), n2 = $n2 (along mode.y)")

# Check: does mode field indexing match?
# mode_Ey[i,j] should correspond to mode.x[i], mode.y[j]
println("\n  mode_Ey[n1÷2, n2÷2] = $(mode_Ey[n1÷2, n2÷2])")
println("  mode_Ey max = $(maximum(abs.(mode_Ey)))")
println("  mode_Hz max = $(maximum(abs.(mode_Hz)))")

# Compute P_mode using mode.x/mode.y grid directly
P_mode_direct = let
    P = 0.0
    for j in 1:n2, i in 1:n1
        P += 0.5 * real(
            mode_Ey[i, j] * conj(mode_Hz[i, j]) -
            mode_Ez[i, j] * conj(mode_Hy[i, j])
        ) * dA
    end
    P
end

println("\nP_mode (direct on mode grid): $P_mode_direct")

# Now compute the overlap of the mode with itself
# a+ = ∫∫ (E × H* + E* × H) · n̂ dA / (4 P_mode)
overlap_plus, overlap_minus = let
    op = ComplexF64(0)
    om = ComplexF64(0)
    for j in 1:n2, i in 1:n1
        e1 = mode_Ey[i, j]
        e2 = mode_Ez[i, j]
        h1 = mode_Hy[i, j]
        h2 = mode_Hz[i, j]

        # sim fields = mode fields (self-overlap test)
        sim_cross_mode = e1 * conj(h2) - e2 * conj(h1)
        mode_cross_sim = conj(e1) * h2 - conj(e2) * h1

        op += (sim_cross_mode + mode_cross_sim) * dA
        om += (sim_cross_mode - mode_cross_sim) * dA
    end
    (op, om)
end

a_plus_self = overlap_plus / (4.0 * P_mode_direct)
a_minus_self = overlap_minus / (4.0 * P_mode_direct)

println("\nSelf-overlap results:")
println("  overlap_plus  = $overlap_plus")
println("  overlap_minus = $overlap_minus")
println("  a+            = $a_plus_self  (should be 1.0)")
println("  a-            = $a_minus_self (should be 0.0)")
println("  |a+|²         = $(abs2(a_plus_self))")
println("  |a-|²         = $(abs2(a_minus_self))")

# Verification:
# overlap_plus = ∫ (E×H* + E*×H) · n̂ dA = 2 Re{∫ (E×H*) · n̂ dA} = 2 * (2 P_mode) = 4 P_mode
# So a+ = 4 P_mode / (4 P_mode) = 1 ✓

# Now test with bilinear interpolation (like compute_mode_amplitudes does)
println("\n" * "=" ^ 60)
println("Testing with bilinear interpolation (DFT grid → mode grid)")
println("=" ^ 60)

# Simulate a DFT grid with coarser resolution (25 pts/μm like the FDTD sim)
fdtd_res = 25
dx_fdtd = 1.0 / fdtd_res
dy_fdtd = dx_fdtd
dA_fdtd = dx_fdtd * dy_fdtd

# DFT grid spans same physical region as mode solver
y_start = first(mode.x) + dx_fdtd / 2
y_end = last(mode.x) - dx_fdtd / 2
z_start = first(mode.y) + dy_fdtd / 2
z_end = last(mode.y) - dy_fdtd / 2
n1_fdtd = floor(Int, (y_end - y_start) / dx_fdtd) + 1
n2_fdtd = floor(Int, (z_end - z_start) / dy_fdtd) + 1

println("DFT grid: n1=$n1_fdtd, n2=$n2_fdtd, dx=$dx_fdtd, dy=$dy_fdtd")

# Interpolate mode onto DFT grid
interp_Ey = zeros(ComplexF64, n1_fdtd, n2_fdtd)
interp_Ez = zeros(ComplexF64, n1_fdtd, n2_fdtd)
interp_Hy = zeros(ComplexF64, n1_fdtd, n2_fdtd)
interp_Hz = zeros(ComplexF64, n1_fdtd, n2_fdtd)

mode_x_arr = collect(mode.x)
mode_y_arr = collect(mode.y)

for i2 in 1:n2_fdtd, i1 in 1:n1_fdtd
    py = y_start + (i1 - 1) * dx_fdtd
    pz = z_start + (i2 - 1) * dy_fdtd
    pt = [py, pz]
    global interp_Ey[i1, i2] = Khronos.bilinear_interpolator(mode_x_arr, mode_y_arr, mode_Ey, pt)
    global interp_Ez[i1, i2] = Khronos.bilinear_interpolator(mode_x_arr, mode_y_arr, mode_Ez, pt)
    global interp_Hy[i1, i2] = Khronos.bilinear_interpolator(mode_x_arr, mode_y_arr, mode_Hy, pt)
    global interp_Hz[i1, i2] = Khronos.bilinear_interpolator(mode_x_arr, mode_y_arr, mode_Hz, pt)
end

println("  interp Ey max = $(maximum(abs.(interp_Ey)))")
println("  interp Hz max = $(maximum(abs.(interp_Hz)))")

# P_mode on interpolated grid
P_mode_interp = let
    P = 0.0
    for i2 in 1:n2_fdtd, i1 in 1:n1_fdtd
        P += 0.5 * real(
            interp_Ey[i1, i2] * conj(interp_Hz[i1, i2]) -
            interp_Ez[i1, i2] * conj(interp_Hy[i1, i2])
        ) * dA_fdtd
    end
    P
end

println("  P_mode (interpolated grid): $P_mode_interp")
println("  P_mode ratio (interp/direct): $(P_mode_interp / P_mode_direct)")

# Self-overlap with interpolated mode
overlap_interp = let
    ov = ComplexF64(0)
    for i2 in 1:n2_fdtd, i1 in 1:n1_fdtd
        e1 = interp_Ey[i1, i2]
        e2 = interp_Ez[i1, i2]
        h1 = interp_Hy[i1, i2]
        h2 = interp_Hz[i1, i2]

        sim_cross_mode = e1 * conj(h2) - e2 * conj(h1)
        mode_cross_sim = conj(e1) * h2 - conj(e2) * h1

        ov += (sim_cross_mode + mode_cross_sim) * dA_fdtd
    end
    ov
end

a_plus_interp = overlap_interp / (4.0 * P_mode_interp)
println("  a+ (interpolated, self-overlap): $a_plus_interp  (should be 1.0)")
println("  |a+|² = $(abs2(a_plus_interp))")

println("\n" * "=" ^ 60)
println("Done")
println("=" ^ 60)
