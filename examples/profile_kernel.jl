# Minimal profiling script: pre-compiles everything, then profiles stepping
import Khronos
using CUDA

N = parse(Int, ARGS[1])

Khronos.choose_backend(Khronos.CUDADevice(), Float64)

cell_val = Float64(N) / 10.0
dpml = 1.0

sources = [
    Khronos.UniformSource(
        time_profile = Khronos.ContinuousWaveSource(fcen=1.0),
        component = Khronos.Ez(),
        center = [0.0, 0.0, 0.0],
        size   = [0.0, 0.0, 0.0],
    ),
]

sim = Khronos.Simulation(
    cell_size   = [cell_val, cell_val, cell_val],
    cell_center = [0.0, 0.0, 0.0],
    resolution  = 10,
    sources     = sources,
    boundaries  = [[dpml, dpml], [dpml, dpml], [dpml, dpml]],
)

# Prepare and warmup fully (all JIT happens here)
Khronos.prepare_simulation!(sim)
for _ in 1:20
    Khronos.step!(sim)
end
CUDA.synchronize()

println("Warmup complete. Starting profiled region...")

# Profile just 3 timesteps using CUPTI range
CUDA.@profile external=true begin
    for _ in 1:3
        Khronos.step!(sim)
    end
    CUDA.synchronize()
end

println("Profiling complete.")
