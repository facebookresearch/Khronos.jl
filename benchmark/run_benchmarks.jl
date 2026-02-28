# Copyright (c) Meta Platforms, Inc. and affiliates.

using CUDA

println("===========================================")
println("Running benchmark suite")
println("===========================================")

# Print out hardware specs
println("\nHardware specs:")
println(Sys.cpu_info()[1].model)
num_cores = length(Sys.cpu_info())
println("Number of cores: $num_cores")
if CUDA.functional()
    println(CUDA.name(CUDA.device()))
end

# All available benchmarks in order of complexity
ALL_BENCHMARKS = [
    ("dipole",              "dipole.jl"),
    ("periodic_stack",      "periodic_stack.jl"),
    ("sphere",              "sphere.jl"),
    ("waveguide_mode",      "waveguide_mode.jl"),
    ("directional_coupler", "directional_coupler.jl"),
    ("mmi_coupler",         "mmi_coupler.jl"),
    ("ring_coupler",        "ring_coupler.jl"),
    ("periodic_bloch",      "periodic_bloch.jl"),
    ("uled",                "uled.jl"),
    ("metalens",            "metalens.jl"),
]

# Parse --select flag to run specific benchmarks
# Usage: julia run_benchmarks.jl --select=dipole,sphere,metalens
# The --select flag is stripped from ARGS before passing to individual benchmarks
# so that ArgParse in each benchmark does not see it as an unknown option.
local_selected = nothing
filtered_args = String[]
for arg in ARGS
    if startswith(arg, "--select=")
        names = split(arg[length("--select=")+1:end], ",")
        local_selected = String.(strip.(names))
    else
        push!(filtered_args, arg)
    end
end

# Replace ARGS with filtered version (removes --select)
empty!(ARGS)
append!(ARGS, filtered_args)

if !isnothing(local_selected)
    println("\nSelected benchmarks: $(join(local_selected, ", "))")
end

# Run the benchmarks
for (name, file) in ALL_BENCHMARKS
    if !isnothing(local_selected) && !(name in local_selected)
        continue
    end

    println("\n===========================================")
    println("Starting $name benchmarks...")
    println("===========================================")
    include(file)
end

println("\n===========================================")
println("Benchmarks complete.")
println("===========================================")
