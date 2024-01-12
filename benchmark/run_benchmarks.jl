# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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


# TODO enable flag parsing to select specific benchmarks to run or skip

# TODO print out estimated time of benchmark

# Run the benchmarks
println("\n===========================================")
println("Starting dipole benchmarks...")
include("dipole.jl")
println("\n===========================================")
println("Starting periodic stack benchmarks...")
include("periodic_stack.jl")
println("\n===========================================")
println("Starting sphere benchmarks...")
include("sphere.jl")
println("\n===========================================")

println("\nBenchmarks complete.")
