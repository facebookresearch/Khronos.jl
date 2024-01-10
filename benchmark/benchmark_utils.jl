# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

module BenchmarkUtils

import fdtd

using CUDA
using ArgParse
using Test

export detect_and_set_backend, get_hardware_key, benchmark_result

# How much better does the current test need to perform before we issue a
# suggestion to update the baseline
UPDATE_FACTOR = 2.0

backend_string_to_struct = Dict([
    ("CUDA", fdtd.CUDADevice()),
    ("Metal", fdtd.MetalDevice()),
    ("CPU", fdtd.CPUDevice()),
])

precision_string_to_type = Dict([
    ("Float32", Float32),
    ("Float64", Float64),
])

"""
    detect_and_set_backend()

TBW
"""
function detect_and_set_backend()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--backend"
            help = "Specify a backend (`CUDA`, `METAL`, `CPU`)"
            default = nothing
            arg_type = Union{Nothing,String}
        "--precision"
            help = "Specify the precision (`Float32`, `Float64`)"
            default = nothing
            arg_type = Union{Nothing,String}
        "--profile"
            help = "Profile the current benchmark (e.g. to update values)"
            action = :store_true
    end

    parsed_args = parse_args(s)
    backend_struct, precion_struct = detect_and_set_backend(parsed_args["backend"], parsed_args["precision"])
    return backend_struct, precion_struct, parsed_args["profile"]
end

function detect_and_set_backend(backend::String, precision::String)
    if !haskey(backend_string_to_struct, backend)
        error("Invalid backend string specified: $backend.\n Choose between `CUDA`, `Metal`,  and `CPU`.")
    end

    if !haskey(precision_string_to_type, precision)
        error("Invalid precion string specified: $precision.\n Choose between `Float32` and `Float64`.")
    end

    backend_struct = backend_string_to_struct[backend]
    precision_type = precision_string_to_type[precision]
    fdtd.choose_backend(backend_struct, precision_type)

    return backend, precision
end

function detect_and_set_backend(backend::Nothing, precision::String)
    if !haskey(precision_string_to_type, precision)
        error("Invalid precion string specified: $precision.\n Choose between `Float32` and `Float64`.")
    end

    # default backend is CPU
    default_backend = "CPU"
    return detect_and_set_backend(default_backend, precision)
end

function detect_and_set_backend(backend::String, precision::Nothing)
    if !haskey(backend_string_to_struct, backend)
        error("Invalid backend string specified: $backend.\n Choose between `CUDA`, `Metal`,  and `CPU`.")
    end

    if backend == "Metal"
        default_precision = "Float32"
    else
        default_precision = "Float64"
    end

    return detect_and_set_backend(backend, default_precision)
end

function detect_and_set_backend(backend::Nothing, precision::Nothing)

    if fdtd.CUDA.functional()
         # Check for CUDA
         default_backend = "CUDA"
         default_precision = "Float64"
    elseif fdtd.Metal.functional()
        # Check for Metal
        default_backend = "Metal"
        default_precision = "Float32"
    else
        # Default to CPU
        default_backend = "CPU"
        default_precision = "Float64"
    end

    return detect_and_set_backend(default_backend,default_precision)
end

"""
    benchmark_rate(timestep_rate::Number, benchmark_rate::Number, tolerance::Number, profile_run::Bool)

TBW
"""
function benchmark_result(
    timestep_rate::Number,
    benchmark_rate::Number,
    tolerance::Number,
    profile_run::Bool,
    benchmark_substruct::Dict,
    )

    if profile_run
        benchmark_substruct["timestep_rate"] = timestep_rate
    else
        @test (timestep_rate * tolerance) > benchmark_rate
        if timestep_rate > UPDATE_FACTOR * benchmark_rate
            @warn ("Current test ($timestep_rate) significantly outperforms benchmark. Consider updating the benchmark.")
        end
    end

    return
end

function get_hardware_key()
    hardware_key = Sys.cpu_info()[1].model
    if CUDA.functional()
        hardware_key = CUDA.name(CUDA.device())
    end

    return hardware_key
end


end