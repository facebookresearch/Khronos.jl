# (c) Meta Platforms, Inc. and affiliates.
#
# Handle all of the backend loading.
using CUDA
using CUDA.CUDAKernels
using Metal
using Metal.MetalKernels

export CUDADevice, MetalDevice, CPUDevice

abstract type Device end
struct CUDADevice <: Device end
struct MetalDevice <: Device end
struct AMDDevice <: Device end
struct CPUDevice <: Device end

backend_engine = CPU()
backend_number = Float64
backend_array = Array{backend_number,N} where {N}
complex_backend_number = ComplexF64
complex_backend_array = Array{Complex{backend_number},N} where {N}

function choose_backend(::MetalDevice, number_type::DataType)
    println("Using Metal backend...")

    global backend_engine = MetalBackend()
    global backend_number = number_type
    global complex_backend_number = Complex{backend_number}
    global backend_array = MtlArray{backend_number,N} where {N}
    global complex_backend_array = MtlArray{complex_backend_number,N} where {N}
    return
end

function choose_backend(::CUDADevice, number_type::DataType)
    println("Using CUDA backend...")

    global backend_engine = CUDABackend()
    global backend_number = number_type
    global complex_backend_number = Complex{backend_number}
    global backend_array = CuArray{backend_number,N} where {N}
    global complex_backend_array = CuArray{complex_backend_number,N} where {N}
    return
end

function choose_backend(::CPUDevice, number_type::DataType)
    println("Using CPU backend...")

    global backend_number = number_type
    global complex_backend_number = Complex{backend_number}
    global backend_array = Array{backend_number,N} where {N}
    global complex_backend_array = Array{complex_backend_number,N} where {N}
    return
end

function choose_backend(number_type::DataType)
    choose_backend(CPUDevice(), number_type)
    return
end

# CPU already initialized
function choose_backend()
    println("Using CPU backend...")
    return
end
