using CUDA
using BenchmarkTools

nx = 10
# matrix vector product on GPU single vs double precision
# set seed on cuda
CUDA.seed!(1234)
dtype = Float32
A = CUDA.randn(dtype, nx, nx);
b = CUDA.randn(dtype, nx);

@btime A*b;


# double precision
CUDA.seed!(1234);
dtype = Float64;
A = CUDA.rand(dtype, nx, nx);
x = CUDA.rand(dtype, nx);

@btime A*b;


