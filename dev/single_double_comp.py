import reluqp.reluqpth as reluqp
import reluqp.utils as utils
import torch
import timeit 

@torch.jit.script
def matrix_vector_product(A, b):
    return A@b

if __name__ == '__main__':
    # time and print matrix vector product in single vs doubel precision on gpu
    # set seed
    torch.manual_seed(0)
    nx = 500
    type = torch.float64
    A = torch.randn(nx, nx, device="cuda", dtype=type)
    b = torch.randn(nx, device="cuda", dtype=type)
    # d_time = timeit.timeit(lambda: torch.mv(A, b), number=1000)/1000
    # d_time = timeit.timeit(lambda: A@b, number=1000)/1000
    d_time = timeit.timeit(lambda: matrix_vector_product(A, b), number=1000)/1000
    print("double precision time: ", d_time)


    type = torch.float32
    torch.manual_seed(0)
    A = torch.randn(nx, nx, device="cuda", dtype=type)
    b = torch.randn(nx, device="cuda", dtype=type)
    # s_time = timeit.timeit(lambda: torch.mv(A, b), number=1000)/1000
    # s_time = timeit.timeit(lambda: A@b, number=1000)/1000
    s_time = timeit.timeit(lambda: matrix_vector_product(A, b), number=1000)/1000
    print("single precision time: ", s_time)

    print("double/single ratio: ", d_time/s_time)

