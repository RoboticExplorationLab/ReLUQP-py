import reluqp.reluqpth as reluqp
import reluqp.utils as utils
import torch

if __name__ == '__main__':
    nx = 100
    n_eq = int(nx/4)
    n_ineq = int(nx/4)
    H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=1)

    # double
    # solve multiple times to avoid overhead
    for _ in range(5):
        model = reluqp.ReLU_QP()
        model.setup(H=H, g=g, A=A, l=l, u=u, eps_abs=1e-2, precision=torch.float64, verbose=False)
        results = model.solve()

    print("double precision:")
    print(results.info.status)
    # print(results.x)
    print(results.info.iter)
    print(results.info.solve_time)

    # single 
    for _ in range(5):
        model = reluqp.ReLU_QP()
        model.setup(H=H, g=g, A=A, l=l, u=u, eps_abs=1e-2, precision=torch.float32, verbose=False)
        results = model.solve()

    print("single precision:")
    print(results.info.status)
    # print(results.x)
    print(results.info.iter)
    print(results.info.solve_time)
    