import reluqp.reluqpth as reluqp
import reluqp.utils as utils

if __name__ == '__main__':
    nx = 10
    n_eq = 5
    n_ineq = 5
    H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq)

    model = reluqp.ReLU_QP()
    model.setup(H=H, g=g, A=A, l=l, u=u)
    results = model.solve()

    print(results.info.status)
    print(results.x)

    