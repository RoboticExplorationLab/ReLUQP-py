import numpy as np
import cvxpy as cp

# lazy randn
def randn(*dims):
    return np.random.randn(*dims)

def rand(*dims):
    return np.random.rand(*dims)

def rand_qp(nx=10, n_eq=5, n_ineq=5, seed=1, compute_sol=True):
    np.random.seed(seed)
    H = randn(nx, nx)
    H = H.T @ H + np.eye(nx)
    H = H + H.T

    A = randn(n_eq, nx)
    C = randn(n_ineq, nx)

    active_ineq = randn(n_ineq) > 0.5

    mu = randn(n_eq)
    lamb = (randn(n_ineq))*active_ineq

    x = randn(nx)
    b = A@x
    d = C@x - randn(n_ineq)*(~active_ineq)

    g = -H@x - A.T@mu - C.T@lamb
    
    if compute_sol:
        x = cp.Variable(nx)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(H)) + g.T@x), [A@x == b, C@x >= d])
        prob.solve()
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)), 
            np.concatenate((b, np.full(n_ineq, np.inf))), x.value)
    else:
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)), 
            np.concatenate((b, np.full(n_ineq, np.inf))), None)


def update_qp(H, A, n_eq, n_ineq, seed=1, compute_sol=True):
    """
    Update the QP problem with vectors
    """
    np.random.seed(seed)
    nx = H.shape[0]
    C = A[n_eq:]
    A = A[:n_eq]
    
    active_ineq = randn(n_ineq) > 0.5
    mu = randn(n_eq)
    lamb = (randn(n_ineq))*active_ineq

    x = randn(nx)
    b = A@x
    d = C@x - randn(n_ineq)*(~active_ineq)

    g = -H@x - A.T@mu - C.T@lamb

    if compute_sol:
        x = cp.Variable(nx)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(H)) + g.T@x), [A@x == b, C@x >= d])
        prob.solve()

        return (H, g, np.vstack((A, C)), np.concatenate((b, d)),
            np.concatenate((b, np.full(n_ineq, np.inf))), x.value)
    else:
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)),
            np.concatenate((b, np.full(n_ineq, np.inf))), None)
