"""
benchmarking random QP
"""

import reluqp.reluqpth as reluqpth
import reluqp.utils as utils
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
import proxsuite
import tqdm

class Random_QP_benchmark():
    def __init__(self) -> None:
        pass

    def reluqpth_solve(self, nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-4):
        H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=seed)
        model = reluqpth.ReLU_QP()
        model.setup(H=H, g=g, A=A, l=l, u=u, eps_abs=tol)
        results = model.solve()
        assert results.info.status == 'solved'
        return results.info.solve_time, results.x
    
    def osqp_solve(self, nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-4):
        H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=seed)
        model = osqp.OSQP()
        model.setup(P=sparse.csc_matrix(H), q=g, A=sparse.csc_matrix(A), l=l, u=u, eps_abs=tol, eps_rel=0, verbose=False)
        results = model.solve()
        assert results.info.status == 'solved'
        return results.info.solve_time, results.x
    
    def proxqp_solve(self, nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-4):
        H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=seed)
        model = proxsuite.proxqp.dense.QP(nx, n_eq, n_ineq)
        model.settings.eps_abs = tol
        model.settings.eps_rel = 0
        model.settings.verbose = False
        model.settings.compute_timings= True
        model.settings.initial_guess = proxsuite.proxqp.NO_INITIAL_GUESS
        model.init(H, g, A[:n_eq], l[:n_eq], A[n_eq:], l[n_eq:], u[n_eq:])
        model.solve()

        return model.results.info.run_time/1e6, model.results.x

    def random_initial_solve(self, nx_min=10, nx_max=1000, n_sample=10, n_seeds=10, n_trials=1, tol=1e-4):
        nx_list = np.geomspace(nx_min, nx_max, num=n_sample)
        timing_dict = dict(nx_list=nx_list, osqp_mean=[], osqp_std=[],
                           reluqpth_mean=[], reluqpth_std=[],
                           proxqp_mean=[], proxqp_std=[])
        
        # make sure reluqp is compiled
        for _ in range(10):
            _, _ = self.reluqpth_solve()

        for nx in nx_list:
            reluqpth_times = []
            osqp_times = []
            proxqp_times = [] 
            print("nx: ", int(nx))

            for seed in tqdm.tqdm(range(n_seeds)):
                reluqpth_solve_time, reluqpth_sol = self.reluqpth_solve(nx=int(nx), n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol)
                osqp_solve_time, osqp_sol = self.osqp_solve(nx=int(nx), n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol)
                proxqp_solve_time, proxqp_sol = self.proxqp_solve(nx=int(nx), n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol)

                assert np.linalg.norm(reluqpth_sol.cpu().detach().numpy() - osqp_sol, ord=np.inf) < tol

                reluqpth_times.append(reluqpth_solve_time)
                osqp_times.append(osqp_solve_time)
                proxqp_times.append(proxqp_solve_time)

            timing_dict["osqp_mean"].append(np.mean(osqp_times))
            timing_dict["osqp_std"].append(np.std(osqp_times))
            timing_dict["reluqpth_mean"].append(np.mean(reluqpth_times))
            timing_dict["reluqpth_std"].append(np.std(reluqpth_times))
            timing_dict["proxqp_mean"].append(np.mean(proxqp_times))
            timing_dict["proxqp_std"].append(np.std(proxqp_times))

        self.plot_timing_results(timing_dict)
    
    def plot_timing_results(self, timing_dict):
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.set_xscale("log")
        # ax.plot(timing_dict['nx_list'], timing_dict['reluqpth_mean'], label='reluqpth')
        # ax.plot(timing_dict['nx_list'], timing_dict['osqp_mean'], label='osqp')
        ax.errorbar(timing_dict['nx_list'], timing_dict['reluqpth_mean'], yerr=timing_dict['reluqpth_std'], label='reluqpth')
        ax.errorbar(timing_dict['nx_list'], timing_dict['osqp_mean'], yerr=timing_dict['osqp_std'], label='osqp')
        ax.errorbar(timing_dict['nx_list'], timing_dict['proxqp_mean'], yerr=timing_dict['proxqp_std'], label='proxqp')
        ax.set_xlabel('problem size')
        ax.set_ylabel('solve time (s)')
        ax.legend()
        # plt.show()
        plt.savefig("results/random_qp_benchmark.png")
        # plt.plot(timing_dict['nx_list'], timing_dict['reluqpth'], label='reluqpth')
        # plt.plot(timing_dict['nx_list'], timing_dict['osqp'], label='osqp')
        # plt.xlabel('problem index')
        # plt.ylabel('solve time (s)')
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    benchmark = Random_QP_benchmark()

    benchmark.random_initial_solve(nx_min=10, nx_max=500, n_sample=10, n_seeds=5, tol=1e-6)

