"""
benchmarking reluqp single vs double precision 
"""
import reluqp.reluqpth as reluqp
import reluqp.utils as utils
import torch 
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def solve(nx, n_eq, n_ineq, seed=1, tol=1e-2, precision=torch.float64):
    H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=seed, compute_sol=False)
    model = reluqp.ReLU_QP()
    model.setup(H=H, g=g, A=A, l=l, u=u, eps_abs=tol, precision=precision, verbose=False)
    results = model.solve()
    # assert results.info.status == 'solved'
    # Warning(results.info.status)
    if results.info.status != 'solved':
        print(results.info.status)
    return results.info.solve_time, results.x

def benchmark(nx_min=10, nx_max=1000, n_sample=10, n_seeds=5, tol=1e-2):

    nx_list = np.geomspace(nx_min, nx_max, num=n_sample)
    timing_dict = dict(nx_list=nx_list, double_mean=[], double_std=[],
                       single_mean=[], single_std=[])

    # make sure reluqp is compiled
    for _ in range(10):
        _, _ = solve(nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-2, precision=torch.float64)
        _, _ = solve(nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-2, precision=torch.float32)

    for nx in nx_list:
        nx = int(nx)
        print("nx: ", nx)
        double_times = []
        single_times = []
        for seed in tqdm.tqdm(range(n_seeds)):
            double_time, _ = solve(nx=nx, n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol, precision=torch.float64)
            single_time, _ = solve(nx=nx, n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol, precision=torch.float32)
            double_times.append(double_time)
            single_times.append(single_time)
        timing_dict['double_mean'].append(np.mean(double_times))
        timing_dict['double_std'].append(np.std(double_times))
        timing_dict['single_mean'].append(np.mean(single_times))
        timing_dict['single_std'].append(np.std(single_times))

    return timing_dict

def plot_timing(timing_dict):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.errorbar(timing_dict['nx_list'], timing_dict['double_mean'], yerr=timing_dict['double_std'], label='double')
    ax.errorbar(timing_dict['nx_list'], timing_dict['single_mean'], yerr=timing_dict['single_std'], label='single')
    ax.set_xlabel('problem size')
    ax.set_ylabel('solve time (s)')
    ax.set_title('reluqp single vs double precision, tol=1e-2')
    ax.legend()
    # plt.show()
    plt.savefig("results/single_vs_double_precision.png")
    
if __name__ == '__main__':

    timing_dict = benchmark(nx_min=10, nx_max=2000, n_sample=10, n_seeds=5, tol=1e-2)

    plot_timing(timing_dict)