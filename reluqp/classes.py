import torch
import numpy as np

class QP(object):
    def __init__(self, H: torch.tensor or np.ndarray, 
                        g: torch.tensor or np.ndarray, 
                        A: torch.tensor or np.ndarray, 
                        l: torch.tensor or np.ndarray, u: torch.tensor or np.ndarray, 
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), precision=torch.double):
        
        # convert to torch tensors if it's numpy array
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)
        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g)
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        if isinstance(l, np.ndarray):
            l = torch.from_numpy(l)
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        
        self.H = H.to(device=device, dtype=precision).contiguous()
        self.g = g.to(device=device, dtype=precision).contiguous()
        self.A = A.to(device=device, dtype=precision).contiguous()
        self.l = l.to(device=device, dtype=precision).contiguous()
        self.u = u.to(device=device, dtype=precision).contiguous()

        self.nx = H.shape[0] # number of decision variables
        self.nc = A.shape[0] # number of constraints
    
class Settings(object):
    def __init__(self, verbose=False,
                        warm_starting=True,
                        scaling=False, #todo: implement scaling
                        rho=0.1,
                        rho_min=1e-6,
                        rho_max=1e6,
                        sigma=1e-6,
                        adaptive_rho=True,
                        adaptive_rho_interval=1,
                        adaptive_rho_tolerance=5,
                        max_iter=4000,
                        eps_abs=1e-3,
                        eq_tol=1e-6,
                        check_interval=25,
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        precision= torch.float64):
        
        self.verbose = verbose
        self.warm_starting = warm_starting
        self.scaling = scaling
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.sigma = sigma
        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_interval = adaptive_rho_interval
        self.adaptive_rho_tolerance = adaptive_rho_tolerance
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eq_tol = eq_tol
        self.check_interval = check_interval
        self.device = device
        self.precision = precision

class Info(object):
    def __init__(self, iter=None, 
                        status=None, 
                        obj_val=None,
                        pri_res=None,
                        dua_res=None,
                        setup_time=0,
                        solve_time=0,
                        update_time=0,
                        run_time=0,
                        rho_estimate=None,
                 ):
        self.iter = iter
        self.status = status
        self.obj_val = obj_val
        self.pri_res = pri_res
        self.dua_res = dua_res
        self.setup_time = setup_time
        self.solve_time = solve_time
        self.update_time = update_time
        self.run_time = run_time
        self.rho_estimate = rho_estimate


class Results(object):
    def __init__(self, x=None, z=None, info: Info=None):
        self.x = x
        self.z = z
        self.info = info

