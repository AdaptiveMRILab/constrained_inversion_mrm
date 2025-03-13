import numpy as np 
import torch 

from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from utils import *

def objective_or_jacobian(
    x_n: np.ndarray,
    subspace: np.ndarray,
    freq: np.ndarray,
    freq_mask: np.ndarray,
    b1: np.ndarray,
    mz_target: np.ndarray,
    dt: float,
    return_jacobian: bool=False,
    device=torch.device('cpu')
):
    
    # convert current guess to complex torch tensor 
    rank = subspace.shape[1]
    x = torch.tensor(x_n, dtype=torch.float64, device=device, requires_grad=return_jacobian)
    x_cplx = torch.complex(x[:rank], x[rank:])

    # convert other arrays to tensors 
    A = torch.tensor(subspace, dtype=torch.complex128, device=device) 
    freq = torch.tensor(freq, dtype=torch.float64, device=device) 
    b1 = torch.tensor(b1, dtype=torch.float64, device=device)
    target = torch.tensor(mz_target, dtype=torch.float64, device=device)
    mask = torch.tensor(freq_mask, dtype=torch.float64, device=device)[:,None]

    # simulate longitudinal magnetization using custom C++ libtorch function
    rf = torch.matmul(A, x_cplx)
    mz_sim = simulate_longitudinal_magnetization_libtorch(rf, freq, b1, dt)

    # use the sum of squared error as loss function 
    loss_fn = torch.nn.MSELoss(reduction='sum') 
    loss = loss_fn((mz_sim*mask).unsqueeze(0), (target*mask).unsqueeze(0))

    # we will either return the scalar objective or the jacobian
    if return_jacobian:
        loss.backward()
        output = x.grad.detach().cpu().numpy()
    else:
        output = loss.item()

    del x
    return output


def optimize_inversion_pulse_autograd(
    rf_init: np.ndarray,
    freq_arr: np.ndarray,
    freq_mask: np.ndarray,
    subspace: np.ndarray,
    mz_target: np.ndarray,
    duration: float,
    max_rf_energy_hz_sqrd_sec: float,
    max_nominal_b1_hz: float,
    b1_scale_arr: np.ndarray=np.ones((1,), dtype=np.float64),
    energy_tol: float=1.0,
    min_nominal_b1_hz: float=0.0,
    sqp_max_iters: int=100,
    sqp_tol: float=1e-6,
    device=torch.device('cpu')
):
    
    # get duration of each sample in pulse
    dt = duration / rf_init.size 

    # convert subspace coefficients to complex-valued RF pulse
    get_rf_wav = lambda x: subspace @ (x[:len(x)//2] + 1j*x[len(x)//2:])

    # function to get the amplitude of an RF pulse (for hardware B1 constraint)
    get_rf_ampl = lambda x: np.abs(get_rf_wav(x))

    # function to calculate the power of an RF pulse (integral of B1 squared)
    get_rf_power = lambda x: np.sum( np.real(get_rf_wav(x) * np.conj(get_rf_wav(x))) * dt)

    # objective function and its jacobian
    obj = lambda x: objective_or_jacobian(x, subspace, freq_arr, freq_mask, b1_scale_arr, mz_target, dt, return_jacobian=False, device=device)
    jac = lambda x: objective_or_jacobian(x, subspace, freq_arr, freq_mask, b1_scale_arr, mz_target, dt, return_jacobian=True,  device=device)

    # set up the constraints 
    constraints = [] 

    # peak B1 constraint
    if max_nominal_b1_hz is not None:
        ampl_constraint = NonlinearConstraint(get_rf_ampl, lb=min_nominal_b1_hz, ub=max_nominal_b1_hz)
        constraints += [ampl_constraint]

    # energy constraint
    if max_rf_energy_hz_sqrd_sec is not None:
        power_constraint = NonlinearConstraint(get_rf_power, lb=(1-energy_tol)*max_rf_energy_hz_sqrd_sec, ub=max_rf_energy_hz_sqrd_sec)
        constraints += [power_constraint]

    # get the initial guess based on least squares fit
    x0 = np.linalg.lstsq(subspace, rf_init, rcond=None)[0]
    x0 = np.concatenate([x0.real, x0.imag], axis=0)

    # run it!
    options = {'maxiter': sqp_max_iters, 'ftol':sqp_tol}
    result = minimize(obj, x0, method='slsqp', constraints=constraints, jac=jac, options=options)
    print(result)
    x_opt = result.x 
    rf_opt = get_rf_wav(x_opt)

    # get the final magnetization
    mz_final = simulate_mz(rf_opt, freq_arr, b1_scale_arr, dt, device=device, return_numpy=True)
    
    return rf_opt, mz_final, x_opt


