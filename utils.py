import numpy as np 
import torch 
import sigpy.mri.rf as sprf 

from simulate_longitudinal_magnetization_libtorch import simulate_longitudinal_magnetization_libtorch

def simulate_mz(rf, freq, b1, dt, device=None, return_numpy=False):
    ''' wrapper to libtorch function for numpy or torch array inputs '''
    if not isinstance(rf, torch.Tensor):
        rf = torch.tensor(rf, dtype=torch.complex128, device=device)
    if not isinstance(freq, torch.Tensor):
        freq = torch.tensor(freq, dtype=torch.float64, device=device) 
    if not isinstance(b1, torch.Tensor):
        b1 = torch.tensor(b1, dtype=torch.float64, device=device)
    mz = simulate_longitudinal_magnetization_libtorch(rf, freq, b1, dt)
    if return_numpy:
        mz = mz.detach().cpu().numpy()
    return mz

# from sigpy:
def hypsec(n=512, beta=800, mu=4.9, dur=0.012):
    r"""Design a hyperbolic secant adiabatic pulse.

    mu * beta becomes the amplitude of the frequency sweep

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        mu (float): a constant, determines amplitude of frequency sweep.
        dur (float): pulse time (s).

    Returns:
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References:
        Baum, J., Tycko, R. and Pines, A. (1985). 'Broadband and adiabatic
        inversion of a two-level system by phase-modulated pulses'.
        Phys. Rev. A., 32:3435-3447.
    """

    t = np.arange(-n // 2, n // 2) / n * dur

    a = np.cosh(beta * t) ** -1
    om = -mu * beta * np.tanh(beta * t)

    return a, om

def get_hs_pulse(num_samples, beta, mu, dur):
    ''' wrapper to sigpy.rf.adiabatic.hypsec function to return complex wave instead of AM/FM'''
    mag, pha = sprf.adiabatic.hypsec(num_samples, beta, mu, dur)
    pha = np.cumsum(pha * dur / num_samples)
    return mag * np.exp(1j*pha)

def sech_waveform(dur, nsamples, bandwidth, beta):
    ''' another parameterization of the hyperbolic secant pulse '''
    tv = np.linspace(0.0, dur, nsamples, endpoint=False, dtype=np.float64)
    tmp = np.exp(beta * (1.0 - 2.0*tv/dur)) + np.exp(-beta * (1.0 - 2.0*tv/dur))
    mag = 2.0 / tmp 
    pha = -np.pi * bandwidth * 0.5 * dur * (1.0/beta) * np.log(abs(0.5 * tmp))
    rf = mag * np.exp(1j * pha)
    return rf 

def get_hs_basis(num_samples, num_pulses, dur_range, mu_range, beta_range, ranseed=1234):
    np.random.seed(ranseed)
    hs_pulses = np.zeros((num_samples, num_pulses), dtype=np.complex128)
    for n in range(num_pulses):
        dur = (max(dur_range) - min(dur_range))*np.random.rand() + min(dur_range)
        mu = (max(mu_range) - min(mu_range))*np.random.rand() + min(mu_range)
        beta = (max(beta_range) - min(beta_range))*np.random.rand() + min(beta_range)
        hs_pulses[:,n] = get_hs_pulse(num_samples, beta, mu, dur)
    basis,s,_ = np.linalg.svd(hs_pulses, full_matrices=False)
    return basis, s 

def get_adiabatic_threshold(rf, raster_time, init_ampl_uT=4.0, gamma_bar=42.58e06, mz_thresh=-0.99, incr=1.01, device=None):
    ampl = init_ampl_uT * 1e-6 * gamma_bar 
    freq_arr = np.zeros((1,), dtype=np.float64)
    b1_arr = np.ones_like(freq_arr)
    while simulate_mz(rf*ampl, freq_arr, b1_arr, raster_time, device=device, return_numpy=True)[0,0] > mz_thresh:
        ampl *= incr 
    return ampl
        
def scale_adiabatic_inv_to_overdrive_factor(rf, overdrive, raster_time, device=None):
    return overdrive * get_adiabatic_threshold(rf, raster_time, device=device) * rf 


