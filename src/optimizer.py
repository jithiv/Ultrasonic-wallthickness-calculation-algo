from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
from src.config import AscanConfig
from src.model import GaussianWaveletParams, compute_gaussian_pulse


@dataclass
class OptimizationResult:
    """
    Data structure holding the optimization output parameters and calculated thickness.
    """
    estimated_tof_us: float
    calculated_thickness_mm: float
    thickness_error_mm: float
    opt_fw_params: GaussianWaveletParams
    opt_bw_params: GaussianWaveletParams
    residuals: np.ndarray


class AscanLMOptimizer:
    """
    Model-Based Estimation (MBE) optimization engine using Levenberg-Marquardt
    algorithm for ultra-precise time-of-flight (TOF) extraction.
    """
    def __init__(self, config: Optional[AscanConfig] = None) -> None:
        self.config = config if config is not None else AscanConfig()

    def _dual_echo_residual(
        self, 
        params_vec: np.ndarray, 
        t_axis: np.ndarray, 
        observed_signal: np.ndarray
    ) -> np.ndarray:
        """
        Computes residual difference between observed signal and modeled composite echoes.
        Parameters order in params_vec: [alpha1, tau1, fc1, phi1, amp1, alpha2, tau2, fc2, phi2, amp2]
        """
        fw = GaussianWaveletParams(*params_vec[0:5])
        bw = GaussianWaveletParams(*params_vec[5:10])

        modeled_signal = compute_gaussian_pulse(t_axis, fw) + compute_gaussian_pulse(t_axis, bw)
        return observed_signal - modeled_signal

    def fit_ascan(
        self, 
        t_axis: np.ndarray, 
        noisy_signal: np.ndarray, 
        true_thickness_mm: Optional[float] = None
    ) -> OptimizationResult:
        """
        Performs Levenberg-Marquardt non-linear least squares optimization to fit Gaussian wavelets.
        """
        # 1. Initial Guess Generation based on signal envelope peaks
        fw_tau_guess = 1.0  # Default initial delay around 1 µs
        bw_tau_guess = 4.5  # Estimated secondary delay

        # [alpha, tau, fc, phi, amplitude]
        initial_guess = [
            1.5, fw_tau_guess, self.config.CENTER_FREQ_MHZ, 0.0, 0.8,   # FW Guess
            1.2, bw_tau_guess, self.config.CENTER_FREQ_MHZ, 0.5, 0.4    # BW Guess
        ]

        # 2. Levenberg-Marquardt Optimization Execution
        res = least_squares(
            fun=self._dual_echo_residual,
            x0=initial_guess,
            args=(t_axis, noisy_signal),
            method='lm',
            max_nfev=2000
        )

        p = res.x
        opt_fw = GaussianWaveletParams(alpha=p[0], tau=p[1], fc=p[2], phi=p[3], amplitude=p[4])
        opt_bw = GaussianWaveletParams(alpha=p[5], tau=p[6], fc=p[7], phi=p[8], amplitude=p[9])

        # 3. Time-of-Flight & Thickness Calculation
        estimated_tof_us = abs(opt_bw.tau - opt_fw.tau)
        
        # d = (v * TOF) / 2  (v in mm/us -> v_m_s * 1e-3)
        velocity_mm_us = self.config.SOUND_VELOCITY_M_S * 1e-3
        calculated_thickness_mm = (velocity_mm_us * estimated_tof_us) / 2.0

        thickness_error = 0.0
        if true_thickness_mm is not None: 
            thickness_error = abs(calculated_thickness_mm - true_thickness_mm)

        return OptimizationResult(
            estimated_tof_us=estimated_tof_us,
            calculated_thickness_mm=calculated_thickness_mm,
            thickness_error_mm=thickness_error,
            opt_fw_params=opt_fw,
            opt_bw_params=opt_bw,
            residuals=res.fun
        )
