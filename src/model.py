from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from src.config import AscanConfig


@dataclass
class GaussianWaveletParams:
    """
    Dataclass encapsulating parameter bounds for an ultrasonic Gaussian wavelet pulse.
    """
    alpha: float      # Envelope bandwidth factor (1/us^2)
    tau: float        # Time-of-Arrival / Center position (us)
    fc: float         # Center frequency (MHz)
    phi: float        # Phase offset (radians)
    amplitude: float  # Pulse amplitude scaling factor


def compute_gaussian_pulse(
    t_axis: np.ndarray, 
    params: GaussianWaveletParams
) -> np.ndarray:
    """
    Calculates a single Gaussian-modulated ultrasonic echo pulse segment.
    """
    time_delta = t_axis - params.tau
    gaussian_envelope = np.exp(-params.alpha * (time_delta ** 2))
    carrier_wave = np.cos(2.0 * np.pi * params.fc * time_delta + params.phi)
    
    return params.amplitude * gaussian_envelope * carrier_wave


class SyntheticAscanGenerator:
    """
    Industrial-grade synthetic NDT A-scan generator simulating multi-echo 
    propagation with customizable signal-to-noise ratios (SNR).
    """
    def __init__(self, config: Optional[AscanConfig] = None) -> None:
        self.config = config if config is not None else AscanConfig()

    def generate_pipe_ascan(
        self, 
        thickness_mm: float = 10.0, 
        snr_db: float = 25.0, 
        attenuation_factor: float = 0.5,
        random_seed: Optional[int] = 42
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Simulates dual-echo ultrasonic reflection (Front-Wall & Back-Wall) for pipe wall inspection.
        """
        if thickness_mm <= 0:
            raise ValueError("Pipe wall thickness must be a positive non-zero value.")

        if random_seed is not None:
            np.random.seed(random_seed)

        # 1. Establish Time Domain
        time_axis = np.arange(0.0, self.config.TIME_WINDOW_US, self.config.sampling_interval_us)

        # 2. Compute Exact Theoretical TOF (Microseconds)
        true_tof_us = (2.0 * thickness_mm) / self.config.SOUND_VELOCITY_M_S * 1e3

        # 3. Model Front-Wall (FW) Reflection Echo
        fw_pulse_time = 1.0  # Default initial delay at 1.0 us
        fw_params = GaussianWaveletParams(
            alpha=1.6,
            tau=fw_pulse_time,
            fc=self.config.CENTER_FREQ_MHZ,
            phi=0.0,
            amplitude=0.85
        )
        front_wall_echo = compute_gaussian_pulse(time_axis, fw_params)

        # 4. Model Back-Wall (BW) Reflection Echo
        bw_pulse_time = fw_pulse_time + true_tof_us
        bw_params = GaussianWaveletParams(
            alpha=1.3,
            tau=bw_pulse_time,
            fc=self.config.CENTER_FREQ_MHZ,
            phi=np.pi / 3.0,
            amplitude=fw_params.amplitude * attenuation_factor
        )
        back_wall_echo = compute_gaussian_pulse(time_axis, bw_params)

        # 5. Composite Clean Waveform
        clean_ascan = front_wall_echo + back_wall_echo

        # 6. Add White Gaussian Noise
        signal_power = np.mean(clean_ascan ** 2)
        noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
        noise_vector = np.random.normal(0.0, np.sqrt(noise_variance), size=len(time_axis))

        noisy_ascan = clean_ascan + noise_vector

        return time_axis, noisy_ascan, true_tof_us
