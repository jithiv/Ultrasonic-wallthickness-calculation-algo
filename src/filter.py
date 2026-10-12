import numpy as np
from scipy.signal import butter, filtfilt
from src.config import AscanConfig

class AscanFilter:
    """
    Filtering engine for ultrasonic A-scan signals.
    Provides zero-phase noise reduction using Butterworth bandpass topology.
    """
    def __init__(self, config: AscanConfig = AscanConfig()):
        self.config = config

    def apply_bandpass(
        self, 
        signal: np.ndarray, 
        lowcut_mhz: float = None, 
        highcut_mhz: float = None, 
        order: int = None
    ) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter to suppress out-of-band noise.
        
        Parameters:
            signal (np.ndarray): Raw ultrasonic time-domain signal.
            lowcut_mhz (float): Lower cutoff frequency in MHz.
            highcut_mhz (float): Upper cutoff frequency in MHz.
            order (int): Filter order.
            
        Returns:
            np.ndarray: Cleaned signal with preserved phase/timing.
        """
        low = lowcut_mhz if lowcut_mhz is not None else self.config.LOWCUT_MHZ
        high = highcut_mhz if highcut_mhz is not None else self.config.HIGHCUT_MHZ
        filt_order = order if order is not None else self.config.FILTER_ORDER
        
        # Calculate Nyquist frequency (half of sampling frequency)
        nyquist = 0.5 * self.config.SAMPLING_RATE_MHZ
        
        # Normalized cutoff frequencies
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Design Butterworth bandpass filter coefficients
        b, a = butter(filt_order, [low_norm, high_norm], btype='band')
        
        # Zero-phase filtering to prevent time-of-flight (TOF) shifting
        cleaned_signal = filtfilt(b, a, signal)
        return cleaned_signal
