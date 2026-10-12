from dataclasses import dataclass

@dataclass(frozen=True)
class AscanConfig:
    """
    Configuration parameters for Ultrasonic A-Scan processing & MBE framework.
    Holds hardware specifications, sound speed in steel, and filter settings.
    """
    # Acoustic & Hardware Constants
    SAMPLING_RATE_MHZ: float = 100.0   # 100 MHz sampling frequency (10ns interval)
    SOUND_VELOCITY_M_S: float = 5900.0 # Longitudinal sound speed in steel (m/s)

    # Butterworth Bandpass Filter Limits
    LOWCUT_MHZ: float = 0.5            # Lower cutoff frequency (MHz)
    HIGHCUT_MHZ: float = 5.0           # Upper cutoff frequency (MHz)
    FILTER_ORDER: int = 4              # Filter order

    # Time Domain Settings
    TIME_WINDOW_US: float = 8.0        # Default inspection time window (microseconds)

    @property
    def sampling_interval_us(self) -> float:
        """Calculates sampling interval (dt) in microseconds."""
        return 1.0 / self.SAMPLING_RATE_MHZ
