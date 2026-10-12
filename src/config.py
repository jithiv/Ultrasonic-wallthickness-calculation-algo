from dataclasses import dataclass

@dataclass
class AscanConfig:
    """
    Central configuration class for Ultrasonic A-scan signal processing.
    """
    SAMPLING_RATE_MHZ: float = 100.0   # Sampling rate in MHz (100 MHz = 10 ns interval)
    SOUND_VELOCITY_M_S: float = 5900.0  # Longitudinal sound velocity in steel (m/s)
    CENTER_FREQ_MHZ: float = 2.25      # Center frequency of ultrasonic transducer (MHz)
    TIME_WINDOW_US: float = 8.0        # Time window duration in microseconds
    LOWCUT_MHZ: float = 0.5            # Butterworth bandpass lower cutoff (MHz)
    HIGHCUT_MHZ: float = 5.0           # Butterworth bandpass upper cutoff (MHz)
    FILTER_ORDER: int = 4              # Bandpass filter order

    @property
    def sampling_interval_us(self) -> float:
        """Returns time step per sample in microseconds."""
        return 1.0 / self.SAMPLING_RATE_MHZ
