"""
Spectral analysis functions for TWSCA package.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_spectrum(
    signal: np.ndarray,
    window_size: Optional[int] = None,
    sampling_rate: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute the frequency spectrum of a time series.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    window_size : int, optional
        Size of the sliding window for spectral analysis
    sampling_rate : float, default=1.0
        Sampling rate of the signal
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys 'frequencies' and 'magnitude'
    """
    # Ensure signal is numpy array
    signal = np.array(signal)
    
    # Set default window size if not provided
    if window_size is None:
        window_size = len(signal)
    
    # Apply Hann window
    window = np.hanning(window_size)
    
    # Zero-pad signal if needed
    if len(signal) < window_size:
        padded_signal = np.zeros(window_size)
        padded_signal[:len(signal)] = signal
        signal = padded_signal
    
    # Use middle portion of the signal if it's longer than window_size
    if len(signal) > window_size:
        start = len(signal) // 2 - window_size // 2
        signal = signal[start:start + window_size]
    
    # Apply window
    windowed_signal = signal * window
    
    # Compute FFT
    fft_result = np.fft.rfft(windowed_signal)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    
    # Compute frequency axis
    frequencies = np.fft.rfftfreq(window_size, d=1.0/sampling_rate)
    
    return {
        'frequencies': frequencies,
        'magnitude': magnitude,
        'phase': np.angle(fft_result)
    }


def spectral_correlation(
    s1: np.ndarray,
    s2: np.ndarray,
    window_size: Optional[int] = None,
    method: str = "magnitude"
) -> float:
    """
    Compute spectral correlation between two time series.
    
    Parameters
    ----------
    s1 : np.ndarray
        First time series
    s2 : np.ndarray
        Second time series
    window_size : int, optional
        Size of the sliding window for spectral analysis
    method : str, default="magnitude"
        Method for correlation computation:
        - "magnitude": correlation between magnitude spectra
        - "coherence": magnitude squared coherence
        
    Returns
    -------
    float
        Spectral correlation value
    """
    # Ensure signals are numpy arrays
    s1 = np.array(s1)
    s2 = np.array(s2)
    
    # Set default window size if not provided
    if window_size is None:
        window_size = min(len(s1), len(s2))
    
    # Compute spectra
    spec1 = compute_spectrum(s1, window_size)
    spec2 = compute_spectrum(s2, window_size)
    
    if method == "magnitude":
        # Compute correlation between magnitude spectra
        mag1 = spec1['magnitude']
        mag2 = spec2['magnitude']
        
        # Ensure both spectra have the same length
        min_len = min(len(mag1), len(mag2))
        mag1 = mag1[:min_len]
        mag2 = mag2[:min_len]
        
        # Compute correlation
        if np.std(mag1) == 0 or np.std(mag2) == 0:
            # Handle constant spectra case
            if np.allclose(mag1, mag2):
                return 1.0
            else:
                return 0.0
        else:
            return np.corrcoef(mag1, mag2)[0, 1]
    
    elif method == "coherence":
        # Compute magnitude squared coherence
        fft1 = np.fft.rfft(s1[:window_size])
        fft2 = np.fft.rfft(s2[:window_size])
        
        # Cross-spectrum
        cross_spectrum = fft1 * np.conj(fft2)
        
        # Auto-spectra
        auto_spectrum1 = fft1 * np.conj(fft1)
        auto_spectrum2 = fft2 * np.conj(fft2)
        
        # Magnitude squared coherence
        coherence = np.abs(cross_spectrum)**2 / (auto_spectrum1 * auto_spectrum2)
        
        # Average coherence across frequencies
        return np.mean(coherence[1:])  # Skip DC component
    
    else:
        raise ValueError(f"Unknown spectral correlation method: {method}") 