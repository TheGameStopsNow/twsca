"""
Spectral Analysis Module

This module provides functionality for analyzing time series in the frequency domain
and computing spectral correlations.
"""

import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any
from scipy.stats import pearsonr


def compute_spectrum(
    series: Union[np.ndarray, List[float]],
    padding: bool = True,
    sampling_rate: float = 100.0,  # Default sampling rate for 100 points in 1 second
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency spectrum of a time series.

    Parameters:
    -----------
    series : array-like
        The time series data
    padding : bool
        Whether to apply zero-padding to improve frequency resolution
    sampling_rate : float
        The sampling rate of the time series in Hz

    Returns:
    --------
    frequencies : np.ndarray
        Array of frequency values in Hz
    spectrum : np.ndarray
        Complex Fourier coefficients
    """
    # Convert to numpy array
    data = np.array(series, dtype=float)

    # Apply zero padding if requested
    if padding:
        # Pad to next power of 2 for efficient FFT
        n = len(data)
        padded_length = 2 ** int(np.ceil(np.log2(n)))
        data = np.pad(data, (0, padded_length - n), "constant")

    # Apply windowing to reduce spectral leakage
    window = np.hanning(len(data))
    windowed_data = data * window

    # Compute FFT
    fft = np.fft.fft(windowed_data)

    # Get frequencies scaled by sampling rate
    freqs = np.fft.fftfreq(len(data)) * sampling_rate

    # Return frequencies and spectrum (only positive frequencies)
    n_positive = len(freqs) // 2
    return freqs[:n_positive], fft[:n_positive]


def spectral_correlation(spec1: np.ndarray, spec2: np.ndarray, method: str = "pearson") -> float:
    """
    Compute the correlation between two spectra.

    Parameters:
    -----------
    spec1 : np.ndarray
        First spectrum (complex Fourier coefficients)
    spec2 : np.ndarray
        Second spectrum (complex Fourier coefficients)
    method : str
        Method for computing correlation ('pearson', 'magnitude', 'coherence')

    Returns:
    --------
    correlation : float
        Correlation coefficient between the two spectra
    """
    # Convert to numpy array if not already
    s1 = np.array(spec1)
    s2 = np.array(spec2)

    # Ensure both spectra are of the same length
    min_len = min(len(s1), len(s2))
    s1 = s1[:min_len]
    s2 = s2[:min_len]

    if method == "pearson":
        # Compute correlation between magnitude spectra
        mag1 = np.abs(s1)
        mag2 = np.abs(s2)

        # If either spectrum is constant, return 0 correlation
        if np.std(mag1) == 0 or np.std(mag2) == 0:
            return 0.0

        # Calculate Pearson correlation
        corr, _ = pearsonr(mag1, mag2)
        return corr

    elif method == "magnitude":
        # Compute normalized dot product of magnitude spectra
        mag1 = np.abs(s1)
        mag2 = np.abs(s2)

        # Normalize each vector to unit length
        mag1_norm = mag1 / (
            np.linalg.norm(mag1) + 1e-10
        )  # Add small epsilon to avoid division by zero
        mag2_norm = mag2 / (np.linalg.norm(mag2) + 1e-10)

        # Dot product of normalized vectors
        return float(np.sum(mag1_norm * mag2_norm))  # Convert to float to avoid numpy type issues

    elif method == "coherence":
        # Compute magnitude-squared coherence
        mag_s1s2 = np.abs(s1 * np.conj(s2)) ** 2
        mag_s1 = np.abs(s1) ** 2
        mag_s2 = np.abs(s2) ** 2

        # Avoid division by zero
        denom = mag_s1 * mag_s2
        mask = denom > 0

        if not np.any(mask):
            return 0.0

        # Compute coherence
        coherence = np.zeros_like(mag_s1s2, dtype=float)
        coherence[mask] = mag_s1s2[mask] / denom[mask]

        # Return mean coherence
        return np.mean(coherence)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson', 'magnitude', or 'coherence'.")
