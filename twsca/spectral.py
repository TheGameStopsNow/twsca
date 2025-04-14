"""
Spectral Analysis Module

This module provides functionality for analyzing time series in the frequency domain
and computing spectral correlations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.stats import pearsonr


def compute_spectrum(
    series: Union[np.ndarray, List[float]],
    padding: bool = True,
    sampling_rate: float = 100.0,  # Default sampling rate for 100 points in 1 second
    detrend: bool = True,
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
    detrend : bool
        Whether to remove linear trend before spectral analysis

    Returns:
    --------
    frequencies : np.ndarray
        Array of frequency values in Hz
    spectrum : np.ndarray
        Complex Fourier coefficients
    """
    # Convert to numpy array
    data = np.array(series, dtype=float)

    # Remove NaN values
    data = np.nan_to_num(data, nan=np.nanmean(data))

    # Apply detrending to remove linear trends that may affect spectral analysis
    if detrend:
        data = signal.detrend(data)

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


def spectral_correlation(
    spec1: np.ndarray, spec2: np.ndarray, method: str = "pearson"
) -> float:
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
        return float(np.sum(mag1_norm * mag2_norm))  # Convert to float

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
        raise ValueError(
            f"Unknown method: {method}. Use 'pearson', 'magnitude', or 'coherence'."
        )


def compute_wavelet_coherence(
    series1: Union[np.ndarray, List[float]],
    series2: Union[np.ndarray, List[float]],
    scales: Optional[np.ndarray] = None,
    wavelet: str = "morlet",
    sampling_period: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wavelet coherence between two time series.

    Parameters:
    -----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    scales : array-like, optional
        Scales for wavelet transform. If None, automatically determined
    wavelet : str
        Wavelet function to use
    sampling_period : float
        Time between samples

    Returns:
    --------
    wct : np.ndarray
        Wavelet coherence (2D array, time x scale)
    scales : np.ndarray
        Scales used for wavelet transform
    freqs : np.ndarray
        Frequencies corresponding to scales
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "PyWavelets package is required for wavelet coherence analysis"
        )

    # Convert to numpy arrays
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)

    # Remove NaN values
    s1 = np.nan_to_num(s1, nan=np.nanmean(s1))
    s2 = np.nan_to_num(s2, nan=np.nanmean(s2))

    # Detrend the data
    s1 = signal.detrend(s1)
    s2 = signal.detrend(s2)

    # Normalize the data
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)

    # Determine scales if not provided
    if scales is None:
        # Use dyadic scales covering from 2 samples to 1/4 of the time series length
        min_scale = 2
        max_scale = len(s1) // 4
        n_scales = int(np.log2(max_scale / min_scale)) * 4 + 1  # 4 voices per octave
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)

    # Compute continuous wavelet transform for both series
    coef1, freqs = pywt.cwt(s1, scales, wavelet, sampling_period)
    coef2, _ = pywt.cwt(s2, scales, wavelet, sampling_period)

    # Compute cross-wavelet transform
    xwt = coef1 * np.conj(coef2)

    # Compute auto-spectra
    power1 = np.abs(coef1) ** 2
    power2 = np.abs(coef2) ** 2

    # Smooth the spectra (simple moving average)
    def smooth(x, window_len=5):
        w = np.ones(window_len, "d") / window_len
        # Smooth along the time axis
        return np.apply_along_axis(
            lambda m: np.convolve(m, w, mode="same"), axis=1, arr=x
        )

    power1_smooth = smooth(power1)
    power2_smooth = smooth(power2)
    xwt_smooth = smooth(xwt)

    # Compute wavelet coherence
    wct = np.abs(xwt_smooth) ** 2 / (power1_smooth * power2_smooth)

    # Ensure values are in [0, 1]
    wct = np.clip(wct, 0, 1)

    return wct, scales, freqs


def validate_spectral_integrity(
    original_series: Union[np.ndarray, List[float]],
    warped_series: Union[np.ndarray, List[float]],
    threshold: float = 0.7,
) -> Tuple[bool, float]:
    """
    Validate that warping has not introduced spectral artifacts.

    Parameters:
    -----------
    original_series : array-like
        Original time series before warping
    warped_series : array-like
        Time series after warping/alignment
    threshold : float
        Minimum similarity threshold (0-1)

    Returns:
    --------
    is_valid : bool
        True if spectral integrity is maintained
    similarity : float
        Measure of spectral similarity (0-1)
    """
    # Compute spectra
    _, spec1 = compute_spectrum(original_series)
    _, spec2 = compute_spectrum(warped_series)

    # Compute spectral correlation
    similarity = spectral_correlation(spec1, spec2, method="magnitude")

    # Check if similarity exceeds threshold
    return similarity >= threshold, similarity
