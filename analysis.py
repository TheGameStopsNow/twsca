"""
Time-Warped Spectral Correlation Analysis (TWSCA) Module

This module provides the high-level API for TWSCA, combining the Dynamic Time Warping
and Spectral Analysis modules to detect correlations between time-warped series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.stats import pearsonr

from .dtw import dtw_distance, align_series
from .spectral import compute_spectrum, spectral_correlation


def compute_twsca(
    series1: Union[np.ndarray, List[float], pd.Series],
    series2: Union[np.ndarray, List[float], pd.Series],
    window: Optional[int] = None,
    detrend: bool = True,
    spectral_method: str = "pearson",
    padding: bool = True,
) -> Dict[str, Any]:
    """Compute Time-Warped Spectral Correlation Analysis between two time series.

    Parameters
    ----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    window : int, optional
        Window constraint for DTW. If None, full DTW is performed
    detrend : bool
        Whether to apply detrending to the data before analysis
    spectral_method : str
        Method for computing spectral correlation ('pearson', 'magnitude', 'coherence')
    padding : bool
        Whether to apply zero-padding to improve frequency resolution

    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'dtw_distance': float - DTW distance between series
        - 'aligned_series1': np.ndarray - Aligned version of series1
        - 'aligned_series2': np.ndarray - Aligned version of series2
        - 'time_domain_correlation': float - Correlation in time domain after alignment
        - 'spectral_correlation': float - Correlation in frequency domain after alignment
    """
    # Convert inputs to numpy arrays
    s1 = _prepare_series(series1, detrend)
    s2 = _prepare_series(series2, detrend)

    # Compute DTW distance and path
    dist, path = dtw_distance(s1, s2, window=window)

    # Align series based on DTW path
    aligned_s1, aligned_s2 = align_series(s1, s2, path)

    # Compute time domain correlation
    if np.std(aligned_s1) == 0 or np.std(aligned_s2) == 0:
        time_corr = 0.0
    else:
        time_corr, _ = pearsonr(aligned_s1, aligned_s2)

    # Compute spectra of aligned series
    _, spectrum1 = compute_spectrum(aligned_s1, padding=padding)
    _, spectrum2 = compute_spectrum(aligned_s2, padding=padding)

    # Compute spectral correlation
    spec_corr = spectral_correlation(spectrum1, spectrum2, method=spectral_method)

    # Return results
    return {
        "dtw_distance": dist,
        "aligned_series1": aligned_s1,
        "aligned_series2": aligned_s2,
        "time_domain_correlation": time_corr,
        "spectral_correlation": spec_corr,
    }


def _prepare_series(
    series: Union[np.ndarray, List[float], pd.Series], detrend: bool = True
) -> np.ndarray:
    """Prepare a time series for analysis by converting to numpy array and optionally detrending.

    Parameters
    ----------
    series : array-like
        Time series data
    detrend : bool
        Whether to apply detrending

    Returns
    -------
    prepared_series : np.ndarray
        Prepared time series
    """
    # Convert to numpy array
    if isinstance(series, pd.Series):
        data = series.values
    else:
        data = np.array(series, dtype=float)

    # Apply detrending if requested
    if detrend and len(data) > 2:
        # Simple linear detrending
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)
        trend = slope * x + intercept
        detrended = data - trend
        return detrended

    return data


def compute_twsca_matrix(x, fs=1.0):
    """Compute TWSCA similarity matrix.

    Parameters
    ----------
    x : array_like
        Input time series data
    fs : float, optional
        Sampling frequency, by default 1.0

    Returns
    -------
    ndarray
        TWSCA similarity matrix
    """
    # ... existing code ...
