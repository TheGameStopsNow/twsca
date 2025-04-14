"""
Time-Warped Spectral Correlation Analysis (TWSCA) Module

This module provides the high-level API for TWSCA, combining the Dynamic Time Warping
and Spectral Analysis modules to detect correlations between time-warped series.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Use relative imports for modules within the package
from .dtw import align_series, dtw_distance
from .spectral import compute_spectrum, spectral_correlation


def compute_twsca(
    series1: Union[np.ndarray, List[float], pd.Series],
    series2: Union[np.ndarray, List[float], pd.Series],
    window: Optional[int] = None,
    detrend: bool = True,
    spectral_method: str = "pearson",
    padding: bool = True,
) -> Dict[str, Any]:
    """
    Compute Time-Warped Spectral Correlation Analysis between two time series.

    Parameters:
    -----------
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

    Returns:
    --------
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
    if np.std(aligned_s1) == 0 and np.std(aligned_s2) == 0:
        # Both series are constant, check if they are proportional
        if np.mean(aligned_s1) == 0 and np.mean(aligned_s2) == 0:
            time_corr = 1.0  # Both series are zero
            spec_corr = 1.0  # Spectral correlation for identical zero-mean series
        else:
            time_corr = 1.0  # Both series are constant and proportional
            spec_corr = 1.0  # Spectral correlation for proportional constant series
    elif np.std(aligned_s1) == 0 or np.std(aligned_s2) == 0:
        time_corr = 0.0  # One series is constant, the other is not
        spec_corr = 0.0  # Spectral correlation when one series is constant
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


def compute_twsca_matrix(
    data: pd.DataFrame,
    window: Optional[int] = None,
    detrend: bool = True,
    spectral_method: str = "pearson",
    padding: bool = True,
) -> pd.DataFrame:
    """
    Compute Time-Warped Spectral Correlation matrix for multiple time series.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame where each column is a time series
    window : int, optional
        Window constraint for DTW. If None, full DTW is performed
    detrend : bool
        Whether to apply detrending to the data before analysis
    spectral_method : str
        Method for computing spectral correlation ('pearson', 'magnitude', 'coherence')
    padding : bool
        Whether to apply zero-padding to improve frequency resolution

    Returns:
    --------
    corr_matrix : pd.DataFrame
        Correlation matrix where each cell represents the TWSCA correlation between series
    """
    # Get column names and number of series
    columns = data.columns
    n_series = len(columns)

    # Initialize correlation matrix with zeros
    corr_matrix = np.zeros((n_series, n_series))

    # Fill diagonal with 1.0 (perfect self-correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    # Compute correlation for each pair of series
    for i in range(n_series):
        for j in range(i + 1, n_series):
            # Get the two series
            series1 = data.iloc[:, i]
            series2 = data.iloc[:, j]

            # Skip if either series has constant values
            if series1.std() == 0 or series2.std() == 0:
                correlation = 0.0
            else:
                # Compute TWSCA
                result = compute_twsca(
                    series1,
                    series2,
                    window=window,
                    detrend=detrend,
                    spectral_method=spectral_method,
                    padding=padding,
                )

                # Extract spectral correlation
                correlation = result["spectral_correlation"]

            # Fill the matrix (it's symmetric)
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation

    # Convert to DataFrame with column names
    return pd.DataFrame(corr_matrix, index=columns, columns=columns)


def _prepare_series(
    series: Union[np.ndarray, List[float], pd.Series], detrend: bool = True
) -> np.ndarray:
    """
    Prepare a time series for analysis by converting to numpy array and optionally detrending.

    Parameters:
    -----------
    series : array-like
        Time series data
    detrend : bool
        Whether to apply detrending

    Returns:
    --------
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
