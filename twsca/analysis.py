"""
Core analysis functions for the TWSCA package
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

from .dtw import align_series, dtw_distance
from .spectral import compute_spectrum, spectral_correlation
from .smoothing import llt_filter
from .utils import validate_time_series, normalize_series


def compute_twsca(
    s1: Union[np.ndarray, pd.Series],
    s2: Union[np.ndarray, pd.Series],
    *,
    window_size: Optional[int] = None,
    dtw_radius: Optional[int] = None,
    normalize: bool = True,
    detrend: bool = True,
    spectral_method: str = "magnitude",
    window: Optional[int] = None,
    optimize_params: bool = False,
    use_llt: bool = True,  # New parameter to control LLT filtering
    llt_sigma: float = 1.0,  # LLT filter sigma parameter
    llt_alpha: float = 0.5,  # LLT filter alpha parameter
) -> Dict[str, Any]:
    """
    Compute Time-Warped Spectral Correlation Analysis between two time series.

    Parameters
    ----------
    s1 : array-like
        First time series
    s2 : array-like
        Second time series
    window_size : int, optional
        Size of the sliding window for spectral analysis
    dtw_radius : int, optional
        Radius for DTW computation
    normalize : bool, default=True
        Whether to normalize the input series
    detrend : bool, default=True
        Whether to remove linear trend from the input series
    spectral_method : str, default="magnitude"
        Method for spectral correlation computation ("magnitude" or "coherence")
    window : int, optional
        Window constraint for DTW
    optimize_params : bool, default=False
        Whether to optimize parameters automatically
    use_llt : bool, default=True
        Whether to apply LLT (Local Laplacian Transform) filtering to smooth the signals.
        LLT filtering is applied by default for better noise handling.
    llt_sigma : float, default=1.0
        Standard deviation parameter for LLT filter (only used if use_llt=True)
    llt_alpha : float, default=0.5
        Smoothing parameter for LLT filter (only used if use_llt=True)

    Returns
    -------
    dict
        Dictionary containing analysis results including:
        - time_domain_correlation: Correlation in time domain after DTW
        - spectral_correlation: Correlation in frequency domain
        - dtw_path: Warping path from DTW
        - dtw_distance: DTW distance between series
        - aligned_series1: First series after alignment
        - aligned_series2: Second series after alignment
        - spectral_components: Spectral components of the analysis
    """
    # Validate and convert input series
    s1 = validate_time_series(s1)
    s2 = validate_time_series(s2)

    # Apply LLT filtering if enabled
    if use_llt:
        s1 = llt_filter(s1, sigma=llt_sigma, alpha=llt_alpha)
        s2 = llt_filter(s2, sigma=llt_sigma, alpha=llt_alpha)

    # Normalize if requested
    if normalize:
        s1 = normalize_series(s1)
        s2 = normalize_series(s2)

    # Compute DTW to align the signals
    dtw_dist, path = dtw_distance(s1, s2, window=window)
    
    # Align the signals using the DTW path
    aligned_s1, aligned_s2 = align_series(s1, s2, path)
    
    # Compute time-domain correlation after alignment
    time_corr = np.corrcoef(aligned_s1, aligned_s2)[0, 1]
    
    # Compute spectral correlation
    spec_corr = spectral_correlation(aligned_s1, aligned_s2, window_size=window_size, method=spectral_method)
    
    # Compute spectrum for visualization (can be used by plotting functions)
    spec1 = compute_spectrum(aligned_s1, window_size=window_size)
    spec2 = compute_spectrum(aligned_s2, window_size=window_size)
    
    # Return results
    return {
        "time_domain_correlation": time_corr,
        "spectral_correlation": spec_corr,
        "dtw_path": path,
        "dtw_distance": dtw_dist,
        "aligned_series1": aligned_s1,
        "aligned_series2": aligned_s2,
        "spectral_components": {
            "frequencies": spec1["frequencies"],
            "spectrum1": spec1["magnitude"],
            "spectrum2": spec2["magnitude"],
        }
    }


def compute_twsca_matrix(
    data: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Compute TWSCA correlation matrix for multiple time series.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing multiple time series as columns
    **kwargs
        Additional arguments to pass to compute_twsca

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    n_series = len(data.columns)
    corr_matrix = np.zeros((n_series, n_series))
    
    # Compute correlations between all pairs
    for i in range(n_series):
        for j in range(i, n_series):
            if i == j:
                # Self-correlation is 1.0
                corr_matrix[i, j] = 1.0
            else:
                # Get series
                s1 = data.iloc[:, i].values
                s2 = data.iloc[:, j].values
                
                # Compute TWSCA
                try:
                    result = compute_twsca(s1, s2, **kwargs)
                    corr = result["spectral_correlation"]
                except Exception:
                    # In case of computation errors, set correlation to 0
                    corr = 0.0
                
                # Store correlation (symmetric matrix)
                corr_matrix[i, j] = corr_matrix[j, i] = corr
    
    # Convert to DataFrame with labels
    corr_df = pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
    
    return corr_df 