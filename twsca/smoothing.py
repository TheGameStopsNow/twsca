"""
Smoothing functions for TWSCA package.
"""

import numpy as np
from typing import Optional, List, Union
from scipy import signal


def llt_filter(
    data: np.ndarray,
    sigma: float = 1.0,
    alpha: float = 0.5,
    iterations: int = 3,
    optimize_params: bool = False
) -> np.ndarray:
    """
    Apply Local Laplacian Transform filter to time series data.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    sigma : float, default=1.0
        Standard deviation for Gaussian kernel
    alpha : float, default=0.5
        Smoothing parameter (0 < alpha < 1)
    iterations : int, default=3
        Number of iterations to apply the filter
    optimize_params : bool, default=False
        Whether to automatically optimize filter parameters
    
    Returns
    -------
    np.ndarray
        Smoothed time series
    """
    # Ensure data is numpy array
    data = np.array(data)
    
    # Auto-optimize parameters if requested
    if optimize_params:
        # This is a simple heuristic - in a real implementation, this would
        # use more sophisticated optimization
        data_std = np.std(data)
        sigma = max(0.5, data_std * 0.5)
        alpha = min(0.9, max(0.1, 1.0 - data_std / np.max(np.abs(data))))
    
    # Validate parameters
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    # Gaussian kernel for convolution
    window_size = int(6 * sigma) | 1  # ensure odd size
    x = np.linspace(-3*sigma, 3*sigma, window_size)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / np.sum(kernel)  # normalize
    
    # Apply filter iteratively
    smoothed = data.copy()
    
    for _ in range(iterations):
        # Compute first and second derivatives using convolution
        # Pad the signal to handle boundaries
        padded = np.pad(smoothed, window_size//2, mode='reflect')
        
        # Apply Gaussian convolution
        smoothed_signal = np.convolve(padded, kernel, mode='valid')
        
        # Compute detail layer (difference between original and smoothed)
        detail = data - smoothed_signal
        
        # Apply alpha modulation to detail layer
        attenuated_detail = detail * alpha
        
        # Add back attenuated details to smoothed signal
        smoothed = smoothed_signal + attenuated_detail
    
    return smoothed


def savitzky_golay(
    data: np.ndarray,
    window_size: int = 5,
    poly_order: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to time series data.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    window_size : int, default=5
        Size of the window (must be odd)
    poly_order : int, default=2
        Order of the polynomial fit
    
    Returns
    -------
    np.ndarray
        Smoothed time series
    """
    # Validate parameters
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    if poly_order >= window_size:
        raise ValueError("Polynomial order must be less than window size")
    
    # Apply Savitzky-Golay filter
    return signal.savgol_filter(data, window_size, poly_order)


def adaptive_smoothing(
    data: np.ndarray,
    base_sigma: float = 1.0,
    sensitivity: float = 0.1
) -> np.ndarray:
    """
    Apply adaptive smoothing based on local volatility.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    base_sigma : float, default=1.0
        Base standard deviation for Gaussian kernel
    sensitivity : float, default=0.1
        Sensitivity to local volatility
    
    Returns
    -------
    np.ndarray
        Smoothed time series
    """
    # Ensure data is numpy array
    data = np.array(data)
    
    # Compute local volatility using rolling standard deviation
    window_size = max(3, int(len(data) * 0.05))  # 5% of data length, minimum 3
    
    # Compute rolling standard deviation using convolution
    # Create a normalized box kernel
    box_kernel = np.ones(window_size) / window_size
    
    # Pad the data for convolution
    padded_data = np.pad(data, window_size//2, mode='reflect')
    
    # Compute rolling mean
    rolling_mean = np.convolve(padded_data, box_kernel, mode='valid')
    
    # Compute rolling variance
    rolling_var = np.convolve(padded_data**2, box_kernel, mode='valid') - rolling_mean**2
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))  # avoid negative values due to numerical issues
    
    # Normalize volatility to [0, 1]
    if np.max(rolling_std) > 0:
        normalized_volatility = rolling_std / np.max(rolling_std)
    else:
        normalized_volatility = np.zeros_like(rolling_std)
    
    # Compute adaptive sigma based on volatility
    adaptive_sigma = base_sigma * (1.0 + sensitivity * (1.0 - normalized_volatility))
    
    # Apply LLT filter with adaptive sigma
    smoothed = np.copy(data)
    
    for i in range(len(data)):
        # Apply LLT filter with local sigma
        local_window = max(3, min(len(data) // 5, int(adaptive_sigma[i] * 6)))
        local_window = local_window + 1 if local_window % 2 == 0 else local_window  # ensure odd
        
        start = max(0, i - local_window // 2)
        end = min(len(data), i + local_window // 2 + 1)
        
        if end - start >= 3:  # minimum window size for meaningful smoothing
            local_data = data[start:end]
            smoothed[i] = llt_filter(local_data, sigma=adaptive_sigma[i])[min(i - start, len(local_data) - 1)]
    
    return smoothed 