"""
Utility functions for TWSCA package.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List


def validate_time_series(series: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """
    Validate and convert input to numpy array.
    
    Parameters
    ----------
    series : array-like
        Input time series
    
    Returns
    -------
    np.ndarray
        Validated numpy array
    """
    # Convert pandas Series to numpy array
    if isinstance(series, pd.Series):
        series = series.values
    
    # Convert list to numpy array
    if isinstance(series, list):
        series = np.array(series)
    
    # Ensure it's a numpy array
    if not isinstance(series, np.ndarray):
        raise TypeError(f"Expected array-like input, got {type(series)}")
    
    # Ensure it's 1D
    if series.ndim > 1:
        if series.shape[1] == 1:
            series = series.flatten()
        else:
            raise ValueError(f"Expected 1D array, got shape {series.shape}")
    
    # Check for NaN values
    if np.isnan(series).any():
        raise ValueError("Time series contains NaN values")
    
    # Check for infinite values
    if np.isinf(series).any():
        raise ValueError("Time series contains infinite values")
    
    return series


def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Normalize a time series to zero mean and unit variance.
    
    Parameters
    ----------
    series : np.ndarray
        Input time series
    
    Returns
    -------
    np.ndarray
        Normalized time series
    """
    # Validate input
    series = validate_time_series(series)
    
    # Check if series has zero standard deviation
    std = np.std(series)
    if std == 0:
        return np.zeros_like(series)  # Return zeros if constant series
    
    # Normalize
    normalized = (series - np.mean(series)) / std
    
    return normalized


def chunk_data(
    data: np.ndarray, 
    chunk_size: int
) -> List[np.ndarray]:
    """
    Split data into chunks of specified size.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to be chunked
    chunk_size : int
        Size of each chunk
    
    Returns
    -------
    List[np.ndarray]
        List of data chunks
    """
    # Validate input
    data = validate_time_series(data)
    
    # Split into chunks
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]


def compute_progress(current: int, total: int) -> str:
    """
    Compute progress string.
    
    Parameters
    ----------
    current : int
        Current progress value
    total : int
        Total expected value
    
    Returns
    -------
    str
        Progress string (e.g., "50%")
    """
    percentage = int(100 * current / total)
    return f"{percentage}%" 