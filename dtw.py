"""
Dynamic Time Warping Module

This module provides functionality for aligning time series using
Dynamic Time Warping (DTW) algorithm.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


def dtw_distance(
    series1: Union[np.ndarray, List[float]],
    series2: Union[np.ndarray, List[float]],
    window: Optional[int] = None,
    adaptive_window: bool = False,
    volatility_scaling: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Calculate the Dynamic Time Warping distance between two time series.

    Parameters:
    -----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    window : int, optional
        Window constraint for DTW. If None, full DTW is performed.
    adaptive_window : bool
        Whether to adjust the window size based on time series volatility
    volatility_scaling : float
        Scaling factor for adaptive window (higher values allow more warping)

    Returns:
    --------
    distance : float
        DTW distance between the two series
    path : np.ndarray
        Optimal warping path as an array of (i, j) index pairs
    """
    # Convert inputs to numpy arrays
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)

    # Check for empty sequences
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Empty sequences")

    # Get lengths of both sequences
    n, m = len(s1), len(s2)

    # Initialize cost matrix with infinity
    cost_matrix = np.ones((n + 1, m + 1)) * np.inf
    cost_matrix[0, 0] = 0

    # Set window size if not provided
    if window is None:
        window = max(n, m)

    # Calculate volatility if adaptive window is enabled
    if adaptive_window:
        # Calculate rolling standard deviation as a measure of volatility
        s1_volatility = np.zeros(n)
        s2_volatility = np.zeros(m)

        # Use a rolling window of 5 or 10% of series length, whichever is larger
        vol_window = max(5, int(min(n, m) * 0.1))

        for i in range(n):
            start_idx = max(0, i - vol_window)
            s1_volatility[i] = np.std(s1[start_idx : i + 1])

        for i in range(m):
            start_idx = max(0, i - vol_window)
            s2_volatility[i] = np.std(s2[start_idx : i + 1])

        # Normalize volatility to [0, 1] range
        if np.max(s1_volatility) > 0:
            s1_volatility = s1_volatility / np.max(s1_volatility)
        if np.max(s2_volatility) > 0:
            s2_volatility = s2_volatility / np.max(s2_volatility)

    # Compute DTW matrix
    for i in range(1, n + 1):
        # Apply window constraint, potentially adapting based on volatility
        if adaptive_window and i > 1 and i <= n:
            # Adjust window based on local volatility - more volatile = larger window
            local_volatility = s1_volatility[i - 1]
            # Scale the window by volatility and the scaling factor
            local_window = int(window * (1 + local_volatility * volatility_scaling))
            window_start = max(1, i - local_window)
            window_end = min(m + 1, i + local_window + 1)
        else:
            window_start = max(1, i - window)
            window_end = min(m + 1, i + window + 1)

        for j in range(window_start, window_end):
            # Calculate distance between points
            cost = (s1[i - 1] - s2[j - 1]) ** 2

            # Update cost as minimum of possible previous steps
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],  # insertion
                cost_matrix[i, j - 1],  # deletion
                cost_matrix[i - 1, j - 1],  # match
            )

    # Retrieve the optimal path
    path = _get_path(cost_matrix)

    # Return distance and path
    return np.sqrt(cost_matrix[n, m]), path


def _get_path(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Retrieve the optimal warping path from the cost matrix.

    Parameters:
    -----------
    cost_matrix : np.ndarray
        DTW cost matrix

    Returns:
    --------
    path : np.ndarray
        Optimal warping path as an array of (i, j) index pairs
    """
    i, j = cost_matrix.shape[0] - 1, cost_matrix.shape[1] - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost_idx = np.argmin(
                [
                    cost_matrix[i - 1, j - 1],
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                ]
            )

            if min_cost_idx == 0:
                i -= 1
                j -= 1
            elif min_cost_idx == 1:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    # Reverse path to start from beginning
    path.reverse()
    return np.array(path)


def align_series(
    series1: Union[np.ndarray, List[float]],
    series2: Union[np.ndarray, List[float]],
    path: Optional[np.ndarray] = None,
    resample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two time series based on DTW path.

    Parameters:
    -----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    path : np.ndarray, optional
        The alignment path. If None, computed using dtw_distance
    resample : bool
        Whether to resample the aligned series to ensure uniform sampling

    Returns:
    --------
    aligned_s1 : np.ndarray
        Aligned version of series1
    aligned_s2 : np.ndarray
        Aligned version of series2
    """
    # Convert inputs to numpy arrays
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)

    # Compute DTW path if not provided
    if path is None:
        _, path = dtw_distance(s1, s2)

    # Create aligned series using the path
    aligned_s1 = np.array([s1[i - 1] for i, _ in path[1:]])
    aligned_s2 = np.array([s2[j - 1] for _, j in path[1:]])

    # Resample if needed to ensure uniform sampling for spectral analysis
    if resample and len(aligned_s1) > 2:
        # Create a uniform time index
        orig_indices = np.arange(len(aligned_s1))
        uniform_indices = np.linspace(0, len(aligned_s1) - 1, len(aligned_s1))

        # Interpolate to create uniformly sampled series
        from scipy.interpolate import interp1d

        f1 = interp1d(
            orig_indices,
            aligned_s1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        f2 = interp1d(
            orig_indices,
            aligned_s2,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        aligned_s1 = f1(uniform_indices)
        aligned_s2 = f2(uniform_indices)

    return aligned_s1, aligned_s2
