"""
Dynamic Time Warping (DTW) functions for TWSCA package.
"""

import numpy as np
from typing import Tuple, List, Optional, Union


def dtw_distance(
    s1: np.ndarray,
    s2: np.ndarray,
    window: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute Dynamic Time Warping distance between two time series.
    
    Parameters
    ----------
    s1 : np.ndarray
        First time series
    s2 : np.ndarray
        Second time series
    window : int, optional
        Sakoe-Chiba band width for constraining the warping path
        
    Returns
    -------
    Tuple[float, np.ndarray]
        DTW distance and warping path
    """
    # Check for empty sequences
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Empty sequences are not allowed for DTW computation")
    
    # Convert to numpy arrays if needed
    s1 = np.array(s1)
    s2 = np.array(s2)
    
    # Get sequence lengths
    n, m = len(s1), len(s2)
    
    # Set window size (if None, use the full matrix)
    if window is None:
        window = max(n, m)
    
    # Initialize cost matrix with infinity
    cost_matrix = np.ones((n + 1, m + 1)) * float('inf')
    cost_matrix[0, 0] = 0
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],     # Insertion
                cost_matrix[i, j - 1],     # Deletion
                cost_matrix[i - 1, j - 1]  # Match
            )
    
    # Extract the distance
    distance = np.sqrt(cost_matrix[n, m])
    
    # Extract the warping path by backtracking
    path = []
    i, j = n, m
    
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        
        min_cost = min(
            cost_matrix[i - 1, j - 1],
            cost_matrix[i - 1, j],
            cost_matrix[i, j - 1]
        )
        
        if min_cost == cost_matrix[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif min_cost == cost_matrix[i - 1, j]:
            i = i - 1
        else:
            j = j - 1
    
    # Add the origin point
    path.append((0, 0))
    
    # Reverse the path to get it in ascending order
    path.reverse()
    
    return distance, np.array(path)


def align_series(
    s1: np.ndarray,
    s2: np.ndarray,
    path: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two time series according to a DTW warping path.
    
    Parameters
    ----------
    s1 : np.ndarray
        First time series
    s2 : np.ndarray
        Second time series
    path : np.ndarray
        Warping path from DTW
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aligned time series
    """
    # Convert path to integers if it's not already
    path = np.array(path, dtype=int)
    
    # Initialize aligned sequences
    aligned_s1 = []
    aligned_s2 = []
    
    # Create aligned sequences by duplicating points according to the path
    for i, j in path:
        aligned_s1.append(s1[i])
        aligned_s2.append(s2[j])
    
    return np.array(aligned_s1), np.array(aligned_s2) 