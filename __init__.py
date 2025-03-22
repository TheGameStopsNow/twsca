"""
Time-Warped Spectral Correlation Analysis (TWSCA) Package

This package provides functionality for detecting correlations between time series
that may be misaligned in time or have nonlinear temporal distortions.
"""

__version__ = "0.1.0"

from .analysis import compute_twsca, compute_twsca_matrix
from .dtw import dtw_distance, align_series
from .spectral import compute_spectrum, spectral_correlation

__all__ = [
    "compute_twsca",
    "compute_twsca_matrix",
    "dtw_distance",
    "align_series",
    "compute_spectrum",
    "spectral_correlation",
]
