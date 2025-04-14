"""
Time-Warped Spectral Correlation Analysis (TWSCA) Package

This package provides functionality for detecting correlations between time series
that may be misaligned in time or have nonlinear temporal distortions.
"""

__version__ = "0.1.0"

# Use relative imports for modules within the package
from .analysis import compute_twsca, compute_twsca_matrix
from .dtw import align_series, dtw_distance
from .spectral import (
    compute_spectrum,
    compute_wavelet_coherence,
    spectral_correlation,
    validate_spectral_integrity,
)

__all__ = [
    "compute_twsca",
    "compute_twsca_matrix",
    "dtw_distance",
    "align_series",
    "compute_spectrum",
    "spectral_correlation",
    "compute_wavelet_coherence",
    "validate_spectral_integrity",
]
