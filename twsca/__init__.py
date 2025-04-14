"""
Time-Warped Spectral Correlation Analysis (TWSCA)

A package for analyzing correlations between time series that may be misaligned in time
or have different speeds, combining dynamic time warping (DTW) with spectral analysis.
"""

from .analysis import compute_twsca, compute_twsca_matrix
from .smoothing import llt_filter, savitzky_golay, adaptive_smoothing
from .dtw import dtw_distance, align_series
from .spectral import compute_spectrum, spectral_correlation
from .utils import normalize_series, validate_time_series
from .config import get_config, set_config, reset_config
from .cli import main as cli_main

__all__ = [
    'compute_twsca',
    'compute_twsca_matrix',
    'llt_filter',
    'savitzky_golay',
    'adaptive_smoothing',
    'dtw_distance',
    'align_series',
    'compute_spectrum',
    'spectral_correlation',
    'normalize_series',
    'validate_time_series',
    'get_config',
    'set_config',
    'reset_config',
    'cli_main',
] 