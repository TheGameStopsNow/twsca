API Reference
============

Core Functions
-------------

compute_twsca
~~~~~~~~~~~~

.. py:function:: compute_twsca(s1, s2, **kwargs)

   Compute Time-Warped Spectral Correlation Analysis between two time series.

   :param np.ndarray s1: First time series
   :param np.ndarray s2: Second time series
   :param int window_size: Size of the sliding window for spectral analysis (optional)
   :param int dtw_radius: Radius for DTW computation (optional)
   :param bool normalize: Whether to normalize the input series (optional)
   :param bool use_llt: Whether to apply LLT filtering to smooth the signals. Default is True.
   :param float llt_sigma: Standard deviation parameter for LLT filter (default=1.0, only used if use_llt=True)
   :param float llt_alpha: Smoothing parameter for LLT filter (default=0.5, only used if use_llt=True)
   :return: Dictionary containing analysis results
   :rtype: dict

   The returned dictionary contains:
   
   * ``time_domain_correlation``: Correlation in time domain
   * ``spectral_correlation``: Correlation in frequency domain
   * ``dtw_path``: DTW alignment path
   * ``spectral_components``: Spectral components of the analysis

Visualization
------------

plot_twsca_results
~~~~~~~~~~~~~~~~

.. py:function:: plot_twsca_results(result, title=None)

   Plot TWSCA analysis results.

   :param dict result: Results from compute_twsca
   :param str title: Optional plot title
   :return: None

Spectral Analysis
---------------

compute_spectral_correlation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: compute_spectral_correlation(s1, s2, window_size=None)

   Compute spectral correlation between two time series.

   :param np.ndarray s1: First time series
   :param np.ndarray s2: Second time series
   :param int window_size: Size of the sliding window (optional)
   :return: Spectral correlation value
   :rtype: float

Dynamic Time Warping
------------------

compute_dtw
~~~~~~~~~

.. py:function:: compute_dtw(s1, s2, radius=None)

   Compute Dynamic Time Warping between two time series.

   :param np.ndarray s1: First time series
   :param np.ndarray s2: Second time series
   :param int radius: DTW radius for computation (optional)
   :return: DTW path and distance
   :rtype: tuple

   Returns a tuple containing:
   
   * The DTW path as a list of (i, j) pairs
   * The DTW distance as a float

Data Preprocessing
----------------

normalize_series
~~~~~~~~~~~~~

.. py:function:: normalize_series(series)

   Normalize a time series to zero mean and unit variance.

   :param np.ndarray series: Input time series
   :return: Normalized time series
   :rtype: np.ndarray

remove_trend
~~~~~~~~~~

.. py:function:: remove_trend(series, order=1)

   Remove polynomial trend from a time series.

   :param np.ndarray series: Input time series
   :param int order: Order of the polynomial trend to remove
   :return: Detrended time series
   :rtype: np.ndarray 