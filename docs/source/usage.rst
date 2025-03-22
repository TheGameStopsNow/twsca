Usage Guide
===========

This guide provides detailed information on how to use TWSCA for time series analysis.

Basic Usage
----------

The most common use case is to analyze the correlation between two time series that may be misaligned:

.. code-block:: python

   from twsca import compute_twsca
   import numpy as np

   # Generate example time series
   t = np.linspace(0, 10, 100)
   s1 = np.sin(t)
   s2 = np.sin(t + 1)  # Phase-shifted version

   # Compute TWSCA
   result = compute_twsca(s1, s2)

   # Access results
   print(f"Time-domain correlation: {result['time_domain_correlation']}")
   print(f"Spectral correlation: {result['spectral_correlation']}")

   # Visualize results
   from twsca import plot_twsca_results
   plot_twsca_results(result, title="TWSCA Analysis Results")

Advanced Usage
------------

Customizing Analysis Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize various parameters of the analysis:

.. code-block:: python

   result = compute_twsca(
       s1, s2,
       window_size=50,      # Custom window size for spectral analysis
       dtw_radius=10,       # DTW computation radius
       normalize=True       # Normalize input series
   )

Preprocessing Data
~~~~~~~~~~~~~~~~

TWSCA provides utilities for preprocessing time series data:

.. code-block:: python

   from twsca import normalize_series, remove_trend

   # Normalize the series
   s1_normalized = normalize_series(s1)
   s2_normalized = normalize_series(s2)

   # Remove trend
   s1_detrended = remove_trend(s1, order=2)  # Remove quadratic trend
   s2_detrended = remove_trend(s2, order=2)

   # Compute TWSCA on preprocessed data
   result = compute_twsca(s1_detrended, s2_detrended)

Working with Real Data
--------------------

Loading Data
~~~~~~~~~~

TWSCA works with numpy arrays, so you can load data from various sources:

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load data from CSV
   df = pd.read_csv('data.csv')
   
   # Extract time series
   s1 = df['series1'].values
   s2 = df['series2'].values

   # Ensure data is numeric and handle missing values
   s1 = pd.to_numeric(s1, errors='coerce').fillna(method='ffill')
   s2 = pd.to_numeric(s2, errors='coerce').fillna(method='ffill')

   # Compute TWSCA
   result = compute_twsca(s1, s2)

Handling Different Lengths
~~~~~~~~~~~~~~~~~~~~~~~

If your time series have different lengths, you can use DTW to align them:

.. code-block:: python

   from twsca import compute_dtw

   # Compute DTW path
   path, distance = compute_dtw(s1, s2)

   # Align series using the path
   s1_aligned = s1[path[:, 0]]
   s2_aligned = s2[path[:, 1]]

   # Now compute TWSCA on aligned series
   result = compute_twsca(s1_aligned, s2_aligned)

Best Practices
------------

1. Data Preprocessing
   * Always check for missing values
   * Consider normalizing your data
   * Remove trends if they're not relevant to your analysis
   * Handle outliers appropriately

2. Parameter Selection
   * Choose window_size based on your data's characteristics
   * Adjust dtw_radius based on expected misalignment
   * Use normalize=True for better comparison of different scales

3. Interpretation
   * Consider both time-domain and spectral correlations
   * Look at the DTW path to understand the alignment
   * Use visualization tools to gain insights

Common Pitfalls
-------------

1. Data Quality
   * Missing values can affect the analysis
   * Outliers can distort correlations
   * Different sampling rates need to be handled

2. Parameter Selection
   * Too small window_size may miss important patterns
   * Too large dtw_radius can be computationally expensive
   * Normalization may not be appropriate for all cases

3. Interpretation
   * Correlation doesn't imply causation
   * Spectral correlation may be affected by noise
   * DTW paths should be validated

Troubleshooting
-------------

1. Performance Issues
   * Use smaller window_size for faster computation
   * Reduce dtw_radius if alignment is not critical
   * Consider downsampling for very long series

2. Memory Issues
   * Process data in chunks for large datasets
   * Clear unnecessary variables
   * Use appropriate data types

3. Accuracy Issues
   * Check data preprocessing steps
   * Validate parameter choices
   * Compare with alternative methods 