Examples
========

This section provides detailed examples of using TWSCA in various scenarios.

Basic Examples
------------

Simple Sine Waves
~~~~~~~~~~~~~~

This example demonstrates TWSCA analysis on simple sine waves with different phases:

.. code-block:: python

   from twsca import compute_twsca, plot_twsca_results
   import numpy as np

   # Generate time series
   t = np.linspace(0, 10, 100)
   s1 = np.sin(t)
   s2 = np.sin(t + 1)  # Phase-shifted version

   # Compute TWSCA
   result = compute_twsca(s1, s2)

   # Plot results
   plot_twsca_results(result, title="Sine Wave Analysis")

Market Analysis
-------------

Stock Price Correlation
~~~~~~~~~~~~~~~~~~~~

This example shows how to analyze correlation between stock prices:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from twsca import compute_twsca, normalize_series

   # Load stock data
   df = pd.read_csv('stock_data.csv')
   
   # Extract price series
   prices1 = df['stock1'].values
   prices2 = df['stock2'].values

   # Normalize prices
   prices1_norm = normalize_series(prices1)
   prices2_norm = normalize_series(prices2)

   # Compute TWSCA
   result = compute_twsca(
       prices1_norm, prices2_norm,
       window_size=20,  # 20-day window
       dtw_radius=5     # Allow for 5-day misalignment
   )

   # Plot results
   plot_twsca_results(result, title="Stock Price Correlation")

Signal Processing
--------------

ECG Signal Analysis
~~~~~~~~~~~~~~~~

This example demonstrates TWSCA analysis on ECG signals:

.. code-block:: python

   import numpy as np
   from scipy.io import loadmat
   from twsca import compute_twsca, remove_trend

   # Load ECG data
   data = loadmat('ecg_data.mat')
   ecg1 = data['ecg1'].flatten()
   ecg2 = data['ecg2'].flatten()

   # Remove baseline drift
   ecg1_clean = remove_trend(ecg1, order=2)
   ecg2_clean = remove_trend(ecg2, order=2)

   # Compute TWSCA
   result = compute_twsca(
       ecg1_clean, ecg2_clean,
       window_size=100,  # Adjust based on signal characteristics
       normalize=True
   )

   # Plot results
   plot_twsca_results(result, title="ECG Signal Analysis")

Advanced Examples
--------------

Multiple Time Series
~~~~~~~~~~~~~~~~~

This example shows how to analyze multiple time series:

.. code-block:: python

   import numpy as np
   from twsca import compute_twsca

   # Generate multiple time series
   t = np.linspace(0, 10, 100)
   series = [
       np.sin(t),
       np.sin(t + 1),
       np.sin(t + 2),
       np.sin(t + 3)
   ]

   # Compute pairwise correlations
   n_series = len(series)
   correlations = np.zeros((n_series, n_series))
   
   for i in range(n_series):
       for j in range(i+1, n_series):
           result = compute_twsca(series[i], series[j])
           correlations[i,j] = result['time_domain_correlation']
           correlations[j,i] = correlations[i,j]

   # Plot correlation matrix
   import matplotlib.pyplot as plt
   plt.imshow(correlations, cmap='viridis')
   plt.colorbar()
   plt.title("Time Series Correlation Matrix")
   plt.show()

Real-World Applications
--------------------

Climate Data Analysis
~~~~~~~~~~~~~~~~~~

This example demonstrates TWSCA analysis on climate data:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from twsca import compute_twsca, remove_trend

   # Load climate data
   df = pd.read_csv('climate_data.csv')
   
   # Extract temperature and precipitation series
   temp = df['temperature'].values
   precip = df['precipitation'].values

   # Remove seasonal trends
   temp_detrended = remove_trend(temp, order=4)  # Higher order for seasonal effects
   precip_detrended = remove_trend(precip, order=4)

   # Compute TWSCA
   result = compute_twsca(
       temp_detrended, precip_detrended,
       window_size=30,  # Monthly window
       dtw_radius=15    # Allow for 15-day misalignment
   )

   # Plot results
   plot_twsca_results(result, title="Climate Data Analysis")

Financial Time Series
~~~~~~~~~~~~~~~~~~

This example shows TWSCA analysis on financial time series:

.. code-block:: python

   import yfinance as yf
   import numpy as np
   from twsca import compute_twsca, normalize_series

   # Download financial data
   tickers = ['AAPL', 'MSFT']
   data = {}
   
   for ticker in tickers:
       stock = yf.Ticker(ticker)
       data[ticker] = stock.history(period='1y')['Close'].values

   # Normalize prices
   prices_norm = {ticker: normalize_series(prices) 
                 for ticker, prices in data.items()}

   # Compute TWSCA
   result = compute_twsca(
       prices_norm['AAPL'], prices_norm['MSFT'],
       window_size=20,  # 20-day window
       dtw_radius=5     # 5-day misalignment
   )

   # Plot results
   plot_twsca_results(result, title="Stock Price Correlation Analysis") 