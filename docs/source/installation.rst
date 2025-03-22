Installation Guide
================

TWSCA is a Python package for analyzing correlations between time series that may be misaligned in time or have different speeds. It combines dynamic time warping (DTW) with spectral analysis to identify hidden relationships.

Features
--------

* Dynamic Time Warping (DTW) for time series alignment
* Spectral analysis for frequency domain correlation
* Support for multiple time series comparison
* Visualization tools for correlation analysis
* Comprehensive test suite
* Type hints for better IDE support

Installation
-----------

.. code-block:: bash

   # Install from GitHub
   pip install git+https://github.com/TheGameStopsNow/twsca.git

   # Or install with development dependencies
   pip install -e "git+https://github.com/TheGameStopsNow/twsca.git#egg=twsca[dev]"

Quick Start
----------

.. code-block:: python

   from twsca import compute_twsca
   import numpy as np

   # Example with sine waves
   t = np.linspace(0, 10, 100)
   s1 = np.sin(t)
   s2 = np.sin(t + 1)  # Phase-shifted version

   # Compute TWSCA
   result = compute_twsca(s1, s2)
   print(f"Time-domain correlation: {result['time_domain_correlation']}")
   print(f"Spectral correlation: {result['spectral_correlation']}")

Development Setup
---------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/TheGameStopsNow/twsca.git
      cd twsca

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Running Tests
-----------

.. code-block:: bash

   pytest

Building Documentation
-------------------

.. code-block:: bash

   cd docs
   make html

Citation
-------

If you use TWSCA in your research, please cite:

.. code-block:: bibtex

   @software{twsca2024,
     author = {Dennis Nedry},
     title = {TWSCA: Time-Warped Spectral Correlation Analysis},
     year = {2024},
     publisher = {GitHub},
     url = {https://github.com/TheGameStopsNow/twsca}
   } 