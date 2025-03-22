Contributing Guide
===============

Thank you for your interest in contributing to TWSCA! This guide will help you get started.

Code of Conduct
-------------

By participating in this project, you agree to abide by our Code of Conduct. Please read the full text in `CODE_OF_CONDUCT.md`.

Getting Started
-------------

1. Fork the repository
2. Clone your fork
3. Set up development environment
4. Create a feature branch
5. Make your changes
6. Submit a pull request

Detailed Steps
------------

Forking and Cloning
~~~~~~~~~~~~~~~~

1. Go to https://github.com/TheGameStopsNow/twsca
2. Click the "Fork" button
3. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/twsca.git
      cd twsca

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create and activate virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

2. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Making Changes
------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes following our coding standards:
   * Follow PEP 8
   * Use type hints
   * Write docstrings
   * Add tests
   * Update documentation

3. Commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "feat: Add your feature description"

4. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

Submitting a Pull Request
----------------------

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template:
   * Description of changes
   * Related issues
   * Testing performed
   * Documentation updates

4. Submit the PR

Code Review Process
----------------

1. Maintainers will review your PR
2. Address any feedback
3. Make requested changes
4. Push updates to your branch
5. PR will be merged when approved

Coding Standards
-------------

Python Style
~~~~~~~~~~

* Follow PEP 8
* Use Black for formatting
* Use isort for import sorting
* Use flake8 for linting

Example:

.. code-block:: python

   from typing import Dict, List, Optional

   import numpy as np
   from scipy import signal

   def process_data(
       data: np.ndarray,
       window_size: int = 100,
       normalize: bool = True
   ) -> Dict[str, np.ndarray]:
       """Process input data.

       Args:
           data: Input data array
           window_size: Size of processing window
           normalize: Whether to normalize data

       Returns:
           Dictionary of processed data
       """
       if normalize:
           data = (data - np.mean(data)) / np.std(data)
       
       return {
           'normalized': data,
           'windowed': signal.windows.hann(window_size)
       }

Documentation
~~~~~~~~~~

* Use Google style docstrings
* Include type hints
* Provide examples
* Keep documentation up to date

Example:

.. code-block:: python

   def compute_correlation(
       series1: np.ndarray,
       series2: np.ndarray,
       method: str = 'pearson'
   ) -> float:
       """Compute correlation between two time series.

       Args:
           series1: First time series
           series2: Second time series
           method: Correlation method ('pearson' or 'spearman')

       Returns:
           Correlation coefficient

       Example:
           >>> import numpy as np
           >>> series1 = np.array([1, 2, 3])
           >>> series2 = np.array([2, 4, 6])
           >>> compute_correlation(series1, series2)
           1.0
       """
       pass

Testing
~~~~~~

* Write unit tests for new features
* Maintain test coverage
* Use pytest fixtures
* Include edge cases

Example:

.. code-block:: python

   def test_compute_correlation():
       """Test correlation computation."""
       import numpy as np
       from twsca import compute_correlation

       # Test perfect correlation
       series1 = np.array([1, 2, 3])
       series2 = np.array([2, 4, 6])
       assert compute_correlation(series1, series2) == 1.0

       # Test zero correlation
       series3 = np.array([1, 0, -1])
       assert abs(compute_correlation(series1, series3)) < 0.1

       # Test with NaN values
       series4 = np.array([1, np.nan, 3])
       with pytest.raises(ValueError):
           compute_correlation(series1, series4)

Common Issues
-----------

1. Code Style
   * Run Black before committing
   * Fix all flake8 warnings
   * Sort imports with isort

2. Documentation
   * Include all required sections in docstrings
   * Update all affected documentation
   * Check for broken links

3. Testing
   * Add tests for new features
   * Fix failing tests
   * Update test documentation

Getting Help
----------

* Check existing issues
* Join discussions
* Ask in pull requests
* Contact maintainers

Thank You
--------

Thank you for contributing to TWSCA! Your contributions help make the project better for everyone. 