Development Guide
===============

This guide provides information for developers who want to contribute to TWSCA.

Development Setup
--------------

Prerequisites
~~~~~~~~~~~

* Python 3.8 or higher
* Git
* pip (Python package installer)
* A code editor (VS Code recommended)

Setting Up the Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Project Structure
--------------

The project follows a standard Python package structure:

.. code-block::

   twsca/
   ├── docs/              # Documentation
   ├── examples/          # Example scripts
   ├── tests/            # Test suite
   ├── twsca/            # Main package
   │   ├── __init__.py
   │   ├── analysis.py   # Core analysis functions
   │   ├── dtw.py        # DTW implementation
   │   └── spectral.py   # Spectral analysis
   ├── setup.py          # Package setup
   ├── pyproject.toml    # Project configuration
   └── README.md         # Project overview

Code Style
---------

TWSCA follows PEP 8 style guidelines. The project uses:

* Black for code formatting
* isort for import sorting
* flake8 for linting

To format your code:

.. code-block:: bash

   # Format code
   black .
   
   # Sort imports
   isort .

   # Run linter
   flake8

Testing
------

Running Tests
~~~~~~~~~~~

The project uses pytest for testing. To run tests:

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_analysis.py

   # Run with coverage report
   pytest --cov=twsca

Writing Tests
~~~~~~~~~~~

Tests should be written in the `tests/` directory. Follow these guidelines:

1. Test file naming: `test_*.py`
2. Test function naming: `test_*`
3. Use descriptive test names
4. Include docstrings for test functions
5. Use appropriate fixtures

Example test:

.. code-block:: python

   def test_compute_twsca_basic():
       """Test basic TWSCA computation with simple sine waves."""
       import numpy as np
       from twsca import compute_twsca

       # Generate test data
       t = np.linspace(0, 10, 100)
       s1 = np.sin(t)
       s2 = np.sin(t + 1)

       # Run test
       result = compute_twsca(s1, s2)

       # Assertions
       assert 'time_domain_correlation' in result
       assert 'spectral_correlation' in result
       assert isinstance(result['time_domain_correlation'], float)
       assert isinstance(result['spectral_correlation'], float)

Documentation
-----------

Building Documentation
~~~~~~~~~~~~~~~~~~

The documentation is built using Sphinx:

.. code-block:: bash

   cd docs
   make html

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~

1. All public functions must have docstrings
2. Follow Google style docstrings
3. Include type hints
4. Provide examples in docstrings
5. Keep documentation up to date

Example docstring:

.. code-block:: python

   def compute_twsca(s1: np.ndarray, s2: np.ndarray, **kwargs) -> dict:
       """Compute Time-Warped Spectral Correlation Analysis.

       Args:
           s1: First time series
           s2: Second time series
           **kwargs: Additional arguments
               window_size: Size of sliding window
               dtw_radius: DTW computation radius
               normalize: Whether to normalize input

       Returns:
           Dictionary containing analysis results

       Example:
           >>> import numpy as np
           >>> from twsca import compute_twsca
           >>> t = np.linspace(0, 10, 100)
           >>> s1 = np.sin(t)
           >>> s2 = np.sin(t + 1)
           >>> result = compute_twsca(s1, s2)
       """
       pass

Release Process
-------------

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Run tests and linting
4. Build documentation
5. Create release on GitHub
6. Update PyPI package

Example release commands:

.. code-block:: bash

   # Run all checks
   pytest
   black .
   isort .
   flake8

   # Build documentation
   cd docs
   make html

   # Build package
   python setup.py sdist bdist_wheel

   # Upload to PyPI
   twine upload dist/*

Contributing
----------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Update documentation
6. Submit a pull request

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~

1. Keep PRs focused and small
2. Include tests for new features
3. Update documentation
4. Follow code style guidelines
5. Provide clear commit messages

Example commit message:

.. code-block::

   feat: Add support for multiple time series analysis

   - Add new function for pairwise correlation computation
   - Add visualization for correlation matrix
   - Update documentation with new examples
   - Add tests for new functionality

   Closes #123

Getting Help
----------

* Check the documentation
* Open an issue on GitHub
* Join the discussion forum
* Contact the maintainers

Maintenance
---------

Regular Tasks
~~~~~~~~~~

1. Update dependencies
2. Run security audits
3. Update documentation
4. Review and merge PRs
5. Monitor issue tracker

Long-term Goals
~~~~~~~~~~~~

1. Improve performance
2. Add more features
3. Expand test coverage
4. Enhance documentation
5. Build community 