# Time-Warped Spectral Correlation Analysis (TWSCA)

TWSCA is a Python package for analyzing correlations between time series that may be misaligned in time or have different speeds. It combines dynamic time warping (DTW) with spectral analysis to identify hidden relationships.

## Features

- Dynamic Time Warping (DTW) for time series alignment
- Spectral analysis for frequency domain correlation
- Support for multiple time series comparison
- Visualization tools for correlation analysis
- Comprehensive test suite
- Type hints for better IDE support

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/TheGameStopsNow/twsca.git

# Or install with development dependencies
pip install -e "git+https://github.com/TheGameStopsNow/twsca.git#egg=twsca[dev]"
```

## Quick Start

```python
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
```

## API Reference

### Core Functions

#### `compute_twsca(s1, s2, **kwargs)`

Compute Time-Warped Spectral Correlation Analysis between two time series.

**Parameters:**
- `s1` (np.ndarray): First time series
- `s2` (np.ndarray): Second time series
- `window_size` (int, optional): Size of the sliding window for spectral analysis
- `dtw_radius` (int, optional): Radius for DTW computation
- `normalize` (bool, optional): Whether to normalize the input series

**Returns:**
- `dict`: Dictionary containing:
  - `time_domain_correlation`: Correlation in time domain
  - `spectral_correlation`: Correlation in frequency domain
  - `dtw_path`: DTW alignment path
  - `spectral_components`: Spectral components of the analysis

### Visualization

#### `plot_twsca_results(result, title=None)`

Plot TWSCA analysis results.

**Parameters:**
- `result` (dict): Results from `compute_twsca`
- `title` (str, optional): Plot title

## Examples

See the `examples` directory for detailed examples:

1. Basic Usage: `examples/basic_usage.py`
2. Market Analysis: `examples/market_analysis.py`
3. Signal Processing: `examples/signal_processing.py`

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TheGameStopsNow/twsca.git
cd twsca
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TWSCA in your research, please cite:

```bibtex
@software{twsca2024,
  author = {Dennis Nedry},
  title = {TWSCA: Time-Warped Spectral Correlation Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/TheGameStopsNow/twsca}
}
``` 