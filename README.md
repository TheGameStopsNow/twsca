# Time-Warped Spectral Correlation Analysis (TWSCA)

TWSCA is a Python package for analyzing correlations between time series that may be misaligned in time or have different speeds. It combines dynamic time warping (DTW) with spectral analysis to identify hidden relationships.

## Features

- Dynamic Time Warping (DTW) for time series alignment
- Spectral analysis for frequency domain correlation
- Support for multiple time series comparison
- Advanced smoothing techniques with automatic parameter optimization
- Interactive visualizations using Plotly
- Animated warping path visualization
- 3D correlation surface plots
- Comprehensive analysis dashboards
- Adaptive smoothing based on local volatility
- Dark/light theme support
- Configuration management
- Comprehensive test suite
- Type hints for better IDE support
- Command-line interface for quick demonstrations

## Installation

Install the latest stable release from PyPI:

```bash
pip install twsca>=0.3.0
```

For interactive visualizations and notebooks support:
```bash
pip install "twsca[interactive]>=0.3.0"
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/TheGameStopsNow/twsca.git
```

## Quick Start

```python
from twsca import compute_twsca, setup_plotting_style, create_interactive_plot
import numpy as np

# Create example data
t = np.linspace(0, 10, 100)
s1 = np.sin(t)
s2 = np.sin(t + 1)  # Phase-shifted version

# Compute TWSCA with automatic parameter optimization
# LLT filtering is on by default for better noise handling
result = compute_twsca(s1, s2, optimize_params=True)
print(f"Time-domain correlation: {result['time_domain_correlation']}")
print(f"Spectral correlation: {result['spectral_correlation']}")

# If you need to disable LLT filtering:
# result = compute_twsca(s1, s2, use_llt=False)

# Create interactive plot
fig = create_interactive_plot({
    'Original': s1,
    'Shifted': s2
}, plot_type='time_series')
fig.show()
```

## Command-Line Interface

TWSCA includes a command-line interface for quickly demonstrating its features:

### Running the CLI

```bash
# Method 1: Using Python module
python -m twsca <command> [options]

# Method 2: Using the provided script
python twsca_cli.py <command> [options]
```

### Available Commands

1. **LLT Filter Demonstration**:
   ```bash
   python -m twsca llt --sigma 1.0 --alpha 0.5 --noise 0.3
   ```

2. **Comparing Different Smoothing Methods**:
   ```bash
   python -m twsca smooth --noise 0.4 --window 7
   ```

3. **TWSCA Analysis with and without LLT**:
   ```bash
   python -m twsca twsca --noise 0.3 --phase 1.5 --warp
   ```

### Common Options

- `--points`: Number of points in the generated time series (default: 100)
- `--noise`: Level of noise to add to the signal (default: 0.3)
- `--sigma`: Sigma parameter for LLT filter (default: 1.0)
- `--alpha`: Alpha parameter for LLT filter (default: 0.5)

For more options, run:
```bash
python -m twsca <command> --help
```

## Advanced Usage

### Advanced Smoothing with Parameter Optimization

```python
from twsca import llt_filter, adaptive_smoothing

# Create noisy data
t = np.linspace(0, 10, 100)
noisy_signal = np.sin(t) + 0.2 * np.random.randn(100)

# Apply LLT filter with automatic parameter optimization
smoothed_opt = llt_filter(noisy_signal, optimize_params=True)

# Apply adaptive smoothing based on local volatility
smoothed_adaptive = adaptive_smoothing(
    noisy_signal,
    base_sigma=1.0,
    sensitivity=0.1
)

# Create interactive comparison plot
fig = create_interactive_plot({
    'Original': noisy_signal,
    'Optimized LLT': smoothed_opt,
    'Adaptive': smoothed_adaptive
})
fig.show()
```

### LLT Filtering in TWSCA

TWSCA applies LLT (Local Laplacian Transform) filtering by default to input signals before analysis to improve robustness against noise. You can control this behavior with the `use_llt` parameter:

```python
from twsca import compute_twsca
import numpy as np

# Create noisy signals
t = np.linspace(0, 10, 100)
s1 = np.sin(t) + 0.3 * np.random.randn(100)
s2 = np.sin(t + 1) + 0.3 * np.random.randn(100)

# With LLT filtering (default)
result_with_llt = compute_twsca(s1, s2)

# Without LLT filtering
result_without_llt = compute_twsca(s1, s2, use_llt=False)

# Control LLT parameters
result_custom_llt = compute_twsca(s1, s2, use_llt=True, llt_sigma=2.0, llt_alpha=0.3)

print(f"Correlation with LLT: {result_with_llt['spectral_correlation']:.4f}")
print(f"Correlation without LLT: {result_without_llt['spectral_correlation']:.4f}")
```

### Interactive Visualization

```python
from twsca import plot_analysis_dashboard, plot_3d_correlation_surface

# Create comprehensive analysis dashboard
dashboard = plot_analysis_dashboard({
    'time_series': {
        'Original': s1,
        'Warped': s2
    },
    'correlation_matrix': result['correlation_matrix'],
    'spectra': result['spectral_components'],
    'warping_path': result['dtw_path']
})
dashboard.show()

# Create interactive 3D correlation surface
surface = plot_3d_correlation_surface(
    times=t,
    frequencies=result['frequencies'],
    correlations=result['time_freq_correlation'],
    interactive=True
)
surface.show()
```

### Animated Warping Visualization

```python
from twsca import plot_warping_animation

# Create animation of the warping process
anim = plot_warping_animation(
    original=s1,
    warped=s2,
    path=result['dtw_path'],
    interval=50  # milliseconds between frames
)
plt.show()
```

### Smoothing Time Series

```python
from twsca import llt_filter, savitzky_golay

# Create noisy data
t = np.linspace(0, 10, 100)
noisy_signal = np.sin(t) + 0.2 * np.random.randn(100)

# Apply Local Laplacian Transform filter
smoothed_llt = llt_filter(noisy_signal, sigma=1.0, alpha=0.5)

# Apply Savitzky-Golay filter
smoothed_sg = savitzky_golay(noisy_signal, window_size=5, poly_order=2)

# Plot results
plot_time_series(
    [noisy_signal, smoothed_llt, smoothed_sg],
    labels=['Original', 'LLT', 'Savitzky-Golay']
)
```

### Configuration Management

```python
from twsca import set_config, get_config, reset_config

# Set global configuration
set_config({
    'plot_style': 'seaborn',
    'dtw_radius': 20,
    'window_size': 100,
    'normalize': True
})

# Get current settings
config = get_config()
print(f"Current plot style: {config.get('plot_style')}")

# Reset to defaults
reset_config()
```

### Batch Processing

```python
from twsca import chunk_data, compute_progress

# Process large datasets in chunks
data = np.random.randn(1000)
chunks = chunk_data(data, chunk_size=100)

for i, chunk in enumerate(chunks):
    # Process each chunk
    result = compute_twsca(chunk, chunk)
    print(compute_progress(i + 1, len(chunks)))
```

### Multiple Time Series Analysis

```python
import pandas as pd
from twsca import compute_twsca_matrix

# Create multiple time series
data = pd.DataFrame({
    'series1': np.sin(t),
    'series2': np.cos(t),
    'series3': np.sin(2 * t)
})

# Compute correlation matrix
correlation_matrix = compute_twsca_matrix(data)
print("Correlation matrix:")
print(correlation_matrix)
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

### Smoothing Functions

#### `llt_filter(data, sigma=1.0, alpha=0.5)`

Apply Local Laplacian Transform filter to time series data.

**Parameters:**
- `data` (np.ndarray): Input time series
- `sigma` (float): Standard deviation for Gaussian kernel
- `alpha` (float): Smoothing parameter (0 < alpha < 1)

#### `savitzky_golay(data, window_size=5, poly_order=2)`

Apply Savitzky-Golay filter to time series data.

**Parameters:**
- `data` (np.ndarray): Input time series
- `window_size` (int): Size of the window (must be odd)
- `poly_order` (int): Order of the polynomial fit

### Plotting Functions

#### `setup_plotting_style(style='default', **kwargs)`

Set up the default plotting style for TWSCA visualizations.

**Parameters:**
- `style` (str): Style name ('default', 'seaborn', 'dark_background')
- `**kwargs`: Additional style parameters

#### `plot_time_series(data, time=None, labels=None, **kwargs)`

Plot one or more time series.

**Parameters:**
- `data` (np.ndarray or list): Time series data
- `time` (np.ndarray, optional): Time points
- `labels` (list, optional): Labels for each series

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