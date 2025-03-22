# TWSCA Examples

This directory contains example scripts demonstrating how to use the Time-Warped Spectral Correlation Analysis (TWSCA) package.

## Basic Demo

The `twsca_demo.py` script demonstrates the basic functionality of TWSCA using synthetic sine wave data:

```bash
# Basic usage with default parameters (phase shift of 1.0 radians)
python twsca_demo.py

# With time warping instead of just phase shift
python twsca_demo.py --warp

# With custom phase shift and more data points
python twsca_demo.py --phase 1.5 --points 200

# Save the output to a file
python twsca_demo.py --output sine_wave_demo.png
```

## Meme Stock Comparison

The `meme_stock_comparison.py` script applies TWSCA to analyze correlations between different meme stocks:

```bash
# Basic usage with default meme stocks
python meme_stock_comparison.py 

# Specify custom stocks to analyze
python meme_stock_comparison.py --symbols GME AMC BB NOK

# Analyze the last 180 days with daily data
python meme_stock_comparison.py --days 180 --interval 1d

# Also generate standard correlation heatmap for comparison
python meme_stock_comparison.py --standard

# Save output to a specific directory
python meme_stock_comparison.py --output-dir ~/Documents/analysis
```

### Meme Stock Comparison Features

The script provides the following features:

1. **Correlation Heatmap**: Creates a heatmap visualization showing how different meme stocks correlate with each other after time-warping alignment.

2. **Detailed Comparisons**: For pairs with significant correlation (above 0.4), generates detailed comparison plots showing:
   - Original price series
   - Normalized series with standard correlation
   - Time-warped aligned series
   - Correlation scatter plots before and after alignment

3. **Analysis Summary**: Prints an interpretation of the results, highlighting stocks with strong hidden correlations.

## Required Dependencies

To run these examples, you'll need the following dependencies in addition to the TWSCA package:

- numpy
- pandas
- matplotlib
- seaborn (for heatmaps)
- yfinance (for fetching stock data)

You can install them with:

```bash
pip install numpy pandas matplotlib seaborn yfinance
``` 