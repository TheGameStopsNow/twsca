#!/usr/bin/env python3
"""
Meme Stock Comparison Script

This script demonstrates the use of Time-Warped Spectral Correlation Analysis (TWSCA)
to detect hidden correlations between different meme stocks. It compares movement patterns
across specified meme stocks and generates visualizations showing their relationships.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from twsca import compute_twsca

# Define common meme stocks
DEFAULT_SYMBOLS = ["GME", "U", "CHWY", "IHRT", "SIRI", "BYON"]


def fetch_stock_data(symbols, start_date=None, end_date=None, interval="1d"):
    """
    Fetch historical stock data for the given symbols

    Parameters:
    -----------
    symbols : list
        List of stock symbols to fetch
    start_date : str or datetime, optional
        Start date for data retrieval
    end_date : str or datetime, optional
        End date for data retrieval
    interval : str, optional
        Data interval ('1d', '1h', etc.)

    Returns:
    --------
    dict
        Dictionary of dataframes with stock data, keyed by symbol
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()

    stock_data = {}

    for symbol in symbols:
        try:
            data = yf.download(
                symbol, start=start_date, end=end_date, interval=interval
            )
            if len(data) > 20:  # Ensure we have enough data
                stock_data[symbol] = data
                print(f"Downloaded {len(data)} rows for {symbol}")
            else:
                print(f"Warning: Insufficient data for {symbol} ({len(data)} rows)")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return stock_data


def normalize_series(series):
    """
    Normalize a time series to zero mean and unit standard deviation

    Parameters:
    -----------
    series : array-like
        Input time series

    Returns:
    --------
    numpy.ndarray
        Normalized time series
    """
    return (series - np.mean(series)) / np.std(series)


def calculate_correlation_matrix(stock_data, use_twsca=True):
    """
    Calculate correlation matrix for the given stock data

    Parameters:
    -----------
    stock_data : dict
        Dictionary of dataframes with stock data, keyed by symbol
    use_twsca : bool, optional
        Whether to use TWSCA (True) or standard correlation (False)

    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    symbols = list(stock_data.keys())
    n_symbols = len(symbols)

    # Initialize correlation matrix
    corr_matrix = np.zeros((n_symbols, n_symbols))

    for i in range(n_symbols):
        for j in range(i, n_symbols):
            sym1 = symbols[i]
            sym2 = symbols[j]

            # Extract closing prices
            series1 = stock_data[sym1]["Close"]
            series2 = stock_data[sym2]["Close"]

            # Ensure same length by finding the period with data for both stocks
            common_index = series1.index.intersection(series2.index)
            if len(common_index) < 20:
                print(f"Warning: Insufficient common data between {sym1} and {sym2}")
                corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                continue

            # Resample to common dates
            clean_series1 = series1.loc[common_index]
            clean_series2 = series2.loc[common_index]

            # Normalize
            norm_series1 = normalize_series(clean_series1.values)
            norm_series2 = normalize_series(clean_series2.values)

            if use_twsca:
                # Use TWSCA for correlation
                try:
                    result = compute_twsca(norm_series1, norm_series2)
                    # Use spectral correlation as it captures hidden relationships better
                    correlation = result["spectral_correlation"]
                except Exception as e:
                    print(f"Error computing TWSCA between {sym1} and {sym2}: {e}")
                    correlation = np.nan
            else:
                # Use standard Pearson correlation
                correlation = np.corrcoef(norm_series1, norm_series2)[0, 1]

            # Store in matrix (symmetric)
            corr_matrix[i, j] = corr_matrix[j, i] = correlation

    # Create DataFrame for better visualization
    corr_df = pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
    return corr_df


def plot_correlation_heatmap(corr_matrix, title, output_file=None):
    """
    Create a heatmap visualization for the correlation matrix

    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    title : str
        Plot title
    output_file : str, optional
        File path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Correlation coefficient"},
        annot=True,
        fmt=".2f",
    )

    plt.title(title, fontsize=16)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_file}")

    plt.show()


def create_detailed_comparison(stock_data, symbol1, symbol2, output_file=None):
    """
    Create a detailed comparison between two stocks

    Parameters:
    -----------
    stock_data : dict
        Dictionary of dataframes with stock data, keyed by symbol
    symbol1 : str
        First stock symbol
    symbol2 : str
        Second stock symbol
    output_file : str, optional
        File path to save the plot
    """
    # Extract closing prices
    series1 = stock_data[symbol1]["Close"]
    series2 = stock_data[symbol2]["Close"]

    # Ensure same length by finding the period with data for both stocks
    common_index = series1.index.intersection(series2.index)
    if len(common_index) < 20:
        print(f"Warning: Insufficient common data between {symbol1} and {symbol2}")
        return

    # Resample to common dates
    clean_series1 = series1.loc[common_index]
    clean_series2 = series2.loc[common_index]

    # Normalize
    norm_series1 = normalize_series(clean_series1.values)
    norm_series2 = normalize_series(clean_series2.values)

    # Calculate standard correlation
    std_corr = np.corrcoef(norm_series1, norm_series2)[0, 1]

    # Calculate TWSCA
    twsca_result = compute_twsca(norm_series1, norm_series2)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original price series
    axes[0, 0].plot(clean_series1.index, clean_series1, label=symbol1)
    axes[0, 0].plot(clean_series2.index, clean_series2, label=symbol2)
    axes[0, 0].set_title(f"Original Price Series")
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].grid(True)

    # Normalized series
    axes[0, 1].plot(clean_series1.index, norm_series1, label=f"{symbol1} (normalized)")
    axes[0, 1].plot(clean_series2.index, norm_series2, label=f"{symbol2} (normalized)")
    axes[0, 1].set_title(f"Normalized Series (Pearson Corr: {std_corr:.4f})")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Aligned series
    axes[1, 0].plot(
        range(len(twsca_result["aligned_series1"])),
        twsca_result["aligned_series1"],
        label=f"{symbol1} (aligned)",
    )
    axes[1, 0].plot(
        range(len(twsca_result["aligned_series2"])),
        twsca_result["aligned_series2"],
        label=f"{symbol2} (aligned)",
    )
    axes[1, 0].set_title(
        f'Time-Warped Aligned Series (DTW Corr: {twsca_result["time_domain_correlation"]:.4f})'
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Scatter plot of correlation
    axes[1, 1].scatter(norm_series1, norm_series2, alpha=0.7, label="Standard")
    axes[1, 1].scatter(
        twsca_result["aligned_series1"],
        twsca_result["aligned_series2"],
        alpha=0.7,
        label="After Time-Warping",
    )
    axes[1, 1].set_title(
        f'Correlation Scatter (Spectral Corr: {twsca_result["spectral_correlation"]:.4f})'
    )
    axes[1, 1].set_xlabel(symbol1)
    axes[1, 1].set_ylabel(symbol2)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f"Detailed Comparison: {symbol1} vs {symbol2}", fontsize=16)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved detailed comparison to {output_file}")

    plt.show()

    # Print analysis summary
    print("\nAnalysis Summary:")
    print(f"Standard correlation: {std_corr:.4f}")
    print(
        f"Time-domain correlation (DTW): {twsca_result['time_domain_correlation']:.4f}"
    )
    print(f"Spectral correlation: {twsca_result['spectral_correlation']:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if twsca_result["spectral_correlation"] > 0.7:
        print("The stocks show strong hidden correlation patterns!")
    elif twsca_result["spectral_correlation"] > 0.4:
        print("The stocks show moderate hidden correlation patterns.")
    else:
        print("The stocks do not show significant hidden correlation patterns.")

    if twsca_result["spectral_correlation"] > std_corr + 0.2:
        print(
            "TWSCA reveals much stronger correlations than standard methods, "
            "suggesting hidden relationships."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between meme stocks using TWSCA"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Stock symbols to analyze"
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days of historical data to use"
    )
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1h, etc.)")
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Also create standard correlation heatmap for comparison",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to save output files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Analyzing meme stocks: {', '.join(args.symbols)}")
    print(f"Using {args.days} days of historical data at {args.interval} interval")

    # Prepare dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Fetch data
    stock_data = fetch_stock_data(args.symbols, start_date, end_date, args.interval)

    if len(stock_data) < 2:
        print("Error: Need at least 2 stocks with sufficient data for comparison")
        return

    # Calculate and plot TWSCA correlation matrix
    twsca_corr = calculate_correlation_matrix(stock_data, use_twsca=True)
    plot_correlation_heatmap(
        twsca_corr,
        f"Time-Warped Spectral Correlation Analysis: Meme Stocks ({args.interval})",
        os.path.join(args.output_dir, f"meme_stock_twsca_correlation.png"),
    )

    # Optionally calculate and plot standard correlation matrix
    if args.standard:
        std_corr = calculate_correlation_matrix(stock_data, use_twsca=False)
        plot_correlation_heatmap(
            std_corr,
            f"Standard Correlation Analysis: Meme Stocks ({args.interval})",
            os.path.join(args.output_dir, f"meme_stock_standard_correlation.png"),
        )

    # Create detailed comparisons for pairs with strong correlations
    symbols = list(stock_data.keys())
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym1 = symbols[i]
            sym2 = symbols[j]

            # Check if correlation is significant (above 0.4)
            if not np.isnan(twsca_corr.iloc[i, j]) and twsca_corr.iloc[i, j] > 0.4:
                print(f"\nGenerating detailed comparison for {sym1} vs {sym2}...")
                create_detailed_comparison(
                    stock_data,
                    sym1,
                    sym2,
                    os.path.join(args.output_dir, f"{sym1}_vs_{sym2}_comparison.png"),
                )


if __name__ == "__main__":
    main()
