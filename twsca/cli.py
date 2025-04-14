"""
Command-line interface for TWSCA package.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from .analysis import compute_twsca
from .smoothing import llt_filter, savitzky_golay, adaptive_smoothing


def create_sample_data(
    n_points: int = 100,
    noise_level: float = 0.3,
    phase_shift: float = 1.5,
    time_warp: bool = False
) -> tuple:
    """
    Create synthetic time series data for demonstration.
    
    Parameters
    ----------
    n_points : int
        Number of points in the time series
    noise_level : float
        Level of noise to add (standard deviation)
    phase_shift : float
        Phase shift between signals
    time_warp : bool
        Whether to apply time warping
        
    Returns
    -------
    tuple
        (t, signal1, signal2) - time points and two signals
    """
    # Generate time points
    t = np.linspace(0, 10, n_points)
    
    # Generate first signal with noise
    signal1 = np.sin(t) + noise_level * np.random.randn(n_points)
    
    # Generate second signal
    if time_warp:
        # Apply non-linear time warping
        warp_factor = 1.5
        warped_t = t + 0.5 * np.sin(t * warp_factor)
        signal2 = np.sin(warped_t + phase_shift) + noise_level * np.random.randn(n_points)
    else:
        signal2 = np.sin(t + phase_shift) + noise_level * np.random.randn(n_points)
    
    return t, signal1, signal2


def plot_comparison(
    t: np.ndarray,
    signal1: np.ndarray,
    signal2: np.ndarray,
    title: str = "Signal Comparison"
) -> None:
    """
    Plot two signals for comparison.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal1, 'b-', label='Signal 1')
    plt.plot(t, signal2, 'r-', label='Signal 2')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demo_llt_filter(args: argparse.Namespace) -> None:
    """
    Demonstrate LLT filtering.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create noisy data
    t = np.linspace(0, 10, args.points)
    signal = np.sin(t) + args.noise * np.random.randn(args.points)
    
    # Apply LLT filter
    filtered = llt_filter(signal, sigma=args.sigma, alpha=args.alpha)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, 'b-', alpha=0.6, label='Original (Noisy)')
    plt.plot(t, filtered, 'r-', label=f'LLT Filtered (σ={args.sigma}, α={args.alpha})')
    plt.title('LLT Filter Demonstration')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demo_smoothing_comparison(args: argparse.Namespace) -> None:
    """
    Demonstrate and compare different smoothing methods.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create noisy data
    t = np.linspace(0, 10, args.points)
    signal = np.sin(t) + args.noise * np.random.randn(args.points)
    
    # Apply different filters
    filtered_llt = llt_filter(signal, sigma=args.sigma, alpha=args.alpha)
    filtered_sg = savitzky_golay(signal, window_size=args.window, poly_order=2)
    filtered_adaptive = adaptive_smoothing(signal, base_sigma=args.sigma, sensitivity=0.2)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t, signal)
    plt.title('Original (Noisy)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(t, filtered_llt)
    plt.title(f'LLT Filter (σ={args.sigma}, α={args.alpha})')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(t, filtered_sg)
    plt.title(f'Savitzky-Golay (window={args.window})')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(t, filtered_adaptive)
    plt.title('Adaptive Smoothing')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def demo_twsca(args: argparse.Namespace) -> None:
    """
    Demonstrate Time-Warped Spectral Correlation Analysis.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create sample data
    t, signal1, signal2 = create_sample_data(
        n_points=args.points,
        noise_level=args.noise,
        phase_shift=args.phase,
        time_warp=args.warp
    )
    
    # Compute TWSCA with LLT enabled
    result_with_llt = compute_twsca(
        signal1, 
        signal2, 
        use_llt=True, 
        llt_sigma=args.sigma, 
        llt_alpha=args.alpha
    )
    
    # Compute TWSCA with LLT disabled
    result_without_llt = compute_twsca(
        signal1, 
        signal2, 
        use_llt=False
    )
    
    # Apply LLT manually for visualization
    signal1_llt = llt_filter(signal1, sigma=args.sigma, alpha=args.alpha)
    signal2_llt = llt_filter(signal2, sigma=args.sigma, alpha=args.alpha)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot original signals
    axes[0, 0].plot(t, signal1, 'b-', alpha=0.5, label='Signal 1 (noisy)')
    axes[0, 0].plot(t, signal2, 'r-', alpha=0.5, label='Signal 2 (noisy)')
    axes[0, 0].set_title('Original Noisy Signals')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot LLT filtered signals
    axes[0, 1].plot(t, signal1_llt, 'b-', label='Signal 1 (LLT filtered)')
    axes[0, 1].plot(t, signal2_llt, 'r-', label='Signal 2 (LLT filtered)')
    axes[0, 1].set_title('LLT Filtered Signals')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot aligned signals without LLT
    aligned_t = np.arange(len(result_without_llt["aligned_series1"]))
    axes[1, 0].plot(
        aligned_t, 
        result_without_llt["aligned_series1"], 
        'b-', 
        label='Signal 1 (aligned)'
    )
    axes[1, 0].plot(
        aligned_t, 
        result_without_llt["aligned_series2"], 
        'r-', 
        label='Signal 2 (aligned)'
    )
    axes[1, 0].set_title(
        f'Aligned Without LLT (Corr: {result_without_llt["time_domain_correlation"]:.4f})'
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot aligned signals with LLT
    aligned_t = np.arange(len(result_with_llt["aligned_series1"]))
    axes[1, 1].plot(
        aligned_t, 
        result_with_llt["aligned_series1"], 
        'b-', 
        label='Signal 1 (aligned)'
    )
    axes[1, 1].plot(
        aligned_t, 
        result_with_llt["aligned_series2"], 
        'r-', 
        label='Signal 2 (aligned)'
    )
    axes[1, 1].set_title(
        f'Aligned With LLT (Corr: {result_with_llt["time_domain_correlation"]:.4f})'
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Add overall title
    plt.suptitle('Effect of LLT Filtering on TWSCA Analysis', fontsize=16)
    
    # Print correlation results
    print("\nTWSCA Correlation Results:")
    print(f"With LLT filtering (σ={args.sigma}, α={args.alpha}):")
    print(f"  - Time-domain correlation: {result_with_llt['time_domain_correlation']:.4f}")
    print(f"  - Spectral correlation: {result_with_llt['spectral_correlation']:.4f}")
    print(f"Without LLT filtering:")
    print(f"  - Time-domain correlation: {result_without_llt['time_domain_correlation']:.4f}")
    print(f"  - Spectral correlation: {result_without_llt['spectral_correlation']:.4f}")
    
    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()


def main() -> None:
    """
    Main CLI entry point.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="TWSCA - Time-Warped Spectral Correlation Analysis CLI"
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Sub-command to run"
    )
    
    # Parser for LLT filter demo
    parser_llt = subparsers.add_parser(
        "llt",
        help="Demonstrate LLT filtering"
    )
    parser_llt.add_argument(
        "--points", type=int, default=100,
        help="Number of points in the time series"
    )
    parser_llt.add_argument(
        "--noise", type=float, default=0.3,
        help="Noise level to add to the signal"
    )
    parser_llt.add_argument(
        "--sigma", type=float, default=1.0,
        help="Sigma parameter for LLT filter"
    )
    parser_llt.add_argument(
        "--alpha", type=float, default=0.5,
        help="Alpha parameter for LLT filter"
    )
    parser_llt.set_defaults(func=demo_llt_filter)
    
    # Parser for smoothing comparison
    parser_smooth = subparsers.add_parser(
        "smooth",
        help="Compare different smoothing methods"
    )
    parser_smooth.add_argument(
        "--points", type=int, default=100,
        help="Number of points in the time series"
    )
    parser_smooth.add_argument(
        "--noise", type=float, default=0.3,
        help="Noise level to add to the signal"
    )
    parser_smooth.add_argument(
        "--sigma", type=float, default=1.0,
        help="Sigma parameter for LLT filter"
    )
    parser_smooth.add_argument(
        "--alpha", type=float, default=0.5,
        help="Alpha parameter for LLT filter"
    )
    parser_smooth.add_argument(
        "--window", type=int, default=5,
        help="Window size for Savitzky-Golay filter"
    )
    parser_smooth.set_defaults(func=demo_smoothing_comparison)
    
    # Parser for TWSCA demo
    parser_twsca = subparsers.add_parser(
        "twsca",
        help="Demonstrate Time-Warped Spectral Correlation Analysis"
    )
    parser_twsca.add_argument(
        "--points", type=int, default=100,
        help="Number of points in the time series"
    )
    parser_twsca.add_argument(
        "--noise", type=float, default=0.3,
        help="Noise level to add to the signal"
    )
    parser_twsca.add_argument(
        "--phase", type=float, default=1.5,
        help="Phase shift between the two signals"
    )
    parser_twsca.add_argument(
        "--warp", action="store_true",
        help="Apply time warping to the second signal"
    )
    parser_twsca.add_argument(
        "--sigma", type=float, default=1.0,
        help="Sigma parameter for LLT filter"
    )
    parser_twsca.add_argument(
        "--alpha", type=float, default=0.5,
        help="Alpha parameter for LLT filter"
    )
    parser_twsca.set_defaults(func=demo_twsca)
    
    # Parse arguments and run appropriate function
    args = parser.parse_args()
    
    # If no command was provided, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the selected command
    args.func(args)


if __name__ == "__main__":
    main() 