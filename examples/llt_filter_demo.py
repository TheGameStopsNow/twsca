#!/usr/bin/env python3
"""
LLT Filter Demo Script

This script demonstrates how to use the LLT filter parameter in the compute_twsca function.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import TWSCA
from twsca import compute_twsca, llt_filter


def create_noisy_signals(n_points=100, noise_level=0.3):
    """
    Create two noisy sine waves with phase shift.
    
    Parameters:
    -----------
    n_points : int
        Number of points in the time series
    noise_level : float
        Level of noise to add (standard deviation)
        
    Returns:
    --------
    tuple
        (t, signal1, signal2) - time points and two signals
    """
    # Generate time points
    t = np.linspace(0, 10, n_points)
    
    # Generate first signal with noise
    signal1 = np.sin(t) + noise_level * np.random.randn(n_points)
    
    # Generate second signal (phase-shifted) with noise
    signal2 = np.sin(t + 1.5) + noise_level * np.random.randn(n_points)
    
    return t, signal1, signal2


def main():
    # Create noisy signals
    t, signal1, signal2 = create_noisy_signals(noise_level=0.3)
    
    # Compute TWSCA with LLT filtering (default)
    result_with_llt = compute_twsca(signal1, signal2, use_llt=True)
    
    # Compute TWSCA without LLT filtering
    result_without_llt = compute_twsca(signal1, signal2, use_llt=False)
    
    # Apply LLT manually for visualization
    signal1_llt = llt_filter(signal1, sigma=1.0, alpha=0.5)
    signal2_llt = llt_filter(signal2, sigma=1.0, alpha=0.5)
    
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
    plt.suptitle(
        'Effect of LLT Filtering on TWSCA Analysis', 
        fontsize=16
    )
    
    # Print correlation results
    print("\nTWSCA Correlation Results:")
    print(f"With LLT filtering (default):")
    print(f"  - Time-domain correlation: {result_with_llt['time_domain_correlation']:.4f}")
    print(f"  - Spectral correlation: {result_with_llt['spectral_correlation']:.4f}")
    print(f"Without LLT filtering:")
    print(f"  - Time-domain correlation: {result_without_llt['time_domain_correlation']:.4f}")
    print(f"  - Spectral correlation: {result_without_llt['spectral_correlation']:.4f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()


if __name__ == "__main__":
    main() 