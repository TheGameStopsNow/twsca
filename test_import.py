#!/usr/bin/env python3
"""
Test script to verify that the modules can be imported correctly.
This is used to test installation before running the full test suite.
"""

# Import standard modules
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # First, import the modules directly
    import analysis  # noqa: F401
    import dtw  # noqa: F401
    import spectral  # noqa: F401

    # Now attempt to import specific functions (imported but used for testing)
    from analysis import compute_twsca, compute_twsca_matrix  # noqa: F401
    from dtw import align_series, dtw_distance  # noqa: F401
    from spectral import compute_spectrum, spectral_correlation  # noqa: F401

    print("Individual modules imported successfully!")

    # Test imports of key functions
    print("Key functions available:")
    for func_name in [
        "compute_twsca",
        "compute_twsca_matrix",
        "dtw_distance",
        "align_series",
        "compute_spectrum",
        "spectral_correlation",
    ]:
        print(f"  ✓ {func_name}")

    print("\nTWSCA modules imported successfully!")

except Exception as e:
    print(f"Import error: {e}")
