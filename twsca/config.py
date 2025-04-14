"""
Configuration management for TWSCA package.
"""

from typing import Dict, Any, Optional

# Default configuration
_DEFAULT_CONFIG = {
    'plot_style': 'default',
    'dtw_radius': None,
    'window_size': None,
    'normalize': True,
    'detrend': True,
    'spectral_method': 'magnitude',
    'use_llt': True,
    'llt_sigma': 1.0,
    'llt_alpha': 0.5,
}

# Current configuration
_current_config = _DEFAULT_CONFIG.copy()


def get_config() -> Dict[str, Any]:
    """
    Get current configuration.
    
    Returns
    -------
    Dict[str, Any]
        Current configuration dictionary
    """
    return _current_config.copy()


def set_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set configuration parameters.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration parameters to set
    
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    global _current_config
    
    # Update configuration
    for key, value in config.items():
        if key in _DEFAULT_CONFIG:
            _current_config[key] = value
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return get_config()


def reset_config() -> Dict[str, Any]:
    """
    Reset configuration to default values.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary
    """
    global _current_config
    _current_config = _DEFAULT_CONFIG.copy()
    return get_config() 