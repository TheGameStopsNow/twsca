from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="twsca",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.4.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "seaborn>=0.11.0",
            "yfinance>=0.1.63",
        ],
    },
    author="Dennis Nedry",
    author_email="vines_woofers.0j@icloud.com",
    description="Time-Warped Spectral Correlation Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheGameStopsNow/twsca",
    project_urls={
        "Bug Tracker": "https://github.com/TheGameStopsNow/twsca/issues",
        "Documentation": "https://twsca.readthedocs.io/",
        "Source Code": "https://github.com/TheGameStopsNow/twsca",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="time-series, correlation, spectral-analysis, dtw, financial-analysis, signal-processing",
    python_requires=">=3.8",
)
