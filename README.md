# Natural Products Analysis

A Python toolkit for analyzing chemical and taxonomic relationships in natural products data.

## Overview

This project provides tools to analyze the relationship between chemical structures and taxonomic classifications of natural products. It focuses on:

- Computing chemical distances between molecules using various embeddings
- Analyzing taxonomic relationships across different levels
- Statistical analysis of chemical-taxonomic correlations
- Visualization of molecular pairs and statistical distributions

## Core Components

### utils.py
Core utility functions for:
- Loading and processing chemical data
- Computing chemical distances between molecules
- Analyzing taxonomic relationships
- Visualizing results
- Statistical analysis

### p_value_histograms.py
Statistical analysis tools for:
- Generating p-value distributions
- Visualizing statistical results
- Analyzing chemical distances across taxonomic levels

### run_all.py
Main execution script that:
- Loads data and configurations
- Runs analysis pipeline
- Generates visualizations
- Saves results

## Usage

### Basic Usage

```bash
python run_all.py --data_folder ./data --encoding chemformer
```

### Parameters

- `--data_folder`: Path to data directory (default: './data')
- `--encoding`: ML Encoding type ('chemformer' or 'smitrans')
- `--upper_limit_ref_size`: Maximum reference species size (default: 1000)
- `--lower_limit_ref_size`: Minimum reference species size (default: 180)

### Data Requirements

Input data should include:
- SMILES representations of molecules
- Taxonomic classifications
- Chemical embeddings (optional)

## Dependencies

- RDKit
- NumPy
- Pandas
- Matplotlib
- SciPy
- mpmath

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install rdkit numpy pandas matplotlib scipy mpmath
```

## Project Structure

```
NaturalProducts/
├── utils.py           # Core utility functions
├── p_value_histograms.py  # Statistical analysis
├── run_all.py        # Main execution script
├── data/             # Data directory
└── README.md         # Documentation
```

## Output

The analysis generates:
- Statistical data files
- Visualization plots
- Molecule pair images
- P-value distributions
