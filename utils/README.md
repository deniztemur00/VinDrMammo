# Mammography Data Processing and Analysis

This folder contains code for processing and analyzing mammography data, with a focus on handling class imbalance and preparing datasets for training detection models.

## Key Files

### data_pruning.ipynb
Main notebook for data preprocessing and balancing that:
- Handles severe class imbalance in mammography findings
- Reduces "No Finding" samples from 18k+ to ~2500 samples
- Maintains BIRADS and density distribution
- Ensures proper train/test split representation

Key strategies:
1. Filters BIRADS scores
   - Keeps all samples with BIRADS > 2
   - Selectively samples from BIRADS ≤ 2 cases
2. Balanced sampling across:
   - BIRADS scores (1-5)
   - Breast density categories (A-D)
3. Maintains representation of rare findings in both training and test sets

## Data Distribution Overview

The dataset contains mammography findings with the following characteristics:

Key finding categories:
- No Finding (~2500 samples after pruning)  
- Mass (~1100 samples)
- Suspicious Calcification (~400 samples)
- Focal Asymmetry (~230 samples)
- Architectural Distortion (~95 samples)
- Other findings (varying sample sizes)

