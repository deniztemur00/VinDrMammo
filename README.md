# VinDr-Mammo Object Detection

Deep learning project for automated detection of breast abnormalities in mammogram images using the VinDr-Mammo dataset.

## Key Features

- Faster R-CNN and RetinaNet architecture for multi-class object detection
- Balanced dataset creation through stratified sampling based on:
  - BI-RADS classifications (1-5)
  - Breast density categories (A-D) 
  - Finding categories (Mass, Calcification, Asymmetry, etc.)
- Data pruning to address class imbalance:
  - Reduced overrepresented "No Finding" class from 18k to 2.5k samples
  - Preserved rare finding combinations across train/test splits



## Setup
### Clone the repository
```bash
git clone https://github.com/yourusername/vindr-mammo.git
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Results

TODO: Add results as a notebook
