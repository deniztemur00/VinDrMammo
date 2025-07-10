import numpy as np
import torch
from torch import nn

# import pyradiomics


class RadiomicsExtractor(nn.Module):
    """Radiomics feature extractor using pyradiomics.
    pip install is broken, i will implement when it is fixed.

    """

    def __init__(self, num_features=10):
        super(RadiomicsExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.compile
    def _compute_radiomics_features(self, image: np.ndarray):
        """I need to compute radiomics features here to leverage cuda computing
        of numpy arrays.
        """
        ...

    def forward(self, x):
        # Flatten the input tensor to (N, C*H*W)
        x_flat = x.view(x.size(0), -1)

        with torch.device(self.device):
            out = self._compute_radiomics_features(...)

        return out
