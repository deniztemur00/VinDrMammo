import torch
from torch import nn
from config import GMICConfig


class GMIC(nn.Module):
    def __init__(self,config:GMICConfig):
        super(GMIC, self).__init__()
        self.config = config



