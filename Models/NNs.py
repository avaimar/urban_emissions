"""
Defines the concatenated neural network model
   * ConcatNet: Modified ResNet18 to accommodate for different number of
   channels
"""

import numpy as np
import torch
import torch.nn as nn


class ConcatNet(nn.Module):
    """
    Define the model to train the concatenated features.
    """

    def __init__(self, feat_size=1024, out_size=1):
        super(ConcatNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=feat_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=out_size)
        )

    def forward(self, x):
        return self.model(x)
