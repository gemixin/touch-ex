"""
A module containing a ResNet-18 model for continuous tactile-target regression.

Author: Gemma McLean
Date: August 2026
"""

import torch.nn as nn
from torchvision import models


class ResNet18Regressor(nn.Module):
    """
    An ImageNet-pretrained ResNet-18 wrapper with a single-value regression head.
    """

    feature_dim = 512

    def __init__(self, freeze_backbone=False):
        """
        Initialise the ResNet-18 regressor.

        Args:
            freeze_backbone (bool, optional): Whether to train only the regression head.
                Defaults to False.
        """

        super().__init__()
        # Store whether the pretrained backbone should remain frozen
        self.freeze_backbone = freeze_backbone
        # Load the pretrained ResNet-18 backbone and remove its classification head
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        # Add a single-output regression head
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, 1),
        )

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        """
        Set training mode while keeping a frozen backbone in evaluation mode.

        Args:
            mode (bool): If True, sets the model to training mode. If False, sets
                the model to evaluation mode.

        Returns:
            ResNet18Regressor: The model instance.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images, return_features=False):
        """
        Perform a forward pass through the regressor.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            return_features (bool): If True, return features before the regression head.

        Returns:
            torch.Tensor: A prediction of shape (batch_size), or feature vectors of shape
                (batch_size, feature_dim) when return_features is True.
        """

        # Pass images through the pretrained backbone to obtain feature vectors
        features = self.backbone(images)
        # Return image features for feature visualisation when requested
        if return_features:
            return features
        # Otherwise return one continuous prediction per input image
        return self.regressor(features).squeeze(1)
