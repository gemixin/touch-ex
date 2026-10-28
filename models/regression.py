import torch.nn as nn
from models.pretrained_classifier import get_pretrained_backbone


class PretrainedRegressor(nn.Module):
    """
    A configurable pretrained encoder with a single-value regression head.

    Author: Gemma McLean
    Date: August 2026
    """

    def __init__(self, model_type="resnet18", freeze_backbone=False):
        """
        Initialise the regressor with the selected pretrained backbone.

        Args:
            model_type (str): Backbone name supported by PretrainedClassifier.
            freeze_backbone (bool, optional): Whether to train only the regression head.
                Defaults to False.
        """

        super().__init__()
        self.freeze_backbone = freeze_backbone

        self.backbone, self.feature_dim = get_pretrained_backbone(model_type)

        # Set the regressor
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, 1),
        )

        # Freeze pretrained backbone parameters when training only the regression head
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
            PretrainedRegressor: The model instance.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images, return_features=False):
        """
        Perform a forward pass through the model.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            return_features (bool): If True, return features before the regression head.

        Returns:
            torch.Tensor: A prediction of shape (batch_size), or feature vectors of shape
                (batch_size, feature_dim) when return_features is True.
        """

        # Pass through the backbone to get features
        features = self.backbone(images)

        # If return_features is True, return the features before the regressor
        if return_features:
            return features

        # Otherwise, pass through the regressor for regression
        return self.regressor(features).squeeze(1)
