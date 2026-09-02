import torch
import torch.nn as nn
from torchvision import models


class ResNet18ConditionedClassifier(nn.Module):
    """
    An ImageNet-pretrained ResNet-18 tactile classifier conditioned on the ground-truth
    force level.

    Author: Gemma McLean
    Date: September 2026
    """

    def __init__(
        self,
        num_classes,
        force_level_labels,
        freeze_backbone=False,
        fusion_hidden_dim=256,
        fusion_dropout=0.2,
    ):
        """
        Initialise the conditioned classifier.

        Args:
            num_classes (int): The number of object or object-region classes.
            force_level_labels (list): Force-level labels in their encoded class order.
            freeze_backbone (bool, optional): Whether to train only the fusion and
                classification layers. Defaults to False.
            fusion_hidden_dim (int, optional): Dimension of the fused representation.
                Defaults to 256.
            fusion_dropout (float, optional): Dropout probability applied before the
                classification layer. Defaults to 0.2.
        """

        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_dropout = fusion_dropout

        # Load the pretrained ResNet-18 backbone and remove its classification head
        self.feature_dim = 512
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        # Freeze pretrained backbone parameters when training only the fusion and head
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Map encoded force classes to evenly spaced normalised scalar values
        try:
            normalised_force_levels = [
                (float(force_level) - 1.0) / 2.0 for force_level in force_level_labels
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "force_level_labels must contain the numeric labels '1', '2', and '3'."
            ) from error
        if sorted(normalised_force_levels) != [0.0, 0.5, 1.0]:
            raise ValueError(
                "force_level_labels must contain each of '1', '2', and '3' exactly once."
            )
        self.register_buffer(
            "normalised_force_levels",
            torch.tensor(normalised_force_levels, dtype=torch.float32),
        )

        # Fuse the image representation with the normalised force-level scalar
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + 1, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Classify the fused representation
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.fusion_dropout),
            nn.Linear(self.fusion_hidden_dim, num_classes),
        )

    def train(self, mode=True):
        """
        Set training mode while keeping a frozen backbone in evaluation mode.

        Args:
            mode (bool): If True, sets the model to training mode. If False, sets the
                model to evaluation mode.

        Returns:
            ResNet18ConditionedClassifier: The model instance.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images, force_levels, return_features=False):
        """
        Perform a conditioned forward pass using image and force-level inputs.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
            force_levels (torch.Tensor): Encoded force-level class indices of shape
                (batch_size).
            return_features (bool): If True, return the fused features before the
                classifier instead of the final output.

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes), or fused
                features of shape (batch_size, fusion_hidden_dim) when return_features
                is True.
        """

        # Extract an image feature vector for every sample
        image_features = self.backbone(images)

        # Look up each encoded force level's normalised scalar value
        normalised_force = self.normalised_force_levels[force_levels].unsqueeze(1)

        # Concatenate and fuse the image and force representations
        combined_features = torch.cat((image_features, normalised_force), dim=1)
        fused_features = self.fusion(combined_features)

        # If return_features is True, return the conditioned representation
        if return_features:
            return fused_features

        # Otherwise, pass the fused representation through the classifier
        return self.classifier(fused_features)
