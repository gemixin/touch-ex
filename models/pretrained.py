import torch.nn as nn
import timm
from torchvision import models
from models.t3 import T3TinyBackbone


# Pretrained deit model checkpoint path
DEIT_TINY_CHECKPOINT = "timm/deit_tiny_patch16_224.fb_in1k"


class PretrainedModel(nn.Module):
    """
    A pretrained model wrapper with a configurable backbone and classifier.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, model_type, num_classes, freeze_backbone=False):
        """
        Initialise the PretrainedModel.

        Args:
            model_type (str): The type of the pretrained model to use. Options are
                'resnet18', 'efficientnet_b0', 'vit_b_16', 'deit_tiny', or 't3_tiny'.
            num_classes (int): The number of classes to classify.
            freeze_backbone (bool, optional): Whether to train only the classifier.
                Defaults to False.
        """

        super(PretrainedModel, self).__init__()
        self.freeze_backbone = freeze_backbone

        # Depending on the model_type, load the appropriate pretrained model and modify
        # it to output features instead of class predictions
        if model_type == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_type == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity()

        elif model_type == "vit_b_16":
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()

        elif model_type == "deit_tiny":
            self.backbone = timm.create_model(
                DEIT_TINY_CHECKPOINT, pretrained=True, num_classes=0
            )
            self.feature_dim = self.backbone.num_features

        elif model_type == "t3_tiny":
            self.backbone = T3TinyBackbone()
            self.feature_dim = self.backbone.feature_dim

        else:
            raise ValueError(f"Invalid model type: {model_type}.")

        # Set the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Freeze pretrained backbone parameters when training only the classifier
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
            self: The model instance.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, return_features=False):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            return_features (bool): If True, return the features before the classifier
                instead of the final output.

        Returns:
            torch.Tensor: If return_features is False, returns the output of the
                classifier (batch_size, num_classes). If return_features is True,
                returns the features before the classifier (batch_size, feature_dim).
        """

        # Pass through the backbone to get features
        features = self.backbone(x)

        # If return_features is True, return the features before the classifier
        if return_features:
            return features

        # Otherwise, pass through the classifier for classification
        return self.classifier(features)
