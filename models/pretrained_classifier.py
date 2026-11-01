import torch.nn as nn
import timm
from torchvision import models

from models.t3 import T3_REPOSITORY, T3_REVISION, T3TinyBackbone


# Pretrained timm model checkpoint paths
VIT_SMALL_CHECKPOINT = "timm/vit_small_patch16_224.augreg_in1k"
DEIT_TINY_CHECKPOINT = "timm/deit_tiny_patch16_224.fb_in1k"

# Pretrained torchvision weights used to load each supported backbone
TORCHVISION_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
}

# Record resolved weight names rather than DEFAULT aliases in experiment results
# The baseline is trained from scratch and has no pretrained checkpoint
PRETRAINED_CHECKPOINTS = {
    "baseline": None,
    **{
        model_type: f"torchvision/{type(weights).__name__}.{weights.name}"
        for model_type, weights in TORCHVISION_WEIGHTS.items()
    },
    "vit_s_16": VIT_SMALL_CHECKPOINT,
    "deit_tiny": DEIT_TINY_CHECKPOINT,
    "t3_tiny": f"{T3_REPOSITORY}@{T3_REVISION}",
}


def get_pretrained_backbone(model_type):
    """
    Load the selected pretrained model and remove its original classification head.
    Return the feature-extraction backbone and its output feature dimension.

    Args:
        model_type (str): The type of pretrained backbone to use. Options are
            'resnet18', 'resnet50', 'efficientnet_b0', 'vit_s_16', 'deit_tiny',
            or 't3_tiny'.

    Returns:
        tuple: A tuple containing the pretrained backbone and its output feature dimension.
    """

    # Load the selected pretrained model and remove its original classification head
    if model_type == "resnet18":
        backbone = models.resnet18(weights=TORCHVISION_WEIGHTS[model_type])
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "resnet50":
        backbone = models.resnet50(weights=TORCHVISION_WEIGHTS[model_type])
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=TORCHVISION_WEIGHTS[model_type])
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()

    elif model_type == "vit_s_16":
        backbone = timm.create_model(VIT_SMALL_CHECKPOINT, pretrained=True, num_classes=0)
        feature_dim = backbone.num_features

    elif model_type == "deit_tiny":
        backbone = timm.create_model(DEIT_TINY_CHECKPOINT, pretrained=True, num_classes=0)
        feature_dim = backbone.num_features

    elif model_type == "t3_tiny":
        backbone = T3TinyBackbone()
        feature_dim = backbone.feature_dim

    else:
        raise ValueError(f"Invalid model type: {model_type}.")

    # Return the backbone and the number of features supplied to the task-specific head
    return backbone, feature_dim


class PretrainedClassifier(nn.Module):
    """
    A tactile classifier with a configurable pretrained feature-extraction backbone.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, model_type, num_classes, freeze_backbone=False):
        """
        Initialise the PretrainedClassifier.

        Args:
            model_type (str): The type of pretrained backbone to use. Options are
                'resnet18', 'resnet50', 'efficientnet_b0', 'vit_s_16', 'deit_tiny',
                or 't3_tiny'.
            num_classes (int): The number of classes to classify.
            freeze_backbone (bool, optional): Whether to train only the classifier.
                Defaults to False.
        """

        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load the selected pretrained model without its original classification head
        self.backbone, self.feature_dim = get_pretrained_backbone(model_type)

        # Set the task-specific classifier
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
            mode (bool): If True, sets the model to training mode. If False, sets the
                model to evaluation mode.

        Returns:
            PretrainedClassifier: The classifier instance.
        """

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, return_features=False):
        """
        Perform a forward pass through the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
            return_features (bool): If True, return the features before the classifier
                instead of the final output.

        Returns:
            torch.Tensor: If return_features is False, returns the output of the
                classifier with shape (batch_size, num_classes). If return_features is
                True, returns the features with shape (batch_size, feature_dim).
        """

        # Pass through the backbone to get features
        features = self.backbone(x)

        # If return_features is True, return the features before the classifier
        if return_features:
            return features

        # Otherwise, pass through the classifier for classification
        return self.classifier(features)
