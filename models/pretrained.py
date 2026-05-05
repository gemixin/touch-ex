import torch.nn as nn
from torchvision import models


class PretrainedModel(nn.Module):
    """
    A pretrained model wrapper with a configurable backbone and classifier.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, model_type, num_classes):
        """
        Initialise the PretrainedModel.

        Args:
            model_type (str): The type of the pretrained model to use. Options are
            'resnet18', 'resnet50', and 'vit'.
            num_classes (int): The number of classes to classify.
        """

        super(PretrainedModel, self).__init__()

        # Depending on the model_type, load the appropriate pretrained model and modify
        # it to output features instead of class predictions
        if model_type == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the original fully connected layer

        elif model_type == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the original fully connected layer

        elif model_type == "vit":
            self.backbone = models.vit.b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()  # Remove the original fully connected layer

        else:
            raise ValueError(f"Invalid model type: {model_type}.")

        # Set the classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

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
