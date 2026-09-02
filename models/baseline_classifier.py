import torch.nn as nn


class BaselineCNNClassifier(nn.Module):
    """
    A simple baseline CNN classifier for tactile images.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, num_classes):
        """
        Initialise the BaselineCNNClassifier.

        Args:
            num_classes (int): The number of classes to classify.
        """

        super().__init__()

        # Extract increasingly abstract spatial features from the input image
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Reduce the final feature maps to one feature vector per image
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

        # Classify the pooled features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        """
        Create a two-layer convolutional feature-extraction block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential container of the convolutional block.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

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

        # Extract and pool convolutional features
        x = self.features(x)
        features = self.global_pool(x).flatten(1)

        # If return_features is True, return the features before the classifier
        if return_features:
            return features

        # Otherwise, pass through the classifier for classification
        return self.classifier(features)
