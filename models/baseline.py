import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNNModel(nn.Module):
    """
    A simple baseline CNN model for tactile image feature extraction/classification.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, num_classes):
        """
        Initialise the BaselineCNNModel.

        Args:
            num_classes (int): The number of classes to classify.
        """

        super(BaselineCNNModel, self).__init__()

        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output after pool: 16 x 112 x 112

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Output after pool: 32 x 56 x 56

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Output after pool: 64 x 28 x 28
        self.feature_dim = 64 * 28 * 28

        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.classifier = nn.Linear(512, num_classes)

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

        # Pass through convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layer
        features = torch.flatten(x, 1)

        # If return_features is True, return the features before the classifier
        if return_features:
            return features

        # Otherwise, pass through the fully connected layers for classification
        x = F.relu(self.fc1(features))
        x = self.classifier(x)
        return x
