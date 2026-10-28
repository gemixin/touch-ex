"""Exercise the ResNet-50 wrapper without downloading pretrained weights."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision import models

from models.pretrained_classifier import PretrainedClassifier


class ResNet50Tests(unittest.TestCase):
    def test_resnet50_forward_and_gradients_in_frozen_and_finetuned_modes(self):
        factory = models.resnet50
        for frozen in (False, True):
            with self.subTest(freeze_backbone=frozen), patch(
                'models.pretrained_classifier.models.resnet50',
                side_effect=lambda **kwargs: factory(weights=None),
            ) as build:
                model = PretrainedClassifier('resnet50', num_classes=3, freeze_backbone=frozen)
                build.assert_called_once_with(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.assertEqual(model.feature_dim, 2048)
                self.assertIsInstance(model.backbone.fc, nn.Identity)
                model.train()
                self.assertEqual(model.backbone.training, not frozen)
                self.assertTrue(model.classifier.training)

                images = torch.randn(2, 3, 224, 224)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 3))
                nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                self.assertIsNotNone(model.classifier[1].weight.grad)
                self.assertEqual(any(p.grad is not None for p in model.backbone.parameters()),
                                 not frozen)
                with torch.no_grad():
                    self.assertEqual(tuple(model(images, return_features=True).shape), (2, 2048))


if __name__ == '__main__':
    unittest.main()
