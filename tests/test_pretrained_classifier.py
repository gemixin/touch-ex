"""Exercise the ResNet-50 wrapper without downloading pretrained weights."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision import models

from models.pretrained_classifier import PretrainedClassifier
from models.conditioned import ConditionedClassifier
from models.regression import PretrainedRegressor


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

    def test_selected_backbones_work_with_regression_and_conditioning(self):
        factory = models.resnet50
        for frozen in (False, True):
            for task in ('regression', 'conditioned'):
                with self.subTest(task=task, frozen=frozen), patch(
                    'models.pretrained_classifier.models.resnet50',
                    side_effect=lambda **kwargs: factory(weights=None),
                ) as build:
                    if task == 'regression':
                        model = PretrainedRegressor('resnet50', freeze_backbone=frozen)
                        args = (torch.randn(2, 3, 224, 224),)
                        expected = (2,)
                    else:
                        model = ConditionedClassifier(
                            3, ['3', '1', '2'], model_type='resnet50',
                            freeze_backbone=frozen, fusion_hidden_dim=8,
                        )
                        args = (torch.randn(2, 3, 224, 224), torch.tensor([0, 2]))
                        expected = (2, 3)
                    build.assert_called_once_with(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                    self.assertEqual(model.feature_dim, 2048)
                    model.train()
                    self.assertEqual(model.backbone.training, not frozen)
                    outputs = model(*args)
                    self.assertEqual(tuple(outputs.shape), expected)
                    outputs.sum().backward()
                    self.assertEqual(any(p.grad is not None for p in model.backbone.parameters()),
                                     not frozen)
                    head = model.regressor if task == 'regression' else model.classifier
                    self.assertIsNotNone(head[-1].weight.grad)


if __name__ == '__main__':
    unittest.main()
