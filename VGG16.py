"""
VGG16 model file for the shared runner (runner_project.py).
Upload it into the SAME FOLDER as runner_project.py; the runner imports it for "vgg16_pretrained"
(like the Assignment 2 runner imported VGGNet.py). Needs only torch and torchvision.
"""
import torch.nn as nn
from torchvision import models


def get_vgg16(num_classes=2, freeze=True):
    """[Method] ImageNet-pretrained VGG16 with a new num_classes output layer.

    Same logic as the VGG16 branch of your transfer_learning.py:
      1. load torchvision's VGG16 with ImageNet weights
      2. if freeze: freeze the convolution stack (features) and the three dense layers (classifier)
      3. replace layer 6 of the classifier (4096 -> 1000) with a NEW 4096 -> num_classes layer
    The new layer is created AFTER freezing, so it is the only trainable part while frozen.
    The runner trains it first (head only), then unfreezes everything for full fine-tuning.
    VGG16 has no batch-norm layers. Your homework VGGNet.py (a small VGG-style network) is NOT used.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False
    last = model.classifier[6]
    model.classifier[6] = nn.Linear(last.in_features, num_classes)
    return model
