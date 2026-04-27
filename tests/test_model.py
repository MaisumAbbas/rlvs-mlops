import pytest
import torch
import torch.nn as nn
from torchvision import models

class ResNetWithHidden(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        old_conv = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            6, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False
        )
    def forward(self, x):
        return self.base(x)

def test_model_output_shape():
    """Model output shape sahi honi chahiye"""
    model = ResNetWithHidden(num_classes=2)
    model.eval()
    # 6 channel input (3 motion + 3 RGB)
    dummy_input = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2), f"Expected (1,2) got {output.shape}"

def test_model_probability_sum():
    """Softmax probabilities ka sum 1 hona chahiye"""
    import torch.nn.functional as F
    model = ResNetWithHidden(num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        probs  = F.softmax(output, dim=1)
    total = probs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"Probs sum should be 1, got {total}"

def test_model_classes():
    """Model sirf 2 classes predict kare"""
    import torch.nn.functional as F
    model = ResNetWithHidden(num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        output     = model(dummy_input)
        probs      = F.softmax(output, dim=1)
        prediction = probs.argmax(dim=1).item()
    assert prediction in [0, 1], f"Prediction must be 0 or 1, got {prediction}"

def test_6channel_input():
    """6 channel input accept hona chahiye"""
    model = ResNetWithHidden(num_classes=2)
    first_layer_channels = model.base.conv1.in_channels
    assert first_layer_channels == 6, \
        f"Expected 6 input channels, got {first_layer_channels}"

def test_batch_processing():
    """Batch processing kaam kare"""
    model = ResNetWithHidden(num_classes=2)
    model.eval()
    batch = torch.randn(4, 6, 224, 224)
    with torch.no_grad():
        output = model(batch)
    assert output.shape[0] == 4, "Batch size mismatch"