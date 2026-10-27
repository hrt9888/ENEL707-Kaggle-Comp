"""Model definitions for the Task 3-5 notebooks."""
from functools import partial

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 8


def conv_block(cin, cout):
    """3x3 convolution -> batch normalisation -> ReLU."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CustomCNN(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, cin = [], 3
        for i, w in enumerate(widths):
            layers.append(conv_block(cin, w))
            if i >= 2:                                   # deeper blocks get a second conv
                layers.append(conv_block(w, w))
            layers.append(nn.MaxPool2d(2))               # halve the height and width
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_custom_cnn(pretrained=True):                   # `pretrained` is ignored (trained from scratch)
    return CustomCNN()


def build_resnet18(pretrained=True, freeze_early=False):
    m, loaded = None, False
    if pretrained:
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # downloads once (needs internet)
            loaded = True
        except Exception as e:
            print("WARNING: could not load pretrained weights, using random init instead:", e)
    if m is None:
        m = models.resnet18(weights=None)
    if freeze_early:
        if pretrained and not loaded:
            print("WARNING: early layers are frozen but the weights are NOT pretrained")
        for part in (m.conv1, m.bn1, m.layer1, m.layer2):   # freeze the early, high-resolution layers
            for p in part.parameters():
                p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m


MODEL_CONFIGS = {
    "custom_cnn":       dict(build=build_custom_cnn, stats="train",
                             lr=2e-3, weight_decay=5e-4, epochs=12),
    "resnet18_pt":      dict(build=build_resnet18, stats="imagenet",
                             lr=1e-3, weight_decay=1e-4, epochs=6),
    "resnet18_partial": dict(build=partial(build_resnet18, freeze_early=True), stats="imagenet",
                             lr=1e-3, weight_decay=1e-4, epochs=6),
}
