"""
Defines the convolutional neural network sub-models, loss functions and metrics
   * BaseResNet: Modified ResNet18 to accommodate for different number of
   channels
   * ResNetRegression: Wrapper class for a BaseResNet regressor
   * ResNetClassification: Wrapper class for a BaseResNet classifier
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class BaseResNet(nn.Module):
    """
    Define the modified ResNet18 model
    """
    def __init__(self, no_channels=3, out_features=1000):
        super(BaseResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Modify the input layer to accommodate for res and channels
        self.model.conv1 = nn.Conv2d(
            in_channels=no_channels, out_channels=64, kernel_size=7,
            stride=2, padding=3, bias=False)

        # Modify the feature size in the final output layer
        self.model.fc = nn.Linear(out_features=out_features)

    def forward(self, x):
        return self.model(x)


class ResNetRegression(nn.Module):
    """
    Define the wrapper model to train BaseResNet with regression as the
    final layer
    """
    def __init__(self, no_channels=3, out_features=1000):
        super(ResNetRegression, self).__init__()
        self.model = BaseResNet(no_channels, out_features)
        self.final_fc = nn.Linear(
            in_features=out_features, out_features=1, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.final_fc(x)
        return x


class ResNetClassifier(nn.Module):
    """
    Define the wrapper model to train BaseResNet as a classifier
    """
    def __init__(self, no_channels=3, out_features=1000, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.model = BaseResNet(no_channels, out_features)
        self.final_fc = nn.Linear(
            in_features=out_features, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.final_fc(x)
        return x


def loss_fn_regression(outputs, labels):
    """
    Compute the MSE loss given outputs and labels
    """
    loss = nn.MSELoss()
    return loss(outputs, labels)


def loss_fn_classification(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels
    """
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy of model outputs given example labels
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


metrics_regression = {
}

metrics_classification = {
    'accuracy': accuracy
}
