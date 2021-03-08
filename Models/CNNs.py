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

    def __init__(self, no_channels=3, p=0.5, add_block=False, num_frozen=0):
        super(BaseResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Modify the input layer to accommodate for res and channels if
        # training satellite imagery
        if no_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels=no_channels, out_channels=64, kernel_size=7,
                stride=2, padding=3, bias=False)

        # Additional FC -> DO block if selected (in between resnet and
        # final layer)
        self.add_block = add_block
        if self.add_block or no_channels == 3: # TODO Note: this second part to the if statement should be removed.
            # TODO: if the street model is ever retrained.
            self.additional_block = nn.Sequential(
                nn.Linear(in_features=1000, out_features=1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p),
                nn.Linear(in_features=1000, out_features=1000)
            )

        # Add final FC + Dropout layer
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(in_features=512, out_features=512)
        )

        # Freeze initial num_frozen layers. We only freeze layers from the
        # resnet model with pretrained weights.
        # The resnet model has 62 elements with trainable weights. Note that
        # each element is treated separately: i.e., a layer having weights and
        # biases is treated as two separate elements.
        assert(62 >= num_frozen >= 0)
        counter = 0
        for name, param in self.resnet.named_parameters():
            if counter < num_frozen:
                param.requires_grad = False
            counter += 1

    def forward(self, x):
        x = self.resnet(x)

        # Add extra FC -> DO layer
        if self.add_block:
            x = self.additional_block(x)

        x = self.final_layers(x)
        return x


class ResNetRegression(nn.Module):
    """
    Define the wrapper model to train BaseResNet with regression as the
    final layer
    """

    def __init__(self, no_channels=3, p=0.5, add_block=False, num_frozen=0):
        super(ResNetRegression, self).__init__()
        self.model = BaseResNet(no_channels, p, add_block=add_block,
                                num_frozen=num_frozen)
        self.model.final_layers[3] = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        return self.model(x)


class ResNetClassifier(nn.Module):
    """
    Define the wrapper model to train BaseResNet as a classifier
    """

    def __init__(self, no_channels=3, num_classes=3, p=0.5, add_block=False,
                 num_frozen=0):
        super(ResNetClassifier, self).__init__()
        self.model = BaseResNet(no_channels, p, add_block=add_block,
                                num_frozen=num_frozen)
        self.model.final_layers[3] = nn.Linear(
            in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.model(x)


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


def rmse(outputs, labels):
    """
    Compute the root mean square error of model outputs given example labels
    :param outputs: (np.array)
    :param labels: (np.array)
    :return:
    """
    return np.sqrt(np.sum(np.square(outputs - labels)))


metrics_regression = {
    'RMSE': rmse
}

metrics_classification = {
    'accuracy': accuracy
}
