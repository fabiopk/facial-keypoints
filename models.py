# TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = self._create_conv_layer_pool(
            in_channels=1, out_channels=16)
        self.conv2 = self._create_conv_layer_pool(
            in_channels=16, out_channels=32)
        self.conv3 = self._create_conv_layer_pool(
            in_channels=32, out_channels=64)
        self.conv4 = self._create_conv_layer_pool(
            in_channels=64, out_channels=64)
        self.conv5 = self._create_conv_layer_pool(
            in_channels=64, out_channels=64)

        in_features = 7*7*64

        self.linear1 = self._create_linear_layer(in_features, 2048, p=0.7)
        self.linear2 = self._create_linear_layer(2048, 1024, p=0.7)
        self.linear3 = self._create_linear_layer(1024, 68*2, p=0)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Get batch_size
        batch_size = x.shape[0]

        # CNN Layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(batch_size, -1)

        # Linear Layers
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def _create_linear_layer(self, in_features, out_features, p=0.6):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=p)
        )

    def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), pool=(2, 2)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )
