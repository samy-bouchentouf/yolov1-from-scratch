"""YOLOv1 model architecture implemented in PyTorch.

Defines the convolutional neural network and fully connected layers
used to predict bounding boxes, object confidence scores, and class
probabilities for each grid cell.
"""

import torch
from torch import nn

# input_shape = (N, 3, 448, 448)
# output_shape = (N, 7, 7, 30)

class YOLOv1(nn.Module):

    def __init__(self):
        super().__init__()

        # first block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # second block
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)

        # third block
        self.conv3_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # fourth block
        self.conv4_1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        # fifth block
        self.conv5_1_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_2_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_1_2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_2_2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)

        # sixth block
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # conn layers
        self.conn_layer1 = nn.Linear(7*7*1024, 4096)
        self.conn_layer2 = nn.Linear(4096, 7*7*30)

        # activation
        self.activation = nn.LeakyReLU(0.1)

        # pooling
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # dropout
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):

        # first block
        x = self.activation(self.conv1(x))
        x = self.pooling(x)

        # second block
        x = self.activation(self.conv2(x))
        x = self.pooling(x)

        # third block
        x = self.activation(self.conv3_1(x))
        x = self.activation(self.conv3_2(x))
        x = self.activation(self.conv3_3(x))
        x = self.activation(self.conv3_4(x))
        x = self.pooling(x)

        # fourth block
        x = self.activation(self.conv4_1_1(x))
        x = self.activation(self.conv4_2_1(x))
        x = self.activation(self.conv4_1_2(x))
        x = self.activation(self.conv4_2_2(x))
        x = self.activation(self.conv4_1_3(x))
        x = self.activation(self.conv4_2_3(x))
        x = self.activation(self.conv4_1_4(x))
        x = self.activation(self.conv4_2_4(x))
        x = self.activation(self.conv4_3(x))
        x = self.activation(self.conv4_4(x))
        x = self.pooling(x)

        # fifth block
        x = self.activation(self.conv5_1_1(x))
        x = self.activation(self.conv5_2_1(x))
        x = self.activation(self.conv5_1_2(x))
        x = self.activation(self.conv5_2_2(x))
        x = self.activation(self.conv5_3(x))
        x = self.activation(self.conv5_4(x))

        # sixth block
        x = self.activation(self.conv6_1(x))
        x = self.activation(self.conv6_2(x))

        # conn layers
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.conn_layer1(x))
        x = self.dropout(x)
        x = self.conn_layer2(x)
        x = x.reshape(-1, 7, 7, 30)

        return x