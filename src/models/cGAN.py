import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch import optim

class Critic(nn.Module): # TODO: Take a look when to add the label
    def __init__(self, num_channels, num_feature_maps, img_size, dis_dropout_rate):
        super().__init__()
        self.num_channels = num_channels
        self.num_feature_maps = num_feature_maps
        self.img_size = img_size
        self.dis_dropout_rate = dis_dropout_rate

        self.num_layers = int(np.log(img_size) / np.log(2))
        self.features = [2 ** i for i in range(self.num_layers - 2)]

        # 'Prelayer' - Makes the critic stronger
        self.prelayer = conv3d_same = nn.Conv3d(
            in_channels=num_channels + label_dim,  # Number of input channels + number of label features
            out_channels=num_channels,  # Number of output channels
            kernel_size=3,  # 3x3x3 kernel
            stride=1,  # Stride of 1
            padding=1,  # Padding set to 1 to maintain dimensions
            bias=False  # Whether to use bias or not
        )

        # Initial layer
        self.initial_layer = nn.Conv3d(
            in_channels=num_channels,
            out_channels=self.num_feature_maps,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        # Hidden layers
        layers = []

        for i in range(self.num_layers - 3):
            i_in = num_feature_maps // self.features[i]
            i_out = num_feature_maps // self.features[i+1]
            i_kernel_size = 4
            i_stride = 2
            i_padding = 2

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(i_in, i_out, i_kernel_size, i_stride, i_padding, bias=False),
                    nn.BatchNorm3d(i_out),
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(self.dis_dropout_rate)
                )
            )
        
        # Transforms the List of Layers into ModuleList, so that PyTorch is aware of the modules (optimizes handling)
        self.hidden_layers = nn.ModuleList(layers)

        # Final layer
        self.final_layer = nn.Conv3d(  # 2x2x2 -> 1
            in_channels=num_feature_maps * self.features[-1], out_channels=1,
            kernel_size=4, stride=2, padding=0, bias=False
        )

    def forward(self, z):
        z = nn.LeakyReLU(0.2)(self.prelayer(z))
        z = nn.LeakyReLU(0.2)(self.initial_layer(z))
        for layer in self.hidden_layers:
            z = layer(z)
        z = self.final_layer(z)
        return z

class Generator(nn.Module):
    def __init__(self, num_channels, num_feature_maps, img_size, gen_dropout_rate):
        super().__init__()
        self.num_channels = num_channels
        self.num_feature_maps = num_feature_maps
        self.img_size = img_size
        self.dropout_rate = gen_dropout_rate

        self.num_layers = int(np.log(img_size) / np.log(2))
        self.features = [2 ** i for i in range(self.num_layers - 2)]
        self.features.reverse()

        self.padding = nn.ReflectionPad3d(1)

        # Initial layer
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=2, out_channels=num_feature_maps//self.features[0],
                kernel_size=4, stride=2, padding=4, bias=False
            ), # in_channels=2 --> noise and label
            nn.BatchNorm3d(num_feature_maps//self.features[0])
        )

        # Hidden layers
        layers = []
        for i in range(self.num_layers-3):
            i_in = num_feature_maps // self.features[i]
            i_out = num_feature_maps // self.features[i+1]
            i_kernel_size = 4
            i_stride = 2
            i_padding = 2

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(i_in, i_out, i_kernel_size, i_stride, i_padding, bias=False),
                    nn.BatchNorm3d(i_out),
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(self.dropout_rate)
                )
            )
        
        # Transforms the List of Layers into ModuleList, so that PyTorch is aware of the modules (optimizes handling)
        self.hidden_layers = nn.ModuleList(layers)

        # Final layer
        self.final_layer = nn.ConvTranspose3d(  # 32x32x32 -> 64x64x64, c=1
            in_channels=i_out,  # Corrected in_channels
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            bias=False
        )

    def forward(self, z):
        z = self.padding(z)
        z = self.initial_layer(z)
        z = nn.LeakyReLU(0)(z)
        z = self.padding(z)
        for layer in self.hidden_layers:
            z = nn.LeakyReLU(0)(layer(z))
            z = self.padding(z)
        z = nn.LeakyReLU(0)(self.final_layer(z))
        z = self.padding(z)
        z = torch.sigmoid(z)
        return(z)