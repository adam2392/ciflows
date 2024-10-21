from typing import List

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from normflows.flows.affine import AffineCouplingBlock
from normflows.flows.base import Flow
from normflows.flows.normalization import ActNorm

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)))
        # Decoding path
        d1 = self.decoder1(e2)
        d2 = self.decoder2(d1 + e1)  # Skip connection
        return d2
    

def affine_transform(x_B, t, s):
    return x_B * torch.exp(s) + t  # Element-wise operation

def inverse_affine_transform(y_B, t, s):
    return (y_B - t) * torch.exp(-s)  # Inverse affine transformation

class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels):
        super(AffineCouplingLayer, self).__init__()
        self.unet = UNet(in_channels // 2, 2 * (in_channels // 2))  # U-Net to output t and s

    def forward(self, x):
        x_A, x_B = torch.chunk(x, 2, dim=1)
        t_s = self.unet(x_A)  # Use U-Net to get t and s
        t, s = torch.chunk(t_s, 2, dim=1)
        y_B = affine_transform(x_B, t, s)
        return torch.cat((x_A, y_B), dim=1)

    def inverse(self, y):
        y_A, y_B = torch.chunk(y, 2, dim=1)
        # Obtain t and s using the same U-Net but now applied to y_A
        t_s = self.unet(y_A)
        t, s = torch.chunk(t_s, 2, dim=1)
        # Perform the inverse affine transformation
        x_B = inverse_affine_transform(y_B, t, s)
        return torch.cat((y_A, x_B), dim=1)
    

if __name__ == '__main__':

    # Verify the UNet model
    def test_unet():
        # Define input parameters
        input_channels = 1  # For example, RGB images
        output_channels = 16  # For binary segmentation
        input_height = 32   # Height of the input image
        input_width = 32    # Width of the input image

        # Create a random input tensor
        x = torch.randn(1, input_channels, input_height, input_width)  # Batch size of 1

        # Instantiate the UNet model
        model = UNet(input_channels, output_channels)

        # Forward pass through the model
        output = model(x)

        # Print the output shape
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)


    test_unet()
