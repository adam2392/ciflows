import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.leaky_relu(out + self.skip(x))


class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.skip = nn.Sequential()
        if in_channels != out_channels or upsample:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.up(x) if self.upsample else x
        out = self.leaky_relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return self.leaky_relu(out + self.skip(x))


# ------------------------
#   BiGAN Core Modules
# ------------------------
class Encoder(nn.Module):
    def __init__(self, img_channels=3, z_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 7 * 7, z_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(x)
        x = self.flatten(x)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, z_dim=32, img_channels=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.resblocks = nn.Sequential(
            ResBlockTranspose(256, 256),
            ResBlockTranspose(256, 128, upsample=True),
            ResBlockTranspose(128, 64, upsample=True)
        )
        self.final = nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 7, 7)
        x = self.resblocks(x)
        return self.activation(self.final(x))


class JointDiscriminator(nn.Module):
    def __init__(self, img_channels=3, z_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7 + z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x = self.initial(x)
        x = self.resblocks(x)
        x_feat = self.flatten(x)
        return self.fc(torch.cat([x_feat, z], dim=1))


if __name__ == "__main__":
    # Define dimensions
    batch_size = 8
    img_channels = 3
    height = width = 28
    z_dim = 32

    # Instantiate models
    encoder = Encoder(img_channels=img_channels, z_dim=z_dim)
    decoder = Decoder(z_dim=z_dim, img_channels=img_channels)
    discriminator = JointDiscriminator(img_channels=img_channels, z_dim=z_dim)

    # Dummy input image
    x = torch.randn(batch_size, img_channels, height, width)

    # Pass through encoder
    z = encoder(x)
    print("Latent vector shape:", z.shape)  # Expect: (batch_size, z_dim)

    # Decode back to image
    x_rec = decoder(z)
    print("Reconstructed image shape:", x_rec.shape)  # Expect: (batch_size, 3, 28, 28)

    # Discriminator prediction
    score = discriminator(x, z)
    print("Discriminator output shape:", score.shape)  # Expect: (batch_size, 1)
