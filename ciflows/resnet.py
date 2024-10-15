import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, latent_dim, inner_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(latent_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.fc3 = nn.Linear(inner_dim, latent_dim)
        self.silu1 = nn.SiLU()  # Swish-like activation function
        self.silu2 = nn.SiLU()  # Swish-like activation function

    def forward(self, x):
        """Forward pass of x latent_dim to latent_dim with residual connection.

        Parameters
        ----------
        x : torch.Tensor of shape (B, latent_dim)
            Input tensor.

        Returns
        -------
        torch.Tensor of shape (B, latent_dim)
            Residual block output.
        """
        residual = x
        x = self.silu1(self.fc1(x))
        x = self.silu2(self.fc2(x))
        x = self.fc3(x)
        return x + residual  # Skip connection


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvTBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvNetEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels, hidden_dim=512, start_channels=32, debug=False):
        super(ConvNetEncoder, self).__init__()

        self.convblocks = nn.Sequential()
        for idx in range(4):
            out_channels = start_channels * 2**idx
            if debug:
                print(
                    f"Adding conv{idx+1} with in_channels={in_channels} and out_channels={out_channels}"
                )
            self.convblocks.append(
                ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            in_channels = out_channels

        # Linear layer to latent space
        self.fc = nn.Linear(in_channels, latent_dim)

        # 4x Residual Blocks with inner_dim=512
        self.resblocks = nn.Sequential(
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
        )

    def forward(self, x):
        # apply convblocks
        x = self.convblocks(x)
        # for layer in self.convblocks:
        #     x = layer(x)
        #     print("Inside encoder: ", x.shape)
        # print("Inside encoder: ", x.shape)
        x = x.view(x.size(0), -1)  # Flatten before FC
        x = self.fc(x)
        x = self.resblocks(x)
        return x


class ConvNetDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, hidden_dim=512, start_channels=32, debug=False):
        super(ConvNetDecoder, self).__init__()

        # 4x Residual Blocks with inner_dim=512
        self.resblocks = nn.Sequential(
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
            ResidualBlock(latent_dim, hidden_dim),
        )

        # Linear layer from latent space
        self.projection_dim = start_channels * 8
        self.fc = nn.Linear(latent_dim, self.projection_dim)

        # Transposed Convolution layers to reconstruct the input
        self.convblocks = nn.Sequential()
        n_layers = 3

        # First, upsample without padding (1) -> (3) -> (7)
        kernel_size = 3
        padding = 0
        in_channels = start_channels * 2 ** (n_layers)
        out_chs = out_channels
        for idx in range(2):
            out_channels = int(start_channels * 2 ** (n_layers - 1 - idx))
            if debug:
                print(
                    f"Adding convT{idx} with in_channels={in_channels} and out_channels={out_channels}"
                )
            self.convblocks.append(
                ConvTBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                )
            )
            in_channels = out_channels

        # Then, upsample with padding (7) -> (14) -> (28)
        kernel_size = 4
        padding = 1
        for idx in range(2, n_layers):
            out_channels = int(start_channels * 2 ** (n_layers - 1 - idx))
            if debug:
                print(
                    f"Adding convT{idx} with in_channels={in_channels} and out_channels={out_channels}"
                )
            self.convblocks.append(
                ConvTBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels

        self.convblocks.append(
            nn.ConvTranspose2d(
                in_channels, out_chs, kernel_size=kernel_size, stride=2, padding=1
            )
        )
        if debug:
            print(
                f"Adding convT{n_layers+1} with in_channels={in_channels} and out_channels={out_channels}"
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        B = z.size(0)
        z = self.resblocks(z)
        z = self.fc(z)
        img_size = int(np.sqrt(self.projection_dim / self.projection_dim))
        z = z.view(B, -1, img_size, img_size)  # Reshape to image-like dimensions

        z = self.convblocks(z)
        # for layer in self.convblocks:
        #     z = layer(z)
        #     print("Inside decoder: ", z.shape)
        z = self.sigmoid(z)
        return z


# Main testing function
if __name__ == "__main__":

    def test_residual_block():
        print("Testing ResidualBlock...")
        block = ResidualBlock(latent_dim=256, inner_dim=512)
        x = torch.randn(1, 256)
        out = block(x)
        assert out.shape == x.shape, f"Expected shape {x.shape}, but got {out.shape}"
        print("ResidualBlock test passed!")

    def test_fiffconvnet_encoder():
        print("Testing FIFFConvNetEncoder...")
        encoder = ConvNetEncoder(latent_dim=128, in_channels=1, start_channels=32*4)
        x = torch.randn(2, 1, 28, 28)  # MNIST image input
        out = encoder(x)
        assert out.shape == (2, 128), f"Expected shape (1, 128), but got {out.shape}"
        print(f"FIFFConvNetEncoder test passed with output shape: {out.shape}")

    def test_fiffconvnet_decoder():
        print("Testing FIFFConvNetDecoder...")
        decoder = ConvNetDecoder(latent_dim=128, out_channels=1, start_channels=32*4)
        z = torch.randn(2, 128)  # Latent vector input
        out = decoder(z)
        assert out.shape == (
            2,
            1,
            28,
            28,
        ), f"Expected shape (1, 1, 28, 28), but got {out.shape}"
        print(f"FIFFConvNetDecoder test passed with output shape: {out.shape}")

    test_residual_block()
    test_fiffconvnet_encoder()
    test_fiffconvnet_decoder()
