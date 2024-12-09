import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic convolutional block with optional batch normalization and activation
class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


# Encoder with skip connections
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)  # Downsample 1
        self.enc2 = ConvBlock(64, 128, stride=2)  # Downsample 2
        self.enc3 = ConvBlock(128, 256, stride=2)  # Downsample 3
        self.enc4 = ConvBlock(256, 512, stride=2)  # Downsample 4
        self.enc5 = ConvBlock(512, 1024, stride=2)  # Bottleneck

        # Latent space parameters
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

    def forward(self, x):
        # Save skip connections
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        bottleneck = self.enc5(skip4)

        # Flatten and compute latent parameters
        bottleneck = bottleneck.view(bottleneck.size(0), -1)
        mu = self.fc_mu(bottleneck)
        logvar = self.fc_logvar(bottleneck)

        return mu, logvar, [skip1, skip2, skip3, skip4]


# Decoder with skip connections
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)

        self.up5 = ConvBlock(1024, 512)
        self.up4 = ConvBlock(512 + 512, 256)  # Combine with skip
        self.up3 = ConvBlock(256 + 256, 128)  # Combine with skip
        self.up2 = ConvBlock(128 + 128, 64)  # Combine with skip
        self.up1 = nn.Conv2d(64 + 64, out_channels, kernel_size=3, padding=1)

    def forward(self, z, skips):
        # Expand latent vector
        x = self.fc(z).view(-1, 1024, 8, 8)

        # Up-sampling and skip connections
        x = F.interpolate(
            self.up5(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat([x, self._resize_to_match(x, skips[3])], dim=1)

        x = F.interpolate(
            self.up4(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat([x, self._resize_to_match(x, skips[2])], dim=1)

        x = F.interpolate(
            self.up3(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat([x, self._resize_to_match(x, skips[1])], dim=1)

        x = F.interpolate(
            self.up2(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat([x, self._resize_to_match(x, skips[0])], dim=1)

        # Final layer
        x = self.up1(x)
        return x

    def _resize_to_match(self, x, skip):
        """Resize skip connection tensor to match the spatial dimensions of x."""
        return F.interpolate(
            skip, size=x.shape[2:], mode="bilinear", align_corners=False
        )


# VAE-U-Net model
class VAEUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, skips)
        return recon_x, mu, logvar

    def encode(self, x):
        mu, logvar, skips = self.encoder(x)
        return mu, logvar, skips

    def decode(self, z, skips):
        return self.decoder(z, skips)


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)

    # Model parameters
    in_channels = 3
    out_channels = 3
    latent_dim = 48

    # Create the model
    model = VAEUNet(
        in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim
    )

    # Generate a random input tensor of shape (batch_size, channels, height, width)
    batch_size = 1  # Single image for testing
    test_input = torch.randn(batch_size, in_channels, 128, 128)

    # Pass the input through the model
    recon_x, mu, logvar = model(test_input)

    # Print results
    print(f"Input Shape: {test_input.shape}")
    print(f"Reconstructed Shape: {recon_x.shape}")
    print(f"Latent Mean Shape: {mu.shape}")
    print(f"Latent Log Variance Shape: {logvar.shape}")
