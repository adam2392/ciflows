import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights


# ResNet-based Encoder (with VAE-specific output)
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )  # Using ResNet50
        self.resnet = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # Remove the final classification layer
        self.fc_mu = nn.Linear(
            resnet.fc.in_features, latent_dim
        )  # For the mean of the latent space
        self.fc_logvar = nn.Linear(
            resnet.fc.in_features, latent_dim
        )  # For the log variance of the latent space

    def forward(self, x):
        x = self.resnet(x)  # Forward pass through ResNet layers
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc_mu(x)  # Mean vector of latent space
        logvar = self.fc_logvar(x)  # Log-variance vector of latent space
        return self.reparameterize(mu, logvar)

    def encode(self, x):
        x = self.resnet(x)  # Forward pass through ResNet layers
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc_mu(x)  # Mean vector of latent space
        logvar = self.fc_logvar(x)  # Log-variance vector of latent space
        return mu, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DeepResNetDecoder(nn.Module):
    def __init__(self, latent_dim, num_blocks_per_stage=5):
        super(DeepResNetDecoder, self).__init__()
        self.latent_to_feature = nn.Linear(
            latent_dim, 512 * 8 * 8
        )  # Expand latent dim to feature map size

        # Create multiple residual blocks for each upsampling stage
        self.stage1 = self._make_stage(512, 256, num_blocks_per_stage)  # 8x8 -> 16x16
        self.stage2 = self._make_stage(256, 128, num_blocks_per_stage)  # 16x16 -> 32x32
        self.stage3 = self._make_stage(128, 64, num_blocks_per_stage)  # 32x32 -> 64x64
        self.stage4 = self._make_stage(64, 32, num_blocks_per_stage)  # 64x64 -> 128x128

        self.final_conv = nn.Conv2d(
            32, 3, kernel_size=3, stride=1, padding=1
        )  # Output RGB image

    def _make_stage(self, in_channels, out_channels, num_blocks):
        """
        Create a stage with multiple residual blocks and an upsampling layer.
        """
        layers = []
        # Add an initial upsampling layer to increase spatial resolution
        layers.append(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Add multiple residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, z):
        # Map latent vector to initial feature map
        x = self.latent_to_feature(z).view(z.size(0), 512, 8, 8)

        # Pass through upsampling stages
        x = self.stage1(x)  # 8x8 -> 16x16
        x = self.stage2(x)  # 16x16 -> 32x32
        x = self.stage3(x)  # 32x32 -> 64x64
        x = self.stage4(x)  # 64x64 -> 128x128

        # Final convolution to generate RGB image
        x = torch.sigmoid(self.final_conv(x))
        return x


# VAE model combining Encoder and Decoder
class DeepResNetVAE(nn.Module):
    def __init__(self, latent_dim, num_blocks_per_stage=5):
        super(DeepResNetVAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim)  # Same encoder as before
        self.decoder = DeepResNetDecoder(latent_dim, num_blocks_per_stage)

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * 128 * 128), x.view(-1, 3 * 128 * 128), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


if __name__ == "__main__":
    # Example instantiation of the model
    latent_dim = 48  # Latent dimension
    model = DeepResNetVAE(latent_dim)
    # print(model)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Sample input for testing
    sample_input = torch.randn(8, 3, 128, 128)  # 8 images of size 128x128 (RGB)
    output, mu, logvar = model(sample_input)
    print(f"Output shape: {output.shape}")

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
