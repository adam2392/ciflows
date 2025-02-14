import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv layer to prevent aggressive downsampling
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove max pooling to retain spatial size
        resnet = nn.Sequential(*list(resnet.children())[:4], *list(resnet.children())[5:-1])

        self.resnet = resnet
        in_features = 512  # ResNet18 final feature size

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)  # Flatten feature maps
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return self.reparameterize(mu, logvar)

    def encode(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)  # Flatten feature maps
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


# Residual Block for the decoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class DeepResNetDecoder(nn.Module):
    def __init__(self, latent_dim, num_blocks_per_stage=2):
        super(DeepResNetDecoder, self).__init__()
        self.latent_to_feature = nn.Linear(latent_dim, 512 * 2 * 2)  # Adjusted for 32x32

        # Upsampling stages
        self.stage1 = self._make_stage(512, 256, num_blocks_per_stage)  # 2x2 -> 4x4
        self.stage2 = self._make_stage(256, 128, num_blocks_per_stage)  # 4x4 -> 8x8
        self.stage3 = self._make_stage(128, 64, num_blocks_per_stage)   # 8x8 -> 16x16
        self.stage4 = self._make_stage(64, 32, num_blocks_per_stage)    # 16x16 -> 32x32

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.latent_to_feature(z).view(z.size(0), 512, 2, 2)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.sigmoid(self.final_conv(x))
        return x


# VAE Model for 32x32 images
class DeepResNetMNISTVAE(nn.Module):
    def __init__(self, latent_dim, img_size=32, num_blocks_per_stage=2, learn_sigma_x=False):
        super(DeepResNetMNISTVAE, self).__init__()
        self.img_size = img_size
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = DeepResNetDecoder(latent_dim, num_blocks_per_stage)
        if learn_sigma_x:
            self.log_sigma_x = nn.Parameter(torch.full((1,), 0)[0], requires_grad=True)
        else:
            self.log_sigma_x = torch.tensor(0, requires_grad=False, dtype=torch.float32)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.img_size * self.img_size), 
                                     x.view(-1, 3 * self.img_size * self.img_size), 
                                     reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


# Testing the model
if __name__ == "__main__":
    latent_dim = 48
    model = DeepResNetMNISTVAE(latent_dim, img_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sample_input = torch.randn(8, 3, 32, 32)  # 8 images of size 32x32 (RGB)
    output, mu, logvar = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
