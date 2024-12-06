import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResNetCelebAEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetCelebAEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2, padding=1
            ),  # (128, 8, 8)    # added to make 128x128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # (128, 4, 4)
            nn.ReLU(),
            ResBlock(128, 32, 128),
            ResBlock(128, 32, 128),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNetCelebADecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetCelebADecoder, self).__init__()

        self.latent_dim = latent_dim

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (128, 8, 8)
            nn.ReLU(),
            ResBlock(128, 32, 128),
            ResBlock(128, 32, 128),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1, output_padding=0
            ),  # (128, 16, 16)
            nn.ReLU(),  # added to make 128x128
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 3, kernel_size=4, stride=2, padding=1
            ),  # (3, 64, 64)
            nn.Sigmoid(),
        )

        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)  # From latent_dim

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 4, 4)
        return self.decoder(x)


class ResNetCelebA(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetCelebA, self).__init__()

        # Encoder
        self.encoder = ResNetCelebAEncoder(latent_dim=latent_dim)
        # Decoder
        self.decoder = ResNetCelebADecoder(latent_dim=latent_dim)
        # self.fc2 = nn.Linear(latent_dim, 128 * 4 * 4)  # From latent_dim
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        #     ),  # (128, 8, 8)
        #     nn.ReLU(),
        #     ResBlock(128, 32, 128),
        #     ResBlock(128, 32, 128),
        #     nn.ConvTranspose2d(
        #         128, 128, kernel_size=4, stride=2, padding=1, output_padding=0
        #     ),  # (128, 16, 16)
        #     nn.ReLU(),  # added to make 128x128
        #     nn.ConvTranspose2d(
        #         128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        #     ),  # (64, 16, 16)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         64, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        #     ),  # (64, 32, 32)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         64, 3, kernel_size=4, stride=2, padding=1
        #     ),  # (3, 64, 64)
        #     nn.Sigmoid(),
        # )

    def encode(self, x):
        return self.encoder(x)
        # x = x.view(x.size(0), -1)
        # return self.fc1(x)

    def decode(self, z):
        # x = self.fc2(z)
        # x = x.view(x.size(0), 128, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    # Example usage
    model = ResNetCelebA(latent_dim=16 * 3)
    x = torch.randn(16, 3, 128, 128)  # Batch of 16 images
    output = model(x)
    print(output.shape)  # Should be (16, 3, 64, 64)

    print(model.encode(x).shape)
