import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Shortcut connection to downsample residual
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # Add dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu1(self.conv1(x))  # First convolution + ReLU
        out = self.dropout(out)
        out = self.conv2(out)  # Second convolution
        out = out +  self.shortcut(x)  # Add the shortcut (input)
        out = self.relu2(out)
        out = self.dropout(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass to upsample.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W)
            Input tensor.

        Returns
        -------
        out : Tensor of shape (B, out_channels, 2 * H, 2 * W)
            Upsample block.
        """
        out = self.upsample(x)
        out = self.relu1(self.conv(out))
        out = self.dropout(out)
        out = out + self.upsample(self.shortcut(x))
        out = self.relu2(out)
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size,
        embed_dim,
        n_compression_layers=5,
        n_bottleneck_layers=6,
        hidden_dim=1024,
        dropout=0.0,
        debug=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        assert n_compression_layers > 1, "n_compression_layers must be greater than 1"
        # Wrapping the layers in a Sequential block
        self.layers = nn.Sequential(
            ResidualBlock(
                in_channels, 32, stride=2, dropout=dropout
            ),  # Initial channels
        )
        for idx in range(1, n_compression_layers):
            in_ch = int(32 * (2 ** (idx - 1)))
            out_ch = int(32 * 2 ** (idx))
            if debug:
                print(in_ch, out_ch)
            self.layers.append(ResidualBlock(in_ch, out_ch, stride=2, dropout=dropout))

        for idx in range(1, n_bottleneck_layers + 1):
            in_ch = int(32 * (2 ** (n_compression_layers - idx)))
            out_ch = int(32 * (2 ** (n_compression_layers - idx - 1)))
            if debug:
                print(in_ch, out_ch)
            self.layers.append(ResidualBlock(in_ch, out_ch, stride=1, dropout=dropout))

        # Calculate the size after five downsampling layers
        self.final_conv_size = img_size // (
            2**n_compression_layers
        )  # Each layer reduces the size by a factor of 2

        # Fully connected layers for mean and log variance
        final_conv_dim = out_ch * self.final_conv_size * self.final_conv_size
        if debug:
            print(out_ch, self.final_conv_size, final_conv_dim)
        # self.ffm = nn.Sequential(
        #     nn.Linear(final_conv_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, embed_dim),
        #     nn.Dropout(dropout),
        # )
        self.fc_mu = nn.Linear(
            out_ch * self.final_conv_size * self.final_conv_size, embed_dim
        )
        self.fc_logvar = nn.Linear(
            out_ch * self.final_conv_size * self.final_conv_size, embed_dim
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

        # (B, final_conv_size, final_conv_size, 8) -> (B, embed_dim)
        # x = self.ffm(x)
        # return x
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z ~ N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std  # Sample from N(mu, std)



class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size,
        out_channels,
        n_bottleneck_layers=6,
        n_upsample_layers=5,
        hidden_dim=1024,
        dropout=0.0,
        debug=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.out_channels = out_channels
        self.n_upsample_layers = n_upsample_layers
        # Calculate the size after five downsampling layers
        self.final_conv_size = img_size // (2**n_upsample_layers)

        # Fully connected layers to upscale to bottleneck
        out_ch = int(32 * (2 ** (n_upsample_layers - n_bottleneck_layers - 1)))
        final_conv_dim = out_ch * self.final_conv_size * self.final_conv_size
        self.ffm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_conv_dim),
            nn.Dropout(dropout),
        )

        self.bottleneck_layers = nn.Sequential()

        for idx in range(n_bottleneck_layers, 0, -1):
            in_ch = int(32 * (2 ** (n_upsample_layers - idx - 1)))
            out_ch = int(32 * (2 ** (n_upsample_layers - idx)))
            if debug:
                print(in_ch, out_ch)
            self.bottleneck_layers.append(
                ResidualBlock(in_ch, out_ch, stride=1, dropout=dropout)
            )

        self.upsample_layers = nn.Sequential()
        for idx in range(n_upsample_layers - 1, 0, -1):
            in_ch = int(32 * (2 ** (idx)))
            out_ch = int(32 * (2 ** (idx - 1)))
            if debug:
                print(in_ch, out_ch)
            self.upsample_layers.append(UpsampleBlock(in_ch, out_ch, dropout=dropout))

        self.final_conv = UpsampleBlock(32, out_channels, dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ffm(x)
        x = x.view(
            x.size(0),
            -1,
            self.final_conv_size,
            self.final_conv_size,
        )

        x = self.bottleneck_layers(x)
        x = self.upsample_layers(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

