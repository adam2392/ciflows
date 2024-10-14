import torch
import torch.nn as nn


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
        out = out + self.shortcut(x)  # Add the shortcut (input)
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

        # for idx in range(1, n_bottleneck_layers + 1):
        #     in_ch = int(32 * (2 ** (n_compression_layers - idx)))
        #     out_ch = int(32 * (2 ** (n_compression_layers - idx - 1)))
        #     if debug:
        #         print(in_ch, out_ch)
        #     self.layers.append(ResidualBlock(in_ch, out_ch, stride=1, dropout=dropout))
        for idx in range(1, n_bottleneck_layers + 1):
            in_ch = int(32 * (2 ** (idx)))
            out_ch = int(32 * (2 ** (idx + 1)))
            if debug:
                print(in_ch, out_ch)
            self.layers.append(ResidualBlock(in_ch, out_ch, stride=1, dropout=dropout))

        for idx in range(1, n_compression_layers):
            in_ch = int(32 * (2 ** (idx - 1)))
            out_ch = int(32 * 2 ** (idx))
            if debug:
                print(in_ch, out_ch)
            self.layers.append(ResidualBlock(in_ch, out_ch, stride=2, dropout=dropout))

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


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvEncoder(nn.Module):
    def __init__(self, channels: int, hidden_size: int, height: int, width: int):
        super(ConvEncoder, self).__init__()

        assert (
            height % 4 == 0 and width % 4 == 0
        ), "Choose height and width to be divisible by 4"

        self.channels = channels
        self.height = height
        self.width = width

        final_height = (self.height // 4 - 3) // 2 + 1
        final_width = (self.width // 4 - 3) // 2 + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 32x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: 128x3x3
            Flatten(),
            nn.Linear(
                128 * final_height * final_width, 32 * final_height * final_width
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32 * final_height * final_width),
            nn.Linear(32 * final_height * final_width, hidden_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, hidden_size: int, channels: int, height: int, width: int):
        super(ConvDecoder, self).__init__()

        final_height = (height // 4 - 3) // 2 + 1
        final_width = (width // 4 - 3) // 2 + 1

        # Calculate the size of the flattened output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 32 * final_height * final_width, bias=False),
            nn.BatchNorm1d(32 * final_height * final_width),
            nn.ReLU(),
            nn.Linear(
                32 * final_height * final_width,
                128 * final_height * final_width,
                bias=False,
            ),
            nn.BatchNorm1d(128 * final_height * final_width),
            nn.ReLU(),
            Stack(128, 3, 3),  # Custom Stack layer
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat


class Conv_VAE(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        hidden_size: int,
        trainable_latent: bool = True,
    ):
        super(Conv_VAE, self).__init__()

        assert (
            height % 4 == 0 and width % 4 == 0
        ), "Choose height and width to be divisible by 4"

        self.channels = channels
        self.height = height
        self.hidden_size = hidden_size
        self.width = width

        final_height = (self.height // 4 - 3) // 2 + 1
        final_width = (self.width // 4 - 3) // 2 + 1

        self.latent_space = "fff"

        # Encoder
        self.encoder = ConvEncoder(channels, hidden_size, height, width)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(self.channels, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # Output: 32x7x7
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # Output: 128x3x3
        #     Flatten(),
        #     nn.Linear(
        #         128 * final_height * final_width, 32 * final_height * final_width
        #     ),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(32 * final_height * final_width),
        #     nn.Linear(32 * final_height * final_width, hidden_size),
        #     nn.LeakyReLU(),
        # )

        # Calculate the size of the flattened output
        self.flattened_size = 32 * final_height * final_width
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

        # latent
        self.fcm = nn.Linear(hidden_size, hidden_size)
        if trainable_latent:
            self.loc = nn.Parameter(torch.zeros(1, hidden_size))
            self.log_scale = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.register_buffer("loc", torch.zeros(1, hidden_size))
            self.register_buffer("log_scale", torch.zeros(1, hidden_size))

        # self.latent = torch.distributions.Independent(
        #     torch.distributions.Normal(
        #         loc=torch.zeros(latent_dim, device=device),
        #         scale=torch.ones(latent_dim, device=device),
        #     ),
        #     1
        # )
        # Decoder
        self.decoder = ConvDecoder(hidden_size, channels, height, width)
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_size, 32 * final_height * final_width),
        #     nn.BatchNorm1d(32 * final_height * final_width),
        #     nn.ReLU(),
        #     nn.Linear(
        #         32 * final_height * final_width, 128 * final_height * final_width
        #     ),
        #     nn.BatchNorm1d(128 * final_height * final_width),
        #     nn.ReLU(),
        #     Stack(128, 3, 3),  # Custom Stack layer
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(8, self.channels, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )

    def encode(self, x):
        """Encode input data into the latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent variables back to the input space."""
        return self.decoder(z)

    def reparameterize(self, mean, log_var):
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def log_prob(self, vhat):
        # compute the log probability of the latent encoding
        log_p = -0.5 * self.hidden_size * torch.log(
            torch.Tensor([2 * torch.pi]).to(vhat.device)
        ) - torch.sum(
            self.log_scale
            + 0.5 * torch.pow((vhat - self.loc) / torch.exp(self.log_scale), 2)
        )
        return log_p

    def forward(self, x, return_latent=False):
        """Forward pass for the VAE."""
        z = self.encode(x)

        # latent encoding are samples from the posterior P(z | x; \theta_encoder)
        z = self.fcm(z)

        if self.latent_space == "vae":
            mean = self.fc_mu(z)
            log_var = self.fc_logvar(z)
            # # Assume z is already the latent representation; split into mean and log variance
            z_sample = self.reparameterize(mean, log_var)
        else:
            z_sample = z

        reconstructed_x = self.decode(z_sample)
        if return_latent:
            return reconstructed_x, z_sample
        else:
            return reconstructed_x


if __name__ == "__main__":
    # Parameters for the test
    channels = 3  # For grayscale images (like MNIST); set to 3 for RGB (like CelebA)
    height = 28  # Height of the input image (28 for MNIST)
    width = 28  # Width of the input image (28 for MNIST)
    hidden_size = 16  # Size of the latent space

    # Create an instance of the Conv_VAE model
    model = Conv_VAE(channels, height, width, hidden_size)

    # Generate a random input tensor (batch of 4 images)
    batch_size = 4
    input_images = torch.randn(batch_size, channels, height, width)

    # Pass the input through the model
    # reconstructed_images, mean, log_var = model(input_images, return_latent=)
    reconstructed_images = model(input_images, return_latent=False)
    # reconstructed_images, log_p = model(input_images, return_latent=True)

    # Print the shapes of the outputs
    print("Input Shape: ", input_images.shape)
    print("Reconstructed Shape: ", reconstructed_images.shape)
    # print("Log Probability Shape: ", log_p.shape)
    # print("Mean Shape: ", mean.shape)
    # print("Log Variance Shape: ", log_var.shape)

    # Verify if the output shapes match expectations
    print(reconstructed_images.shape)
    assert reconstructed_images.shape == (
        batch_size,
        channels,
        height,
        width,
    ), "Reconstructed images shape mismatch!"
    # assert log_p.shape == (batch_size,), "Log probability shape mismatch!"
    # assert mean.shape == (batch_size, hidden_size), "Mean shape mismatch!"
    # assert log_var.shape == (batch_size, hidden_size), "Log variance shape mismatch!"
    print("All tests passed!")
