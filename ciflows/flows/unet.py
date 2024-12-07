import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=128):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=latent_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=128):
        super(Unet, self).__init__()

        # Define the number of filters for each level
        self.encoder_filters = [16, 32]
        self.decoder_filters = [x * 2 for x in [32, 16]]

        self.initial_conv = ConvBlock(
            input_channels, self.encoder_filters[0], latent_dim=latent_dim
        )

        # Encoder conv blocks
        self.conv_blocks1 = nn.ModuleList()
        for in_ch, out_ch in zip(self.encoder_filters[:-1], self.encoder_filters[1:]):
            self.conv_blocks1.append(ConvBlock(in_ch, out_ch))

        # Bridge
        self.conv_block_bridge = ConvBlock(
            self.encoder_filters[-1], self.encoder_filters[-1]
        )

        # Decoder conv blocks
        self.conv_blocks2 = nn.ModuleList()
        for in_ch, out_ch in zip(self.decoder_filters[:-1], self.decoder_filters[1:]):
            self.conv_blocks2.append(ConvBlock(in_ch, out_ch))

        # Pooling and upsampling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Final convolution
        self.final_conv = nn.Conv2d(
            in_channels=self.decoder_filters[-1],
            out_channels=output_channels,
            kernel_size=1,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_x = []

        x = self.initial_conv(x)

        ## Encoder
        for conv_block in self.conv_blocks1:
            x = conv_block(x)
            skip_x.append(x)

            # XXX: Note this will not work if we downsample too much
            x = self.maxpool(x)
            # else:
            # x = self.maxpool(x)

        ## Bridge
        x = self.conv_block_bridge(x)

        ## Decoder
        for conv_block, skip in zip(self.conv_blocks2, reversed(skip_x)):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)  # Concatenation along the channel dimension
            x = conv_block(x)

        ## Output
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":

    x = torch.randn(2, 32, 128, 128)  # Batch size 1, 32 channels, 128x128 image
    model = ConvBlock(32, 32)
    output = model(x)
    print(output.shape)  # Expected output shape: (1, 32, 128, 128)

    x = torch.randn(10, 32, 16, 16)
    model = ConvBlock(32, 64)
    model2 = ConvBlock(64, 128)

    output = model(x)
    output = model2(output)
    print(output.shape)  # Expected output shape: (1, 32, 128, 128)

    # x = torch.randn(10, 2, 32, 32)  # Batch size 1, 32 channels, 128x128 image
    x = torch.randn(10, 24, 2, 2)  # Batch size 1, 32 channels, 128x128 image
    model = Unet(input_channels=24, output_channels=48)
    output = model(x)
    print(output.shape)  # Expected output shape: (1, 3, 128, 128
