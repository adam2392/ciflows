import torch
import torch.nn as nn


# Double Convolution Block (Conv -> ReLU -> Conv -> ReLU)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Encoder Block (Double Conv -> Max Pool)
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x_pooled = self.pool(x)
        return x_pooled, x  # return both pooled and unpooled (for skip connection)


# Decoder Block (Upsample/ConvTranspose -> Double Conv)
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.upconv(x)
        # Concatenate along the channel axis
        x = torch.cat((x, skip_x), dim=1)
        x = self.double_conv(x)
        return x


# Full U-Net Model for MNIST
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_layers=2,
        n_channels_start=32,
        debug=False,
    ):
        super(UNet, self).__init__()
        self.n_layers = n_layers
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.debug = debug

        # Encoder: Downsampling path
        self.encoders = nn.ModuleList()
        for i in range(n_layers):
            in_channels = in_channels if i == 0 else out_channels
            out_channels = n_channels_start * (2**i)
            print(f"Encoder {i}: {in_channels} -> {out_channels}")
            self.encoders.append(Encoder(in_channels, out_channels))

        # Bottleneck
        self.bottleneck = DoubleConv(out_channels, out_channels * 2)  # 64x7x7 -> 128x7x7
        out_channels *= 2

        # Decoder: Upsampling path
        self.decoders = nn.ModuleList()
        for i in range(n_layers):
            in_channels = out_channels
            out_channels = n_channels_start * (2 ** (n_layers - i - 1))
            print(f"Decoder {i}: {in_channels} -> {out_channels}")
            self.decoders.append(Decoder(in_channels, out_channels))

        # Final output layer
        self.final_conv = nn.Conv2d(
            out_channels, self.out_channels, kernel_size=1
        )  # 32x28x28 -> 1x28x28

    def encode(self, x, include_skips=True):
        skips = []
        x_pooled = x
        for _, encoder in enumerate(self.encoders):
            x_pooled, x = encoder(x_pooled)
            skips.append(x)
        if include_skips:
            return x_pooled, skips
        return x_pooled

    def decode(self, bottleneck, skips):
        for i, decoder in enumerate(self.decoders):
            bottleneck = decoder(bottleneck, skips[-i - 1])
        return bottleneck

    def forward(self, x):
        # Encoder - downsampling blocks
        skips = []
        x_pooled = x
        for _, encoder in enumerate(self.encoders):
            # (B, C, H, W) -> (B, 2C, H/2, W/2)
            x_pooled, x = encoder(x_pooled)
            skips.append(x)

        # x1_pooled, x1 = self.encoder1(x)  # Downsample 1st block
        # x2_pooled, x2 = self.encoder2(x1_pooled)  # Downsample 2nd block

        # Bottleneck the last encoder output -> (B, C, H, W) -> (B, 2C, H, W)
        bottleneck = self.bottleneck(x_pooled)

        # Decoder - upsampling blocks
        for i, decoder in enumerate(self.decoders):
            if self.debug:
                print(f"Decoder {i}: ", bottleneck.shape, skips[-i - 1].shape)
            # (B, 2C, H, W) -> (B, C, 2H, 2W)
            bottleneck = decoder(bottleneck, skips[-i - 1])

        # Final Convolution
        x = self.final_conv(bottleneck)
        return x


# Example usage
if __name__ == "__main__":
    # Create the U-Net model for MNIST
    model = UNet(in_channels=1, out_channels=1, n_layers=2)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get the number of trainable parameters in the encoder and decoder
    num_encoder_params = count_trainable_parameters(model)

    # Print the results
    print(f"Total trainable parameters in Encoder: {num_encoder_params}")
    # Example input: MNIST image (1 channel, 28x28 pixels)
    input_tensor = torch.randn(1, 1, 28, 28)

    # Forward pass through the model
    output = model(input_tensor)
    encoding = model.encode(input_tensor, include_skips=False)
    print(output.shape)  # Should be [1, 1, 28, 28]
    print(encoding.shape)
