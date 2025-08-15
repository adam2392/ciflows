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
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.skip = nn.Sequential()
        if in_channels != out_channels or upsample:
            skip_layers = []
            if upsample:
                skip_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            skip_layers += [
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            ]
            self.skip = nn.Sequential(*skip_layers)

    def forward(self, x):
        out = self.up(x) if self.upsample else x
        out = self.leaky_relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return self.leaky_relu(out + self.skip(x))


# helper to build channel list from depth
def channels_from_depth(depth, base_channels=32, growth=2):
    """
    Create a channels list of length `depth`.
    depth: int >= 1
    base_channels: starting channels (int)
    growth: multiplicative growth per stage (int)
    """
    assert depth >= 1 and isinstance(depth, int)
    channels = [base_channels * (growth ** i) for i in range(depth)]
    return channels


# def build_encoder_resblocks(channels):
#     blocks = []
#     for i in range(1, len(channels)):
#         blocks.append(ResBlock(channels[i-1], channels[i], downsample=True))
#     # final same-channel block (no downsample)
#     blocks.append(ResBlock(channels[-1], channels[-1], downsample=False))
#     return nn.Sequential(*blocks)

def build_encoder_resblocks(channels, blocks_per_stage=2):
    blocks = []
    for i in range(1, len(channels)):
        # First block in the stage downsamples
        blocks.append(ResBlock(channels[i-1], channels[i], downsample=True))
        # Extra blocks at same resolution (no downsample)
        for _ in range(blocks_per_stage - 1):
            blocks.append(ResBlock(channels[i], channels[i], downsample=False))
    return nn.Sequential(*blocks)

def build_decoder_resblocks(channels, blocks_per_stage=2):
    blocks = []
    for i in range(len(channels) - 1, 0, -1):
        # First block in the stage upsamples
        blocks.append(ResBlockTranspose(channels[i], channels[i-1], upsample=True))
        # Extra blocks at same resolution (no upsample)
        for _ in range(blocks_per_stage - 1):
            blocks.append(ResBlockTranspose(channels[i-1], channels[i-1], upsample=False))
    return nn.Sequential(*blocks)



# def build_decoder_resblocks(channels):
#     blocks = []
#     # first: no upsample block at bottleneck channels[-1] -> channels[-1]
#     blocks.append(ResBlockTranspose(channels[-1], channels[-1], upsample=False))
#     # then upsample through reversed channel pairs
#     for i in range(len(channels)-1, 0, -1):
#         blocks.append(ResBlockTranspose(channels[i], channels[i-1], upsample=True))
#     return nn.Sequential(*blocks)


class Encoder(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, img_size=32,blocks_per_stage=2,
                 channels=None, depth=None, base_channels=32, growth=2, debug=False):
        """
        Provide either `channels` (list) OR `depth` (int). If both are None, defaults to channels [32,64,128,256].
        img_size must be divisible by 2**(len(channels)-1).
        """
        super().__init__()
        self.debug = debug

        if channels is None:
            if depth is None:
                channels = [32, 64, 128, 256]
            else:
                channels = channels_from_depth(depth, base_channels=base_channels, growth=growth)
        self.channels = channels

        self.initial = nn.Conv2d(img_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.resblocks = build_encoder_resblocks(channels, blocks_per_stage=blocks_per_stage)

        downsample_count = len(channels) - 1
        assert img_size % (2 ** downsample_count) == 0, \
            f"img_size {img_size} must be divisible by 2**{downsample_count}"
        final_spatial = img_size // (2 ** downsample_count)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * final_spatial * final_spatial, z_dim)
        self._final_spatial = final_spatial

    def forward(self, x):
        if self.debug:
            print("input:", x.shape)
        x = self.initial(x)
        if self.debug:
            print("after initial:", x.shape)
        x = self.resblocks(x)
        if self.debug:
            print("after resblocks:", x.shape)
        x = self.flatten(x)
        if self.debug:
            print("flatten:", x.shape)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, z_dim=32, img_channels=3, img_size=32,blocks_per_stage=2,
                 channels=None, depth=None, base_channels=32, growth=2, debug=False):
        super().__init__()
        self.debug = debug

        if channels is None:
            if depth is None:
                channels = [32, 64, 128, 256]
            else:
                channels = channels_from_depth(depth, base_channels=base_channels, growth=growth)
        self.channels = channels

        downsample_count = len(channels) - 1
        assert img_size % (2 ** downsample_count) == 0, \
            f"img_size {img_size} must be divisible by 2**{downsample_count}"
        final_spatial = img_size // (2 ** downsample_count)

        self.fc = nn.Linear(z_dim, channels[-1] * final_spatial * final_spatial)
        self.resblocks = build_decoder_resblocks(channels, blocks_per_stage=blocks_per_stage)
        self.final = nn.Conv2d(channels[0], img_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()
        self._start_spatial = final_spatial

    def forward(self, z):
        if self.debug:
            print("z:", z.shape)
        x = self.fc(z).view(-1, self.channels[-1], self._start_spatial, self._start_spatial)
        if self.debug:
            print("after fc reshape:", x.shape)
        x = self.resblocks(x)
        if self.debug:
            print("after resblocks:", x.shape)
            print("final conv output:", self.final(x).shape)
        return self.activation(self.final(x))


class JointDiscriminator(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, img_size=32,
                 channels=None, depth=None, base_channels=32, growth=2, blocks_per_stage=2):
        super().__init__()
        if channels is None:
            if depth is None:
                channels = [32, 64, 128, 256]
            else:
                channels = channels_from_depth(depth, base_channels=base_channels, growth=growth)
        self.channels = channels

        self.initial = nn.Conv2d(img_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.resblocks = build_encoder_resblocks(channels, blocks_per_stage=blocks_per_stage)

        downsample_count = len(channels) - 1
        assert img_size % (2 ** downsample_count) == 0, \
            f"img_size {img_size} must be divisible by 2**{downsample_count}"
        final_spatial = img_size // (2 ** downsample_count)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(channels[-1] * final_spatial * final_spatial + z_dim, 512),
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
    # quick examples
    batch_size = 8
    img_channels = 3
    height = width = 32
    z_dim = 64

    # 1) old default via channels list (identical behavior)
    default_channels = [32, 64, 128, 256]
    enc = Encoder(img_channels=img_channels, z_dim=z_dim, img_size=32, channels=default_channels, blocks_per_stage=3)
    dec = Decoder(z_dim=z_dim, img_channels=img_channels, img_size=32, channels=default_channels, blocks_per_stage=3)
    dis = JointDiscriminator(img_channels=img_channels, z_dim=z_dim, img_size=32, channels=default_channels, blocks_per_stage=3)

    x = torch.randn(batch_size, img_channels, height, width)
    z = enc(x)
    print("latent (default):", z.shape)
    print("recon (default):", dec(z).shape)
    print("disc (default):", dis(x, z).shape)

    # 2) specify depth instead of explicit channels: depth=5 (deeper)
    #    channels will be [32, 64, 128, 256, 512] with default base/growth
    # depth = 7
    # enc_d = Encoder(img_channels=img_channels, z_dim=z_dim, img_size=32, depth=depth)
    # dec_d = Decoder(z_dim=z_dim, img_channels=img_channels, img_size=32, depth=depth)
    # dis_d = JointDiscriminator(img_channels=img_channels, z_dim=z_dim, img_size=32, depth=depth)

    # z2 = enc_d(x)   # works when img_size divisible by 2**(depth-1)
    # print("latent (depth=5):", z2.shape)
    # print("recon (depth=5):", dec_d(z2).shape)
    # print("disc (depth=5):", dis_d(x, z2).shape)
