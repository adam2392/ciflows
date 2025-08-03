import torch
import torch.nn as nn


class CNN_Module(nn.Module):
    def __init__(self, i_channels, o_size, feature_maps=64, h_layers=3, use_batch_norm=True):
        """
        Maps images to real vectors with the following shapes:
            input shape: (i_channels, img_size, img_size)
            output shape: o_size * ((img_size / (2 ** (h_layers + 1))) - 3) ** 2
            output shape is o_size if img_size == 2 ** (h_layers + 3)

        Other parameters:
            feature_maps: number of feature maps between convolutional layers
            use_batch_norm: set True to use batch norm after each layer
        """
        super().__init__()
        self.i_channels = i_channels
        self.feature_maps = feature_maps
        self.o_size = o_size

        bias = not use_batch_norm

        # Shape: (b, i_channels, img_size, img_size)
        conv_layers = [
            nn.Conv2d(i_channels, feature_maps, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Shape: (b, feature_maps, img_size / 2, img_size / 2)

        for h in range(h_layers):
            # Shape: (b, 2 ** h * feature_maps, img_size / (2 ** (h + 1)), img_size / (2 ** (h + 1)))
            conv_layers.append(
                nn.Conv2d(2**h * feature_maps, 2 ** (h + 1) * feature_maps, 4, 2, 1, bias=bias)
            )
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(2 ** (h + 1) * feature_maps))
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            # Shape: (b, 2 ** (h + 1) * feature_maps, img_size / (2 ** (h + 2)), img_size / (2 ** (h + 2)))

        # Shape: (b, 2 ** h_layers * feature_maps, img_size / (2 ** (h_layers + 1)), img_size / (2 ** (h_layers + 1)))
        conv_layers.append(nn.Conv2d(2**h_layers * feature_maps, o_size, 4, 1, 0, bias=True))
        # Shape: (b, o_size, (img_size / (2 ** (h_layers + 1))) - 3, (img_size / (2 ** (h_layers + 1))) - 3)

        self.conv_nn = nn.Sequential(*conv_layers)

        self.device_param = nn.Parameter(torch.empty(0))

        self.conv_nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, include_inp=False):
        out = torch.reshape(self.conv_nn(x), (x.shape[0], -1))
        if include_inp:
            return out, x
        return out


class CNN_Deconv_Module(nn.Module):
    def __init__(self, i_size, o_channels, feature_maps=64, h_layers=3, use_batch_norm=True):
        """
        Maps real vectors to images with the following shapes:
            input shape: i_size
            output shape: (o_channels, 2 ** (h_layers + 3), 2 ** (h_layers + 3))

        Other parameters:
            feature_maps: number of feature maps between deconvolutional layers
            use_batch_norm: set True to use batch norm after each layer
        """
        super().__init__()
        self.i_size = i_size
        self.o_channels = o_channels
        self.feature_maps = feature_maps

        bias = not use_batch_norm

        # Shape: (b, i_size, 1, 1)
        layers = [nn.ConvTranspose2d(self.i_size, 2**h_layers * feature_maps, 4, 1, 0, bias=bias)]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(2**h_layers * feature_maps))
        layers.append(nn.ReLU(True))
        # Shape: (b, 2 ** h_layers * feature_maps, 4, 4)

        for h in range(h_layers):
            # Shape: (b, 2 ** (h_layers - h) * feature_maps, 2 ** (h + 2), 2 ** (h + 2))
            layers.append(
                nn.ConvTranspose2d(
                    2 ** (h_layers - h) * feature_maps,
                    2 ** (h_layers - h - 1) * feature_maps,
                    4,
                    2,
                    1,
                    bias=bias,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(2 ** (h_layers - h - 1) * feature_maps))
            layers.append(nn.ReLU(True))
            # Shape: (b, 2 ** (h_layers - h - 1) * feature_maps, 2 ** (h + 3), 2 ** (h + 3))

        # Shape: (b, feature_maps, 2 ** (h_layers + 2), 2 ** (h_layers + 2))
        layers.append(nn.ConvTranspose2d(feature_maps, o_channels, 4, 2, 1, bias=True))
        layers.append(nn.Tanh())
        # Shape: (b, o_channels, 2 ** (h_layers + 3), 2 ** (h_layers + 3))

        self.conv_nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(torch.empty(0))

        self.conv_nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, include_inp=False):
        x = torch.reshape(x, (x.shape[0], -1, 1, 1))
        if include_inp:
            return self.conv_nn(x), x
        return self.conv_nn(x)


def smoke_test():
    # Simulate a batch of 8 colored MNIST-like images (3, 32, 32)
    batch_size = 8
    img_channels = 3
    img_size = 32
    latent_dim = 16

    # needs to be at most 2.
    # stride of 2 makes 32 -> 16 for the initial conv layer. Then there are
    # n more hidden layers, each halving the size of the image.
    # then 16 -> 8
    # then 8 -> 4
    h_layers = 2

    print("===> Creating encoder...")
    encoder = CNN_Module(i_channels=img_channels, o_size=latent_dim, h_layers=h_layers)
    print(encoder)

    print("\n===> Creating decoder...")
    decoder = CNN_Deconv_Module(i_size=latent_dim, o_channels=img_channels, h_layers=h_layers)
    print(decoder)

    print("\n===> Generating input image batch...")
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")

    print("\n===> Running through encoder...")
    z = encoder(x)
    print(f"Encoded shape: {z.shape}")

    print("\n===> Running through decoder...")
    x_recon = decoder(z)
    print(f"Reconstructed image shape: {x_recon.shape}")

    print("\n===> Smoke test complete!")


if __name__ == "__main__":
    smoke_test()

    # s = CNN_Deconv_Module(4, 3, h_layers=3)
    # print(s)
    # x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]).float()
    # out_samp = s(x)

    # print(out_samp)
    # print(out_samp.shape)

    # s2 = CNN_Module(3, 5, h_layers=3)
    # print(s2)
    # out = s2(out_samp)

    # print(out)
    # print(out.shape)
