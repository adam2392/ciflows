from typing import List

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from normflows.flows.affine import AffineCouplingBlock
from normflows.flows.base import Flow
from normflows.flows.normalization import ActNorm


class GlowBlock(Flow):
    """Glow: Generative Flow with Invertible 1×1 Convolutions, [arXiv: 1807.03039](https://arxiv.org/abs/1807.03039)

    One Block of the Glow model, comprised of

    - MaskedAffineFlow (affine coupling layer)
    - Invertible1x1Conv (dropped if there is only one channel)
    - ActNorm (first batch used for initialization)
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        scale=True,
        scale_map="sigmoid",
        split_mode="channel",
        leaky=0.0,
        init_zeros=True,
        use_lu=True,
        net_actnorm=False,
        dropout_probability=0.0
    ):
        """Constructor

        Args:
          channels: Number of channels of the data
          hidden_channels: number of channels in the hidden layer of the ConvNet
          scale: Flag, whether to include scale in affine coupling layer
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
          leaky: Leaky parameter of LeakyReLUs of ConvNet2d
          init_zeros: Flag whether to initialize last conv layer with zeros
          use_lu: Flag whether to parametrize weights through the LU decomposition in invertible 1x1 convolution layers
          logscale_factor: Factor which can be used to control the scale of the log scale factor, see [source](https://github.com/openai/glow)
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1
        if "channel" == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif "channel_inv" == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif "checkerboard" in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError("Mode " + split_mode + " is not implemented.")
        param_map = nf.nets.ConvNet2d(
            channels_, kernel_size, leaky, init_zeros, actnorm=net_actnorm
        )
        # print(channels_)
        # assert len(channels_) == 3, len(channels_)
        in_chs, hidden_chs, _, out_chs = channels_
        param_map = nf.nets.ConvResidualNet(
            in_channels=in_chs,
            out_channels=out_chs,
            hidden_channels=hidden_chs,
            use_batch_norm=False,
            num_blocks=2,
            dropout_probability=dropout_probability,
            # context_channels=hidden_chs,
            # activation="relu",
        )
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Invertible 1x1 convolution
        self.flows += [nf.flows.Invertible1x1Conv(channels, use_lu)]
        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


class Squeeze(Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z):
        """Squeeze channel dimension into spatial dimensions.

        Parameters
        ----------
        z : torch.Tensor of shape (batch_size, channels, height, width)
            Input data.

        Returns
        -------
        z : torch.Tensor of shape (batch_size, channels * 4, height // 2, width // 2)
            Squeezed data.
        log_det : float
            Log determinant of the Jacobian of the transformation. This is 0, since
            no transformation is applied.
        """
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        """Unsqueeze channel dimension from spatial dimensions.

        Parameters
        ----------
        z : torch.Tensor of shape (batch_size, channels, height, width)
            The input data.

        Returns
        -------
        z : torch.Tensor of shape (batch_size, channels // 4, height * 2, width * 2)
            The output data.
        """
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det


class Injective1x1Conv(Flow):
    def __init__(
        self,
        num_channels_in: int,
        num_channels_v: int = None,
        activation: str = "linear",
        gamma: float = 0.0,
        preset_W=None,
    ):
        """Injective 1x1 convolution with support for injective cases (e.g., relu).

        Parameters
        ----------
        num_channels_in : int
            The number of input channels for this layer. This should be the number
            of channels from the input towards the latent representation
        activation : str, optional
            The activation, by default 'linear'. Can be 'relu'.
        gamma : float, optional
            L2 regularization for SVD of weight filter matrix, by default 0.0.
        """
        super().__init__()
        self.num_channels_in = num_channels_in
        if num_channels_v is None:
            num_channels_v = num_channels_in // 2
        self.num_channels_v = num_channels_v
        self.activation = activation
        self.gamma = gamma

        if self.activation == "linear":
            Q1, _ = torch.linalg.qr(torch.randn(num_channels_v, num_channels_v))
            Q2, _ = torch.linalg.qr(torch.randn(num_channels_v, num_channels_v))
            W = torch.cat([Q1, Q2], axis=1) / np.sqrt(2.0)

            # print("inside initialization: ", W.shape)
            # Initialize the weight matrix as a random orthogonal matrix as shape (n_chs_in, n_chs_in // 2)
        else:
            W, _ = torch.linalg.qr(torch.randn(num_channels_v, num_channels_v))
        if preset_W is not None:
            W = preset_W
        # Initialize the weight matrix as a random orthogonal matrix as shape (n_chs_in // 2, n_chs_in // 2)
        self.W = nn.Parameter(W)
        # print("inside initialization: ", self.W)

    def inverse(self, v):
        """Forward pass through "encoder" from input to output (latent representation).

        Decreasing channels by a factor of 2 to get output.

        XXX: needs to be inverse if with normflows, and refactor to use n_channels as the
        number of channels in the "X" data.

        Parameters
        ----------
        v : torch.Tensor of shape (batch_size, num_channels_in, height, width)
            The output representation.

        Returns
        -------
        x_ : torch.Tensor of shape (batch_size, num_channels_in * 2, height, width)
            The output.
        log_det : torch.Tensor of shape (batch_size,)
            The log determinant of the Jacobian of the 1x1 operation.
        """
        _, channels, height, width = v.shape

        assert (
            channels == self.num_channels_in
        ), f"Expected {self.num_channels_in} channels, got {channels}."

        # y = f(x) = Wx
        # SVD decomposition of the weight vector to get the log det J_{f}^T J_{f}
        # as the sum of the square singular values.
        svals = torch.linalg.svdvals(self.W)

        # compute log of its regularized singular values and sum them
        log_det = torch.log(svals + self.gamma**2 / (svals + 1e-6))
        log_det = torch.sum(log_det) * (height * width)

        # Volume is now going from v to X
        log_det *= -1

        if self.activation == "relu":
            # For injective, we are combining halves
            x_a = v[:, : self.num_channels_v, :, :]
            x_b = v[:, self.num_channels_v :, :, :]
            x = x_a - x_b
        else:
            x = v

        debug = False
        if debug:
            print(f"v shape: {v.shape}")
            print(f"W shape: {self.W.shape}")
            print(f"num_channels_in: {self.num_channels_in}")

        # Apply the weight to project data from latent space to output space
        # x_ = convolve(W, v)
        W = self.W.view(self.W.shape[0], self.W.shape[1], 1, 1)
        x_ = torch.nn.functional.conv2d(x, W, stride=1, padding=0)

        return x_, log_det

    def forward(self, x):
        """Reverse' pass, going through the decoder to get the input images.

        Increasing channels by a factor of 2.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_channels_in // 2, height, width)
            Input data

        Returns
        -------
        x : torch.Tensor of shape (batch_size, num_channels_in, height, width)
            Output data.
        """
        _, channels, height, width = x.shape

        assert (
            channels == self.num_channels_v
        ), f"Expected {self.num_channels_v} channels, got {channels}."

        # y = f(x) = Wx
        # SVD decomposition of the weight vector to get the log det J_{f}^T J_{f}
        # as the sum of the square singular values.
        try:
            svals = torch.linalg.svdvals(self.W).to(x.device)
        except Exception as e:
            print(self.W)
            print(self.W.shape)
            raise Exception(e)

        # compute log of its regularized singular values and sum them
        log_det = torch.log(
            svals + self.gamma**2 / (svals + torch.Tensor([1e-6]).to(x.device))
        )
        log_det = torch.sum(log_det) * (height * width)

        # compute the pseudo-inverse of the weight matrix: (W W^T + gamma^2 I)^{-1} W^T
        # Assume self.w is a tensor (W^T W + gamma^2 I)
        prefactor = torch.matmul(self.W, self.W.T) + self.gamma**2 * torch.eye(
            self.W.shape[0]
        ).to(x.device)

        # Inverse of prefactor
        w_pinv = torch.matmul(self.W.T, torch.linalg.inv(prefactor))

        if self.activation == "relu":
            conv_filter = torch.cat([w_pinv, -w_pinv], dim=0).to(x.device)
        else:
            conv_filter = w_pinv

        # Reshaping to fit the 1x1 convolution kernel dimensions
        # 1, 1, n_chs_in, n_chs_out
        conv_filter = conv_filter.view(conv_filter.size(0), conv_filter.size(1), 1, 1)

        # Apply convolution (assuming x is in NCHW format by default in PyTorch)
        v_ = torch.nn.functional.conv2d(x, conv_filter, stride=1, padding=0)
        v_ = torch.nn.functional.relu(v_)

        log_det *= -1
        return v_, log_det


class InjectiveGlowBlock(Flow):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        scale=True,
        scale_map="sigmoid",
        split_mode="channel",
        leaky=0.0,
        init_zeros=True,
        activation="linear",
        gamma=1e-3,
        net_actnorm=False,
        debug=False,
    ):
        """Injective generalized Flow with Injective 1x1 Convolutions.

        For full details, see the paper: https://arxiv.org/pdf/2102.10461.

        Parameters
        ----------
        channels : int
            The number of input channels of the flow block at the latent representation.
            Note: This will be doubled, or halved after the injective 1x1 convolution
            in the forward, or inverse pass, respectively.
        hidden_channels : int
            The size of the embedding dimension in the coupling layer.
        scale : bool, optional
            Whether to apply scale, by default True. See
            :class:`nf.flows.affine.AffineCouplingBlock`.
        scale_map : str, optional
            Scale map, by default "sigmoid".
        split_mode : str, optional
            Splitting mode, for possible values see Split class, by default "channel".
        leaky : float, optional
            Leaky parameter of LeakyReLUs of ConvNet2d, by default 0.0.
        init_zeros : bool, optional
            Flag whether to initialize last conv layer with zeros, by default True
        activation : str, optional
            Activation function in the 1x1 convolution, by default "linear".
        gamma : float, optional
            The l2 regularization in 1x1 convolution, by default 1e-3.
        net_actnorm : bool, optional
            Whether to normalize the activations of the convolutional network, by default False.

        """
        super().__init__()
        self.flows: List[Flow] = nn.ModuleList([])
        if channels < 1:
            raise ValueError(
                "Channels must be at least 1, as injective 1x1 convs will "
                "decrease the number of channels by half after each block."
            )

        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1
        if "channel" == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif "channel_inv" == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif "checkerboard" in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError("Mode " + split_mode + " is not implemented.")

        param_map = nf.nets.ConvNet2d(
            channels_, kernel_size, leaky, init_zeros, actnorm=net_actnorm
        )
        in_chs, hidden_chs, _, out_chs = channels_
        # param_map = nf.nets.ConvResidualNet(
        #     in_channels=in_chs,
        #     out_channels=out_chs,
        #     hidden_channels=hidden_chs,
        #     # context_channels=hidden_chs,
        #     # activation="relu",
        # )
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]

        # channels = channels * 2  # after injective 1x1 conv
        # Invertible 1x1 convolution
        channels = channels * 2  # after injective 1x1 conv
        self.flows += [
            Injective1x1Conv(
                num_channels_in=channels,
                num_channels_v=channels // 2,
                activation=activation,
                gamma=gamma,
            )
        ]

        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]
        self.debug = debug

    def forward(self, v):
        """Forward pass of the flow towards input images.

        Parameters
        ----------
        v : torch.Tensor of shape (batch_size, channels, height, width)
            Latent variable.

        Returns
        -------
        v : torch.Tensor of shape (batch_size, channels * 2, height, width)
            The transformed latent variable with increased dimensionality.
        log_det_tot : torch.Tensor of shape (batch_size,)
            Total log determinant of the transformation.
        """
        log_det_tot = torch.zeros(v.shape[0], dtype=v.dtype, device=v.device)
        for flow in self.flows:
            if self.debug:
                print(flow._get_name(), v.shape)
            v, ld = flow(v)
            log_det_tot += ld
        return v, log_det_tot

    def inverse(self, x):
        """Inverse pass of the flow.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, channels * 2, height, width)
            The output data.

        Returns
        -------
        x : torch.Tensor of shape (batch_size, channels, height, width)
            The reconstructed latent variable.
        log_det_tot : torch.Tensor of shape (batch_size,)
            Total log determinant of the transformation.
        """
        log_det_tot = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if self.debug:
                print(self.flows[i], x.shape)
            x, log_det = self.flows[i].inverse(x)
            log_det_tot += log_det
        return x, log_det_tot


# # Define the UNet model
# class ConvBlock(nn.Module):
#     def __init__(self, num_filters):
#         super(ConvBlock, self).__init__()
#         self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         return x


# class UNet(nn.Module):
#     def __init__(self, output_channels):
#         super(UNet, self).__init__()

#         self.num_filters = [32, 64]

#         self.conv_blocks1 = nn.ModuleList([ConvBlock(num_filters) for num_filters in self.num_filters])
#         self.conv_blocks2 = nn.ModuleList([ConvBlock(num_filters) for num_filters in self.num_filters[::-1]])

#         self.conv_block_bridge = ConvBlock(self.num_filters[-1])
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.concat = lambda x: torch.cat(x, dim=1)  # Concatenate along the channel dimension
#         self.conv = nn.Conv2d(self.num_filters[0], output_channels, kernel_size=1, padding=0, bias=False)

#     def forward(self, x):
#         skip_x = []

#         ## Encoder
#         for i in range(len(self.num_filters)):
#             x = self.conv_blocks1[i](x)
#             skip_x.append(x)
#             x = self.maxpool(x)

#         ## Bridge
#         x = self.conv_block_bridge(x)

#         skip_x = skip_x[::-1]  # Reverse the skip connections

#         ## Decoder
#         for i in range(len(self.num_filters)):
#             x = self.upsample(x)
#             x = self.concat([x, skip_x[i]])
#             x = self.conv_blocks2[i](x)

#         ## Output
#         x = self.conv(x)
#         return torch.sigmoid(x)  # Use sigmoid for output activation if output is in [0, 1]


# Verify the UNet model
def test_unet():
    # Define input parameters
    input_channels = 1  # For example, RGB images
    output_channels = 16  # For binary segmentation
    input_height = 32  # Height of the input image
    input_width = 32  # Width of the input image

    # Create a random input tensor
    x = torch.randn(1, input_channels, input_height, input_width)  # Batch size of 1

    # Instantiate the UNet model
    output_channels = input_channels * 2
    hidden_chs = 32
    model = nf.nets.ConvResidualNet(
        in_channels=input_channels,
        out_channels=output_channels,
        hidden_channels=hidden_chs,
        # context_channels=hidden_chs,
        # activation="relu",
    )

    # Forward pass through the model
    output = model(x)

    # Print the output shape
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


def test_convnet():
    # Define input parameters
    input_channels = 1  # For example, RGB images
    output_channels = 16  # For binary segmentation
    input_height = 32  # Height of the input image
    input_width = 32  # Width of the input image

    # Create a random input tensor
    x = torch.randn(1, input_channels, input_height, input_width)  # Batch size of 1

    # Instantiate the UNet model
    output_channels = input_channels * 2
    hidden_chs = 32
    kernel_size = (3, 1, 3)
    channels_ = (input_channels,) + 2 * (hidden_chs,)
    channels_ += (2 * input_channels,)
    model = nf.nets.ConvNet2d(
        channels_, kernel_size,
    )

    # Forward pass through the model
    output = model(x)

    # Print the output shape
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

# Run the test
if __name__ == "__main__":
    test_unet()


    test_convnet()