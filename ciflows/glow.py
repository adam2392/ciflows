from typing import List

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from normflows.flows.affine import AffineCouplingBlock
from normflows.flows.base import Flow
from normflows.flows.normalization import ActNorm


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
        self, num_channels_in: int, activation: str = "linear", gamma: float = 0.0
    ):
        """Injective 1x1 convolution with support for injective cases (e.g., relu).

        Parameters
        ----------
        num_channels_in : int
            The number of input channels for this layer. This should be the number
            of channels from the latent representation.
        activation : str, optional
            The activation, by default 'linear'. Can be 'relu'.
        gamma : float, optional
            L2 regularization for SVD of weight filter matrix, by default 0.0.
        """
        super().__init__()
        self.num_channels_in = num_channels_in
        self.activation = activation
        self.gamma = gamma

        if self.activation == "linear":
            Q1, _ = torch.linalg.qr(
                torch.randn(self.num_channels_in, self.num_channels_in, dtype=torch.float32)
            )
            Q2, _ = torch.linalg.qr(
                torch.randn(self.num_channels_in, self.num_channels_in, dtype=torch.float32)
            )
            W = torch.cat([Q1, Q2], axis=0) / np.sqrt(2.0)

            print("inside initialization: ", W.shape)
            # Initialize the weight matrix as a random orthogonal matrix as shape (n_chs_in, n_chs_in // 2)
            self.W = nn.Parameter(W)
            print("inside initialization: ", self.W)
        else:
            Q, _ = torch.linalg.qr(
                torch.randn(self.num_channels_in, self.num_channels_in)
            )
            # Initialize the weight matrix as a random orthogonal matrix as shape (n_chs_in // 2, n_chs_in // 2)
            self.W = nn.Parameter(Q)

    def forward(self, v):
        """Forward pass through "decoder" from latent representation to output.

        Increasing channels by a factor of 2 to get the output representation again.

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
        log_det = torch.log(svals + self.gamma**2 / (svals + 1e-8))
        log_det = torch.sum(log_det) * (height * width)

        # Volume is now going from v to X
        log_det *= -1

        if self.activation == "relu":
            W = torch.cat([self.W, -self.W], dim=0)
        else:
            W = self.W

        debug = False
        if debug:
            print(f"v shape: {v.shape}")
            print(f"W shape: {self.W.shape}")
            print(f"num_channels_in: {self.num_channels_in}")

        # Apply the weight to project data from latent space to output space
        # x_ = convolve(W, v)
        W = W.view(self.num_channels_in * 2, self.num_channels_in, 1, 1)
        x_ = torch.nn.functional.conv2d(v, W)

        return x_, log_det

    def inverse(self, x):
        """Reverse' pass, going through the encoder to get the latent representation.

        Decreasing channels by a factor of 2.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_channels_in * 2, height, width)
            Input data

        Returns
        -------
        x : torch.Tensor of shape (batch_size, num_channels_in, height, width)
            Output data.
        """
        _, channels, height, width = x.shape

        assert (
            channels == self.num_channels_in * 2
        ), f"Expected {self.num_channels_in * 2} channels, got {channels}."

        # y = f(x) = Wx
        # SVD decomposition of the weight vector to get the log det J_{f}^T J_{f}
        # as the sum of the square singular values.
        try:
            svals = torch.linalg.svdvals(self.W)
            svals = svals.to(x.device)
        except Exception as e:
            print(self.W)
            print(self.W.shape)
            raise Exception(e)

        # compute log of its regularized singular values and sum them
        log_det = torch.log(svals + self.gamma**2 / (svals + torch.Tensor([1e-8]).to(x.device)))
        log_det = torch.sum(log_det) * (height * width)

        # compute the pseudo-inverse of the weight matrix: (W W^T + gamma^2 I)^{-1} W^T
        # Assume self.w is a tensor (W^T W + gamma^2 I)
        prefactor = torch.matmul(self.W.T, self.W) + self.gamma**2 * torch.eye(
            self.W.shape[1]
        ).to(x.device)

        # Inverse of prefactor
        try:
            w_pinv = torch.matmul(torch.inverse(prefactor), self.W.T)
        except Exception as e:
            print()
            print('Prefactor: ')
            print(prefactor)
            print()
            print("Shapes: ")
            print(prefactor.shape, self.W.shape)
            raise Exception(e)

        if self.activation == "relu":
            conv_filter = torch.cat([w_pinv, -w_pinv], dim=0)
        else:
            conv_filter = w_pinv

        # Reshaping to fit the 1x1 convolution kernel dimensions
        # 1, 1, n_chs_in, n_chs_out
        conv_filter = conv_filter.view(
            self.num_channels_in, self.num_channels_in * 2, 1, 1
        )

        # Apply convolution (assuming x is in NCHW format by default in PyTorch)
        v_ = torch.nn.functional.conv2d(x, conv_filter)
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
        if channels < 2:
            raise ValueError(
                "Channels must be at least 2, as injective 1x1 convs will "
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
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]

        # channels = channels * 2  # after injective 1x1 conv
        # Invertible 1x1 convolution
        self.flows += [Injective1x1Conv(channels, activation=activation, gamma=gamma)]

        # Activation normalization
        channels = channels * 2  # after injective 1x1 conv
        self.flows += [ActNorm((channels,) + (1, 1))]
        self.debug = debug

    def forward(self, v):
        """Forward pass of the flow

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
                print(flow, v.shape)
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
