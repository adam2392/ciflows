import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###############################################################################
# Basic building blocks for 2D Glow
###############################################################################

class ActNorm2D(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape: tuple (channels, H, W). H and W are used only for initialization.
        """
        super().__init__()
        self.num_channels = input_shape[0]
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
    
    def forward(self, x, reverse=False):
        if not self.initialized:
            with torch.no_grad():
                # compute per-channel statistics
                mean = x.mean(dim=[0, 2, 3], keepdim=True)
                std = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6
                self.bias.data = -mean
                self.log_scale.data = -torch.log(std)
            self.initialized = True
        if not reverse:
            return (x + self.bias) * torch.exp(self.log_scale)
        else:
            return x * torch.exp(-self.log_scale) - self.bias

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        """
        A learned invertible 1x1 convolution.
        """
        super().__init__()
        # Initialize weight as an orthogonal matrix.
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(W.unsqueeze(-1).unsqueeze(-1))  # shape: (C, C, 1, 1)
    
    def forward(self, x, reverse=False):
        if not reverse:
            return F.conv2d(x, self.weight)
        else:
            W = self.weight.squeeze(-1).squeeze(-1)
            W_inv = torch.inverse(W).unsqueeze(-1).unsqueeze(-1)
            return F.conv2d(x, W_inv)

class AffineCoupling2D(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        """
        Splits the channels into two groups. If num_channels is odd, we pad an extra channel.
        The conditioning network is a small convolutional network that predicts scale and translation.
        """
        super().__init__()
        self.orig_channels = num_channels
        if num_channels % 2 != 0:
            # pad one channel so that the total number is even
            self.pad = True
            num_channels_padded = num_channels + 1
        else:
            self.pad = False
            num_channels_padded = num_channels
        self.split_channels = num_channels_padded // 2
        self.net = nn.Sequential(
            nn.Conv2d(self.split_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, (num_channels_padded - self.split_channels) * 2, kernel_size=3, padding=1)
        )
    
    def forward(self, x, reverse=False):
        # If we padded earlier, add an extra zero channel (at the end).
        if self.pad:
            # pad on the channel dimension; F.pad expects pad in the form (pad_left, pad_right, pad_top, pad_bottom, ...)
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        
        # Now x.shape[1] is even; split the channels.
        x1 = x[:, :self.split_channels, :, :]
        x2 = x[:, self.split_channels:, :, :]
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)  # for numerical stability
        if not reverse:
            x2 = x2 * torch.exp(s) + t
        else:
            x2 = (x2 - t) * torch.exp(-s)
        x_out = torch.cat([x1, x2], dim=1)
        # If padded, remove the extra channel.
        if self.pad:
            x_out = x_out[:, :self.orig_channels, :, :]
        return x_out

###############################################################################
# A single Glow step for 2D images
###############################################################################

class GlowStep2D(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        """
        A single Glow step: ActNorm -> Invertible 1x1 Conv -> Affine Coupling.
        num_channels: expected channels after any prior squeezing.
        """
        super().__init__()
        # We pass a dummy (H, W) for initialization of ActNorm; usually the spatial dims are known.
        self.actnorm = ActNorm2D((num_channels, 32, 32))
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.affine_coupling = AffineCoupling2D(num_channels, hidden_channels)
    
    def forward(self, x, reverse=False):
        if not reverse:
            x = self.actnorm(x, reverse=False)
            x = self.inv_conv(x, reverse=False)
            x = self.affine_coupling(x, reverse=False)
        else:
            x = self.affine_coupling(x, reverse=True)
            x = self.inv_conv(x, reverse=True)
            x = self.actnorm(x, reverse=True)
        return x

###############################################################################
# Glow Flow Block: Stack of Glow steps (2D)
###############################################################################

class GlowFlowBlock2D(nn.Module):
    def __init__(self, num_channels, hidden_channels, num_steps=3):
        """
        A block applying a fixed number of Glow steps sequentially.
        """
        super().__init__()
        self.steps = nn.ModuleList([
            GlowStep2D(num_channels, hidden_channels) for _ in range(num_steps)
        ])
    
    def forward(self, x, reverse=False):
        if not reverse:
            for step in self.steps:
                x = step(x, reverse=False)
        else:
            for step in reversed(self.steps):
                x = step(x, reverse=True)
        return x

###############################################################################
# Squeeze and Split operations (for multi–scale architecture)
###############################################################################

class Squeeze2d(nn.Module):
    def forward(self, x, reverse=False):
        if not reverse:
            B, C, H, W = x.size()
            assert H % 2 == 0 and W % 2 == 0, "H and W must be even for squeeze"
            # Rearrange (B, C, H, W) to (B, 4*C, H//2, W//2)
            x = x.view(B, C, H//2, 2, W//2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, C*4, H//2, W//2)
            return x
        else:
            B, C, H, W = x.size()
            assert C % 4 == 0, "C must be divisible by 4 for unsqueeze"
            x = x.view(B, C//4, 2, 2, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(B, C//4, H*2, W*2)
            return x

class Split2d(nn.Module):
    def forward(self, x, reverse=False, z=None):
        if not reverse:
            B, C, H, W = x.size()
            assert C % 2 == 0, "Number of channels must be even for split"
            x1 = x[:, :C//2, :, :]
            x2 = x[:, C//2:, :, :]
            return x1, x2
        else:
            # In reverse mode, z (the factored-out tensor) is provided.
            return torch.cat([x, z], dim=1)

###############################################################################
# A single multi–scale level for 2D Glow
###############################################################################

class GlowLevel2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_steps_per_level, do_split=True):
        """
        in_channels: number of channels at this level before squeezing.
        After squeeze, channels become (4 * in_channels).
        If do_split is True, a reversible split is applied (factoring out half the channels).
        """
        super().__init__()
        self.squeeze = Squeeze2d()
        # After squeezing, channels = 4 * in_channels.
        self.flow = GlowFlowBlock2D(in_channels * 4, hidden_channels, num_steps=num_steps_per_level)
        self.do_split = do_split
        if do_split:
            self.split = Split2d()
    
    def forward(self, x, reverse=False, z=None):
        if not reverse:
            # Squeeze spatially: (B, C, H, W) -> (B, 4*C, H//2, W//2)
            x = self.squeeze(x, reverse=False)
            # Apply Glow steps.
            x = self.flow(x, reverse=False)
            if self.do_split:
                x, z_new = self.split(x, reverse=False)
                return x, z_new
            else:
                return x, None
        else:
            if self.do_split:
                assert z is not None, "Must provide z when reversing a split."
                x = self.split(x, reverse=True, z=z)
            x = self.flow(x, reverse=True)
            x = self.squeeze(x, reverse=True)
            return x

###############################################################################
# Full Multi–Scale Glow Block for 2D images
###############################################################################

class MultiScaleGlow2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_levels, num_steps_per_level, squeeze_factor=2):
        """
        in_channels: channels of the input image.
        hidden_channels: for the coupling network.
        num_levels: number of multi–scale levels.
        num_steps_per_level: number of Glow steps per level.
        squeeze_factor: factor used to squeeze spatial dimensions (default is 2).
          (For this implementation, a squeeze factor of 2 is assumed.)
        """
        super().__init__()
        self.levels = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_levels):
            do_split = (i < num_levels - 1)
            level = GlowLevel2D(current_channels, hidden_channels, num_steps_per_level, do_split=do_split)
            self.levels.append(level)
            # Update channel count:
            # After squeezing: channels become 4*current_channels.
            # If split, keep the first half: 4*current_channels/2 = 2*current_channels.
            # Otherwise, keep all 4*current_channels.
            if do_split:
                current_channels = 2 * current_channels
            else:
                current_channels = 4 * current_channels
        self.final_channels = current_channels  # Not used further here.
    
    def forward(self, x):
        """
        Forward pass that:
          1. Decomposes x into latent factors over multiple levels.
          2. Immediately reconstructs (inverts) them to produce a “mixed” image.
        The overall mapping is bijective.
        """
        z_list = []
        for level in self.levels:
            x, z_new = level(x, reverse=False)
            if z_new is not None:
                z_list.append(z_new)
        # In this design, we immediately run the inverse to reassemble a mixed image.
        x_mixed = self.inverse(x, z_list)
        return x_mixed
    
    def inverse(self, x, z_list):
        """
        Reconstruct the image by running the inverse pass over the levels.
        """
        for level in reversed(self.levels):
            if level.do_split:
                z = z_list.pop()
                x = level(x, reverse=True, z=z)
            else:
                x = level(x, reverse=True)
        return x

###############################################################################
# Test Script
###############################################################################

def test_multiscale_glow():
    print("Verifying Multi–Scale Glow (2D) for an image (3, 32, 32)...")
    batch_size = 8
    channels = 3
    height, width = 32, 32
    hidden_channels = 64
    num_levels = 3
    num_steps_per_level = 8
    
    # Create a dummy image of shape (B, 3, 32, 32)
    x = torch.randn(batch_size, channels, height, width)
    
    model = MultiScaleGlow2D(in_channels=channels, hidden_channels=hidden_channels,
                             num_levels=num_levels, num_steps_per_level=num_steps_per_level)
    
    # Run full forward pass (mixing)
    x_mixed = model(x)
    print("Mixed image shape:", x_mixed.shape)
    
    # To verify inversion, we perform a forward pass separately to obtain the latent factors
    # and then run the inverse explicitly.
    x_forward, z_list = run_forward_levels(model, x)
    x_recon = model.inverse(x_forward, z_list)
    recon_error = torch.abs(x - x_recon).mean().item()
    print(f"Mean absolute reconstruction error: {recon_error:.6f}")

def run_forward_levels(model, x):
    """
    Helper function that runs the forward pass through each level of the multi–scale Glow,
    returning the final x (which is not yet inverted) and the list of latent factors.
    """
    z_list = []
    for level in model.levels:
        x, z_new = level(x, reverse=False)
        if z_new is not None:
            z_list.append(z_new)
    return x, z_list

if __name__ == '__main__':
    test_multiscale_glow()
