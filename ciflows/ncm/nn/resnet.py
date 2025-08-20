import torch
import torch.nn as nn
from ciflows.ncm.nn.base import SCMFunc


# Define a simple residual block.
class ResidualBlock(nn.Module):
    """
    A simple residual block for a fully-connected network.

    Parameters
    ----------
    in_features : int
        The number of input (and output) features.
    hidden_features : int
        The number of features in the hidden layer.
    use_layer_norm : bool, optional
        Whether to use layer normalization (default: True).
    """

    def __init__(self, in_features, hidden_features, use_layer_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.relu = nn.ReLU(inplace=True)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_features)
            self.norm2 = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if self.use_layer_norm:
            out = self.norm2(out)
        out += residual
        return self.relu(out)


# Define the ResNet-style module.
class ResNet(SCMFunc):
    """
    A ResNet-style module for generating outputs from concatenated inputs.

    This module is designed to mimic the MLP's API but uses residual blocks in
    the hidden layers. It computes the input size by concatenating parent (pa) and
    latent (u) inputs, applies an initial fully-connected layer, passes the result
    through several residual blocks, and finally maps to the desired output.

    Parameters
    ----------
    pa_size : dict
        Dictionary mapping parent variable names to their dimensions.
    u_size : dict
        Dictionary mapping latent variable names to their dimensions.
    o_size : int
        The output dimension.
    h_size : int, optional
        Hidden layer size (default: 128).
    h_layers : int, optional
        Number of residual blocks (default: 2).
    use_sigmoid : bool, optional
        Whether to apply a sigmoid activation at the end (default: True).
    use_layer_norm : bool, optional
        Whether to use layer normalization after linear layers (default: True).
    output_shape : tuple, optional
        If provided, the output tensor will be reshaped to this shape after the forward pass.

    Returns
    -------
    torch.Tensor or tuple
        If include_inp is False, returns the final output tensor.
        Otherwise, returns (output, concatenated_input).
    """

    def __init__(
        self,
        pa_size,
        u_size,
        o_size,
        h_size=128,
        h_layers=2,
        use_sigmoid=True,
        use_layer_norm=True,
        output_shape=None,
    ):
        super().__init__(pa_size, u_size, o_size, output_shape)

        self.h_size = h_size

        # Initial fully connected layer.
        layers = []
        layers.append(nn.Linear(self.i_size, self.h_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU(inplace=True))

        # Add residual blocks
        for _ in range(h_layers):
            layers.append(ResidualBlock(self.h_size, self.h_size, use_layer_norm))

        # Final output layer
        layers.append(nn.Linear(self.h_size, self.o_size))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*layers)

        # Dummy parameter for device tracking.
        self.device_param = nn.Parameter(torch.empty(0))

        # Initialize weights.
        self.apply(self.init_weights)


# --- Test Script ---
def main():
    # We want to test with:
    #   - A latent dimension of 32 (u_size)
    #   - An output dimension of 128*128*3 = 49152
    # We'll use no parent variables.
    pa_size = {}  # No parent variables.
    u_size = {"latent": 32}  # Latent dimension is 32.
    output_dim = 128 * 128 * 3  # 49152

    # Instantiate the ResNet module.
    # To "scale up slowly," we choose a small hidden dimension and several residual blocks.
    # Here, h_size=64 and h_layers=5 will gradually process the input.
    model = ResNet(
        pa_size,
        u_size,
        output_dim,
        h_size=64,
        h_layers=5,
        use_sigmoid=True,
        use_layer_norm=True,
        output_shape=(3, 128, 128),
    )
    model.eval()  # Evaluation mode

    batch_size = 4
    # Since there are no parents, pa is empty.
    pa = {}
    u = {"latent": torch.randn(batch_size, 32)}

    # Forward pass.
    output = model(pa, u)
    print("Output shape:", output.shape)  # Expected: [4, 49152]
    # print("Sample output (first element):", output[0])

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


if __name__ == "__main__":
    main()
