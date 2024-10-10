import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn

from ciflows.loss import volume_change_surrogate
from ciflows.vae import ConvDecoder, ConvEncoder

# model = nn.Linear(16, 128)
# input = torch.randn(4, 16)

# params = {name: p for name, p in model.named_parameters()}
# tangents = {name: torch.rand_like(p) for name, p in params.items()}

# with fwAD.dual_level():
#     for name, p in params.items():
#         delattr(model, name)
#         setattr(model, name, fwAD.make_dual(p, tangents[name]))

#     out = model(input)
#     jvp = fwAD.unpack_dual(out).tangent



# Define a simple linear encoder and decoder
class LinearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.linear(x)


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.linear(z)


# Main testing function
def test_volume_change_surrogate():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define dimensions
    channels = 1  # For example, grayscale images
    height = 28
    width = 28
    hidden_size = 64
    batch_size = 32

    # Create random input data
    x = torch.randn(batch_size, channels, height, width)

    # Instantiate encoder and decoder
    encoder = ConvEncoder(channels, hidden_size, height, width)
    decoder = ConvDecoder(hidden_size, channels, height, width)

    # Run the volume change surrogate function
    surrogate_loss, v, xhat = volume_change_surrogate(
        x, encoder, decoder, hutchinson_samples=2
    )

    # Print the results
    print("Surrogate Loss:", surrogate_loss.item())
    print("Latent Representation Shape:", v.shape)
    print("Reconstructed Output Shape:", xhat.shape)


def test_volume_change_surrogate_linear():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define dimensions
    input_dim = 28 * 28  # For example, MNIST images
    latent_dim = 64
    batch_size = 32

    # Create random input data
    x = torch.randn(batch_size, input_dim)

    # Instantiate encoder and decoder
    encoder = LinearEncoder(input_dim, latent_dim)
    decoder = LinearDecoder(latent_dim, input_dim)

    # Run the volume change surrogate function
    surrogate_loss, v, xhat = volume_change_surrogate(
        x, encoder, decoder, hutchinson_samples=2
    )

    # Print the results
    print("Surrogate Loss:", surrogate_loss.item())
    print("Latent Representation Shape:", v.shape)
    print("Reconstructed Output Shape:", xhat.shape)


# Execute the test
if __name__ == "__main__":
    test_volume_change_surrogate()
