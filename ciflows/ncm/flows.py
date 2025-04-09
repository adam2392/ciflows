import normflows as nf

import torch
import torch.nn as nn
import numpy as np


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, flows, merges, latent_shapes, x_shape):
        """Constructor

        Args:

            flows: List of list of flows for each level
            merges: List of merge/split operations (forward pass must do merge)
            latent_shapes: List of latent variable shapes for each level
        """
        super().__init__()
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.latent_shapes = latent_shapes
        self.x_shape = x_shape
        self.num_levels = len(flows)

        # test if latent shapes passed through are correct shapes
        sample_x_shape = (1, *x_shape)
        sample_x = torch.randn(sample_x_shape)
        sample_z, _ = self.inverse_and_log_det(sample_x)
        for i, z in enumerate(sample_z):
            assert z.shape == (
                1,
                *self.latent_shapes[i],
            ), f"Shape of latent variable {i} is {z.shape}, expected {self.latent_shapes[i]}"

        # now test passing from latent shapes to x shape
        sample_z = [torch.randn((1, *shape)) for shape in self.latent_shapes]
        sample_x_recon, _ = self.forward_and_log_det(sample_z)
        assert (
            sample_x_recon.shape == sample_x_shape
        ), f"Shape of reconstructed x is {sample_x_recon.shape}, expected {sample_x_shape}"

    def forward_and_log_det(self, z):
        """Get observed variable x from list of latent variables z

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z[0].device)
        for i in range(self.num_levels):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_)
                log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


def make_flow_model(input_dim=32, num_layers=32, hidden_dim=128, num_hidden_layers=4):
    # Define list of flows
    flows = []
    for i in range(num_layers):
        # Last layer is initialized by zeros making training more stable
        # Neural network with N hidden layers having 64 units each
        layer_sizes = [input_dim // 2]
        for j in range(num_hidden_layers):
            layer_sizes.append(hidden_dim)
        # last layer is output mean and stdev
        layer_sizes.append(2)
        
        param_map = nf.nets.MLP(layer_sizes, init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(input_dim, mode="shuffle"))

    # Construct flow model
    model = NormalizingFlow(flows)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print()
    return model


def test_model():
    torch.manual_seed(42)

    input_dim = 32
    batch_size = 5
    atol = 1e-4

    model = make_flow_model()
    model.eval()

    # Generate dummy data
    x = torch.randn(batch_size, input_dim)

    # Test invertibility: x -> z -> x_recon
    z, _ = model.inverse_and_log_det(x)
    x_recon, _ = model.forward_and_log_det(z)
    print(f"x shape: {x.shape}, x_recon shape: {x_recon.shape}")
    max_diff = (x - x_recon).abs().max().item()
    print(f"Max diff x -> z -> x_recon: {max_diff:.6f}")
    assert torch.allclose(x, x_recon, atol=atol), "Inversion (x -> z -> x) failed"

    # Test z -> x -> z_recon
    z = torch.randn(batch_size, input_dim)
    x_gen, _ = model.forward_and_log_det(z)
    z_recon, _ = model.inverse_and_log_det(x_gen)
    max_diff = (z - z_recon).abs().max().item()
    print(f"Max diff z -> x -> z_recon: {max_diff:.6f}")
    assert torch.allclose(z, z_recon, atol=atol), "Inversion (z -> x -> z) failed"


def make_img_flow_model():
    # Set up model

    # Define flows
    L = 3
    K = 16
    torch.manual_seed(0)

    input_shape = (3, 32, 32)
    n_dims = np.prod(input_shape)
    channels = 3
    hidden_channels = 256
    split_mode = "channel"
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    merges = []
    flows = []
    latent_shapes = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [
                nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i), hidden_channels, split_mode=split_mode, scale=scale
                )
            ]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (
                input_shape[0] * 2 ** (L - i),
                input_shape[1] // 2 ** (L - i),
                input_shape[2] // 2 ** (L - i),
            )
        else:
            latent_shape = (
                input_shape[0] * 2 ** (L + 1),
                input_shape[1] // 2**L,
                input_shape[2] // 2**L,
            )
        latent_shapes += [latent_shape]

    print("latent_shapes:")
    print(latent_shapes)
    print()
    print()
    print(len(flows))
    print(len(merges))

    # Construct flow model with the multiscale architecture
    model = MultiscaleFlow(flows, merges, latent_shapes=latent_shapes, x_shape=input_shape)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print()


def test_img_flow_model(model):
    input_shape = model.x_shape
    latent_shapes = model.latent_shapes
    model.eval()

    batch_size = 6
    atol = 1e-3
    # Create dummy input image batch
    x = torch.randn(batch_size, *input_shape)

    # --- Test inverse_and_log_det followed by forward_and_log_det ---
    z, _ = model.inverse_and_log_det(x)
    x_recon, _ = model.forward_and_log_det(z)

    print(x.shape, x_recon.shape, [i.shape for i in z])
    max_diff = (x - x_recon).abs().max().item()
    print(f"Max diff x -> z -> x_recon: {max_diff:.6f}")
    assert torch.allclose(x, x_recon, atol=atol), "Inversion (x -> z -> x) failed"

    # --- Test forward_and_log_det followed by inverse_and_log_det ---
    # Sample latent variables using model.q0
    z_sample = []
    for i in range(len(latent_shapes)):
        z_sample.append(torch.randn(batch_size, *latent_shapes[i]))
    x_gen, _ = model.forward_and_log_det(z_sample)
    z_recon, _ = model.inverse_and_log_det(x_gen)

    print([z.shape for z in z_sample])
    for z, z_rec in zip(z_sample, z_recon):
        max_diff = (z - z_rec).abs().max().item()
        print(f"Max diff z -> x -> z_recon: {max_diff:.6f}")
        assert torch.allclose(z, z_rec, atol=atol), "Inversion (z -> x -> z) failed"


if __name__ == "__main__":
    # model = make_flow_model()
    # test_img_flow_model(model)

    test_model()
