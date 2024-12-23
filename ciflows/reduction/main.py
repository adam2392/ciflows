import normflows as nf
import torch

from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.flows.model import CausalNormalizingFlow


def make_nf_model(K=32, debug=False):
    """Make normalizing flow model."""
    # Define list of flows
    if debug:
        # K = 32
        net_hidden_layers = 3
        net_hidden_dim = 128
    else:
        # K = 8  # v1
        # K = 32  # v2
        net_hidden_layers = 3
        net_hidden_dim = 128

    latent_dim = 48

    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_dim, net_hidden_layers, net_hidden_dim
            )
        ]

    node_dimensions = {
        0: 22,
        1: 22,
        2: 4,
    }
    edge_list = [(1, 2)]
    noise_means = {
        0: torch.zeros(node_dimensions[0]),
        1: torch.zeros(node_dimensions[1]),
        2: torch.zeros(node_dimensions[2]),
    }
    noise_variances = {
        0: torch.ones(node_dimensions[0]),
        1: torch.ones(node_dimensions[1]),
        2: torch.ones(node_dimensions[2]),
    }
    intervened_node_means = [
        {2: torch.ones(node_dimensions[2]) + 4},
        {2: torch.ones(node_dimensions[2]) + 8},
    ]
    intervened_node_vars = [
        {2: torch.ones(node_dimensions[2])},
        {2: torch.ones(node_dimensions[2])},
    ]

    # confounded_list = [(0, 1)]
    confounded_list = []

    # independent noise with causal prior
    q0 = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
        trainable_edges=True,
    )

    # Construct flow model with the multiscale architecture
    model = CausalNormalizingFlow(q0, flows)
    return model
