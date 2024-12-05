from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from numpy.testing import assert_array_equal
from torch import nn

from ciflows.distributions.multidistr import MultidistrCausalFlow


class LinearGaussianDag(MultidistrCausalFlow):
    def __init__(
        self,
        node_dimensions,
        edge_list,
        noise_means=None,
        noise_variances=None,
        confounded_list=None,
        intervened_node_means=None,
        intervened_node_vars=None,
    ):
        """
        Note, this class pre-specifies the list of interventions/domain changes that can occur.

        Args:
            node_dimensions (dict): Dictionary mapping node names to their feature dimensions.
                                    Example: {'A': 5, 'B': 4, 'C': 3}
            edge_list (list): List of directed edges as tuples (source, target).
                              Example: [('A', 'B'), ('C', 'B')]
            noise_means (dict): Dictionary mapping node names to their noise means.
                                Example: {'A': 0.0, 'B': 0.0, 'C': 0.0}
            noise_variances (dict): Dictionary mapping node names to their noise variances.
                                Example: {'A': 0.1, 'B': 0.05, 'C': 0.2}
            confounded_list (list): List of tuples of confounded variables. Each pair tuple contains the names of the
                                    confounded variables that are connected via a bidirected edge. This is represented
                                    by a standard gaussian.
            intervened_node_means (list of dict): List of dictionaries mapping intervened
                                node names to their noise means.
                                Each dictionary corresponds to a different distribution index.
            intervened_node_vars (list of dict): List of dictionaries mapping intervened node names to their noise variances.
                                Each dictionary corresponds to a different distribution index.
        """
        super(LinearGaussianDag, self).__init__()

        # default is standard gaussian noise for all nodes and their dimensionalities
        if noise_means is None:
            noise_means = {node: 0.0 for node in node_dimensions}
        if noise_variances is None:
            noise_variances = {node: 1.0 for node in node_dimensions}
        if confounded_list is None:
            confounded_list = []

        # default is no intervened nodes
        if intervened_node_means is None:
            intervened_node_means = []
        if intervened_node_vars is None:
            intervened_node_vars = []
        if len(intervened_node_means) != len(intervened_node_vars):
            raise ValueError(
                "Intervened node means and variances must have the same length."
            )

        self.node_dimensions = node_dimensions
        self.latent_dim = sum(node_dimensions.values())
        self.edge_list = edge_list
        self.noise_variances = noise_variances
        self.noise_means = noise_means
        self.confounded_list = confounded_list

        # mappings for multiple distributions
        self.intervened_node_means = intervened_node_means
        self.intervened_node_vars = intervened_node_vars

        # maps distribution index to a set of variables
        self.distr_idx_map = dict()
        self.distr_idx_map[0] = set()

        # Create a weight dictionary for edges
        self.edge_weights = nn.ParameterDict(
            {
                f"{src}->{tgt}": nn.Parameter(
                    torch.randn(node_dimensions[src], node_dimensions[tgt]) + 1.0,
                    requires_grad=False,
                )
                for src, tgt in edge_list
            }
        )

        # Create a topological ordering of nodes
        self.graph = nx.DiGraph(edge_list)
        for node in node_dimensions:
            if node not in self.graph.nodes:
                self.graph.add_node(node)
        self.topological_order = list(nx.topological_sort(self.graph))

        # Register buffers for node parameters (non-trainable)
        for node in node_dimensions:
            # For each node, register noise mean and variance as buffers
            self.register_buffer(
                f"exog_mean_{node}_0", torch.tensor(noise_means.get(node, 0.0))
            )
            self.register_buffer(
                f"exog_variance_{node}_0", torch.tensor(noise_variances.get(node, 1.0))
            )

        self.confounder_means = defaultdict(dict)
        self.confounder_variances = defaultdict(dict)
        for node_a, node_b in confounded_list:
            confounder_mean = 0.0
            confounder_var = 1.0
            self.confounder_means[node_a][node_b] = confounder_mean
            self.confounder_means[node_b][node_a] = confounder_mean

            self.confounder_variances[node_a][node_b] = confounder_var
            self.confounder_variances[node_b][node_a] = confounder_var

            # For each pair of confounded variables, register a shared noise mean and variance
            self.register_buffer(
                f"confounded_mean_{node_a}_{node_b}",
                torch.tensor(self.confounder_means[node_a][node_b]),
            ),
            self.register_buffer(
                f"confounded_variance_{node_a}_{node_b}",
                torch.tensor(self.confounder_variances[node_a][node_b]),
            )

        # Register intervened node means and variances as buffers
        for idx, (intervened_node_mean, intervened_node_var) in enumerate(
            zip(intervened_node_means, intervened_node_vars)
        ):
            if not all(key in intervened_node_var for key in intervened_node_mean):
                raise ValueError(
                    "Intervened node means and variances must have the same keys."
                )
            idx += 1  # start indexing at 1, since 0 is reserved for observational exogenous noise

            self.distr_idx_map[idx] = set(intervened_node_mean.keys())
            for node, mean in intervened_node_mean.items():
                var = intervened_node_var[node]
                self.register_buffer(
                    f"exog_mean_{node}_{idx}",
                    torch.tensor(mean),
                )
                self.register_buffer(
                    f"exog_variance_{node}_{idx}",
                    torch.tensor(var),
                )

    def forward(self, batch_size, distr_idx=None, as_dict=False):
        """
        Sample data from the Linear Gaussian DAG.

        Args:
            batch_size (int): Number of samples to generate.
            distr_idx (int): Distribution index to sample from.

        Returns:
            samples (dict): A dictionary of node samples. Each key is a node, and the value is
                            a tensor of shape (batch_size, node_dimension).
        """
        if distr_idx is None:
            distr_idx = 0
        device = next(self.parameters()).device

        # Initialize a dictionary to hold the samples
        samples = {node: None for node in self.node_dimensions}

        intervened_vars = self.distr_idx_map[distr_idx]

        # Process nodes in topological order
        for node in self.topological_order:
            # Compute contribution from parents
            node_dim = self.node_dimensions[node]
            parent_contributions = torch.zeros(batch_size, node_dim).to(device)
            for parent in self.graph.predecessors(node):
                weight = self.edge_weights[f"{parent}->{node}"]
                parent_contributions += samples[parent] @ weight

            # Sample from the conditional normal distribution
            mean = parent_contributions  # Mean determined by parents
            # noise_mean = self.noise_means.get(node, 0.0)  # Non-zero mean of the noise
            # Exogenous noise input - depends on the distribution
            if distr_idx == 0 or node not in intervened_vars:
                # Non-zero noise mean and variance
                noise_mean = getattr(
                    self, f"exog_mean_{node}_0", 0.0
                )  # self.noise_means.get(node, 0.0)

                # Compute total noise variance (node-specific + confounders)
                node_noise_var = torch.tensor(
                    getattr(self, f"exog_variance_{node}_0", 1.0)
                )
            else:
                # Intervened noise mean and variance
                noise_mean = getattr(self, f"exog_mean_{node}_{distr_idx}")
                node_noise_var = getattr(self, f"exog_variance_{node}_{distr_idx}")

            noise_std = torch.sqrt(
                torch.tensor(node_noise_var)
            )  # Standard deviation of the noise

            # confounder noise
            confounder_noise = torch.tensor(0.0).to(device)
            for confounded_node in self.confounder_means.get(node, {}):
                confounder_mean = self.confounder_means[node][confounded_node]
                confounder_var = self.confounder_variances[node][confounded_node]
                confounder_noise += confounder_mean + torch.randn(
                    batch_size, node_dim
                ).to(device) * torch.sqrt(torch.tensor(confounder_var))

            # exogenous noise
            noise = noise_mean + noise_std * torch.randn(batch_size, node_dim).to(
                device
            )  # Gaussian noise with non-zero mean

            samples[node] = (
                mean + noise + confounder_noise
            )  # Sampled value for each node in the batch

        if not as_dict:
            samples = self._dict_to_tensor(samples)
            assert_array_equal(samples.shape, (batch_size, self.latent_dim))

        log_p = self.log_prob(
            samples, distr_idx=torch.ones(batch_size, dtype=int) * distr_idx
        )
        return samples, log_p

    def _dict_to_tensor(self, dataset):
        x = []
        for node, node_dim in self.node_dimensions.items():
            x.append(dataset[node])
            assert (
                dataset[node].shape[1] == node_dim
            ), f"Node {node} has incorrect dimension: {dataset[node]} != {node_dim}"
        x = torch.cat(x, dim=1)
        return x

    def _tensor_to_dict(self, x):
        # dataset = TensorDict()
        dataset = dict()
        start = 0
        for node, node_dim in self.node_dimensions.items():
            end = start + node_dim
            dataset[node] = x[:, start:end]
            start = end
        return dataset

    def log_prob(self, x, distr_idx=None):
        """
        Compute the log-probability of a given dataset.

        dataset (dict): A dictionary mapping node names to tensors of shape (batch_size, node_dimension).
                            Example: {'A': tensor of shape (batch_size, 5),
                                      'B': tensor of shape (batch_size, 4),
                                      'C': tensor of shape (batch_size, 3)}

        Args:
            x : tensor of shape (batch_size, latent_dim)
                This is transformed to a dataset dict.
            distr_idx (list): A list of distribution indices for each sample in the batch.
        Returns:
            log_prob (torch.Tensor): A tensor of shape (batch_size,) containing the log-probabilities for each sample.
        """
        if isinstance(x, torch.Tensor):
            dataset = self._tensor_to_dict(x)
            device = x.device
        else:
            dataset = x
            device = "cpu"

        if distr_idx is None:
            distr_idx = [0] * len(dataset[next(iter(dataset))])
        distr_idx = np.array(distr_idx)
        # if any(not isinstance(idx, int) for idx in distr_idx):
        #     raise RuntimeError(f'Distribution indices should be ints, not {type(distr_idx[0])}.')
        batch_size = len(distr_idx)
        log_prob = torch.zeros(batch_size).to(
            device
        )  # Initialize batch-wise log-probability

        unique_distrs = set(distr_idx)
        for idx in unique_distrs:
            # find batch indices for this distribution
            idx_mask = torch.tensor(distr_idx) == idx
            intervened_vars = self.distr_idx_map[idx]

            for node in self.topological_order:
                node_dim = self.node_dimensions[node]

                # extract data for this node, and corresponding distribution
                node_data = dataset[node][idx_mask, :]
                parent_contributions = torch.zeros_like(node_data).to(device)

                # Compute mean based on parent contributions
                for parent in self.graph.predecessors(node):
                    weight = self.edge_weights[f"{parent}->{node}"]
                    parent_contributions += dataset[parent][idx_mask] @ weight

                # Add contributions from confounders
                confounder_contributions = torch.zeros_like(node_data).to(device)
                if node in self.confounder_means:
                    for confounder_nbr, confounder_mean in self.confounder_means[
                        node
                    ].items():
                        # Contribution of confounder to the node's mean
                        confounder_contributions += confounder_mean

                # Exogenous noise input - depends on the distribution
                if idx == 0 or node not in intervened_vars:
                    # Non-zero noise mean and variance
                    noise_mean = getattr(self, f"exog_mean_{node}_0")

                    # Compute total noise variance (node-specific + confounders)
                    node_noise_var = torch.tensor(
                        getattr(self, f"exog_variance_{node}_0")
                    )
                else:
                    # Intervened noise mean and variance
                    noise_mean = getattr(self, f"exog_mean_{node}_{idx}")
                    node_noise_var = getattr(self, f"exog_variance_{node}_{idx}")

                node_noise_std = torch.sqrt(node_noise_var)
                if node in self.confounder_means:
                    for confounder_nbr, confounder_var in self.confounder_means[
                        node
                    ].items():
                        node_noise_var += confounder_var
                node_noise_std = torch.sqrt(torch.tensor(node_noise_var))
                # Combine all contributions to compute the node's conditional mean
                conditional_mean = (
                    parent_contributions + confounder_contributions + noise_mean
                )

                # Compute Gaussian log-probability for this node
                residual = node_data - conditional_mean
                # print()
                # print('Summary of terms: ')
                # print(x.shape, dataset[node].shape, node_data.shape, residual.shape, node_noise_std.shape)
                quadratic_term = -0.5 * torch.sum(
                    (torch.divide(residual, node_noise_std)) ** 2, dim=1
                )  # Quadratic term
                # normalization_term = - 0.5 * node_dim * torch.log(
                #     2 * torch.pi * node_noise_var
                # )  # Normalization constant
                normalization_term = -0.5 * (
                    node_dim * torch.log(torch.tensor(2 * torch.pi))
                    + torch.sum(torch.log(node_noise_var))
                )  # Normalization constant

                node_log_prob = quadratic_term + normalization_term
                log_prob[idx_mask] += node_log_prob  # Accumulate log-probability

        return log_prob


if __name__ == "__main__":
    node_dimensions = {"A": 5, "B": 4, "C": 3}
    edge_list = [("A", "B"), ("C", "B")]
    noise_variances = {"A": 1.0, "B": 1.0, "C": 1.0}
    noise_means = {"A": 2.0, "B": 0.0, "C": 0.0}
    confounded_list = [("A", "C")]
    intervened_node_means = [
        {"A": 0.5, "B": -0.3},  # First intervention
        {"B": 0.1, "C": -0.2},  # Second intervention
    ]
    intervened_node_vars = [
        {"A": 0.02, "B": 0.05},  # Variances for first intervention
        {"B": 0.1, "C": 0.2},  # Variances for second intervention
    ]
    sampler = LinearGaussianDag(
        node_dimensions,
        edge_list,
        noise_means,
        noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )
    batch_size = 1000
    distr_idx = 0
    samples = sampler.sample(batch_size, distr_idx=distr_idx, as_dict=True)
    distr_idx = 0

    for node, data in samples.items():
        print(f"Node {node}: {data.shape}")

    print(sampler.distr_idx_map)

    log_probs = sampler.log_prob(
        samples, distr_idx=np.ones(batch_size, dtype=int) * distr_idx
    )
    print(log_probs.shape)  # Should be (10,)
    # print(log_probs)  # Log-probabilities for each sample
    print(-log_probs.mean())

    # Generate samples
    batch_size = 1000
    # samples = sampler.sample(batch_size, as_dict=True)

    # Verify the sample distribution
    import matplotlib.pyplot as plt

    # Plot histograms for one of the dimensions of 'A', 'B', and 'C'
    plt.figure(figsize=(12, 4))
    for i, node in enumerate(["A", "B", "C"]):
        plt.subplot(1, 3, i + 1)
        plt.hist(
            samples[node][:, 2].detach().cpu().numpy(),
            bins=50,
            density=True,
            alpha=0.7,
            label=f"{node}",
        )
        plt.title(f"Node {node} (Dim 0)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
    plt.tight_layout()
    plt.show()
