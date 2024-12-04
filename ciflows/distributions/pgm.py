from torch import nn
import torch
import networkx as nx


class LinearGaussianDAGSampler:
    def __init__(self, node_dimensions, edge_list, noise_means, noise_variances):
        """
        Args:
            node_dimensions (dict): Dictionary mapping node names to their feature dimensions.
                                    Example: {'A': 5, 'B': 4, 'C': 3}
            edge_list (list): List of directed edges as tuples (source, target).
                              Example: [('A', 'B'), ('C', 'B')]
            noise_variances (dict): Dictionary mapping node names to their noise variances (diagonal covariance).
                                    Example: {'A': 0.1, 'B': 0.05, 'C': 0.2}
        """
        self.node_dimensions = node_dimensions
        self.edge_list = edge_list
        self.noise_variances = noise_variances
        self.noise_means = noise_means

        # Create a weight dictionary for edges
        self.edge_weights = nn.ParameterDict(
            {
                f"{src}->{tgt}": nn.Parameter(
                    torch.randn(node_dimensions[src], node_dimensions[tgt])
                )
                for src, tgt in edge_list
            }
        )

        # Create a topological ordering of nodes
        self.graph = nx.DiGraph(edge_list)
        self.topological_order = list(nx.topological_sort(self.graph))

    def sample(self, batch_size):
        """
        Sample data from the Linear Gaussian DAG.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            samples (dict): A dictionary of node samples. Each key is a node, and the value is
                            a tensor of shape (batch_size, node_dimension).
        """
        # Initialize a dictionary to hold the samples
        samples = {node: None for node in self.node_dimensions}

        # Process nodes in topological order
        for node in self.topological_order:
            # Compute contribution from parents
            node_dim = self.node_dimensions[node]
            parent_contributions = torch.zeros(batch_size, node_dim)
            for parent in self.graph.predecessors(node):
                weight = self.edge_weights[f"{parent}->{node}"]
                parent_contributions += samples[parent] @ weight

            # Add Gaussian noise
            # noise_std = torch.sqrt(torch.tensor(self.noise_variances[node]))
            # noise = noise_std * torch.randn(batch_size, node_dim)
            # samples[node] = parent_contributions + noise
            # Sample from the conditional normal distribution
            mean = parent_contributions  # Mean determined by parents
            noise_mean = self.noise_means.get(node, 0.0)  # Non-zero mean of the noise
            noise_std = torch.sqrt(
                torch.tensor(self.noise_variances[node])
            )  # Standard deviation of the noise
            noise = noise_mean + noise_std * torch.randn(
                batch_size, node_dim
            )  # Gaussian noise with non-zero mean

            samples[node] = mean + noise  # Sampled value for each node in the batch
            # print(mean)

        return samples

    def log_prob(self, dataset):
        """
        Compute the log-probability of a given dataset.

        Args:
            dataset (dict): A dictionary mapping node names to tensors of shape (batch_size, node_dimension).
                            Example: {'A': tensor of shape (batch_size, 5),
                                      'B': tensor of shape (batch_size, 4),
                                      'C': tensor of shape (batch_size, 3)}
        Returns:
            log_prob (torch.Tensor): A tensor of shape (batch_size,) containing the log-probabilities for each sample.
        """
        log_prob = torch.zeros(
            dataset[next(iter(dataset))].shape[0]
        )  # Initialize batch-wise log-probability

        for node in self.topological_order:
            node_dim = self.node_dimensions[node]
            node_data = dataset[node]
            parent_contributions = torch.zeros_like(node_data)

            # Compute mean based on parent contributions
            for parent in self.graph.predecessors(node):
                weight = self.edge_weights[f"{parent}->{node}"]
                parent_contributions += dataset[parent] @ weight

            # Non-zero noise mean and variance
            noise_mean = self.noise_means.get(node, 0.0)
            noise_var = torch.Tensor([self.noise_variances[node]])
            noise_std = torch.sqrt(noise_var)

            # Compute Gaussian log-probability for this node
            residual = node_data - parent_contributions - noise_mean
            node_log_prob = -0.5 * torch.sum(
                (residual / noise_std) ** 2, dim=1
            ) - 0.5 * node_dim * torch.log(  # Quadratic term
                2 * torch.pi * noise_var
            )  # Normalization constant

            log_prob += node_log_prob  # Accumulate log-probability

        return log_prob


node_dimensions = {"A": 5, "B": 4, "C": 3}
edge_list = [("A", "B"), ("C", "B")]
noise_variances = {"A": 1.0, "B": 1.0, "C": 1.0}
noise_means = {"A": 2.0, "B": 0.0, "C": 0.0}
sampler = LinearGaussianDAGSampler(
    node_dimensions, edge_list, noise_means, noise_variances
)
batch_size = 10
samples = sampler.sample(batch_size)

for node, data in samples.items():
    print(f"Node {node}: {data.shape}")


log_probs = sampler.log_prob(samples)
print(log_probs.shape)  # Should be (10,)
print(log_probs)  # Log-probabilities for each sample
print(-log_probs.mean())

# Generate samples
batch_size = 1000
samples = sampler.sample(batch_size)

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
