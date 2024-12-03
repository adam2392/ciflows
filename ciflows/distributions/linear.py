import networkx as nx
import numpy as np
import torch
from torch import Tensor

from ciflows.distributions.multidistr import MultidistrCausalFlow


def sample_linear_gaussian_dag(
    cluster_sizes,
    intervention_targets_per_distr,
    adj_mat,
    confounded_variables=None,
):
    if adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    n_nodes = adj_mat.shape[0]
    dag = nx.from_numpy_array(adj_mat, edge_attr="weight", create_using=nx.DiGraph)
    for src, target in dag.edges:
        # sample well-conditioned matrix
        # Step 1: Sample a random matrix from standard normal distribution
        d_1 = cluster_sizes[src]
        d_0 = cluster_sizes[target]
        W = torch.randn(d_1, d_0)  # W of shape (d_1, d_0)

        # Step 2: Perform SVD on the sampled matrix
        U, S, V = torch.svd(W)

        # Step 3: Control the singular values to ensure well-conditioning
        # Here we just clamp the singular values between min_value and max_value for stability
        min_value = 0.1  # Minimum singular value (avoid near-zero values)
        max_value = 3.0  # Maximum singular value (avoid extremely large values)

        S_clamped = torch.clamp(S, min=min_value, max=max_value)

        # Step 4: Reconstruct the matrix with adjusted singular values
        W_conditioned = U @ torch.diag(S_clamped) @ V.T

        dag.add_edge(src, target, weight=W_conditioned)

    # now, add exogenous variables per node
    for i in range(n_nodes):
        node_dim = cluster_sizes[i]

        for distr_idx in range(intervention_targets_per_distr.shape[0]):
            intervention_targets = np.argwhere(
                intervention_targets_per_distr[distr_idx] == 1
            )
            # print('INtervention targets: ', intervention_targets)

            # add the exogenous node
            if distr_idx == 0 or i in intervention_targets:
                # print(f'Adding exogenous node for {i} in distr_idx {distr_idx}')
                exogenous_node = f"U_{i}^{distr_idx}"
                dag.add_node(
                    exogenous_node,
                    exogenous=True,
                    mean=0.0,
                    variance=1.0,
                    distr_idx=distr_idx,
                    #  mean=torch.Tensor([0.]), variance=torch.Tensor([1.])
                )

            # add the cluster node
            dag.add_node(i, exogenous=False, dim=node_dim)

            # sample exogenous weight
            weight = torch.randn(node_dim)
            dag.add_edge(exogenous_node, i, weight=weight)

    # now, add confounders per c-component
    if confounded_variables is None:
        confounded_variables = []
    for c_pair in confounded_variables:
        node1, node2 = c_pair
        confounder = f"C{node1, node2}"
        dag.add_node(confounder, exogenous=True, mean=0.0, variance=1.0)
        weight = torch.randn(cluster_sizes[i])

        for confounded_node in c_pair:
            dag.add_edge(confounder, confounded_node, weight=weight)
    return dag


def sample_from_dag(dag, n_samples=100, distr_idx=0):
    # sample in topological order
    nodes = [
        node
        for node in nx.topological_sort(dag)
        if not dag.nodes[node].get("exogenous", False)
    ]

    # get cluster sizes per node in topological order
    cluster_sizes = torch.tensor(
        [
            dag.nodes[node]["dim"]
            for node in nodes
            if not dag.nodes[node].get("exogenous", False)
        ]
    )
    samples = torch.zeros((n_samples, cluster_sizes.sum()))

    # confounded_samples =
    # for each variable, sample from its parents, which includes the exogenous
    # variables and confounders
    for idx, node in enumerate(nodes):
        parents = list(dag.predecessors(node))
        if len(parents) == 0:
            raise ValueError(f"Node {node} has no parents")

        # parametrize the multivariate normal distribution
        parametrized_mean = torch.zeros(n_samples, cluster_sizes[idx])
        parametrized_variance = torch.zeros(
            n_samples, cluster_sizes[idx], cluster_sizes[idx]
        )

        node_idx = np.arange(
            cluster_sizes[:idx].sum(), cluster_sizes[: idx + 1].sum(), dtype=int
        )
        node_dim = cluster_sizes[idx]

        # accumulate the mean and variance
        node_sample = torch.zeros(n_samples, node_dim)
        for par in parents:
            par_weight = dag[par][node]["weight"]

            # sample the parent if exogenous
            if dag.nodes[par].get("exogenous", False):
                mean = dag.nodes[par]["mean"]
                std = dag.nodes[par]["variance"] ** 0.5
                mean_tensor = torch.full((n_samples, node_dim), mean)
                std_tensor = torch.full((n_samples, node_dim), std)
                par_sample = torch.normal(mean=mean_tensor, std=std_tensor)

                # parametrized_variance += par_weight ** 2 * dag.nodes[par]["variance"]
                node_sample += par_sample * par_weight
            else:
                par_idx = np.arange(
                    cluster_sizes[:idx].sum(), cluster_sizes[: idx + 1].sum(), dtype=int
                )
                par_sample = samples[:, par_idx]

                node_sample += par_sample @ par_weight
            # parametrized_mean += par_sample * par_weight

        # now, sample from the parametrized distribution
        # samples[:, node_idx] = torch.distributions.MultivariateNormal(parametrized_mean, parametrized_variance ** 0.5).
        samples[:, node_idx] = node_sample

    return samples


def log_prob_from_dag(
    dag, X, distr_idx, intervention_targets=None, hard_interventions=None, debug=False
):
    """Log likelihood of the data given the DAG.

    For each distribution index, we compute the log likelihood of the data
    given the DAG. The log likelihood is computed as the sum of the log
    likelihood of each variable given its parents.

    Parameters
    ----------
    dag : nx.DiGraph
        The latent graph.
    X : tensor of shape (n_samples, n_dims)
        The data.
    distr_idx : tensor of shape (n_samples)
        The distribution index of each sample.
    intervention_targets : tensor of shape (n_samples, n_nodes)
        The intervention targets for each sample.
    hard_interventions : tensor of shape (n_samples, n_dims)
        The hard interventions for each sample.
    """
    batch_size, latent_dim = X.shape

    if intervention_targets is None:
        intervention_targets = torch.zeros((batch_size, latent_dim))
    if hard_interventions is None:
        hard_interventions = torch.zeros_like(intervention_targets)

    n_samples, n_dims = X.shape
    # get the nodes in topological order
    nodes = [
        node
        for node in nx.topological_sort(dag)
        if not dag.nodes[node].get("exogenous", False)
    ]
    cluster_sizes = torch.tensor(
        [
            dag.nodes[node]["dim"]
            for node in nodes
            if not dag.nodes[node].get("exogenous", False)
        ]
    )

    if debug:
        print("\n cluster sizes: ", cluster_sizes)

    unique_distrs = torch.unique(distr_idx)
    log_prob = torch.zeros(n_samples)

    # for each distribution index, compute the log likelihood for those samples
    # that belong to that distribution
    for distr in unique_distrs:
        env_mask = torch.argwhere(distr_idx == distr).flatten()
        print(env_mask)
        # get the intervention targets and hard interventions for this distribution
        intervention_targets_env = intervention_targets[env_mask, :]
        hard_interventions_env = hard_interventions[env_mask, :]

        # iterate in topological order of the latent variable nodes
        for idx, node in enumerate(nodes):
            node_idx = np.arange(
                cluster_sizes[:idx].sum(), cluster_sizes[: idx + 1].sum(), dtype=int
            )
            node_dim = cluster_sizes[idx]

            if debug:
                print("Node idx: ", node_idx, idx, node)
            parents = [
                par
                for par in dag.predecessors(node)
                if not dag.nodes[par].get("exogenous", False)
            ]

            # get exogenous parents for this distribution index
            exogenous_parents = [
                par
                for par in dag.predecessors(node)
                if dag.nodes[par].get("exogenous", False)
                and dag.nodes[par].get("distr_idx", None) == distr
            ]
            if exogenous_parents == []:
                exogenous_parents = [
                    par
                    for par in dag.predecessors(node)
                    if dag.nodes[par].get("exogenous", False)
                    and dag.nodes[par].get("distr_idx", None) == 0
                ]

            # get confounded parents for this distribution index
            confounded_parents = [
                par
                for par in dag.predecessors(node)
                if dag.nodes[par].get("exogenous", False)
                and dag.nodes[par].get("distr_idx", None) is None
            ]

            # initialize the mean and variance contributions from the parents
            # and the exogenous variables (exogenous per distribution and the confounder)
            exo_mean_contributions = torch.zeros((len(env_mask), node_dim))
            exo_var_contributions = torch.zeros((len(env_mask), node_dim))
            par_contributions = torch.zeros((len(env_mask), node_dim))

            # accumulate the mean and variance from the parents
            if (
                intervention_targets_env[0, idx] != 1
                and hard_interventions_env[0, idx] != 1
            ):
                exogenous_parents = exogenous_parents + confounded_parents
                for par in parents:
                    par_weight = torch.atleast_2d(dag[par][node]["weight"])
                    par_node_idx = (
                        np.argwhere(np.array(nodes) == par).flatten().squeeze()
                    )

                    par_idx = np.arange(
                        cluster_sizes[:par_node_idx].sum(),
                        cluster_sizes[: par_node_idx + 1].sum(),
                        dtype=int,
                    )

                    par_sample = torch.atleast_2d(X[env_mask][:, par_idx])
                    par_contributions += par_sample @ par_weight

            for par in exogenous_parents:
                par_weight = dag[par][node]["weight"]
                par_mean = dag.nodes[par]["mean"]
                par_var = dag.nodes[par]["variance"]
                exo_mean_contributions += par_weight * par_mean
                exo_var_contributions += par_weight**2 * par_var

            # get the 1d mean and std vectors for each dimension of the node
            # Note: we assume the dimensions of the cluster node are independent
            mean_vec = (par_contributions + exo_mean_contributions).squeeze()
            std_vec = exo_var_contributions.sqrt().squeeze()

            # accumulate log probability for each sample using conditional normal
            for i, sample_idx in enumerate(env_mask):
                # print(mean_vec[i], std_vec[i])
                # print(X[sample_idx, node_idx].shape)
                log_prob_node = (
                    torch.distributions.Normal(
                        mean_vec[i],
                        std_vec[i],
                    )
                    .log_prob(X[sample_idx, node_idx].squeeze())
                    .sum()
                )
                log_prob[sample_idx] += log_prob_node

    return log_prob


class ClusteredLinearGaussianDistribution(MultidistrCausalFlow):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        cluster_sizes: np.ndarray,
        intervention_targets_per_distr: Tensor,
        hard_interventions_per_distr: Tensor,
        confounded_variables: list = None,
        input_shape=None,
    ):
        """Linear clustered causal graph distribution.

        This class defines a parametric distribution over a clustered causal graph, where
        the causal mechanisms are linear. The noise distributions are assumed to be Gaussian.
        This allows for multiple distributions.

        A clustered causal graph defines a grouping over the variables in the graph.
        Thus the variables that describe the graph can either be the clusters, or the
        fine-grained variables.

        Parameters
        ----------
        adjacency_matrix : np.ndarray of shape (n_clusters, n_clusters)
            The adjacency matrix of the causal graph over the clustered variables.
            Each row/column is another cluster.
        cluster_sizes : np.ndarray of shape (n_clusters, 1)
            The size/dimensionality of each cluster.
        intervention_targets_per_distr : Tensor of shape (n_distributions, n_clusters)
            The intervention targets for each cluster-variable in each environment.
        hard_interventions_per_distr : Tensor of shape (n_distributions, n_clusters)
            Whether the intervention target for each cluster-variable is hard (i.e.
            all parents are removed).
        fix_mechanisms : bool, optional
            Whether to fix the mechanisms, by default False.

        Attributes
        ----------
        coeff_values : nn.ParameterList of length (n_nodes) each of length (cluster_dim + 1)
            The coefficients for the linear mechanisms for each variable in the DAG.
            The last element in the list is the constant term.
        noise_means : nn.ParameterList of length (n_nodes) each of length (cluster_dim)
            The means for the noise distributions for each variable in the DAG.
        noise_stds : nn.ParameterList of length (n_nodes) each of length (cluster_dim)
            The standard deviations for the noise distributions for each variable in the DAG.
        """
        super().__init__()

        self.adjacency_matrix = adjacency_matrix
        self.intervention_targets_per_distr = intervention_targets_per_distr
        self.hard_interventions_per_distr = hard_interventions_per_distr
        self.confounded_variables = confounded_variables

        self.input_shape = input_shape

        if cluster_sizes is None:
            cluster_sizes = [1] * adjacency_matrix.shape[0]
        self.cluster_sizes = cluster_sizes

        # map the node in adjacency matrix to a cluster size in the latent space
        self.cluster_mapping = dict()
        for idx in range(len(cluster_sizes)):
            start = np.sum(cluster_sizes[:idx])
            end = start + cluster_sizes[idx]
            self.cluster_mapping[idx] = (start, end)

        self.dag = nx.DiGraph(adjacency_matrix)
        if input_shape is not None:
            assert self.latent_dim == np.prod(input_shape)

        dag = sample_linear_gaussian_dag(
            cluster_sizes=cluster_sizes,
            intervention_targets_per_distr=intervention_targets_per_distr,
            adj_mat=adjacency_matrix,
            confounded_variables=confounded_variables,
        )
        self.dag = dag

    @property
    def latent_dim(self):
        latent_nodes = [
            node
            for node in self.dag.nodes
            if not self.dag.nodes[node].get("exogenous", False)
        ]
        return sum([self.dag[node].get("dim", 1) for node in latent_nodes])

    def set_noise_means(self, noise_mean, node, distr_idx):
        # get the noise variable for node
        exo_node = [
            nbr
            for nbr in self.dag.predecessors(node)
            if self.dag.nodes[nbr].get("exogenous", False)
            and self.dag.nodes[nbr].get("distr_idx", None) == distr_idx
        ]
        assert len(exo_node) == 1
        exo_node = exo_node[0]

        # set the noise mean
        self.dag.nodes[exo_node]["mean"] = noise_mean

    def sample_noise(self, distr_idx, n_samples=1):
        noise_means = self.noise_means[distr_idx]
        noise_stds = self.noise_stds[distr_idx]
        result = torch.zeros((n_samples, len(noise_means), len(noise_means[0])))

        # Sample according to specified means
        for idx in range(len(noise_means)):
            mean = noise_means[idx]
            stddev = noise_stds[idx]
            cov = torch.diag(stddev**2)
            dist = torch.distributions.MultivariateNormal(mean, cov)
            result[:, idx, :] = dist.sample(sample_shape=((n_samples,))).squeeze()
        return result

    def sample(self, num_samples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        z, _ = self.forward(num_samples, **kwargs)
        return z

    def forward(
        self,
        n_samples=1,
        distr_idx=0,
    ):
        # XXX: figure out how to sample from interventional distributions?
        v_samples = sample_from_dag(self.dag, n_samples=n_samples, distr_idx=distr_idx)

        distr_idx = torch.full((n_samples, 1), distr_idx)
        log_prob = log_prob_from_dag(self.dag, v_samples, distr_idx)
        return v_samples, log_prob

    def log_prob(
        self,
        v_latent: Tensor,
        e: Tensor,
        intervention_targets: Tensor,
        hard_interventions: Tensor = None,
    ) -> Tensor:
        """Multi-environment log probability of the latent variables.

        Parameters
        ----------
        v_latent : Tensor of shape (n_distributions, latent_dim)
            The "representation" layer for latent variables v.
        e : Tensor of shape (n_distributions, 1)
            Indicator of different environments (overloaded to indicate intervention
            and change in domain).
        intervention_targets : Tensor
            The intervention targets for each variable in each distribution.

        Returns
        -------
        log_p : Tensor of shape (n_distributions, 1)
            The log probability of the latent variables in each distribution.
        """
        return log_prob_from_dag(
            self.dag, v_latent, e, intervention_targets, hard_interventions
        )
