from collections import defaultdict

import networkx as nx
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .utils import (
    set_initial_confounder_edge_coeffs,
    set_initial_confounder_parameters,
    set_initial_edge_coeffs,
    set_initial_noise_parameters,
)


class MultidistrCausalFlow(nf.distributions.BaseDistribution):
    """
    Base class for parametric multi-environment causal distributions.

    In typical normalizing flow architectures, the base distribution is a simple distribution
    such as a multivariate Gaussian. In our case, the base distribution has additional multi-environment
    causal structure. Hence, in the parametric case, this class learns the parameters of the causal
    mechanisms and noise distributions. The causal graph is assumed to be known.

    This is a subclass of BaseDistribution, which is a subclass of torch.nn.Module. Hence, this class
    can be used as a base distribution in a normalizing flow.

    Methods
    -------
    log_prob(z, e, intervention_targets) -> Tensor
        Compute the log probability of the latent variables v in environment e, given the intervention targets.
        This is used as the main training objective.
    """

    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
            samples: torch.Tensor of shape (n_samples, ...)
                Samples drawn from the distribution.
            log_p: torch.Tensor of shape (n_samples, ...)
                log probability for each sample.
        """
        raise NotImplementedError

    def log_prob(self, v):
        """Calculate log probability of batch of samples

        Args:
          v: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        raise NotImplementedError

    def log_prob(
        self,
        v: Tensor,
        e: Tensor,
        intervention_targets: Tensor,
        hard_interventions: Tensor = None,
    ) -> Tensor:
        """Log probability of the latent variables.

        Parameters
        ----------
        v : Tensor of shape (n_samples, n_nodes)
            Batch of latent variables.
        e : Tensor of shape (n_samples,)
            Index of the environment for each sample.
        intervention_targets : Tensor of shape (n_samples, n_nodes)
            The targets for each distributions intervention.
        hard_interventions : Tensor of shape (n_samples)
            Whether the intervention is hard or soft. By default None, meaning
            the intervention is soft.

        Returns
        -------
        Tensor
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError


class ClusteredCausalDistribution(MultidistrCausalFlow):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        cluster_sizes: np.ndarray,
        intervention_targets_per_distr: Tensor,
        hard_interventions_per_distr: Tensor,
        confounded_variables: list = None,
        input_shape=None,
        ind_noise_dim=None,
        fix_mechanisms: bool = False,
        use_matrix: bool = False,
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
        use_matrix : bool, optional
            Whether to use a matrix to represent the edge coefficients, by default False.
            If False, a vector is used and the kronecker product is used to compute the
            product of the coefficients with the parent variables.

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
        self.use_matrix = use_matrix
        self.input_shape = input_shape
        self.confounded_variables = confounded_variables

        # if independent noise is defined, it is used in the last dimensions
        if ind_noise_dim is None:
            ind_noise_dim = 0
        self.ind_noise_dim = ind_noise_dim
        if self.ind_noise_dim > 0:
            self.ind_noise_q0 = nf.distributions.DiagGaussian(
                ind_noise_dim, trainable=True
            )

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
        self.latent_dim = (
            self.dag.number_of_nodes()
            if self.cluster_sizes is None
            else np.sum(self.cluster_sizes)
        ) + self.ind_noise_dim

        if input_shape is not None:
            assert np.sum(self.cluster_sizes) + self.ind_noise_dim == np.prod(
                input_shape
            )

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # parametrize the trainable coefficients for the linear mechanisms
        # this will be a full matrix of coefficients for each variable in the DAG
        coeff_values, coeff_values_requires_grad = set_initial_edge_coeffs(
            self.dag,
            min_val=-1.0,
            max_val=1.0,
            cluster_mapping=self.cluster_mapping,
            use_matrix=self.use_matrix,
            # device=device,
        )
        environments = torch.ones(
            intervention_targets_per_distr.shape[0],
            1,
            #   device=device
        )

        # parametrize the trainable means for the noise distributions
        # for each separate distribution
        noise_means, noise_means_requires_grad = set_initial_noise_parameters(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_distr,
            environments=environments,
            n_dim_per_node=cluster_sizes,
            min_val=-0.5,
            max_val=0.5,
            # device=device,
        )

        # parametrize the trainable standard deviations for the noise distributions
        # for each separate distribution
        noise_stds, noise_stds_requires_grad = set_initial_noise_parameters(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_distr,
            n_dim_per_node=cluster_sizes,
            environments=environments,
            min_val=0.5,
            max_val=1.5,
            # device=device,
        )

        # check confounded variables all are in the DAG
        if self.confounded_variables is not None:
            self.confounder_mapping = dict()

            for node1, node2 in self.confounded_variables:
                # self.confounder_mapping[node1].append(node2)
                if node1 not in self.dag.nodes or node2 not in self.dag.nodes:
                    raise ValueError(
                        f"Confounded variable {node1}, or {node2} is not in the DAG."
                    )

            # sample the confounding variable distribution parameters
            confounder_cluster_sizes = []
            for confounder in self.confounded_variables:
                confounder_cluster_sizes.append(cluster_sizes[confounder[0]])

            # means and stds for each confounder (n_confounders)
            confounder_means, confounder_stds = set_initial_confounder_parameters(
                confounded_variables=self.confounded_variables,
                min_val=-3.0,
                max_val=3.0,
                n_dim_per_node=confounder_cluster_sizes,
            )

            # parametrize the trainable coefficients for the linear mechanisms
            # this will be a full matrix of coefficients for each variable in the DAG
            # (n_confounders, n_cluster_dims, 2), where the first index is the edge coefficient to apply
            # the confounder to node1, and the second index is the edge coefficient to apply
            # the confounder to node2 for (node1, node2) confounders
            #
            # list of (n_confounders) length, with each consisting of a list of length 2
            # with each inner element being an array of shape (n_cluster_dims)
            confounder_coeff_values = set_initial_confounder_edge_coeffs(
                self.confounded_variables,
                min_val=-1.0,
                max_val=1.0,
                cluster_sizes=confounder_cluster_sizes,
                use_matrix=self.use_matrix,
                # device=device,
            )

            # map each node to a set of confounder indices
            self.confounder_mapping = defaultdict(dict)

            for idx, (node1, node2) in enumerate(self.confounded_variables):
                self.confounder_mapping[node1][idx] = None
                self.confounder_mapping[node2][idx] = None

            # each is a list of n_confounders length, with the corresponding
            # coefficients, or means, or stds for each confounder
            self.confounder_coeff_values = nn.ParameterList(confounder_coeff_values)
            self.confounder_means = nn.ParameterList(confounder_means)
            self.confounder_stds = nn.ParameterList(confounder_stds)
        else:
            self.confounder_mapping = None
            self.confounder_coeff_values = None
            self.confounder_means = None
            self.confounder_stds = None

        self.coeff_values = nn.ParameterList(coeff_values)
        self.noise_means = nn.ParameterList(noise_means)
        self.noise_stds = nn.ParameterList(noise_stds)
        self.coeff_values_requires_grad = coeff_values_requires_grad
        self.noise_means_requires_grad = noise_means_requires_grad
        self.noise_stds_requires_grad = noise_stds_requires_grad

    def set_noise_means(self, noise_means, node, distr_idx):
        assert len(noise_means) == len(self.noise_means[distr_idx][node])
        self.noise_means[distr_idx][node] = noise_means

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

    def forward(
        self,
        num_samples=1,
        intervention_targets: Tensor = None,
        hard_interventions: Tensor = None,
    ):
        """Sample from the distribution."""
        if intervention_targets is not None:
            if intervention_targets.squeeze().shape[0] != self.dag.number_of_nodes():
                raise ValueError(
                    "Intervention targets must have the same length as the number of nodes in the DAG."
                )
            if hard_interventions is not None and len(intervention_targets) != len(
                hard_interventions
            ):
                raise ValueError(
                    "Intervention targets and hard interventions must have the same length."
                )
        if hard_interventions is None:
            hard_interventions = torch.zeros(self.latent_dim)
        if intervention_targets is None:
            intervention_targets = torch.zeros_like(hard_interventions)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # return (num_samples, latent_dim) samples from the distribution
        samples = torch.zeros(
            (num_samples, self.latent_dim)
            #   , device=device
        )

        # return the log probability of the samples
        log_p = torch.zeros(
            num_samples,
            # device=device
        )

        # start from observational environment
        noise_env_idx = 0

        # sample from the noise distributions for each variable over the DAG
        for idx in range(self.dag.number_of_nodes()):
            parents = list(self.dag.predecessors(idx))

            # get start/end in the representation for this cluster
            start, end = self.cluster_mapping[idx]
            cluster_idx = np.arange(start, end, dtype=int)

            # compute the contribution of the parents
            if len(parents) == 0 or (
                intervention_targets[idx] == 1 and hard_interventions[idx] == 1
            ):
                parent_contribution = 0.0
                var = self.noise_stds[noise_env_idx][idx] ** 2
            else:
                parent_cluster_idx = np.hstack(
                    [np.arange(*self.cluster_mapping[p], dtype=int) for p in parents]
                )
                # get coeffieicnts for the parents
                # which is a vector of coefficients for each parent
                coeffs_raw = self.coeff_values[idx][:-1]
                if isinstance(coeffs_raw, nn.ParameterList):
                    coeffs_raw = torch.cat([c for c in coeffs_raw])
                parent_coeffs = coeffs_raw  # .to(device)
                parent_contribution = parent_coeffs * samples[:, parent_cluster_idx]

                # compute the contribution of the noise
                var = self.noise_stds[noise_env_idx][idx] ** 2 * torch.ones_like(
                    samples[:, cluster_idx]
                )

            # XXX: compute the contributions of the confounders

            noise_coeff = self.coeff_values[idx][-1]  # .to(device)
            noise_contribution = noise_coeff * self.noise_means[noise_env_idx][idx]
            var *= noise_coeff**2
            var *= noise_coeff**2

            # print(parent_contribution, noise_contribution.shape, var.shape, samples[:, idx].shape)
            # samples from the normal distribution for (n_samples, cluster_dims)
            samples[:, cluster_idx] = torch.normal(
                parent_contribution + noise_contribution, var.sqrt()
            )

            # compute the log probability of the variable given the
            # parametrized normal distribution using the parents mean and variance
            log_p += (
                torch.distributions.Normal(
                    parent_contribution + noise_contribution, var.sqrt()
                )
                .log_prob(samples[:, cluster_idx])
                .mean(axis=1)
            )

        # sample from the independent noise distributions
        if self.ind_noise_dim > 0:
            samples[:, -self.ind_noise_dim :], ind_log_p = self.ind_noise_q0.forward(
                num_samples
            )
            log_p += ind_log_p

        # reshape for the rest of the flow
        if self.input_shape is not None:
            samples = samples.view(num_samples, *self.input_shape)
        return samples, log_p

    def contrastive_loss(
        self,
        v_latent: Tensor,
        e: Tensor,
        intervention_targets: Tensor,
        hard_interventions: Tensor = None,
    ) -> Tensor:
        """Compute a contrastive loss metric.

        V1 -> V2 -> V3

        Take a batch.
        - separate samples into their different distributions
        - across variables that are intervened, maximize their distribution distance
            of the latent variable embedding for the intervened variable (e.g. V3)
            across the different distributions

        For all variables in each distribution marked with indices of ``e``,
        for each sample within the same distribution, we will minimize their
        cosine similarity. For each sample within different distributions, we
        will maximize their cosine similarity.

        Each `v_latent` sample though consists of `latent_cluster_dims` corresponding
        to the cluster dimensions of our latent causal graph.

        For any two arbitrary samples, that may come from different distributions, we will
        either maximize, or minimize their similarities:

        1. maximize: if the cluster dimension corresponds to a cluster variable that is a
        descendant of an intervention difference including the intervened variable themselves.
        2. minimize: if the cluster dimension corresponds to a cluster variable that is an
        ancestor of an intervention difference, excluding the intervened variables.

        Options: instead, we can also only maximize intervened variable differences
        and minimize everything else.

        Parameters
        ----------
        v_latent : Tensor of shape (batch_size, latent_cluster_dims)
            _description_
        e : Tensor
            _description_
        intervention_targets : Tensor
            _description_
        hard_interventions : Tensor, optional
            _description_, by default None

        Returns
        -------
        Tensor
            _description_
        """

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
        if self.input_shape is not None:
            n_batch = v_latent.shape[0]

            # flatten representation
            v_latent = v_latent.view(n_batch, np.prod(self.input_shape))

        log_p = torch.zeros(len(v_latent), dtype=v_latent.dtype, device=v_latent.device)
        latent_dim = v_latent.shape[1]
        batch_size = v_latent.shape[0]
        if intervention_targets is None:
            intervention_targets = torch.zeros(batch_size, latent_dim)
        if hard_interventions is None:
            hard_interventions = torch.zeros_like(intervention_targets)
        if e is None:
            e = torch.zeros(batch_size, 1)

        for env in e.unique():
            env_mask = (e == env).flatten()
            env_mask = env_mask  # .to(log_p.device)

            # print('Analyzing environment ', env)
            # print(v_latent.shape, env_mask.shape, intervention_targets.shape)
            # v_env is (n_samples, latent_dim)
            v_env = v_latent[env_mask, :]
            intervention_targets_env = intervention_targets[env_mask, :]
            hard_interventions_env = hard_interventions[env_mask, :]

            # iterate over all variables in the latent space in topological order
            for idx in range(self.dag.number_of_nodes()):
                parents = list(self.dag.predecessors(idx))

                # get start/end in the representation for this cluster
                start, end = self.cluster_mapping[idx]
                cluster_idx = np.arange(start, end, dtype=int)

                noise_env_idx = int(env) if intervention_targets_env[0, idx] == 1 else 0
                # print('Noise env idx: ', noise_env_idx)

                # compute the contribution of the parents
                if len(parents) == 0 or (
                    intervention_targets_env[0, idx] == 1
                    and hard_interventions_env[0, idx] == 1
                ):
                    parent_contribution = 0.0
                    var = self.noise_stds[noise_env_idx][idx] ** 2
                else:
                    # parent_cluster_idx = np.hstack(
                    #     [np.arange(*self.cluster_mapping[p], dtype=int) for p in parents]
                    # )

                    # get coeffieicnts for the parents
                    # which is a vector of coefficients for each parent
                    coeffs_raw = self.coeff_values[idx][:-1]
                    if isinstance(coeffs_raw, nn.ParameterList):
                        coeffs_raw = torch.cat([c for c in coeffs_raw])

                    # get the coefficients per parents and compute their contribution
                    parent_coeffs = coeffs_raw  # .to(v_latent.device)
                    parent_contribution = torch.zeros_like(v_env[:, cluster_idx])
                    for jdx, p in enumerate(parents):
                        p_cluster_idx = np.arange(*self.cluster_mapping[p], dtype=int)
                        parent_contribution += (
                            parent_coeffs[jdx] * v_env[:, p_cluster_idx]
                        )

                    # print('\n\n Parent contribution is nonzero')
                    # print(parent_contribution.shape, parent_coeffs.shape)
                    # print(self.noise_stds[noise_env_idx][idx].shape, v_env[:, parent_cluster_idx].shape)
                    # print('\n\n'
                    # )
                    # compute the contribution of the noise for the current endogenous variable
                    var = self.noise_stds[noise_env_idx][idx] ** 2 * torch.ones_like(
                        v_env[:, cluster_idx]
                    )

                    # print(parent_contribution.shape, parent_coeffs.shape, var.shape, v_env[:, parent_cluster_idx].shape)

                # XXX: compute the contributions of the confounders
                if self.confounded_variables is not None:
                    # add the contribution for each confounder to this variable
                    for confounder_idx in self.confounder_mapping[idx]:
                        confounder_coeff = self.confounder_coeff_values[confounder_idx][
                            str(idx)
                        ]
                        confounder_means = self.confounder_means[confounder_idx]
                        confounder_stds = self.confounder_stds[confounder_idx]

                        # print('\n\nConfounded variable contribution')
                        # print(confounder_means.shape, confounder_stds.shape)
                        # print(confounder_coeff.shape)
                        confounder_contribution = confounder_coeff * confounder_means

                        # scale the variance based on the confounder contribution
                        var *= confounder_coeff**2 * confounder_stds**2
                else:
                    confounder_contribution = 0.0

                noise_coeff = self.coeff_values[idx][-1]  # .to(v_latent.device)
                noise_contribution = noise_coeff * self.noise_means[noise_env_idx][idx]
                var *= noise_coeff**2

                # compute the log probability of the variable given the
                # parametrized normal distribution using the parents mean and variance
                # print(parent_contribution, noise_contribution.shape, var.shape, v_env[:, cluster_idx].shape)
                distr = torch.distributions.Normal(
                    parent_contribution + noise_contribution + confounder_contribution,
                    var.sqrt(),
                )
                log_p_distr = (distr.log_prob(v_env[:, cluster_idx]).sum(axis=1)).to(
                    log_p.device
                )
                log_p[env_mask] += log_p_distr

        # sample from the independent noise distributions
        if self.ind_noise_dim > 0:
            v = v_latent[:, -self.ind_noise_dim :]
            ind_log_p = self.ind_noise_q0.log_prob(v)
            log_p += ind_log_p
        return log_p
