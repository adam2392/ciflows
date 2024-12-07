import networkx as nx
import numpy as np
import torch
from numpy.testing import assert_array_equal
from torch.distributions import Normal

from ciflows.distributions.linear import (ClusteredLinearGaussianDistribution,
                                          log_prob_from_dag, sample_from_dag,
                                          sample_linear_gaussian_dag)


# Define a function to generate data from a known linear Gaussian DAG
def generate_linear_gaussian_dag_data():
    """Generate data from a linear Gaussian DAG."""
    # Create the DAG
    dag = nx.DiGraph()
    dag.add_node("X1", dim=1)  # Root node
    dag.add_node("X2", dim=1)
    dag.add_node("X3", dim=1)

    # Add weighted edges
    dag.add_edge("X1", "X2", weight=torch.tensor([2.0]))
    dag.add_edge("X2", "X3", weight=torch.tensor([3.0]))

    # Node properties for noise
    dag.add_node("U_1", exogenous=True, mean=0.0, variance=1.0)
    dag.add_node("U_2", exogenous=True, mean=0.0, variance=1.0)
    dag.add_node("U_3", exogenous=True, mean=0.0, variance=1.0)

    weight_ux0 = torch.tensor([1.0])
    weight_ux1 = torch.tensor([2.0])
    weight_ux2 = torch.tensor([1.0])
    dag.add_edge("U_1", "X1", weight=weight_ux0)
    dag.add_edge("U_2", "X2", weight=weight_ux1)
    dag.add_edge("U_3", "X3", weight=weight_ux2)

    # Sample data
    n_samples = 10
    X1 = weight_ux0 * Normal(
        dag.nodes["U_1"]["mean"], dag.nodes["U_1"]["variance"] ** 0.5
    ).sample((n_samples,))
    X2 = 2.0 * X1 + weight_ux1 * Normal(
        dag.nodes["U_2"]["mean"], dag.nodes["U_2"]["variance"] ** 0.5
    ).sample((n_samples,))
    X3 = 3.0 * X2 + weight_ux2 * Normal(
        dag.nodes["U_3"]["mean"], dag.nodes["U_3"]["variance"] ** 0.5
    ).sample((n_samples,))

    X = torch.cat([X1.unsqueeze(1), X2.unsqueeze(1), X3.unsqueeze(1)], dim=1)
    return dag, X


def test_log_prob_from_dag():
    """Test the log probability computation for a known linear Gaussian DAG."""
    dag, X = generate_linear_gaussian_dag_data()
    n_samples = X.shape[0]
    distr_idx = torch.zeros(
        n_samples, dtype=torch.long
    )  # All samples belong to the same distribution

    # Compute log probability using the function
    computed_log_prob = log_prob_from_dag(dag, X, distr_idx, intervention_targets=None)

    # Compute expected log probability analytically
    expected_log_prob = torch.zeros(n_samples)
    mean_vec = torch.zeros((n_samples, 3))
    weight_ux0 = dag.edges[("U_1", "X1")]["weight"]
    weight_ux1 = dag.edges[("U_2", "X2")]["weight"]
    weight_ux2 = dag.edges[("U_3", "X3")]["weight"]

    for i in range(n_samples):
        x1, x2, x3 = X[i]
        # print('Manual computation: ', x1, x2, x3)
        # Log prob for X1
        # mean_vec = torch.zeros(1)
        mean_vec[i, 0] = 0
        std_vec = torch.ones(1)
        lp_x1 = Normal(mean_vec[i, 0], (weight_ux0**2 * std_vec) ** 0.5).log_prob(x1)

        # Log prob for X2 | X1
        mean_vec[i, 1] = 2 * x1
        # if i == 0:
        # print(2 * x1)
        lp_x2 = Normal(mean_vec[i, 1], (weight_ux1**2 * std_vec) ** 0.5).log_prob(x2)

        # Log prob for X3 | X2
        mean_vec[i, 2] = 3 * x2
        lp_x3 = Normal(mean_vec[i, 2], (weight_ux2**2 * std_vec) ** 0.5).log_prob(x3)

        expected_log_prob[i] = lp_x1 + lp_x2 + lp_x3

    # Assert close values
    assert torch.allclose(
        computed_log_prob, expected_log_prob, atol=1e-4
    ), "Computed log-probability does not match the expected value!"


def test_log_prob_from_random_dag():
    G = nx.DiGraph([("digit", "color-digit"), ("color-digit", "color-bar")])
    adj_mat = nx.adjacency_matrix(G).todense()

    cluster_sizes = np.array([2, 2, 2])
    intervention_targets_per_distr = torch.Tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    hard_interventions_per_distr = None
    confounded_variables = [(0, 1)]

    dag = sample_linear_gaussian_dag(
        cluster_sizes=cluster_sizes,
        intervention_targets_per_distr=intervention_targets_per_distr,
        adj_mat=adj_mat,
        confounded_variables=confounded_variables,
    )

    # sample
    n_samples = 5
    intervention_targets_per_distr = torch.Tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    v_samples = sample_from_dag(dag, n_samples, distr_idx=0)

    assert v_samples.shape == (n_samples, 6)

    # log prob
    distr_idx = torch.zeros(n_samples)
    log_prob_distr0 = log_prob_from_dag(
        dag,
        v_samples,
        distr_idx=distr_idx,
        intervention_targets=intervention_targets_per_distr,
    )
    distr_idx = torch.ones(n_samples)
    log_prob_distr1 = log_prob_from_dag(
        dag,
        v_samples,
        distr_idx=distr_idx,
        intervention_targets=intervention_targets_per_distr,
    )

    assert log_prob_distr0.shape == (n_samples,)
    assert (
        log_prob_distr0.sum() > log_prob_distr1.sum()
    ), f"{log_prob_distr0}, {log_prob_distr1}"

    prior = ClusteredLinearGaussianDistribution(
        cluster_sizes=cluster_sizes,
        adjacency_matrix=adj_mat,
        confounded_variables=confounded_variables,
        intervention_targets_per_distr=intervention_targets_per_distr,
        hard_interventions_per_distr=hard_interventions_per_distr,
    )
    prior.dag = dag
    prior_log_p = prior.log_prob(
        v_samples, distr_idx, intervention_targets=intervention_targets_per_distr
    )
    print(prior_log_p, log_prob_distr1)
    assert_array_equal(prior_log_p, log_prob_distr1)
