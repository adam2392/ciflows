import torch

from ciflows.distributions.pgm import LinearGaussianDag


def test_linear_gaussian_dag():
    # Define test parameters
    node_dimensions = {"A": 2, "B": 3, "C": 1}
    edge_list = [("A", "B"), ("B", "C")]
    noise_means = {"A": 0.1, "B": -0.2, "C": 0.3}
    noise_variances = {"A": 0.05, "B": 0.1, "C": 0.2}
    intervened_node_means = [
        {"A": 0.5, "B": -0.3},  # First intervention
        {"B": 0.1, "C": -0.2},  # Second intervention
    ]
    intervened_node_vars = [
        {"A": 0.02, "B": 0.05},  # Variances for first intervention
        {"B": 0.1, "C": 0.2},  # Variances for second intervention
    ]
    confounded_list = [("A", "C")]

    # Instantiate the LinearGaussianDAG
    dag = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
        confounded_list=confounded_list,
    )

    # Test node dimensions
    assert dag.node_dimensions == node_dimensions, "Node dimensions are incorrect."

    # Test edge weights
    for src, tgt in edge_list:
        weight_name = f"{src}->{tgt}"
        assert (
            weight_name in dag.edge_weights
        ), f"Edge weight for {src}->{tgt} not initialized."

    # Test topological order
    expected_order = ["A", "B", "C"]
    assert dag.topological_order == expected_order, "Topological ordering is incorrect."

    # Test noise means and variances
    for node, mean in noise_means.items():
        assert torch.isclose(
            getattr(dag, f"exog_mean_{node}_0"), torch.tensor(mean)
        ), f"Noise mean for {node} is incorrect."
    for node, var in noise_variances.items():
        assert torch.isclose(
            getattr(dag, f"exog_variance_{node}_0"), torch.tensor(var)
        ), f"Noise variance for {node} is incorrect."

    # Test intervened variables
    for idx, (mean_dict, var_dict) in enumerate(
        zip(intervened_node_means, intervened_node_vars), start=1
    ):
        for node, mean in mean_dict.items():
            assert torch.isclose(
                getattr(dag, f"exog_mean_{node}_{idx}"),
                torch.tensor(mean),
            ), f"Intervened mean for {node} at index {idx} is incorrect."
            assert torch.isclose(
                getattr(dag, f"exog_variance_{node}_{idx}"),
                torch.tensor(var_dict[node]),
            ), f"Intervened variance for {node} at index {idx} is incorrect."

    # Test confounded variables
    for node_a, node_b in confounded_list:
        assert torch.isclose(
            getattr(dag, f"confounded_mean_{node_a}_{node_b}"), torch.tensor(0.0)
        ), f"Confounded mean for {node_a}-{node_b} is incorrect."
        assert torch.isclose(
            getattr(dag, f"confounded_variance_{node_a}_{node_b}"), torch.tensor(1.0)
        ), f"Confounded variance for {node_a}-{node_b} is incorrect."

    print("All tests passed!")


# Run the test
test_linear_gaussian_dag()
