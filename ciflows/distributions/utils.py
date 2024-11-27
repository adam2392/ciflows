
import networkx as nx
import normflows as nf
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Uniform
from torch.nn import ParameterDict, ParameterList


def make_spline_flows(
    n_flows: int,
    latent_dim: int,
    n_hidden_dim: int,
    n_layers: int,
    permutation: bool = True,
) -> list[nf.flows.Flow]:
    """_summary_

    Parameters
    ----------
    n_flows : int
        The number of flows to generate
    latent_dim : int
        The dimensionality of the latent space.
    n_hidden_dim : int
        The dimensionality of the neural network parametrization of
        the splines.
    n_layers : int
        Number of layers to use to parametrize the neural spline.
    permutation : bool, optional
        Whether to add a permutation of the flow, by default True.

    Returns
    -------
    list[nf.flows.Flow]
        List of normalizing flows.
    """
    flows = []
    for i in range(n_flows):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(latent_dim, n_layers, n_hidden_dim)
        ]
        if permutation:
            flows += [nf.flows.LULinearPermute(latent_dim)]
    return flows


def sample_invertible_matrix(size, min_val, max_val, max_samples=1000):
    """Samples a square invertible matrix."""
    idx = 0
    while True and idx < max_samples:
        # Sample a random matrix with uniform entries
        matrix = (max_val - min_val) * torch.rand((size, size)) + min_val
        idx += 1

        # Check if the matrix is invertible by computing its determinant
        if torch.det(matrix) != 0:
            return matrix
    raise ValueError(f"Could not find an invertible matrix after {max_samples} samples.")


def set_initial_confounder_edge_coeffs(
    confounded_vars,
    min_val: float = -1,
    max_val: float = 1,
    cluster_sizes=None,
    use_matrix: bool = False,
    device: torch.device = None,
) -> list[ParameterDict]:
    """Set initial coefficient values for a linear SCM and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    min_val : float
        The minimum value for the coefficient. Default is -1.
    max_val : float
        The maximum value for the coefficient. Default is 1.
    cluster_mapping : dict
        Mapping of nodes to latent indices. Default is None,
        which assumes each node has a single latent index, and
        each edge is therefore a single coefficient. If the
        latent indices are multivariate, then an invertible matrix
        is sampled for each "edge", which maps the incoming latent
        cluster node to the outgoing node.
        This assumes that each cluster mapping is a list of indices
        that correspond to the latent dimensions.
    use_matrix : bool
        Whether to use a matrix to represent the edge coefficients.
        Default is False, which corresponds to a vector of coefficients
        if cluster_mapping is not None. This then applies a element-by-element
        multiplication of the latent cluster node with the edge coefficient.
    device : torch.device
        PyTorch device. Default is None, and will be inferred.

    Returns
    -------
    coeff_values : list (len n_confounders) of list (length 2) of list of length (2) with tensors
            for coefficients and constant
        The coefficient values linearly mapping each confounder to its nodes in the SCM.
        Each element has a list of coefficients for each node, and the constant. The coefficients
        are ordered by their index in the graph using confounded_vars.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coeff_values = []
    for confounder_idx, (node1, node2) in enumerate(confounded_vars):
        coeff_values_i = dict()
        if cluster_sizes is not None and use_matrix:
            # get start/end in the representation for this cluster
            cluster_size = cluster_sizes[confounder_idx]

            # if len(cluster_mapping[cluster_size1]) != len(cluster_mapping[confounder_idx]):
            #     raise ValueError(
            #         "The number of dimensions for the latent clusters must be the same if "
            #         "using a matrix to represent the edge coefficients."
            #         f"{idx}: {len(cluster_mapping[idx])} != {confounder_idx}: {len(cluster_mapping[confounder_idx])}."
            #     )
            # sample a random invertible matrix to map confounder to node1 and node 2
            matrix = sample_invertible_matrix(cluster_size, min_val, max_val)
            param_node1 = nn.Parameter(matrix, requires_grad=True)  # .to(device)

            # sample a random invertible matrix
            matrix = sample_invertible_matrix(cluster_size, min_val, max_val)
            param_node2 = nn.Parameter(matrix, requires_grad=True)  # .to(device)
        elif cluster_sizes is not None:
            # get start/end in the representation for this cluster
            cluster_size = cluster_sizes[confounder_idx]

            # sample a random vector with each value between min_val and max_val
            vector = (max_val - min_val) * torch.rand((cluster_size,)) + min_val
            param_node1 = nn.Parameter(vector, requires_grad=True)  # .to(device)

            # sample a random vector with each value between min_val and max_val
            vector = (max_val - min_val) * torch.rand((cluster_size,)) + min_val
            param_node2 = nn.Parameter(vector, requires_grad=True)  # .to(device)
        else:
            random_val = Uniform(min_val, max_val).sample((1,))
            val = random_val
            param_node1 = nn.Parameter(val * torch.ones(1), requires_grad=False)  # .to(device)
            param_node2 = nn.Parameter(val * torch.ones(1), requires_grad=False)  # .to(device)

        node1_params = param_node1
        coeff_values_i[str(node1)] = node1_params

        node2_params = param_node2
        coeff_values_i[str(node2)] = node2_params

        coeff_values.append(nn.ParameterDict(coeff_values_i))

    return coeff_values


def set_initial_edge_coeffs(
    dag: nx.DiGraph,
    min_val: float = -1,
    max_val: float = 1,
    cluster_mapping=None,
    use_matrix: bool = False,
    device: torch.device = None,
) -> tuple[list[ParameterList], list[list[bool]]]:
    """Set initial coefficient values for a linear SCM and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    min_val : float
        The minimum value for the coefficient. Default is -1.
    max_val : float
        The maximum value for the coefficient. Default is 1.
    cluster_mapping : dict
        Mapping of nodes to latent indices. Default is None,
        which assumes each node has a single latent index, and
        each edge is therefore a single coefficient. If the
        latent indices are multivariate, then an invertible matrix
        is sampled for each "edge", which maps the incoming latent
        cluster node to the outgoing node.
        This assumes that each cluster mapping is a list of indices
        that correspond to the latent dimensions.
    use_matrix : bool
        Whether to use a matrix to represent the edge coefficients.
        Default is False, which corresponds to a vector of coefficients
        if cluster_mapping is not None. This then applies a element-by-element
        multiplication of the latent cluster node with the edge coefficient.
    device : torch.device
        PyTorch device. Default is None, and will be inferred.

    Returns
    -------
    coeff_values : list[ParameterList]
        The coefficient values for each node in the SCM. Each element has a list
        of coefficients for each parent and the exogenous term. The parents
        are ordered by their index in the graph using `list(dag.predecessors(idx))`.
    coeff_values_requires_grad : list[list[bool]]]
        Whether each coefficient requires a gradient to update it or not.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coeff_values = []
    coeff_values_requires_grad = []
    n_nodes = dag.number_of_nodes()
    for idx in range(n_nodes):
        coeff_values_i = []
        coeff_values_requires_grad_i = []
        num_parents = len(list(dag.predecessors(idx)))
        if cluster_mapping is not None:
            # get start/end in the representation for this cluster
            start, end = cluster_mapping[idx]
            cluster_size = int(end - start)

        for pa_idx in range(num_parents):
            if cluster_mapping is None:
                random_val = Uniform(min_val, max_val).sample((1,))
                val = random_val
                param = nn.Parameter(val * torch.ones(1), requires_grad=True)  # .to(device)
            elif cluster_mapping is not None and use_matrix:
                if len(cluster_mapping[idx]) != len(cluster_mapping[pa_idx]):
                    raise ValueError(
                        "The number of dimensions for the latent clusters must be the same if "
                        "using a matrix to represent the edge coefficients."
                        f"{idx}: {len(cluster_mapping[idx])} != {pa_idx}: {len(cluster_mapping[pa_idx])}."
                    )
                # sample a random invertible matrix
                matrix = sample_invertible_matrix(cluster_size, min_val, max_val)
                param = nn.Parameter(matrix, requires_grad=True)  # .to(device)
            else:
                # sample a random vector with each value between min_val and max_val
                vector = (max_val - min_val) * torch.rand((cluster_size,)) + min_val
                param = nn.Parameter(vector, requires_grad=True)  # .to(device)

            coeff_values_i.append(param)
            coeff_values_requires_grad_i.append(True)

        if cluster_mapping is None:
            const = torch.ones(1, requires_grad=False)  # .to(device)  # variance param
        else:
            const = nn.Parameter(torch.ones(cluster_size, requires_grad=False))  # .to(device)

        coeff_values_i.append(const)
        coeff_values_requires_grad_i.append(False)
        coeff_values.append(nn.ParameterList(coeff_values_i))
        coeff_values_requires_grad.append(coeff_values_requires_grad_i)
    return coeff_values, coeff_values_requires_grad


def set_initial_noise_parameters(
    dag: nx.DiGraph,
    fix_mechanisms: bool,
    intervention_targets: Tensor,
    environments: Tensor,
    min_val: float,
    max_val: float,
    cluster_mapping=None,
    use_matrix: bool = False,
    n_dim_per_node: list[int] = None,
    device: torch.device = None,
) -> tuple[list[ParameterList], list[list[bool]]]:
    """Set initial noise parameters for each node's distribution and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    fix_mechanisms : bool
        Whether to allow fix the mechanisms and intervention targets,
        so it is non-trainable.
    intervention_targets : Tensor
        Intervention targets for each distribution.
    environments : Tensor
        The environment index for each intervention target.
        XXX: included to enable selection diagram inputs. We need a function
        that is part of the graph that queries whether a variable/node is "shifted"
        or not given two domain indices.
    min_val : float
        The minimum value for the noise parameter.
    max_val : float
        The maximum value for the noise parameter.
    n_dim_per_node : list[int]
        The number of dimensions for each node in the graph. Default is None,
        corresponding to a single dimension for each node. If a node has more
        than one dimension, it will have a separate noise parameter for each
        dimension and sample from a multivariate distribution.
    device : torch.device
        The device to put the parameters on. Can be 'cpu' or 'cuda'.

    Returns
    -------
    noise_params : list[ParameterList] or torch.Tensor of shape (num_distributions, num_nodes)
        The noise means/stds for each variable in each distribution.
    noise_params_requires_grad : list[list[bool]]]
        Whether each noise mean/std requires a gradient to update it or not.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_dim_per_node is None:
        n_dim_per_node = [1] * dag.number_of_nodes()

    noise_params = []
    noise_params_requires_grad = []

    # noise_params = torch.zeros((intervention_targets.shape[0], dag.number_of_nodes()))
    num_distributions = intervention_targets.shape[0]

    for idx in range(num_distributions):
        noise_means_e = []
        noise_means_requires_grad_e = []

        # this goes in the order of the nodes in the graph
        # stored as an adjacency matrix
        for node_idx in range(dag.number_of_nodes()):
            n_dims = n_dim_per_node[node_idx]
            is_shifted = intervention_targets[idx][node_idx] == 1
            is_root = len(list(dag.predecessors(node_idx))) == 0

            # fix mechanism if it is not shifted by an intervention
            if fix_mechanisms:
                is_fixed = is_shifted
            else:
                # fix mechanism if it is a root or if it is not shifted
                is_fixed = (is_shifted and not is_root) or (not is_shifted and is_root)
            is_fixed = is_fixed

            # initialize the means
            random_val = Uniform(min_val, max_val).sample((n_dims,))
            params = nn.Parameter(
                random_val * torch.ones(1), requires_grad=not is_fixed
            )  # .to(device)
            noise_means_e.append(params)
            noise_means_requires_grad_e.append(not is_fixed)
        noise_params.append(nn.ParameterList(noise_means_e))
        noise_params_requires_grad.append(noise_means_requires_grad_e)
    return noise_params, noise_params_requires_grad


def set_initial_confounder_parameters(
    confounded_variables: list[tuple[int, int]],
    min_val: float,
    max_val: float,
    n_dim_per_node: list[int] = None,
) -> tuple[list[ParameterList], list[list[bool]]]:
    """Set initial noise parameters for each node's distribution and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    fix_mechanisms : bool
        Whether to allow fix the mechanisms and intervention targets,
        so it is non-trainable.
    intervention_targets : Tensor
        Intervention targets for each distribution.
    min_val : float
        The minimum value for the noise parameter.
    max_val : float
        The maximum value for the noise parameter.
    n_dim_per_node : list[int]
        The number of dimensions for each node in the graph. Default is None,
        corresponding to a single dimension for each node. If a node has more
        than one dimension, it will have a separate noise parameter for each
        dimension and sample from a multivariate distribution.

    Returns
    -------
    noise_means : list[ParameterList]
        The noise means for each confounder.
    noise_stds : list[list[bool]]]
        The noise stds for each confounder.
    """
    noise_means = []
    noise_stds = []
    num_confounders = len(confounded_variables)

    for idx in range(num_confounders):
        n_dims = n_dim_per_node[idx]

        # initialize the means
        random_val = Uniform(min_val, max_val).sample((n_dims,))
        params = nn.Parameter(random_val * torch.ones(1), requires_grad=False)
        noise_means.append(params)

        # initialize the stds
        random_val = Uniform(min_val, max_val).sample((n_dims,))
        params = nn.Parameter(random_val * torch.ones(1), requires_grad=False)
        noise_stds.append(params)

    return noise_means, noise_stds
