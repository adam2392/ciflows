from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import normflows as nf
from normflows.distributions import DiagGaussian
from normflows.core import ConditionalNormalizingFlow
from numpy.testing import assert_array_equal
from torch import nn

from ciflows.distributions.multidistr import MultidistrCausalFlow


class MultiDistrDAGNode(nn.Module):
    def __init__(
            self, dim, q0=None, parent_dims=None,
                 interv_indices=None, hard_interventions=None
                 ):
        super().__init__()
        if interv_indices is None:
            interv_indices = []
        if hard_interventions is None:
            hard_interventions = [False for _ in range(len(interv_indices))]
        if len(interv_indices) != len(hard_interventions):
            raise ValueError("Interventions and hard interventions must be same length "
                             "if specified")

        # create normalizing flow for each parent
        if parent_dims is not None:
            self.multi_distr_flows = nn.ModuleDict()
            total_dim = 0

            for distr_idx in interv_indices:
                self.multi_distr_flows[distr_idx] = nn.ModuleDict()
                
                for parent_name, parent_dim in parent_dims.items:
                    self.multi_distr_flows[distr_idx][parent_name] = nf.flows.AutoregressiveRationalQuadraticSpline(
                        parent_dim, 3, 128
                    )
                    total_dim += parent_dim
            if total_dim != dim:
                raise ValueError(f"Sum of parent dimensions {total_dim} must equal node dimension {dim}")

        if q0 is None:
            q0 = DiagGaussian(shape=dim)
        self.q0 = q0

    def forward():
        pass


class DAGFlow(MultidistrCausalFlow):
    def __init__(
        self,
        node_dimensions,
        edge_list,
        confounded_list=None,
        intervened_nodes=None,
    ):
        super(DAGFlow, self).__init__()

        # default is standard gaussian noise for all nodes and their dimensionalities
        if confounded_list is None:
            confounded_list = []

        self.node_dimensions = node_dimensions
        self.latent_dim = sum(node_dimensions.values())
        self.edge_list = edge_list
        self.confounded_list = confounded_list

        # maps distribution index to a set of variables
        self.distr_idx_map = dict()
        self.distr_idx_map[0] = set()

        # Create a weight dictionary for edges
        self.edge_weights = nn.ParameterDict(
            {
                f"{src}->{tgt}": nn.Parameter(
                    torch.randn(node_dimensions[src], node_dimensions[tgt]) + 1.0,
                    requires_grad=trainable_edges,
                )
                for src, tgt in edge_list
            }
        )

        # create the graph over endogenous variables
        self.endog_graph = nx.DiGraph()
        self.endog_graph.add_edges_from(edge_list)


        # create an initial normalizing flow from the input
        # and then feed in wrt DAG
        self.nodes = nn.ModuleDict()
        for node_name, node_dim in node_dimensions.items():
            if node_name not in self.endog_graph.nodes:
                raise ValueError(f"Node {node_name} must be in the endogenous graph")
            
            # create a DAGNode for each node
            dagnode = MultiDistrDAGNode(
                dim=node_dim,
                q0=DiagGaussian(shape=node_dim),
                parent_dims={src: node_dimensions[src] for src in self.endog_graph.predecessors(node_name)},
            )
            self.nodes[node_name] = dagnode
        

    def sample(self, num_samples=1, **kwargs):
        return super().sample(num_samples, **kwargs)
    
    def forward(self, num_samples=1):
        # start from the q0's of each of the DAGNodes

    
    def log_prob(self, v, e, intervention_targets, hard_interventions = None):
        return super().log_prob(v, e, intervention_targets, hard_interventions)