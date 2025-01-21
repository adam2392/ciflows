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
        confounded_list=None,
        intervened_nodes=None,
    ):
        pass

    def sample(self, num_samples=1, **kwargs):
        return super().sample(num_samples, **kwargs)
    
    def forward(self, num_samples=1):
        return super().forward(num_samples)
    
    def log_prob(self, v, e, intervention_targets, hard_interventions = None):
        return super().log_prob(v, e, intervention_targets, hard_interventions)