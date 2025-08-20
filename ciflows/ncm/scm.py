import itertools

import torch
import torch.nn as nn

from .distribution import Distribution


class SCM(nn.Module):
    """
    Structural Causal Model (SCM) for generating samples under different interventions.

    Parameters
    ----------
    v : list
        List of observed variables in the model.
    f : dict
        Dictionary mapping variables to their generative functions.
    pu : Distribution
        Prior distribution over latent noise variables.
    delta_v_list : list of dict, optional
        A list of dictionaries specifying interventions or domain shifts on observed variables.
        Each dictionary represents a specific intervention setup:
        - If a variable is mapped to "hard", it undergoes a hard intervention, meaning its
          generative process is overridden.
        - If mapped to "conditional", the variable's value depends on other variables via
          an intervention-specific function.
        - If mapped to a constant, the variable is directly assigned that value.
    delta_f : list of dict, optional
        A list of dictionaries specifying modified generative functions corresponding to
        interventions in `delta_v_list`. If a variable undergoes an intervention, `delta_f`
        provides the function that defines its value under that intervention.

    Attributes
    ----------
    device_param : nn.Parameter
        A dummy parameter used to track the device for tensor allocation.

    Methods
    -------
    space(v_size, select=None, tensor=True)
        Generates all possible binary value assignments for selected variables.
    sample(n=None, u=None, select=None, idx=None)
        Samples from the SCM, considering interventions and domain shifts.
    forward(n=None, u=None, select=None, idx=None, evaluating=False)
        Generates samples, with an option to disable gradient tracking.
    """

    def __init__(self, v, f, pu: Distribution, delta_v_list=[{}], delta_f=[{}]):
        super().__init__()
        self.v = v  # List of observed variables
        self.u = list(pu)  # List of latent noise variables
        self.f = f  # Dictionary of structural functions for generating variables
        self.pu = pu  # Prior distribution over latent variables
        self.delta_v_list = delta_v_list  # List of interventions or domain shifts
        self.delta_f = delta_f  # List of generative functions corresponding to interventions
        self.device_param = nn.Parameter(torch.empty(0))  # Parameter for device tracking

    def space(self, v_size, select=None, tensor=True):
        """
        Generates all possible binary value assignments for selected variables.

        Parameters
        ----------
        v_size : dict
            Dictionary mapping variable names to their respective dimensions.
        select : list, optional
            Subset of variables to generate assignments for (default is all variables).
        tensor : bool, optional
            Whether to return values as tensors (default: True).

        Yields
        ------
        dict
            A dictionary mapping variable names to their possible binary values.
        """
        if select is None:
            select = self.v

        for pairs in itertools.product(
            *(
                [
                    (vi, torch.LongTensor(value).to(self.device_param.device) if tensor else value)
                    for value in itertools.product(*([0, 1] for _ in range(v_size[vi])))
                ]
                for vi in select
            )
        ):
            yield dict(pairs)

    def sample(self, n=None, u=None, select=None, idx=None):
        """
        Samples from the SCM, taking into account interventions and domain shifts.

        Parameters
        ----------
        n : int, optional
            Number of samples to draw. Required if `u` is not provided.
        u : dict, optional
            Dictionary of sampled latent variables. Required if `n` is not provided.
        select : list, optional
            Subset of variables to return in the sample (default is all variables).
        idx : list, optional
            List of indices specifying which intervention setup to use from `delta_v_list`.

        Returns
        -------
        list of dict
            A list of sampled dictionaries, each corresponding to one intervention setup.
        """
        assert (n is None) != (
            u is None
        ), "Specify either `n` (for new samples) or `u` (for given noise)."
        assert (idx is None) or (isinstance(idx, list) and max(idx) < len(self.delta_v_list))

        if idx is None:
            idx = list(range(len(self.delta_v_list)))  # Default: consider all intervention setups

        list_results = []

        for i in idx:
            delta = self.delta_v_list[i]  # Select the intervention setup
            if u is None:
                u = self.pu.sample(n)  # Sample latent noise if not provided

            if select is None:
                select = self.v  # Default: select all variables

            v = {}  # Dictionary to store sampled variables
            remaining = set(select)  # Track variables that need to be generated

            for k in self.v:
                if k in delta.keys():  # Check if the variable is affected by an intervention
                    if delta[k] == "hard":
                        # Hard intervention: override generative function with a new sampled value
                        u_tmp = self.pu.sample(n)
                        v[k] = self.delta_f[i][k]({}, u_tmp)
                    elif delta[k] == "conditional":
                        # Conditional intervention: use an alternative function based on current samples
                        v[k] = self.delta_f[i][k](v, u)
                    else:
                        # Directly assign fixed intervention values
                        v[k] = delta[k]
                else:
                    # No intervention, use the original generative function
                    v[k] = self.f[k](v, u)

                remaining.discard(k)
                if not remaining:
                    break  # Stop if all required variables are generated

            list_results.append({k: v[k] for k in select})

        return list_results

    def forward(self, n=None, u=None, select=None, idx=None, evaluating=False):
        """
        Forward pass to generate samples from the SCM.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate (required if `u` is not given).
        u : dict, optional
            Dictionary of latent variables (required if `n` is not given).
        select : list, optional
            Subset of variables to return in the output (default: all).
        idx : list, optional
            List of indices specifying which intervention setup to apply.
        evaluating : bool, optional
            If True, disables gradient computation for efficiency.

        Returns
        -------
        dict or list of dict
            Generated samples. If `evaluating=True`, returns a dictionary with CPU tensors.
        """
        if evaluating:
            with torch.no_grad():
                result = self.sample(n, u, select, idx)
                return {k: result[k].cpu() for k in result}

        return self.sample(n, u, select, idx)
