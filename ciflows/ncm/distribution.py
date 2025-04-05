import numpy as np
import torch
import torch.nn as nn
from .mlp import MLP


class Distribution(nn.Module):
    """
    Base class for probability distributions.

    Parameters
    ----------
    u : list of str
        Names of the random variables.

    Attributes
    ----------
    u : list of str
        Names of the random variables.
    device_param : torch.nn.Parameter
        Dummy parameter to track device placement.

    Methods
    -------
    sample(n=1, device="cpu")
        Samples from the distribution (must be implemented in subclasses).
    forward(n=1)
        Calls `sample(n)`, enabling compatibility with `torch.nn.Module`.
    """

    def __init__(self, u):
        super().__init__()
        self.u = u
        self.device_param = nn.Parameter(torch.empty(0))

    def __iter__(self):
        return iter(self.u)

    def sample(self, n=1, device="cpu"):
        """
        Samples from the distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to draw (default is 1).
        device : str, optional
            Device to store the samples (default is "cpu").

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are sampled tensors.
        """
        raise NotImplementedError()

    def forward(self, n=1):
        """
        Calls `sample(n)`, allowing the class to be used as a `torch.nn.Module`.

        Parameters
        ----------
        n : int, optional
            Number of samples to draw (default is 1).

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are sampled tensors.
        """
        return self.sample(n=n)


class UniformDistribution(Distribution):
    """
    Uniform distribution over specified variables.

    Parameters
    ----------
    u_names : list of str
        Names of the random variables.
    sizes : dict of {str: int}
        Dictionary specifying the dimensionality of each variable.
    seed : int, optional
        Random seed for reproducibility (default is None).

    Attributes
    ----------
    sizes : dict of {str: int}
        Dimensionality of each variable.
    rand_state : np.random.RandomState
        Random number generator.
    """

    def __init__(self, u_names, sizes, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes.get(U, 1) for U in u_names}
        self.rand_state = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )

    def sample(self, n=1, device=None):
        """
        Draws samples from a uniform distribution U(0,1).

        Parameters
        ----------
        n : int, optional
            Number of samples to draw (default is 1).
        device : str or torch.device, optional
            Device to store the samples (default is None, uses module's device).

        Returns
        -------
        dict
            Dictionary mapping variable names to sampled tensors.
        """
        if device is None:
            device = self.device_param.device

        return {
            U: torch.from_numpy(self.rand_state.rand(n, self.sizes[U])).float().to(device)
            for U in self.sizes
        }


class StandardNormalDistribution(Distribution):
    """
    Standard normal (Gaussian) distribution over specified variables.

    Parameters
    ----------
    u_names : list of str
        Names of the random variables.
    sizes : dict of {str: int}
        Dictionary specifying the dimensionality of each variable.
    seed : int, optional
        Random seed for reproducibility (default is None).

    Attributes
    ----------
    sizes : dict of {str: int}
        Dimensionality of each variable.
    rand_state : np.random.RandomState
        Random number generator.
    """

    def __init__(self, u_names, sizes, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes.get(U, 1) for U in u_names}
        self.rand_state = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )

    def sample(self, n=1, device=None):
        """
        Draws samples from a standard normal distribution N(0,1).

        Parameters
        ----------
        n : int, optional
            Number of samples to draw (default is 1).
        device : str or torch.device, optional
            Device to store the samples (default is None, uses module's device).

        Returns
        -------
        dict
            Dictionary mapping variable names to sampled tensors.
        """
        if device is None:
            device = self.device_param.device

        return {
            U: torch.from_numpy(self.rand_state.randn(n, self.sizes[U])).float().to(device)
            for U in self.sizes
        }


class NeuralDistribution(Distribution):
    """
    A neural network-based distribution where each variable is parameterized by an MLP.

    Parameters
    ----------
    u_names : list of str
        Names of the random variables.
    sizes : dict of {str: int}
        Dictionary specifying the dimensionality of each variable.
    hyperparams : dict
        Hyperparameters for the neural network, including:
        - 'h-layers' (int): Number of hidden layers (default: 2).
        - 'h-size' (int): Size of hidden layers (default: 128).
        - 'layer-norm' (bool): Whether to apply layer normalization (default: False).
    default_module : nn.Module, optional
        The neural network module to use for parameterizing the distribution (default: `MLP`).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Attributes
    ----------
    sizes : dict of {str: int}
        Dimensionality of each variable.
    rand_state : np.random.RandomState
        Random number generator.
    func : nn.ModuleDict
        Dictionary of neural network modules for generating samples.

    Methods
    -------
    sample(n=1, device=None)
        Generates `n` samples from the neural distribution.
    """

    def __init__(self, u_names, sizes, hyperparams, default_module=MLP, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes.get(U, 1) for U in u_names}
        self.rand_state = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        self.func = torch.nn.ModuleDict(
            {
                str(u): default_module(
                    {},
                    {u: self.sizes[u]},
                    self.sizes[u],
                    h_layers=hyperparams.get("h-layers", 2),
                    h_size=hyperparams.get("h-size", 128),
                    use_layer_norm=hyperparams.get("layer-norm", False),
                    use_sigmoid=False,
                )
                for u in self.sizes
            }
        )

    def sample(self, n=1, device=None):
        """
        Generates samples from the neural distribution using the learned MLP model.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate (default is 1).
        device : str or torch.device, optional
            Device to store the samples (default is None, which uses the module's device).

        Returns
        -------
        dict
            Dictionary mapping variable names to sampled tensors.
        """
        if device is None:
            device = self.device_param.device

        u_vals = {}
        for U in self.sizes:
            noise = torch.randn((n, self.sizes[U]), device=device, dtype=torch.float32)
            u_vals[U] = self.func[str(U)]({}, {U: noise})

        return u_vals
