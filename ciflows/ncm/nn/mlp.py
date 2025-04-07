import torch
import torch.nn as nn

from ciflows.ncm.nn.base import SCMFunc


class MLP(SCMFunc):
    def __init__(
        self,
        pa_size,
        u_size,
        o_size,
        h_size=128,
        h_layers=2,
        use_sigmoid=True,
        use_layer_norm=True,
        output_shape=None,
    ):
        """
        Multi-Layer Perceptron (MLP) for modeling variable dependencies in SCMs.

        Parameters
        ----------
        pa_size : dict
            Dictionary mapping parent variable names to their dimensionalities.
        u_size : dict
            Dictionary mapping exogenous (latent) variable names to their dimensionalities.
        o_size : int
            Output dimensionality of the MLP.
        h_size : int, optional
            Number of units in each hidden layer (default is 128).
        h_layers : int, optional
            Number of hidden layers (default is 2).
        use_sigmoid : bool, optional
            Whether to apply a sigmoid activation to the final layer (default is True).
        use_layer_norm : bool, optional
            Whether to apply layer normalization after each hidden layer (default is True).
        output_shape : tuple, optional
            If provided, the output tensor will be reshaped to this shape after the forward pass.

        Attributes
        ----------
        nn : nn.Sequential
            The full feedforward network.
        pa, u : list
            Sorted lists of parent and latent variable names used in the input.

        Methods
        -------
        forward(pa, u, include_inp=False, debug=False)
            Performs a forward pass with concatenated parent and latent inputs.

        Notes
        -----
        - The model automatically concatenates parent (`pa`) and latent (`u`) inputs.
        - Input dictionaries should have tensors of shape (batch_size, variable_dim).
        - Output tensor shape is (batch_size, o_size).
        """
        super().__init__(pa_size, u_size, o_size, output_shape)

        self.h_size = h_size
        layers = [nn.Linear(self.i_size, self.h_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU())
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.o_size))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(torch.empty(0))

        self.nn.apply(self.init_weights)


if __name__ == "__main__":
    s = MLP(dict(v1=2, v2=1), dict(u1=1, u2=2), 3)
    print(s)
    pa = {"v1": torch.tensor([[1, 2], [3, 4.0]]), "v2": torch.tensor([[5], [6.0]])}
    u = {"u1": torch.tensor([[7.0], [8]]), "u2": torch.tensor([[9, 10], [11, 12.0]])}
    print(s(pa, u))
