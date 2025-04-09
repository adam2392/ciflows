from typing import Dict
import torch
import torch.nn as nn


class SCMFunc(nn.Module):
    def __init__(
        self,
        pa_size: Dict[str, int],
        u_size: Dict[str, int],
        o_size: int,
        output_shape = None,
    ):
        """
        Base neural network modeling variable dependencies in SCMs.

        Parameters
        ----------
        pa_size : dict
            Dictionary mapping parent variable names to their dimensionalities.
        u_size : dict
            Dictionary mapping exogenous (latent) variable names to their dimensionalities.
        o_size : int
            Output dimensionality of the SCM functional.
        output_shape : tuple, optional
            If provided, the output tensor will be reshaped to this shape after the forward pass.

        Attributes
        ----------
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
        super().__init__()
        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.u_size = u_size
        self.o_size = o_size
        self.output_shape = output_shape

        # total input dimensions
        self.i_size = sum(self.pa_size[k] for k in self.pa_size) + sum(
            self.u_size[k] for k in self.u_size
        )
        
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, pa, u, include_inp=False, debug=False):
        """
        Forward pass of the SCM neural network functional.

        Parameters
        ----------
        pa : dict of str -> torch.Tensor
            Dictionary of parent variable tensors. Each tensor should be of shape (batch_size, feature_dim).
        u : dict of str -> torch.Tensor
            Dictionary of exogenous noise variable tensors. Each tensor should be of shape (batch_size, feature_dim).
        include_inp : bool, optional
            If True, also returns the concatenated input tensor. Default is False.
        debug : bool, optional
            If True, prints debug information about the input keys. Default is False.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            Output of the neural network. If `include_inp` is True, also returns the input tensor used.
        """
        # if debug:
        #     print("In SCM func: ", pa.keys(), u.keys())

        if len(u.keys()) == 0:
            inp = torch.cat([pa[k] for k in self.pa], dim=1)
        elif len(pa.keys()) == 0 or len(set(pa.keys()).intersection(self.set_pa)) == 0:
            inp = torch.cat([u[k] for k in self.u], dim=1)
        else:
            inp_u = torch.cat([u[k] for k in self.u], dim=1)
            inp_pa = torch.cat([pa[k] for k in self.pa], dim=1)
            inp = torch.cat((inp_pa, inp_u), dim=1)

        # If the output dimension is 49152 (i.e. 3*128*128), reshape the output.
        x = self.nn(inp)
        if self.output_shape is not None:
            x = x.view(-1, *self.output_shape)

        if include_inp:
            return x, inp

        return x