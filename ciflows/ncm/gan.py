import normflows as nf
import torch
import torch.nn as nn

from ciflows.ncm.cg import CausalGraph
from ciflows.ncm.distribution import NeuralDistribution, UniformDistribution
from ciflows.ncm.nn.mlp import MLP
from ciflows.ncm.scm import SCM
from ciflows.ncm.utils import expand_do

from normflows import MultiscaleFlow
from normflows.flows import GlowBlock, Squeeze, Merge
from normflows.distributions import DiagGaussian

from ciflows.ncm.flow import GlowFlowBlock1D, GlowFlowBlock2D

# ========= Example A: For images of shape (3, 32, 32) =========

def build_glow_image_model():
    # Settings similar to the example on the normflows page:
    L = 3              # number of scales
    K = 16             # number of Glow blocks per scale
    torch.manual_seed(0)
    
    input_shape = (3, 32, 32)
    channels = input_shape[0]      # for images, channels = 3
    hidden_channels = 256
    split_mode = 'channel'
    scale = True
    num_classes = 10               # for class‐conditional models

    q0 = []     # base distributions per scale
    merges = [] # merge operations at scales > 1
    flows = []  # list of flows per scale

    for i in range(L):
        flows_i = []
        # Create K GlowBlocks at each scale.
        for j in range(K):
            # The number of channels increases at coarser scales.
            num_channels = channels * 2 ** (L + 1 - i)
            flows_i += [GlowBlock(num_channels, hidden_channels,
                                  split_mode=split_mode, scale=scale)]
        # At the end of the scale, apply a squeeze (spatial downsampling)
        flows_i += [Squeeze()]
        flows.append(flows_i)
        
        # For scales i > 0, a merge operation is applied.
        if i > 0:
            merges += [Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i),
                            input_shape[1] // 2 ** (L - i),
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1),
                            input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        q0 += [DiagGaussian(latent_shape)]
    
    # Construct the multiscale flow model.
    model = MultiscaleFlow(q0, flows, merges)
    return model

# ========= Example B: For a “vector” output/input =========
# Here we assume the vector is of length 32. To use the same multiscale Glow architecture,
# we reshape it as a 1-channel “image” of size (32,1). Note that spatial operations like Squeeze
# require the spatial dimensions to be even. For simplicity we use a single-scale (L=1) model without squeezing.

def build_glow_vector_model():
    # We redefine the input shape as a 1-channel “image” of size (32, 1)
    # even if the intended data is a 32-dimensional vector.
    input_shape = (1, 32, 1)  # channels=1, height=32, width=1
    L = 1      # Use a single scale; squeezing might be problematic if width==1
    K = 4      # Fewer Glow blocks are typically enough for low-dimensional data
    torch.manual_seed(0)
    
    channels = input_shape[0]     # Here, channels = 1
    hidden_channels = 64          # Smaller hidden dimension for low-dimensional data
    split_mode = 'channel'
    scale = True
    num_classes = None            # For unconditional flows, or set to an integer if class-conditioning is desired

    q0 = []
    merges = []  # With a single scale, no merges are applied.
    flows = []

    # For L==1, we can simply build a list of GlowBlocks.
    flows_i = []
    for j in range(K):
        # For a single scale, the number of channels remains the same.
        num_channels = channels * 2 ** (L + 1)  # following the convention from the image example
        flows_i += [GlowBlock(num_channels, hidden_channels,
                              split_mode=split_mode, scale=scale)]
    # For vector data, we typically skip the Squeeze operation.
    flows.append(flows_i)
    
    # Compute latent shape according to the same convention.
    latent_shape = (input_shape[0] * 2 ** (L + 1),
                    input_shape[1] // 2 ** L,
                    input_shape[2] // 2 ** L)
    q0 += [DiagGaussian(latent_shape)]  # if num_classes is None, the distribution may be unconditional.

    model = MultiscaleFlow(q0, flows, merges)
    return model


class GAN_NCM(SCM):
    """
    Generative Adversarial Network-based Non-Causal Model (GAN_NCM).

    TODO:
    1. allow for different distributions represented by different latents
    2. allow for different distributions represented by different functions

    Parameters
    ----------
    cg : object
        Causal graph structure defining variable relationships.
    v_size : dict, optional
        Dictionary mapping variable names to their dimensions (default: {}).
    default_v_size : int, optional
        Default size for observed variables (default: 1).
    u_size : dict, optional
        Dictionary mapping latent noise variables to their dimensions (default: {}).
    default_u_size : int, optional
        Default size for latent noise variables (default: 1).
    f : dict, optional
        Dictionary of predefined generative functions for variables (default: {}).
    hyperparams : dict, optional
        Hyperparameters for the model, including:
        - 'h-layers' (int): Number of hidden layers (default: 2).
        - 'h-size' (int): Hidden layer size (default: 128).
        - 'layer-norm' (bool): Whether to use layer normalization (default: False).
        - 'neural-pu' (bool): Whether to use a neural parameterized distribution.
        - 'single-disc' (bool): Whether to use a single discriminator.
        - 'do-var-list' (list): List of intervention variables.
    default_gen_module : nn.Module, optional
        Default neural network module for generating samples (default: `MLP`).
    disc_module : nn.Module, optional
        Module to be used for discriminators (default: `MLP`).
    gen_use_sigmoid : bool, optional
        Whether to apply a sigmoid activation in generators (default: True).
    disc_use_sigmoid : bool, optional
        Whether to apply a sigmoid activation in discriminators (default: True).

    Attributes
    ----------
    gens : nn.ModuleDict
        Dictionary of generative functions for each variable.
    pu_dist : Distribution
        Prior distribution over noise variables.
    f_disc : nn.Module or nn.ModuleList
        Discriminator(s) for adversarial training.
    single_disc : bool
        Whether to use a single discriminator or multiple discriminators.
    delta_v_list : list of dict
        List of difference in distribution setups per variable. Key is variable name,
        value is 'conditional', or 'hard', where 'conditional' means the variable is
        generated conditionally its existing parents, and 'hard' means the variable is
        generated independently of its parents only using the latent exogenous noise.

    Methods
    -------
    convert_evaluation(samples)
        Converts generated samples into binary outputs.
    query_loss(input, val)
        Computes query loss for inference, using squared loss if sigmoid is off.
    get_disc_outputs(samples, index, include_inp=False)
        Computes discriminator outputs for given samples.
    """

    def __init__(
        self,
        cg: CausalGraph,
        delta_v_list=[{}],
        v_size={},
        default_v_size=1,
        u_size={},
        default_u_size=1,
        f={},
        hyperparams=None,
        default_gen_module=MLP,
        disc_module=MLP,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    ):
        if hyperparams is None:
            hyperparams = {}

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}

        self.gen_use_sigmoid = gen_use_sigmoid

        # initialize the generative model
        gens = nn.ModuleDict(
            {
                v: (
                    f[v]
                    if v in f
                    else default_gen_module(
                        {k: self.v_size[k] for k in self.cg.pa[v]},
                        {k: self.u_size[k] for k in self.cg.v2c2[v]},
                        self.v_size[v],
                        h_layers=hyperparams.get("h-layers", 2),
                        h_size=hyperparams.get("h-size", 128),
                        use_layer_norm=hyperparams.get("layer-norm", False),
                        use_sigmoid=gen_use_sigmoid,
                    )
                )
                for v in cg
            }
        )

        # define list of endogenous variable functions that are defined differently
        # for each distribution
        list_delta_f_funcs = []
        for idx in range(len(delta_v_list)):
            dict_delta_f_funcs = dict()
            delta_vs = delta_v_list[idx]
            for v, change_type in delta_vs.items():
                # if the variable is not in the causal graph, skip it
                if v not in cg:
                    raise ValueError(f"Variable {v} in delta_v_list is not in the causal graph.")

                if change_type == "conditional":
                    dict_delta_f_funcs[v] = default_gen_module(
                        {k: self.v_size[k] for k in self.cg.pa[v]},
                        {k: self.u_size[k] for k in self.cg.v2c2[v]},
                        self.v_size[v],
                        h_layers=hyperparams.get("h-layers", 2),
                        h_size=hyperparams.get("h-size", 128),
                        use_layer_norm=hyperparams.get("layer-norm", False),
                        use_sigmoid=gen_use_sigmoid,
                    )
                elif change_type == "hard":
                    dict_delta_f_funcs[v] = default_gen_module(
                        {},
                        {k: self.u_size[k] for k in self.cg.v2c2[v]},
                        self.v_size[v],
                        h_layers=hyperparams.get("h-layers", 2),
                        h_size=hyperparams.get("h-size", 128),
                        use_layer_norm=hyperparams.get("layer-norm", False),
                        use_sigmoid=gen_use_sigmoid,
                    )
                else:
                    raise ValueError(
                        f"Invalid change_type '{change_type}' for variable '{v}'. "
                        "Expected 'conditional' or 'hard'."
                    )
            dic_delta_f = nn.ModuleDict(dict_delta_f_funcs)
            list_delta_f_funcs.append(dic_delta_f)
        list_delta_f_funcs = nn.ModuleList(list_delta_f_funcs)

        # initialize the latent exogenous distributions
        pu_dist = (
            NeuralDistribution(self.cg.c2, self.u_size, hyperparams)
            if hyperparams.get("neural-pu", False)
            else UniformDistribution(self.cg.c2, self.u_size)
        )

        super().__init__(
            v=list(cg), f=gens, pu=pu_dist, delta_v_list=delta_v_list, delta_f=list_delta_f_funcs
        )

        self.single_disc = hyperparams.get("single-disc", False)

        # initialize discriminator(s)
        self._init_discriminator(disc_module, disc_use_sigmoid, hyperparams)

    def _init_discriminator(self, disc_module, disc_use_sigmoid, hyperparams):
        if self.single_disc:
            disc_sizes = {k: v for (k, v) in self.v_size.items()}
            disc_sizes["_delta_choice"] = len(self.delta_v_list)
            self.f_disc = disc_module(
                pa_size=disc_sizes,
                u_size={},
                o_size=1,
                h_layers=hyperparams.get("h-layers", 2),
                h_size=len(self.v_size) * hyperparams.get("h-size", 128),
                use_sigmoid=disc_use_sigmoid,
                use_layer_norm=hyperparams.get("layer-norm", False),
            )
        else:
            self.f_disc = nn.ModuleList(
                [
                    disc_module(
                        self.v_size,
                        {},
                        1,
                        h_layers=hyperparams.get("h-layers", 2),
                        h_size=len(self.v_size) * hyperparams.get("h-size", 128),
                        use_sigmoid=disc_use_sigmoid,
                        use_layer_norm=hyperparams.get("layer-norm", False),
                    )
                    for _ in range(self.delta_v_list)
                ]
            )

    def generator_parameters(self, include_pu=False, include_delta_f=False):
        """
        Returns the parameters of the generator.

        Returns
        -------
        list
            List of generator parameters.
        """
        params = list(self.f.parameters())

        if include_pu:
            params += list(self.pu.parameters())
        if include_delta_f:
            params += list(self.delta_f.parameters())
        return params

    def discriminator_parameters(self):
        """
        Returns the parameters of the discriminator.

        Returns
        -------
        list
            List of discriminator parameters.
        """
        return list(self.f_disc.parameters())

    def convert_evaluation(self, samples):
        """
        Converts generated samples into binary outcomes.

        Parameters
        ----------
        samples : dict
            Dictionary of generated samples.

        Returns
        -------
        dict
            Dictionary with binary values for each variable.
        """
        return {k: (samples[k] > 0.5).float() for k in samples}

    def query_loss(self, input, val):
        """
        Computes the loss for inference queries.

        Parameters
        ----------
        input : torch.Tensor
            The model output.
        val : torch.Tensor or float
            The ground-truth value.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        if self.gen_use_sigmoid:
            return super().query_loss(input, val)
        else:
            if torch.is_tensor(val):
                raise NotImplementedError()
            return torch.sum(torch.square(input - val))

    def get_disc_outputs(self, samples, index, include_inp=False):
        """
        Computes discriminator outputs for given samples.

        Parameters
        ----------
        samples : dict
            Dictionary of generated samples.
        index : int
            Index for selecting the corresponding discriminator.
        include_inp : bool, optional
            Whether to return the input along with the discriminator output.

        Returns
        -------
        torch.Tensor
            Discriminator output.
        """
        if self.single_disc:
            # get the number of samples
            sample = next(iter(samples))
            n = len(samples[sample])

            one_hot_do = torch.zeros(self.delta_v_list, device=self.device_param.device)
            one_hot_do[index] = 1
            inp = {k: v for k, v in samples.items()}
            inp["_delta_choice"] = expand_do(one_hot_do, n)
            return self.f_disc(inp, {}, include_inp=include_inp)
        return self.f_disc[index](samples, {}, include_inp=include_inp)


class GAN_NF_NCM(GAN_NCM):
    """
    GAN_NF_NCM extends GAN_NCM by adding normalizing flows (NF) to mix
    the generated samples. The flows are applied to the concatenated outputs
    of the causal graph's variables to produce a final output sample.

    Attributes
    ----------
    x_size : int
        Total output dimension computed as the sum of the sizes of all variables.
    f_X : nn.ModuleList
        List of normalizing flow modules (Autoregressive Rational Quadratic Spline)
        used to mix the outputs.
    """

    def __init__(
        self,
        cg,
        delta_v_list=[{}],
        v_size={},
        default_v_size=1,
        u_size={},
        default_u_size=1,
        f={},
        hyperparams=None,
        default_gen_module=MLP,
        disc_module=MLP,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    ):
        # get the size of the output generation, which is the sum of the sizes of
        # the variables in the causal graph generated
        self.v_size = {k: v_size.get(k, default_v_size) for k in cg}
        self.x_size = sum(self.v_size.values())

        # initialize the rest of the GAN model
        super().__init__(
            cg,
            delta_v_list,
            v_size,
            default_v_size,
            u_size,
            default_u_size,
            f,
            hyperparams,
            default_gen_module,
            disc_module,
            gen_use_sigmoid,
            disc_use_sigmoid,
        )

        # initialize a set of normalizing flow blocks
        net_hidden_layers = hyperparams.get("h-layers", 2)
        net_hidden_dim = hyperparams.get("h-size", 128)
        K_flows = hyperparams.get("K", 16)
        # flows = []
        # for i in range(K_flows):
        #     flows += [
        #         nf.flows.AutoregressiveRationalQuadraticSpline(
        #             self.x_size, net_hidden_layers, net_hidden_dim
        #         )
        #     ]
        # self.f_X = nn.ModuleList(flows)
        flows = []
        for _ in range(K_flows):
            # Create either a 1D or 2D Glow flow block depending on flow_input_shape
            if len(self.flow_input_shape) == 1:
                flows.append(GlowFlowBlock1D(self.flow_input_shape[0], net_hidden_dim, num_steps=num_steps))
            elif len(self.flow_input_shape) == 3:
                # Here self.flow_input_shape[0] is channels.
                flows.append(GlowFlowBlock2D(self.flow_input_shape[0], net_hidden_dim, num_steps=num_steps))
            else:
                raise ValueError("Unsupported flow_input_shape: {}".format(self.flow_input_shape))
        self.f_X = nn.ModuleList(flows)

    def _init_discriminator(self, disc_module, disc_use_sigmoid, hyperparams):
        """Discriminator operates over X."""
        if self.single_disc:
            disc_sizes = {"X": self.x_size}
            disc_sizes["_delta_choice"] = len(self.delta_v_list)
            self.f_disc = disc_module(
                pa_size=disc_sizes,
                u_size={},
                o_size=1,
                h_layers=hyperparams.get("h-layers", 2),
                h_size=len(self.v_size) * hyperparams.get("h-size", 128),
                use_sigmoid=disc_use_sigmoid,
                use_layer_norm=hyperparams.get("layer-norm", False),
            )
        else:
            self.f_disc = nn.ModuleList(
                [
                    disc_module(
                        {"X": self.x_size},
                        {},
                        1,
                        h_layers=hyperparams.get("h-layers", 2),
                        h_size=len(self.v_size) * hyperparams.get("h-size", 128),
                        use_sigmoid=disc_use_sigmoid,
                        use_layer_norm=hyperparams.get("layer-norm", False),
                    )
                    for _ in range(self.delta_v_list)
                ]
            )

    def apply_mixing_function(self, v):
        """
        Applies the mixing function to the generated samples.

        Parameters
        ----------
        v : torch.Tensor
            Generated samples.

        Returns
        -------
        torch.Tensor
            Mixed samples.
        """
        x = torch.cat([v[k] for k in v], dim=1)
        for flow in self.f_X:
            x, _ = flow(x)
        return x

    def get_disc_outputs(self, samples, index, include_inp=False, debug=False):
        """
        Computes the discriminator outputs for given samples in the NF setting.

        When using a single discriminator, this method also attaches a one-hot
        vector (named '_delta_choice') to indicate the current intervention index.
        Otherwise, it uses the indexed discriminator from a ModuleList.

        Parameters
        ----------
        samples : dict
            Dictionary of generated samples.
        index : int
            Index of the discriminator or intervention to use.
        include_inp : bool, optional
            If True, returns the input to the discriminator along with its output.

        Returns
        -------
        torch.Tensor
            Discriminator output for the provided samples.
        """
        if self.single_disc:
            # get the number of samples
            sample = next(iter(samples))
            n = len(samples[sample])

            one_hot_do = [0 for _ in range(len(self.delta_v_list))]
            one_hot_do[index] = 1
            one_hot_do = torch.FloatTensor(one_hot_do).to(self.device_param)
            inp = {k: v for (k, v) in samples.items()}
            inp["_delta_choice"] = expand_do(one_hot_do, n)
            return self.f_disc(inp, {}, include_inp=include_inp, debug=debug)
        else:
            return self.f_disc[index](samples, {}, include_inp=include_inp)

    def sample_mixture(
        self, v_samples_list=None, n=None, u=None, select=None, idx=None, include_v=False
    ):
        """
        Samples from the model and applies the mixing function to produce final outputs.

        This method calls the base sample() method if v_samples_list is not provided,
        then for each intervention setup it applies the mixing function via `apply_mixing_function`.

        Parameters
        ----------
        v_samples_list : list of dict, optional
            Pre-computed samples for each intervention setup. If None, samples are generated.
        n : int, optional
            Number of samples to generate if v_samples_list is None.
        u : dict, optional
            Exogenous noise values to use for sampling.
        select : list, optional
            List of variable names to include.
        idx : list, optional
            Indices of intervention setups to use.
        include_v : bool, optional
            If True, includes the original variable outputs in the result.

        Returns
        -------
        list of dict
            A list where each element is a dictionary containing the mixed output under key 'X'
            (and, if include_v is True, the original outputs as well).
        """
        if v_samples_list is None:
            v_samples_list = self.sample(n=n, u=u, select=select, idx=idx)
        results_list = []
        for i in range(len(v_samples_list)):
            if include_v:
                result = v_samples_list[i]
            else:
                result = {}
            v_samples = v_samples_list[i]
            result["X"] = self.apply_mixing_function(v_samples)
            results_list.append(result)
        return results_list


# =======================
# Testing the Models
# =======================
def test_nf_models():
    # Example for 1D: Suppose we want to mix a (B, 1) tensor.
    hyperparams_1d = {
        "h-size": 64,
        "num_glow_steps": 2,
        "K": 4,
        "flow_input_shape": (1,)
    }
    # Create a dummy causal graph (details depend on your implementation)
    # cg = {'var1': 1}
    # # Dummy causal graph
    cg = CausalGraph({"A": [], "B": ["A"], "C": ["A", "B"]})

    # v_size here: variable "var1" is size 1
    model_1d = GAN_NF_NCM(cg, v_size={'var1': 1}, hyperparams=hyperparams_1d)
    # Create dummy samples: dictionary with key 'var1', tensor shape (batch, 1)
    samples_1d = {'var1': torch.randn(32, 1)}
    mixed_1d = model_1d.apply_mixing_function(samples_1d)
    print("Mixed 1D shape:", mixed_1d.shape)  # expected (32, 1)

    # Example for 2D: Suppose we want to mix images of shape (3, 32, 32).
    hyperparams_2d = {
        "h-size": 64,
        "num_glow_steps": 2,
        "K": 4,
        "flow_input_shape": (3, 32, 32)
    }
    # For images, assume our causal graph outputs already have the 3 channels.
    model_2d = GAN_NF_NCM(cg, v_size={'var1': 3*32*32}, hyperparams=hyperparams_2d)
    # Create dummy samples: here we mimic an image stored under key 'var1'
    # We need shape (B, 3, 32, 32): so create random tensor and make sure that the
    # concatenation in apply_mixing_function recovers that shape.
    images = torch.randn(32, 3, 32, 32)
    # Here we wrap the image tensor in a dict. (In your application each variable may
    # be a slice or part of the final image.)
    samples_2d = {'var1': images}
    mixed_2d = model_2d.apply_mixing_function(samples_2d)
    print("Mixed 2D shape:", mixed_2d.shape)  # expected (32, 3, 32, 32)

if __name__ == "__main__":
    # test_nf_models()
    from ciflows.ncm.cg import CausalGraph

    # Dummy causal graph
    cg = CausalGraph({"A": [], "B": ["A"], "C": ["A", "B"]})

    # Sample hyperparameters
    hyperparams = {
        "h-layers": 2,
        "h-size": 64,
        "layer-norm": True,
        "neural-pu": False,
        "single-disc": True,
        "do-var-list": ["A", "B"],
    }

    # Initialize GAN_NCM
    gan_ncm = GAN_NCM(cg, hyperparams=hyperparams)

    # Generate samples
    samples = gan_ncm.sample(
        n=5,
        #  idx=[0, 1]
    )

    # Evaluate discriminator output
    print("Generated Samples:", samples)
    disc_output = gan_ncm.get_disc_outputs(samples[0], index=0)

    print("Discriminator Output:", disc_output)
    print(len(samples))
    print(disc_output.shape)
