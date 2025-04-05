import torch
import torch.nn as nn
import numpy as np
from ciflows.ncm.distribution import UniformDistribution  # Assuming this is in your library
from ciflows.ncm.scm import SCM  # Replace with actual module name where SCM is defined


# Define generative functions
def f_x(_, u):
    """Generative function for X: a simple function of noise U."""
    return u["U"] * 2 + 1  # Example transformation of latent noise


def f_y(v, u):
    """Generative function for Y: depends on X and noise U."""
    return 0.5 * v["X"] + u["U"]


# Define noise distribution
pu_dist = UniformDistribution(["U"], {"U": 1})  # Uniform distribution over U

# Define intervention setups
delta_v_list = [
    {},  # No intervention (default setting)
    {"X": 3.0},  # Hard intervention: X is set to 3.0
]
delta_f = [{}, {}]  # No change to functions  # No change needed as X is set manually

# Create SCM instance
scm = SCM(
    v=["X", "Y"], f={"X": f_x, "Y": f_y}, pu=pu_dist, delta_v_list=delta_v_list, delta_f=delta_f
)

# Sample from the default setting (no intervention)
sampled_data = scm.sample(n=5)
print("Samples (No Intervention):")
for sample in sampled_data:
    print(sample)

# Sample from the intervention setting (X set to 3.0)
sampled_data_intervened = scm.sample(n=5, idx=[1])
print("\nSamples (With X=3.0 Intervention):")
for sample in sampled_data_intervened:
    print(sample)
