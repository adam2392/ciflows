import torch as T
import torch
from ciflows.ncm import CausalGraph, GAN_NCM  # Assuming the causal graph class is in this module
from ciflows.ncm.mlp import MLP


def make_gan_ncm_model(
    latent_dim=32,
):
    V_list = ["style", "digit", "digit-color", "bar-color"]
    default_u_size = int(latent_dim / len(V_list))
    default_v_size = int(latent_dim / len(V_list))
    hyperparams = {
        "h-layers": 2,
        "h-size": 128,
        "layer-norm": False,
        "neural-pu": False,
        "single-disc": True,
        "do-var-list": ["digit-color", "bar-color", "bar-color"],
    }

    directed_edges = [
        ("digit", "digit-color"),
        ("digit-color", "bar-color"),
        ("digit", "style"),
    ]
    bidirected_edges = [
        ("style", "digit"),
    ]
    cg = CausalGraph(V_list, directed_edges=directed_edges, bidirected_edges=bidirected_edges)

    gan_model = GAN_NCM(
        cg=cg,
        default_u_size=default_u_size,
        default_v_size=default_v_size,
        # f=None,
        hyperparams=hyperparams,
        disc_module=MLP,
        default_gen_module=MLP,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    )

    return gan_model


def test_gan_ncm():
    # Create the GAN_NCM model
    gan_model = make_gan_ncm_model(latent_dim=32)

    # Ensure model is on the correct device
    device = T.device("cuda" if T.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    gan_model.to(device)

    # Generate samples
    batch_size = 10  # Number of samples to test
    samples = gan_model.sample(n=batch_size)

    # Print a sample result for verification
    print("Generated Samples:")
    for var, val in samples[0].items():  # Printing first set of samples
        print(f"{var}: {val.shape} {val.device}")

    # Ensure forward pass works
    output = gan_model.forward(n=batch_size)
    print("\nForward Pass Output:")
    for var, val in output[0].items():
        print(f"{var}: {val.shape} {val.device}")

    # Verify discriminator outputs
    disc_output = gan_model.get_disc_outputs(output[0], index=0)
    print("\nDiscriminator Output Shape:", disc_output.shape)
    print(disc_output)

    # Check for NaNs (to confirm numerical stability)
    assert not any(T.isnan(v).any() for v in output[0].values()), "NaN detected in forward pass"
    assert not T.isnan(disc_output).any(), "NaN detected in discriminator output"

    print("\nGAN_NCM Model Test Passed Successfully!")


# Run the test
test_gan_ncm()
