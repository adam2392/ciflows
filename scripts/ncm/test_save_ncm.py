import os
import torch
import torch.nn as nn
import normflows as nf  # Ensure normflows is installed
from ciflows.ncm.cg import CausalGraph  # Adjust the import based on your project structure
from ciflows.ncm import GAN_NCM, GAN_NF_NCM
from ciflows.ncm.utils import expand_do

if __name__ == "__main__":
    # Create a dummy causal graph.
    # Here the keys represent variable names and the values their parent lists.
    cg = CausalGraph({"A": [], "B": ["A"], "C": ["A", "B"]})

    # Define sample hyperparameters.
    hyperparams = {
        "h-layers": 2,
        "h-size": 64,
        "layer-norm": True,
        "neural-pu": False,
        "single-disc": True,
        "do-var-list": ["A", "B"],  # two intervention settings
        "K": 8  # number of flows (use a small number for testing)
    }

    # Instantiate the GAN_NF_NCM model.
    gan_nf_ncm = GAN_NF_NCM(cg, hyperparams=hyperparams)
    gan_nf_ncm.eval()  # Set to evaluation mode for testing

    # Generate samples.
    # sample() returns a list of dictionaries (one per intervention setup).
    samples = gan_nf_ncm.sample(n=5)
    print("Generated Samples:")
    for i, sample in enumerate(samples):
        print(f"Intervention {i}:")
        for key, tensor in sample.items():
            print(f"  {key}: shape {tensor.shape}")

    # Test the discriminator output.
    # For single-discriminator mode, get_disc_outputs expects a dictionary from one intervention.
    try:
        disc_output = gan_nf_ncm.get_disc_outputs(samples[0], index=0)
        print("\nDiscriminator Output:")
        print(disc_output)
        print("Discriminator Output shape:", disc_output.shape)
    except Exception as e:
        print("Error during discriminator evaluation:", e)

    # Test the mixing function by applying it to one set of generated samples.
    try:
        mixed_output = gan_nf_ncm.apply_mixing_function(samples[0])
        print("\nMixed Output (after applying normalizing flows):")
        print(mixed_output)
        print("Mixed Output shape:", mixed_output.shape)
    except Exception as e:
        print("Error during mixing function evaluation:", e)

    # Test sample_mixture() method.
    try:
        mixture_samples = gan_nf_ncm.sample_mixture(n=5)
        print("\nSample Mixture Outputs:")
        for i, mix in enumerate(mixture_samples):
            print(f"Intervention {i} mixed sample shape: {mix['X'].shape}")
    except Exception as e:
        print("Error during sample mixture evaluation:", e)

    # Save a checkpoint to verify that saving works as expected.
    checkpoint_path = "./checkpoint_test.pth"
    torch.save(
        {
            "epoch": 1,
            "ncm_model_state_dict": gan_nf_ncm.state_dict(),
            # For this test we do not have actual optimizers, so we use None
            "optimizer_generator_state_dict": None,
            "optimizer_critic_state_dict": None,
        },
        checkpoint_path,
    )
    print("\nCheckpoint saved to:", os.path.abspath(checkpoint_path))

    # Define a function to check if two models have identical parameters
    def compare_models(model1, model2):
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2 or not torch.equal(param1, param2):
                print(f"Mismatch found in parameter: {name1}")
                return False
        return True
    # Instantiate a new model and load the saved checkpoint
    reloaded_gan_nf_ncm = GAN_NF_NCM(cg, hyperparams=hyperparams)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
    reloaded_gan_nf_ncm.load_state_dict(checkpoint["ncm_model_state_dict"])

    # Compare parameters
    if compare_models(gan_nf_ncm, reloaded_gan_nf_ncm):
        print("Reloaded model matches the original.")
    else:
        print("Reloaded model does NOT match the original.")
