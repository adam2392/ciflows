import torch
import yaml
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ciflows.datasets.causalmnist import CausalMNIST, CausalMNISTEmbedding
from ciflows.eval import load_model
from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE


# Load experiment configuration from YAML
def load_experiment_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Function to visualize decoded embeddings alongside original images (same indices)
def visualize_decoded_embeddings(vae_checkpoint, dataset_root, save_path, num_samples=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE configuration
    vae_config_path = Path(vae_checkpoint).parent / "experiment_config.yaml"
    vae_config = load_experiment_config(vae_config_path)

    # Load the trained VAE model
    vae_model = DeepResNetMNISTVAE(
        latent_dim=vae_config["model"]["latent_dim"],
        num_blocks_per_stage=vae_config["model"]["num_blocks_per_stage"],
    )
    vae_model, _ = load_model(vae_model, vae_checkpoint, device)
    vae_model.to(device).eval()

    # Load both datasets
    # Define image transformations
    img_size = 32
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    original_dataset = CausalMNIST(
        root=dataset_root, transform=image_transform, graph_type="chain", fast_dev_run=False
    )
    embedding_dataset = CausalMNISTEmbedding(
        root=dataset_root, graph_type="chain", fast_dev_run=False
    )

    # Select random indices (ensuring we fetch the same ones)
    selected_indices = np.random.choice(len(original_dataset), num_samples, replace=False)

    # Create subset loaders for the selected indices
    original_subset = Subset(original_dataset, selected_indices)
    embedding_subset = Subset(embedding_dataset, selected_indices)

    original_loader = DataLoader(original_subset, batch_size=num_samples, shuffle=False)
    embedding_loader = DataLoader(embedding_subset, batch_size=num_samples, shuffle=False)

    # Fetch the corresponding original images and latent embeddings
    original_images, distr_idx, target, meta_label = next(
        iter(original_loader)
    )  # Get images from original dataset
    latent_embeddings, _, _, _ = next(iter(embedding_loader))  # Get embeddings

    original_images = original_images.to(device)
    latent_embeddings = latent_embeddings.to(device)

    # Decode embeddings into images
    with torch.no_grad():
        reconstructed_images = vae_model.decode(latent_embeddings)
        reconstructed_images = reconstructed_images.view(-1, 3, 32, 32)  # Adjust shape if needed

    # Save original and decoded images side by side
    comparison_images = torch.cat(
        [original_images, reconstructed_images], dim=0
    )  # Stack originals and reconstructions

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_image(
        comparison_images.cpu(),
        save_path / "original_vs_decoded.png",
        nrow=num_samples,
        normalize=True,
    )

    print(f"Original and decoded images saved at {save_path / 'original_vs_decoded.png'}")


# Example usage
if __name__ == "__main__":
    dataset_root = Path("/local/eb/adam2392/")  # Update with actual dataset location
    vae_checkpoint_path = (
        dataset_root
        / "CausalMNIST"
        / "vae_reduction"
        / "causalmnist_exp2_betamax005"
        / "final_model.pt"
    )  # Update with actual path
    output_dir = "visualizations/"

    visualize_decoded_embeddings(vae_checkpoint_path, dataset_root, output_dir)
