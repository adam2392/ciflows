from pathlib import Path
import argparse
import pandas as pd
import shutil
import yaml
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE
from ciflows.datasets.causalmnist import CausalMNIST


def load_config(config_path):
    """Load YAML experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_best_model(checkpoint_dir):
    """Find the best model checkpoint (assumes highest epoch is best)."""
    model_checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
    if (checkpoint_dir / "final_model.pt").exists():
        return checkpoint_dir / "final_model.pt"
    if not model_checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")

    best_model = model_checkpoints[-1]  # Assumes highest epoch is best
    return best_model


def main(exp_path):
    """Load the best VAE model, encode dataset images, and save embeddings."""
    # Ensure experiment directory exists
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory {exp_path} not found!")

    # Load experiment config
    config_path = exp_path / "experiment_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config file {config_path} not found!")

    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Define model parameters
    latent_dim = config["model"]["latent_dim"]
    num_blocks_per_stage = config["model"]["num_blocks_per_stage"]

    # Locate best model checkpoint
    checkpoint_dir = exp_path
    best_model_path = find_best_model(checkpoint_dir)
    print(f"Using best model: {best_model_path}")

    # Load the VAE model
    vae_model = DeepResNetMNISTVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    vae_model.load_state_dict(torch.load(best_model_path, map_location=device)["model_state_dict"])
    vae_model.to(device)
    vae_model.eval()
    print("Model loaded successfully.")

    # Load dataset
    root_dir = Path(config["data"]["root_dir"])
    graph_type = config["data"]["graph_type"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    img_size = config["data"]["img_size"]

    data_dir = root_dir / "CausalMNIST" / graph_type
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = CausalMNIST(
        root=root_dir, graph_type=graph_type, transform=transform, fast_dev_run=False
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Encode dataset
    encodings = []
    meta_df = pd.DataFrame(columns=["digit", "color_digit", "color_bar", "distr_idx"])
    target_tensor = []
    for batch_idx, (images, distr_idx, target, meta_label) in tqdm(enumerate(data_loader), desc="Encoding Images"):
        with torch.no_grad():
            images = images.to(device)
            embedding, _ = vae_model.encoder.encode(images)
        
        meta_df = pd.concat((meta_df, pd.DataFrame(meta_label, columns=["digit", "color_digit", "color_bar", "distr_idx"])), axis=0)
        target_tensor.append(target)
        encodings.append(embedding.cpu())

    target_tensor = torch.cat(target_tensor, dim=0)
    # Save embeddings
    latent_vectors = torch.cat(encodings, dim=0)
    print(target_tensor.shape)
    print(latent_vectors.shape)
    output_path = data_dir / f"{config['exp_name']}_encodings.pt"
    torch.save(latent_vectors, output_path)
    output_path = data_dir / f"{config['exp_name']}_targets.pt"
    torch.save(target_tensor, output_path)
    output_path = data_dir / f"{config['exp_name']}_causal_attrs.csv"
    meta_df.to_csv(output_path)
    print(f"Saved encodings to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode dataset using a trained VAE model.")
    parser.add_argument(
        "--exp_dir",
        required=False,
        default=None,
        type=str,
        help="Path to experiment directory containing experiment.yaml and model checkpoints",
    )
    args = parser.parse_args()

    if args.exp_dir is None:
        exp_dir = (
            Path("/local/eb/adam2392/CausalMNIST/vae_reduction/") / "causalmnist_exp1_betamax01"
        )
    else:
        exp_dir = args.exp_dir

    main(exp_dir)
