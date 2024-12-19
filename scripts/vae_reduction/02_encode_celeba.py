import os
from pathlib import Path

import PIL
import torch
from torchvision import transforms

from ciflows.reduction.resnetvae import DeepResNetVAE
from ciflows.reduction.vae import VAE


# Encode images in a directory
def encode_images_in_directory(
    directory, model: DeepResNetVAE, transform: transforms.Compose, device="cpu"
):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith(".jpg")])
    encodings = []

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(directory, img_file)
        image = PIL.Image.open(img_path).convert("RGB")  # Convert to 3-channel RGB
        img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            mu, log_var = model.encoder.encode(img_tensor)
            latent_vector = model.reparameterize(mu, log_var)
            if idx == 0:
                print(f"Latent vector shape: {latent_vector.shape}")
        encodings.append(latent_vector.cpu())
    return torch.stack(encodings)


if __name__ == "__main__":
    # Main encoding process
    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    graph_type = "chain"

    root = Path("/Users/adam2392/pytorch_data/CausalCelebA/")
    data_dir = root / graph_type / "dim128"
    directories = [data_dir / "obs", data_dir / "int_hair_0", data_dir / "int_hair_1"]
    latent_vectors_per_directory = {}

    model_fname = "model_epoch_100.pt"
    model_fname = "celeba_vaeresnetreduction_batch512_latentdim48_img128_v1.pt"
    vae_model_fpath = root / "vae_reduction" / model_fname.split(".")[0] / model_fname

    # vae_model = VAE()  # Replace with loading logic
    latent_dim = 48
    num_blocks_per_stage = 3
    vae_model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    vae_model.load_state_dict(torch.load(vae_model_fpath, map_location=device))
    vae_model.eval()

    # Define preprocessing for images
    image_size = 128  # Adjust based on model input
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize images to 128x128
            transforms.CenterCrop(image_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    vae_model.to(device)

    for directory in directories:
        print(f"Processing directory: {directory}")
        latent_vectors = encode_images_in_directory(directory, vae_model, transform, device)
        latent_vectors_per_directory[directory] = latent_vectors

        # Save the tensor
        output_path = f"{directory.name}_encodings.pt"
        torch.save(latent_vectors, directory / output_path)
        print(f"Saved encodings to: {output_path}")
        print("Encoding process completed.")
