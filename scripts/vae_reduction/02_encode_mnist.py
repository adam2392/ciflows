from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE
from ciflows.datasets.causalmnist import CausalDigitBarMNIST


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
    debug = False
    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")
        root = Path("/local/eb/adam2392/")

    scm_model = "CausalDigitBarMNIST"

    data_dir = root / scm_model / graph_type

    model_dir = "mnist_cyclicbetal1loss_vaeresnetreduction_batch128_gradaccum_latentdim48_img32_v4"
    model_fname = "model_epoch_11190.pt"
    vae_model_fpath = root / "CausalMNIST" / "vae_reduction" / model_dir.split(".")[0] / model_fname
    batch_size = 512
    num_workers = 4

    # vae_model = VAE()  # Replace with loading logic
    latent_dim = 48
    num_blocks_per_stage = 3
    vae_model = DeepResNetMNISTVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    vae_model.load_state_dict(torch.load(vae_model_fpath, map_location=device)["model_state_dict"])
    vae_model.eval()

    # Define preprocessing for images
    image_size = 32  # Adjust based on model input
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize images to 128x128
            transforms.CenterCrop(image_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )
    vae_model.to(device)

    data_to_encode = data_dir / "chain-imgs-train.pt"
    causal_mnist_dataset = CausalDigitBarMNIST(
        root=root,
        graph_type=graph_type,
        transform=transform,
        # img_size=img_size,
        fast_dev_run=False,  # Set to True for debugging
    )
    data_loader = DataLoader(
        dataset=causal_mnist_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        # shuffle=True,  # Shuffle data during training
        num_workers=num_workers,
        pin_memory=True,  # Enable if using a GPU
        persistent_workers=True,
    )
    data_tensor = torch.load(data_to_encode)

    n_images = len(data_tensor)
    encodings = []
    img_idx = 0

    for batch_idx, (
        images,
        distr_idx,
        targets,
        meta_labels,
    ) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            images = images.to(device)
            embedding, _ = vae_model.encoder.encode(images)
        # encodings.append(latent_vector.cpu())
        encodings.append(embedding.cpu())

    latent_vectors = torch.cat(encodings, dim=0)
    print(latent_vectors.shape)

    # Save the tensor
    # v2 nonorm encodings = sample from latent, rather than the mean
    # output_path = f"{directory.name}_cyclicbeta_noimgaug_encodings.pt"
    output_path = f"chain_alldata_encodings.pt"
    torch.save(latent_vectors, data_dir / output_path)
    print(f"Saved encodings to: {output_path}")
    print("Encoding process completed.")
