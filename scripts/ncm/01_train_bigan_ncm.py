import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from ciflows.datasets.causalmnist import CausalMNIST, CausalMNISTEmbedding
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.ncm.bigan import Decoder, Encoder, JointDiscriminator


def train_bigan(
    output_dir,
    epochs=100,
    batch_size=64,
    z_dim=32,
    lr=1e-4,
    log_every=10,
    val_split=0.1,
    save_top_k=5,
    multi_gpu=False,
    use_scheduler=False,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), lambda x: x.expand(3, -1, -1)]
    )

    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    val_len = int(val_split * len(dataset))
    train_len = len(dataset) - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    encoder = Encoder(z_dim=z_dim)
    decoder = Decoder(z_dim=z_dim)
    discriminator = JointDiscriminator(z_dim=z_dim)

    if multi_gpu and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        discriminator = nn.DataParallel(discriminator)

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    gen_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)

    if use_scheduler:
        scheduler = ReduceLROnPlateau(gen_opt, mode="min", factor=0.5, patience=5)

    best_models = []  # list of tuples: (loss, path)

    for epoch in range(1, epochs + 1):
        encoder.train(), decoder.train(), discriminator.train()
        total_d_loss, total_g_loss = 0, 0

        for x, _ in train_loader:
            x = x.to(device)
            batch_size = x.size(0)

            # Encode
            z_enc = encoder(x)
            x_fake = decoder(z_enc)

            # Random latent
            z_rand = torch.randn(batch_size, z_dim).to(device)
            x_rand = decoder(z_rand).detach()

            # Discriminator step
            dis_opt.zero_grad()
            real_score = discriminator(x, z_enc.detach())
            fake_score = discriminator(x_rand, z_rand)
            d_loss = -torch.mean(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8))
            d_loss.backward()
            dis_opt.step()

            # Generator step
            gen_opt.zero_grad()
            fake_score = discriminator(x_rand, z_rand)
            g_loss = -torch.mean(torch.log(fake_score + 1e-8))
            g_loss.backward()
            gen_opt.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)

        # Validation (recon loss)
        encoder.eval(), decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                z_val = encoder(x_val)
                x_recon = decoder(z_val)
                val_loss += F.mse_loss(x_recon, x_val, reduction="sum").item()
        val_loss /= len(val_loader.dataset)

        if use_scheduler:
            scheduler.step(val_loss)

        # Log
        if epoch % log_every == 0:
            print(
                f"Epoch {epoch}: D Loss = {avg_d_loss:.4f}, G Loss = {avg_g_loss:.4f}, Val Loss = {val_loss:.4f}"
            )
            with torch.no_grad():
                z_sample = torch.randn(64, z_dim).to(device)
                x_sample = decoder(z_sample)
                utils.save_image(
                    x_sample,
                    os.path.join(sample_dir, f"sample_epoch_{epoch}.png"),
                    nrow=8,
                    normalize=True,
                )

        # Save top-k models
        model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            },
            model_path,
        )

        best_models.append((val_loss, model_path))
        best_models = sorted(best_models, key=lambda x: x[0])[:save_top_k]

        for _, path in best_models[save_top_k:]:
            if os.path.exists(path):
                os.remove(path)


def data_loader(root_dir, dataset, graph_type, num_workers, batch_size, img_size, debug=False):
    """Loads CausalMNIST dataset with stratified sampling.

    Loads the original CausalMNIST (32, 32, 3) datasets.
    """
    causal_dataset = CausalMNISTEmbedding(root=root_dir, graph_type=graph_type, fast_dev_run=False)

    if debug:
        print("Debugging data loader...")
        print(causal_dataset[0])

    distr_labels = [x[1] for x in causal_dataset]
    # print("The distribution index labels: ", distr_labels)
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    # Initialize the dataset (replace with your desired dataset settings)
    # Define image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = CausalMNIST(
        root=root_dir,
        graph_type=graph_type,
        transform=image_transform,
    )

    return DataLoader(
        shuffle=False,
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


if __name__ == "__main__":
    # Configuration
    config = {
        "output_dir": "./output/bigan_causal",
        "epochs": 5,
        "batch_size": 128,
        "z_dim": 32,
        "lr": 1e-4,
        "log_every": 10,
        "val_split": 0.1,
        "save_top_k": 5,
        "multi_gpu": torch.cuda.device_count() > 1,
        "use_scheduler": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "img_size": 32,
        "graph_type": "chain",  # or whichever causal structure you want to test
        "debug": False,
    }
    root = Path("/local/eb/adam2392/")

    # Pre-warm CUDA to avoid cuBLAS warning
    if torch.cuda.is_available():
        _ = torch.tensor([0.0], device="cuda")
        torch.cuda.reset_peak_memory_stats()

    print("=================== Training Configuration ===================")
    print(f"Device: {config['device']}")
    print(f"Multi-GPU: {config['multi_gpu']} ({torch.cuda.device_count()} GPUs detected)")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Latent Dimension: {config['z_dim']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Validation Split: {config['val_split']}")
    print(f"Dataset Root: {root}")
    print(f"Causal Graph Type: {config['graph_type']}")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Initial CUDA Memory Usage: {mem:.2f} MB (Peak: {peak_mem:.2f} MB)")

    print("=============================================================")

    # Load training data
    train_loader = data_loader(
        root_dir=root,
        dataset="CausalMNIST",
        graph_type=config["graph_type"],
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        debug=config["debug"],
    )

    # Start training
    train_bigan(
        output_dir=config["output_dir"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        z_dim=config["z_dim"],
        lr=config["lr"],
        log_every=config["log_every"],
        val_split=config["val_split"],
        save_top_k=config["save_top_k"],
        multi_gpu=config["multi_gpu"],
        use_scheduler=config["use_scheduler"],
        device=config["device"],
    )
