import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
import yaml
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from ciflows.datasets.causalmnist import CausalMNIST
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.ncm import GAN_NF_NCM, MLP, CausalGraph, ResNet, log
from ciflows.training import TopKModelSaver


def make_gan_ncm_model(
    latent_dim=32,
    v_dim=128 * 128 * 3,
) -> GAN_NF_NCM:
    V_list = ["style", "digit", "digit-color", "bar-color"]
    default_u_size = int(latent_dim / len(V_list))
    default_v_size = int(v_dim / len(V_list))
    hyperparams = {
        "h-layers": 5,
        "h-size": 128,
        "layer-norm": False,
        "neural-pu": False,
        "single-disc": True,
        "output_shape": (3, 128, 128),
        # "do-var-list": ["digit-color", "bar-color", "bar-color"],
    }

    delta_v_list = [
        {},
        {"bar-color": "conditional"},
        {"bar-color": "conditional"},
        {"digit-color": "hard"},
    ]

    directed_edges = [
        ("digit", "digit-color"),
        ("digit-color", "bar-color"),
        ("digit", "style"),
    ]
    bidirected_edges = [
        ("style", "digit"),
    ]
    cg = CausalGraph(V_list, directed_edges=directed_edges, bidirected_edges=bidirected_edges)

    gan_model = GAN_NF_NCM(
        cg=cg,
        delta_v_list=delta_v_list,
        default_u_size=default_u_size,
        default_v_size=default_v_size,
        hyperparams=hyperparams,
        disc_module=ResNet,
        default_gen_module=ResNet,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    )

    return gan_model


def visualize_images(imgs, title="Generated Images", nrow=4):
    """Visualize a batch of images (assumes shape [B, C, H, W])."""
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    grid = vutils.make_grid(imgs.cpu(), nrow=nrow, normalize=True, scale_each=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def test_ganv2_ncm_generation_and_discrimination(model: GAN_NF_NCM, batch_size=4):
    model.eval()

    # --- Step 1: Generate samples ---
    with torch.no_grad():
        generated_samples = model.sample(n=batch_size, idx=[0])  # index=0 = base distribution
        for img_dict in generated_samples:
            print(f"Generated : {img_dict.keys()}")

        # produce mixture
        generated_samples = model.sample_mixture(generated_samples)

        img = generated_samples[0]  # bar-color is your image node
        print("Generated image tensor shape:", img["X"].shape)  # Should be [B, 3, 128, 128]

        # Visualize
        visualize_images(img, title="Generated Images from GAN_NCM")

    # --- Step 2: Run through discriminator ---
    disc_out = model.get_disc_outputs(generated_samples, index=0)
    print("Discriminator output on generated data:", disc_out)

    # --- Step 3: Create fake "real" sample (random noise shaped like real image) ---
    real_samples = {
        "bar-color": torch.rand(batch_size, 3, 128, 128),  # fake real data
        "digit": torch.randint(0, 10, (batch_size, 1)).float(),
        "digit-color": torch.rand(batch_size, 3),
        "style": torch.rand(batch_size, 5),
    }
    real_disc_out = model.get_disc_outputs(real_samples, index=0)
    print("Discriminator output on real data:", real_disc_out)

    # Optionally compare scores
    print("Avg score on generated:", disc_out.mean().item())
    print("Avg score on 'real':", real_disc_out.mean().item())


def test_gan_ncm_generation_and_discrimination(model: GAN_NF_NCM, batch_size=4):
    model.eval()

    # --- Step 1: Generate samples using sample_mixture ---
    with torch.no_grad():
        mixed_samples = model.sample_mixture(n=batch_size, idx=[0])  # base mixture

        for i, sample in enumerate(mixed_samples):
            print(f"[Sample {i}] Keys: {list(sample.keys())}")

        imgs = torch.stack([d["X"] for d in mixed_samples])  # shape [B, 3, 128, 128]
        print("Mixed output image tensor shape:", imgs.shape)

        # Visualize
        visualize_images(imgs, title="Generated Mixed Samples from GAN_NCM")

    # --- Step 2: Run discriminator on generated samples ---
    disc_out_fake = model.get_disc_outputs(mixed_samples, index=0)
    print("Discriminator output on generated (mixed) data:", disc_out_fake)

    # --- Step 3: Create fake real samples for comparison ---
    real_samples = {
        "bar-color": torch.rand(batch_size, 3, 128, 128),  # image node
        "digit": torch.randint(0, 10, (batch_size, 1)).float(),
        "digit-color": torch.rand(batch_size, 3),
        "style": torch.rand(batch_size, 5),
    }
    disc_out_real = model.get_disc_outputs(real_samples, index=0)
    print("Discriminator output on real (random) data:", disc_out_real)

    # --- Step 4: Print summary stats ---
    print(f"Avg discriminator score on generated (mixed): {disc_out_fake.mean().item():.4f}")
    print(f"Avg discriminator score on fake 'real': {disc_out_real.mean().item():.4f}")


if __name__ == "__main__":
    ncm_gan = make_gan_ncm_model(latent_dim=32)
    print(ncm_gan)
    test_gan_ncm_generation_and_discrimination(ncm_gan)
