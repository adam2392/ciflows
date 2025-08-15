#!/usr/bin/env python
import os
import time
from pathlib import Path

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from ciflows.datasets.causalmnist import CausalMNIST
from ciflows.ncm import GAN_NF_NCM, MLP, CausalGraph
from ciflows.ncm.nn.biggan import BigGANDeconv, BigGANDisc


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_simple_gan(latent_dim, h_layers, h_size, flow_blocks):
    # Single node causal graph
    V_list = ["X"]
    cg = CausalGraph(V_list, directed_edges=[], bidirected_edges=[])
    # one delta‐v setting: unconditional
    delta_v_list = [{}]
    return GAN_NF_NCM(
        cg=cg,
        delta_v_list=delta_v_list,
        default_u_size=latent_dim,
        default_v_size=latent_dim,
        hyperparams={
            "h-layers": h_layers,
            "h-size": h_size,
            "layer-norm": False,
            "neural-pu": False,
            "single-disc": True,
            "K": flow_blocks,
        },
        disc_module=BigGANDisc,
        default_gen_module=BigGANDeconv,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    )


def gradient_penalty(model, real, fake, device, λ=10.0, clamp=1.0):
    b = real.size(0)
    α = torch.rand(b, 1, 1, 1, device=device)
    interp = (α * real + (1 - α) * fake).requires_grad_(True)
    # get critic score
    score, inp = model.get_disc_outputs({"X": interp}, index=0, include_inp=True)
    grads = autograd.grad(
        outputs=score,
        inputs=inp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0].view(b, -1)
    norm = grads.norm(2, dim=1)
    penalty = torch.relu(norm - clamp) ** 2
    return λ * penalty.mean()


def train(model, dataloader, device, opt_D, opt_G, epochs, n_critic, λ_gp, log_dir, save_dir):
    writer = SummaryWriter(log_dir) if log_dir else None
    model.to(device).train()
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for i, (imgs, _, _, _) in enumerate(pbar):
            real = imgs.to(device)
            b = real.size(0)

            # ---- Discriminator ----
            for _ in range(n_critic):
                opt_D.zero_grad()
                # real score
                real_score = model.get_disc_outputs({"X": real}, index=0)
                # fake images
                fake = model.sample_mixture(n=b, idx=[0])[0]["X"].to(device)
                fake_score = model.get_disc_outputs({"X": fake}, index=0)

                # WGAN loss
                loss_D = fake_score.mean() - real_score.mean()
                # gradient penalty
                gp = gradient_penalty(model, real, fake, device, λ=λ_gp)
                (loss_D + gp).backward()
                opt_D.step()

            # ---- Generator ----
            opt_G.zero_grad()
            fake = model.sample_mixture(n=b, idx=[0])[0]["X"].to(device)
            g_score = model.get_disc_outputs({"X": fake}, index=0)
            loss_G = -g_score.mean()
            loss_G.backward()
            opt_G.step()

            if i % 100 == 0:
                pbar.set_postfix(D=loss_D.item(), G=loss_G.item())

        # Logging & checkpointing
        if writer:
            writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch)
            writer.add_scalar("Loss/Generator", loss_G.item(), epoch)
            # sample grid
            with torch.no_grad():
                samples = model.sample_mixture(n=16, idx=[0])[0]["X"]
                grid = utils.make_grid(samples.cpu(), nrow=4, normalize=True)
                writer.add_image("Samples", grid, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), Path(save_dir) / f"gan_epoch_{epoch}.pt")

    if writer:
        writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train simple GAN on raw CausalMNIST")
    parser.add_argument(
        "--config", type=str, default="./ncm_experiment.yml", help="YAML config path"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="Where to save models"
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose(
        [
            transforms.Resize(cfg["data"]["img_size"]),
            transforms.ToTensor(),
        ]
    )
    dataset = CausalMNIST(
        root=cfg["data"]["model_dir"], graph_type=cfg["data"]["graph_type"], transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )

    # Model
    gan = make_simple_gan(
        latent_dim=cfg["model"]["latent_dim"],
        h_layers=cfg["model"].get("h_layers", 2),
        h_size=cfg["model"].get("h_size", 128),
        flow_blocks=cfg["model"].get("K", 8),
    )

    # Optimizers
    opt_D = torch.optim.Adam(
        gan.discriminator_parameters(), lr=cfg["optimizer"]["disc_lr"], betas=(0.0, 0.9)
    )
    opt_G = torch.optim.Adam(
        gan.generator_parameters(), lr=cfg["optimizer"]["gen_lr"], betas=(0.0, 0.9)
    )

    # Train
    train(
        model=gan,
        dataloader=loader,
        device=device,
        opt_D=opt_D,
        opt_G=opt_G,
        epochs=cfg["training"]["max_epochs"],
        n_critic=cfg["training"].get("n_critic", 5),
        λ_gp=cfg["training"].get("lambda_gp", 10.0),
        log_dir=args.log_dir,
        save_dir=args.save_dir,
    )
