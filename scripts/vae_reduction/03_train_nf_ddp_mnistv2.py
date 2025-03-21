import shutil
import os
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter

from ciflows.datasets.causalmnist import CausalMNISTEmbedding
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.reduction import make_mnist_nf_model
from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE
from ciflows.training import TopKModelSaver, delete_old_checkpoints


def load_experiment_config(config_path):
    """Load experiment configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def softclip(tensor, min):
    """Clips tensor values at min in a soft way (from 'Handful of Trials')."""
    return min + F.softplus(tensor - min)


def configure_optimizers(model, learning_rate, betas, weight_decay=0.0):
    """Configures AdamW optimizer with weight decay handling."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    return optimizer


def data_loader(root_dir, dataset, graph_type, num_workers, batch_size, img_size, debug=False):
    """Loads CausalMNIST dataset with stratified sampling."""
    causal_dataset = CausalMNISTEmbedding(root=root_dir, graph_type=graph_type, fast_dev_run=False)

    if debug:
        print("Debugging data loader...")
        print(causal_dataset[0])

    distr_labels = [x[1] for x in causal_dataset]
    # print("The distribution index labels: ", distr_labels)
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    return DataLoader(
        dataset=causal_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


if __name__ == "__main__":
    root = Path("/local/eb/adam2392/CausalMNIST/")

    # Load NF Configuration
    nf_config_path = "/home/adam2392/projects/ciflows/scripts/vae_reduction/nf_experiment.yml"
    nf_config = load_experiment_config(nf_config_path)
    load_from_checkpoint = nf_config.get("load_from_checkpoint", False)

    # Load VAE Configuration
    vae_checkpoint_dir = (
        Path(nf_config["vae"]["checkpoint_dir"]) / nf_config["vae"]["experiment_name"]
    )
    vae_config_path = vae_checkpoint_dir / "experiment_config.yaml"
    vae_config = load_experiment_config(vae_config_path)

    # Extract experiment settings
    exp_name = nf_config["exp_name"]
    model_root = root / "nf_reduction" / exp_name
    model_root.mkdir(parents=True, exist_ok=True)

    # tensorboard writer...
    writer = SummaryWriter(log_dir=str(model_root / "logs"))

    max_epochs = nf_config["training"]["max_epochs"]

    # Save a copy of the NF config
    config_save_path = model_root / "experiment_config.yaml"
    shutil.copy(nf_config_path, config_save_path)

    # Device Setup
    world_size = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype
    ]

    # DDP Setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        gradient_accumulation_steps = nf_config["training"]["grad_accum_steps"] // ddp_world_size
    else:
        master_process = True
        gradient_accumulation_steps = nf_config["training"]["grad_accum_steps"]

    # Load Pretrained VAE Model from VAE Config
    vae_checkpoint_file = vae_checkpoint_dir / nf_config["vae"]["vae_checkpoint_fname"]
    vae_model = DeepResNetMNISTVAE(
        latent_dim=vae_config["model"]["latent_dim"],
        num_blocks_per_stage=vae_config["model"]["num_blocks_per_stage"],
    )
    print(f"Loading from {vae_checkpoint_file}")
    vae_model, _ = load_model(vae_model, vae_checkpoint_file, device)
    vae_model.to(device).eval()

    # Initialize NF Model
    nf_model = make_mnist_nf_model(
        K=nf_config["model"]["num_flows"],
    ).to(device)

    # Load NF Checkpoint if Available
    nf_checkpoint_dir = root / "nf_reduction" / nf_config["exp_name"]
    nf_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    start_epoch = 1
    if load_from_checkpoint:
        nf_checkpoint_file = nf_checkpoint_dir / nf_config["training"]["checkpoint_file"]
        nf_model, start_epoch = load_model(nf_model, nf_checkpoint_file, device)

    # Optimizer and Scheduler
    optimizer = configure_optimizers(
        nf_model,
        learning_rate=nf_config["optimizer"]["lr"],
        betas=tuple(nf_config["optimizer"]["betas"]),
        weight_decay=nf_config["optimizer"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=float(nf_config["scheduler"]["lr_min"]),
    )

    # Wrap NF Model with DDP if needed
    if ddp:
        nf_model = DDP(nf_model, device_ids=[ddp_local_rank])

    # Model Saving Utility
    top_k_saver = TopKModelSaver(nf_checkpoint_dir, k=5)

    # DataLoader
    train_loader = data_loader(
        root_dir=nf_config["data"]["root_dir"],
        dataset=nf_config["data"]["graph_type"],
        graph_type=nf_config["data"]["graph_type"],
        num_workers=nf_config["data"]["num_workers"],
        batch_size=nf_config["data"]["batch_size"],
        img_size=nf_config["data"]["img_size"],
    )

    if master_process:
        total_params = sum(p.numel() for p in nf_model.parameters() if p.requires_grad)
        print(f"Total trainable parameters in NF model: {total_params:,}")

    # Training Loop
    for epoch in tqdm(range(start_epoch, max_epochs), desc="Epochs"):
        nf_model.train()
        total_loss = 0.0
        train_iterator = iter(train_loader)

        for step, batch in enumerate(train_iterator):
            latent_embeddings, distr_idx, targets, meta_labels = batch
            latent_embeddings = latent_embeddings.to(device)

            # print(f"Inside training: ", distr_idx)
            # Encode images with VAE
            # with torch.no_grad():
            #     latent_embeddings = vae_model.encode(images)

            optimizer.zero_grad()
            if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                loss = nf_model.module.forward_kld(
                    latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                )
            else:
                loss = nf_model.forward_kld(
                    latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                nf_model.parameters(), max_norm=nf_config["training"]["grad_clip"]
            )
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if master_process:
            print(f"Epoch [{epoch}/{nf_config['training']['max_epochs']}] - Loss: {total_loss:.4f}")

            # Save Checkpoints
            if epoch % nf_config["training"]["save_every"] == 0:
                top_k_saver.save_model(nf_model, optimizer, epoch, total_loss)
                delete_old_checkpoints(nf_checkpoint_dir, keep_top_k=5)
                print(f"Checkpoint saved at {nf_checkpoint_dir}")

            # Generate and Save Samples
            nf_model.eval()
            with torch.no_grad():
                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    sample_embeddings, _ = nf_model.module.sample(8, distr_idx=0)
                else:
                    sample_embeddings, _ = nf_model.sample(8, distr_idx=0)
                reconstructed_images = vae_model.decode(sample_embeddings).reshape(
                    -1, 3, nf_config["data"]["img_size"], nf_config["data"]["img_size"]
                )
                save_image(
                    reconstructed_images.cpu(),
                    nf_checkpoint_dir / f"epoch_{epoch}_samples.png",
                    nrow=4,
                    normalize=True,
                )

            top_k_saver.save_model(nf_model, optimizer, epoch, total_loss)
            delete_old_checkpoints(nf_checkpoint_dir, keep_top_k=5)

            # write to tensorboard
            if epoch % nf_config["training"]["n_log_steps"] == 0:
                # nf_model.eval()
                with torch.no_grad():
                    # Compute log-probability on a fixed batch from the training and get the log probability
                    if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                        log_prob = nf_model.module.log_prob(
                            latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                        )
                    else:
                        log_prob = nf_model.log_prob(
                            latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                        )
                    mean_log_prob = log_prob.mean().item()
                    writer.add_scalar("LogProb/Mean", mean_log_prob, epoch)

                    # Generate new images from samples
                    if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                        sample_latents, _ = nf_model.module.sample(8, distr_idx=0)
                    else:
                        sample_latents, _ = nf_model.sample(8, distr_idx=0)
                    generated_images = vae_model.decode(sample_latents).reshape(
                        -1, 3, nf_config["data"]["img_size"], nf_config["data"]["img_size"]
                    )
                    writer.add_images(
                        "Generated_Images - 0", generated_images, epoch, dataformats="NCHW"
                    )

                    # Generate new images from samples
                    if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                        sample_latents, _ = nf_model.module.sample(8, distr_idx=1)
                    else:
                        sample_latents, _ = nf_model.sample(8, distr_idx=1)
                    generated_images = vae_model.decode(sample_latents).reshape(
                        -1, 3, nf_config["data"]["img_size"], nf_config["data"]["img_size"]
                    )
                    writer.add_images(
                        "Generated_Images - 1", generated_images, epoch, dataformats="NCHW"
                    )

                    # Generate new images from samples
                    if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                        sample_latents, _ = nf_model.module.sample(8, distr_idx=2)
                    else:
                        sample_latents, _ = nf_model.sample(8, distr_idx=2)
                    generated_images = vae_model.decode(sample_latents).reshape(
                        -1, 3, nf_config["data"]["img_size"], nf_config["data"]["img_size"]
                    )
                    writer.add_images(
                        "Generated_Images - 2", generated_images, epoch, dataformats="NCHW"
                    )

    # Save Final Model
    torch.save(nf_model.state_dict(), nf_checkpoint_file)
    print(f"Final model saved at {nf_checkpoint_file}")
