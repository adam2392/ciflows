import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
import shutil
import yaml
import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from albumentations import CoarseDropout, Compose
from albumentations.pytorch import ToTensorV2

from ciflows.datasets.causalmnist import CausalMNIST
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE
from ciflows.training import TopKModelSaver, delete_old_checkpoints

# -------------------
# Helper functions
# -------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def softclip(tensor, min_val):
    """Clips the tensor values at the minimum value min_val in a soft way."""
    return min_val + F.softplus(tensor - min_val)

def configure_optimizers(model, learning_rate, betas, weight_decay=0.0):
    # Collect parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # Create groups based on dimensionality for weight decay
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    print("Using fused AdamW")
    return optimizer

def data_loader(config):
    # Load data-related parameters from config
    data_cfg = config["data"]
    img_size = data_cfg["img_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    val_split = data_cfg.get("val_split", 0.2)
    graph_type = data_cfg["graph_type"]
    root_dir = Path(data_cfg["root_dir"])

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    # Initialize the dataset (replace with your desired dataset settings)
    dataset = CausalMNIST(
        root=root_dir,
        graph_type=graph_type,
        transform=image_transform,
        fast_dev_run=config.get("debug", False),
    )

    # Split the dataset into train and validation sets
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Create stratified samplers (ensuring batch size meets requirements)
    def create_sampler(ds):
        distr_labels = [x[1] for x in ds]
        unique_distrs = len(np.unique(distr_labels))
        if batch_size < unique_distrs:
            raise ValueError(f"Batch size must be at least {unique_distrs} for stratified sampling.")
        return StratifiedSampler(distr_labels, batch_size)

    train_sampler = create_sampler(train_dataset)
    val_sampler = create_sampler(val_dataset)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
    )
    return train_loader, val_loader

def gaussian_nll(recon_x, log_sigma, x):
    first_term = 0.5 * torch.pow((x - recon_x) / log_sigma.exp(), 2)
    return first_term + log_sigma + 0.5 * np.log(2 * np.pi)

def loss_function(recon_x, x, mu, log_var, log_sigma_x, capacity=0.0, beta=0.00025):
    rec_loss = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = KLD if capacity == 0 else torch.max(KLD - capacity, torch.tensor(0.0).cuda())
    return rec_loss + beta * kl_loss

def cyclic_beta(step, cycle_length, beta_min=0.00025, beta_max=0.1):
    cycle_position = step % cycle_length
    fraction = cycle_position / cycle_length
    return beta_min + (beta_max - beta_min) * (1 - math.cos(math.pi * fraction)) / 2

def get_model_attribute(model, attr):
    return getattr(model.module if isinstance(model, DDP) else model, attr)

# -------------------
# Main training loop
# -------------------

def main(config_fpath):
    config = load_config(config_fpath)

    # Load general configuration values
    debug = config.get("debug", False)
    compile_model = config.get("compile", False)
    load_from_checkpoint = config.get("load_from_checkpoint", True)
    
    # System settings
    sys_cfg = config["system"]
    seed = sys_cfg.get("seed", 1234)
    if torch.cuda.is_available():
        device = torch.device(sys_cfg.get("device", "cuda"))
        accelerator = sys_cfg.get("accelerator", "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Set dtype (example, you may refine this based on config)
    dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    # DDP settings
    ddp = int(os.environ.get("RANK", -1)) != -1
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank

    if master_process:
        print(f"Using device: {device} with {world_size} GPUs")
        print("PyTorch version:", torch.__version__)
        print(f"Using accelerator: {accelerator}")
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
        print("NCCL backend available:", torch.distributed.is_nccl_available())

    if master_process:
        print(f"Training on {world_size} device(s) with seed {seed + seed_offset}")

    # Set seed
    np.random.seed(seed + seed_offset)
    pl.seed_everything(seed + seed_offset, workers=True)

    # Data settings
    train_loader, val_loader = data_loader(config)

    # Model settings
    model_cfg = config["model"]
    latent_dim = model_cfg["latent_dim"]
    num_blocks_per_stage = model_cfg["num_blocks_per_stage"]
    model = DeepResNetMNISTVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    model = model.to(ptdtype).to(device)

    # Optimizer settings
    opt_cfg = config["optimizer"]
    optimizer = configure_optimizers(model, opt_cfg["lr"], tuple(opt_cfg["betas"]), opt_cfg["weight_decay"])

    if compile_model:
        model = torch.compile(model)

    # Load from checkpoint if required
    start_epoch = 1
    root = Path(config["data"]["root_dir"])
    checkpoint_dir = root / "CausalMNIST" / "vae_reduction" / config['exp_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if load_from_checkpoint:
        checkpoint_fname = config.get("checkpoint_fname", "latest.pt")
        ckpt_path = checkpoint_dir / checkpoint_fname
        if ckpt_path.exists():
            model, start_epoch = load_model(model, ckpt_path, device, optimizer=optimizer)
            if ddp:
                dist.barrier()
            print(f"Loaded checkpoint from epoch {start_epoch}")
        
        # XXX: check if there is an existing checkpoint here

    # Save a copy of the config file in the checkpoint directory
    config_save_path = checkpoint_dir / "experiment_config.yaml"
    shutil.copy(config_fpath, config_save_path)
    print(f"Saved experiment config to {config_save_path}")

    # Wrap model for distributed training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Scheduler settings
    sched_cfg = config["scheduler"]
    training_cfg = config["training"]
    max_epochs = training_cfg["max_epochs"]
    print(max_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(sched_cfg["lr_min"]))

    # Set up additional training parameters
    grad_accum_steps = training_cfg["grad_accum_steps"]
    check_samples_every_n_epoch = training_cfg["check_samples_every_n_epoch"]
    cycle_length = len(train_loader) * training_cfg['cycle_length']  # Adjust as needed

    top_k_saver = TopKModelSaver(checkpoint_dir, k=5)

    # For mixed precision (if desired)
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))
    ctx = nullcontext() if device == "cpu" else torch.autocast(device_type=accelerator, dtype=ptdtype)
    ctx = nullcontext()  # Disable autocast if not needed

    # Training loop
    t0 = time.time()
    for epoch in tqdm(range(start_epoch, max_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        train_iterator = iter(train_loader)
        global_step = epoch * len(train_loader)
        # Compute cyclic beta from config parameters
        beta_cfg = config["beta"]
        beta = cyclic_beta(global_step, cycle_length, beta_cfg["beta_min"], beta_cfg["beta_max"])
        if master_process:
            print(f"Epoch {epoch}: beta={beta:.6f}")

        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            images, distr_idx, targets, meta_labels = batch
            images = images.to(device, dtype=ptdtype)
            target_images = images  # In this case, target images are the same

            optimizer.zero_grad()
            with ctx:
                reconstructed, latent_mu, latent_logvar = model(images)
                latent_logvar = torch.clamp_(latent_logvar, -10, 10)
                # log_sigma_x = get_model_attribute(model, "log_sigma_x")
                # log_sigma_x = softclip(log_sigma_x, -6)
                loss = loss_function(
                    reconstructed,
                    target_images,
                    latent_mu,
                    latent_logvar,
                    # log_sigma_x,
                    # capacity=config["capacity"]["initial"],
                    beta=beta,
                )
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            train_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        dt = time.time() - t0
        t0 = time.time()
        current_lr = scheduler.get_last_lr()[0]
        if master_process:
            print(f"Epoch {epoch} took {dt*1000:.2f}ms, train_loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Validation and checkpointing (sample images, save model, etc.)
        if (epoch % check_samples_every_n_epoch == 0) and master_process:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images, _, _, _ = val_batch
                    val_images = val_images.to(device, dtype=ptdtype)
                    reconstructed, mu, logvar = model(val_images)
                    log_sigma_x = get_model_attribute(model, "log_sigma_x")
                    loss = loss_function(reconstructed, val_images, mu, logvar, log_sigma_x, capacity=config["capacity"]["initial"], beta=beta)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch}: Average Val Loss: {val_loss:.4f}")
            # Save sample images and checkpoint
            sample_path = checkpoint_dir / f"epoch_{epoch}_samples.png"
            save_image(reconstructed.cpu(), sample_path, nrow=4, normalize=True)
            top_k_saver.save_model(model, optimizer, epoch, train_loss)
            delete_old_checkpoints(checkpoint_dir, keep_top_k=5)

    # Save final model
    if master_process:
        final_ckpt = checkpoint_dir / "final_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, final_ckpt)
        print(f"Training complete. Final model saved at {final_ckpt}.")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

# -------------------
# Entry point
# -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with YAML configuration")
    parser.add_argument("--config", type=str, default="experiment.yml", help="Path to YAML config file")
    args = parser.parse_args()
    
    main(args.config)
