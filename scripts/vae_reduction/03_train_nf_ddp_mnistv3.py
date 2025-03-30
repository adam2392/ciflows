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
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
import lightning as pl

from ciflows.datasets.causalmnist import CausalMNISTEmbedding, CausalMNIST
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
        dataset=causal_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    ), DataLoader(
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
    # Load configuration
    nf_config_path = "/home/adam2392/projects/ciflows/scripts/vae_reduction/nf_experiment.yml"
    nf_config = load_experiment_config(nf_config_path)

    # Load VAE Configuration
    vae_checkpoint_dir = (
        Path(nf_config["vae"]["checkpoint_dir"]) / nf_config["vae"]["experiment_name"]
    )
    vae_config_path = vae_checkpoint_dir / "experiment_config.yaml"
    vae_config = load_experiment_config(vae_config_path)

    # Paths and Experiment Settings
    root = Path("/local/eb/adam2392/CausalMNIST/")
    exp_name = nf_config["exp_name"]
    model_root = root / "nf_reduction" / exp_name
    model_root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the NF config
    config_save_path = model_root / "experiment_config.yaml"
    shutil.copy(nf_config_path, config_save_path)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(model_root / "logs"))

    # System settings
    sys_cfg = nf_config["system"]
    seed = sys_cfg.get("seed", 1234)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Load device settings
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(sys_cfg.get("device", "cuda"))
        accelerator = sys_cfg.get("accelerator", "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
    world_size = torch.cuda.device_count()
    use_amp = nf_config["training"].get("use_amp", True)  # Enable AMP by default
    grad_accum_steps = nf_config["training"]["grad_accum_steps"]

    # Set up DDP if necessary
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        grad_accum_steps //= torch.distributed.get_world_size()
    else:
        master_process = True

    dtype = (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = "float32"

    # pytorch dtype
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext() if device == "cpu" else torch.autocast(device_type=accelerator, dtype=ptdtype)
    )
    ctx = nullcontext()

    print(f"Training NF model with {world_size} GPUs on {device} with dtype {dtype}")
    print(f"Batch size: {nf_config['data']['batch_size']}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    
    # Load Pretrained VAE Model
    vae_checkpoint_dir = (
        Path(nf_config["vae"]["checkpoint_dir"]) / nf_config["vae"]["experiment_name"]
    )
    vae_checkpoint_file = vae_checkpoint_dir / nf_config["vae"]["vae_checkpoint_fname"]

    img_size = nf_config["data"]["img_size"]
    vae_model = DeepResNetMNISTVAE(
        latent_dim=vae_config["model"]["latent_dim"],
        num_blocks_per_stage=vae_config["model"]["num_blocks_per_stage"],
    )
    vae_model, _ = load_model(vae_model, vae_checkpoint_file, device)
    vae_model.to(device)

    # Initialize NF Model
    nf_model = make_mnist_nf_model(
        K=nf_config["model"]["num_flows"], trainable_edges=nf_config["model"]["trainable_edges"]
    ).to(device)

    # Optimizer and Scheduler
    optimizer_nf = configure_optimizers(
        nf_model,
        learning_rate=nf_config["optimizer"]["lr"],
        betas=tuple(nf_config["optimizer"]["betas"]),
        weight_decay=nf_config["optimizer"]["weight_decay"],
    )
    optimizer_vae = configure_optimizers(
        vae_model,
        learning_rate=nf_config["optimizer"]["lr"],
        betas=tuple(nf_config["optimizer"]["betas"]),
        weight_decay=nf_config["optimizer"]["weight_decay"],
    )

    # Load NF Checkpoint if Available
    nf_checkpoint_dir = root / "nf_reduction" / nf_config["exp_name"]
    start_epoch = 1
    if nf_config.get("load_from_checkpoint", False):
        nf_checkpoint_file = nf_checkpoint_dir / nf_config["checkpoint_file"]
        nf_model, start_epoch = load_model(nf_model, nf_checkpoint_file, device, optimizer=optimizer_nf)
        print(f"Loaded model from {nf_checkpoint_file} and starting from {start_epoch} epoch")

        # Synchronize all processes
        if ddp:
            torch.distributed.barrier()

    # Model Saving Utility
    top_k_saver = TopKModelSaver(nf_checkpoint_dir, k=5)
    top_k_saver_vae = TopKModelSaver(nf_checkpoint_dir, k=5)
    scheduler = CosineAnnealingLR(
        optimizer_nf, T_max=nf_config["training"]["max_epochs"], eta_min=1e-5
    )

    # Enable mixed precision training if AMP is enabled
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))

    grad_clip = nf_config["training"]["grad_clip"]

    # Wrap with DDP if needed
    if ddp:
        nf_model = DDP(nf_model, device_ids=[ddp_local_rank])
        vae_model = DDP(vae_model, device_ids=[ddp_local_rank])

    # DataLoader
    train_loader, orig_train_loader = data_loader(
        root_dir=nf_config["data"]["root_dir"],
        dataset=nf_config["data"]["graph_type"],
        graph_type=nf_config["data"]["graph_type"],
        num_workers=nf_config["data"]["num_workers"],
        batch_size=nf_config["data"]["batch_size"],
        img_size=nf_config["data"]["img_size"],
    )

    if master_process:
        print(
            f"Training NF model with {sum(p.numel() for p in nf_model.parameters() if p.requires_grad):,} parameters"
        )

    # Training Loop
    max_epochs = nf_config["training"]["max_epochs"]
    for epoch in tqdm(range(start_epoch, max_epochs), desc="Epochs"):
        nf_model.train()
        total_loss = 0.0
        total_vae_loss = 0.0
        train_iterator = iter(orig_train_loader)

        for step, batch in enumerate(train_iterator):
            with ctx:
                imgs, distr_idx, targets, meta_labels = batch
                imgs = imgs.to(device)
                distr_idx = distr_idx.to(device)
                targets = targets.to(device)
                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    latent_embeddings = vae_model.module.encode(imgs)
                    ll_loss = nf_model.module.forward_kld(
                        latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                    )
                    inv_latent = nf_model.module.inverse(latent_embeddings)
                    inv_latent += torch.randn_like(inv_latent).to(device) * 1e-5
                    recon_latent_embeddings = nf_model.module.forward(inv_latent)
                    recon_imgs = vae_model.module.decode(recon_latent_embeddings)

                    # Compute the reconstruction loss.
                    rec_loss = F.mse_loss(recon_imgs, imgs)
                else:
                    latent_embeddings = vae_model.encode(imgs)
                    ll_loss = nf_model.forward_kld(
                        latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
                    )
                    inv_latent = nf_model.inverse(latent_embeddings)
                    inv_latent += torch.randn_like(inv_latent).to(device) * 1e-5
                    recon_latent_embeddings = nf_model.forward(inv_latent)
                    recon_imgs = vae_model.decode(recon_latent_embeddings)

                    # Compute the reconstruction loss.
                    rec_loss = F.mse_loss(recon_imgs, imgs)

                total_vae_loss = ll_loss + rec_loss
                total_vae_loss /= grad_accum_steps  # Scale loss for gradient accumulation

            # Backpropagation with AMP
            scaler.scale(total_vae_loss).backward(retain_graph=True)
            
            if (step + 1) % grad_accum_steps == 0:
                if grad_clip != 0.0 and scaler.is_enabled():
                    scaler.unscale_(optimizer_nf)
                    scaler.unscale_(optimizer_vae)
                torch.nn.utils.clip_grad_norm_(nf_model.parameters(), grad_clip)
                scaler.step(optimizer_nf)
                scaler.step(optimizer_vae)
                scaler.update()
                optimizer_nf.zero_grad()
                optimizer_vae.zero_grad()

            total_loss += total_vae_loss.item()

        scheduler.step()

        # Save model periodically
        if master_process and epoch % nf_config["training"]["save_every"] == 0:
            # torch.save(nf_model.state_dict(), nf_checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
            top_k_saver.save_model(nf_model, optimizer_nf, epoch, total_loss)
            top_k_saver_vae.save_model(vae_model, optimizer_vae, epoch, total_loss)
            delete_old_checkpoints(nf_checkpoint_dir, keep_top_k=10)
            print(f"Checkpoint saved at {nf_checkpoint_dir}")

        # Image Generation & Latent Visualization
        if master_process and epoch % nf_config["training"]["n_log_steps"] == 0:
            nf_model.eval()
            with torch.no_grad():
                # 1 & 2. Training images reconstructed using VAE and NF+VAE
                recon_vae = vae_model.decode(latent_embeddings[:8]).reshape(
                    -1, 3, img_size, img_size
                )
                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    nf_recon_latents = nf_model.module.forward(
                        nf_model.module.inverse(latent_embeddings[:8])
                    )
                else:
                    nf_recon_latents = nf_model.forward(nf_model.inverse(latent_embeddings[:8]))
                recon_nf_vae = vae_model.decode(nf_recon_latents).reshape(-1, 3, img_size, img_size)

                # Stack reconstructions vertically
                recon_stack = torch.cat(
                    [recon_vae, recon_nf_vae], dim=0
                )  # Stacking along batch dimension
                save_image(
                    recon_stack.cpu(),
                    nf_checkpoint_dir / f"epoch_{epoch}_reconstructions.png",
                    nrow=8,
                    normalize=True,
                )

                # 3 & 4. Sample images from VAE and NF decoded by VAE
                sample_latents = torch.randn((8, 32), device=device)
                sample_vae = vae_model.decode(sample_latents).reshape(-1, 3, img_size, img_size)

                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=0)
                else:
                    sample_nf_latents, _ = nf_model.sample(8, distr_idx=0)
                sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
                # Stack generated samples vertically
                sample_stack = torch.cat([sample_vae, sample_nf], dim=0)

                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=1)
                else:
                    sample_nf_latents, _ = nf_model.sample(8, distr_idx=1)
                sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
                # Stack generated samples vertically
                sample_stack = torch.cat([sample_stack, sample_nf], dim=0)

                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=2)
                else:
                    sample_nf_latents, _ = nf_model.sample(8, distr_idx=2)
                sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
                # Stack generated samples vertically
                sample_stack = torch.cat([sample_stack, sample_nf], dim=0)
                
                save_image(
                    sample_stack.cpu(),
                    nf_checkpoint_dir / f"epoch_{epoch}_samples.png",
                    nrow=8,
                    normalize=True,
                )

                # 5. Visualizing NF Latent Distribution
                if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
                    nf_latent_distr = nf_model.module.inverse(latent_embeddings)
                else:
                    nf_latent_distr = nf_model.inverse(latent_embeddings)
                for dim in range(32):
                    writer.add_histogram(
                        f"NF_Latents/dim_{dim}",
                        nf_latent_distr[:, dim].cpu().numpy().squeeze(),
                        epoch,
                    )

        # TensorBoard Logging
        if master_process:
            writer.add_scalar("Loss/Train", total_loss, epoch)

    # Save Final Model
    if master_process:
        torch.save(nf_model.state_dict(), nf_checkpoint_dir / "final_model.pt")
        torch.save(vae_model.state_dict(), nf_checkpoint_dir / "final_vae_model.pt")

    print("Training Complete!")
