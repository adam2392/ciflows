import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
import torch.version
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ciflows.reduction.resnetvae import DeepResNetVAE
from ciflows.datasets.causalceleba import CausalCelebA, CausalCelebAEyeGlasses
from ciflows.datasets.multidistr import StratifiedSampler
from torch.utils.data import DataLoader, random_split
from ciflows.eval import load_model
from ciflows.training import TopKModelSaver, delete_old_checkpoints


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


def configure_optimizers(
    model,
    learning_rate,
    betas,
    weight_decay=0.0,
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    print("using fused AdamW")

    return optimizer


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    val_split=0.2,
    img_size=64,
    scm_type="haircolor",
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # Resize images to 128x128
            transforms.CenterCrop(img_size),  # Ensure square crop
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            # ),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )

    if scm_type == "haircolor":
        causal_celeba_dataset = CausalCelebA(
            root=root_dir,
            graph_type=graph_type,
            transform=image_transform,
            img_size=img_size,
            fast_dev_run=False,  # Set to True for debugging
        )
    elif scm_type == "eyeglasses":
        causal_celeba_dataset = CausalCelebAEyeGlasses(
            root=root_dir,
            graph_type=graph_type,
            transform=image_transform,
            img_size=img_size,
            fast_dev_run=False,  # Set to True for debugging
        )

    # Calculate the number of samples for training and validation
    total_len = len(causal_celeba_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(causal_celeba_dataset, [train_len, val_len])

    distr_labels = [x[1] for x in causal_celeba_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(f"Batch size must be at least {unique_distrs} for stratified sampling.")
    sampler = StratifiedSampler(distr_labels, batch_size)

    distr_labels = [x[1] for x in train_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(f"Batch size must be at least {unique_distrs} for stratified sampling.")
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    distr_labels = [x[1] for x in val_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(f"Batch size must be at least {unique_distrs} for stratified sampling.")
    val_sampler = StratifiedSampler(distr_labels, batch_size)

    # Define the DataLoader
    train_loader = DataLoader(
        dataset=causal_celeba_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        # shuffle=True,  # Shuffle data during training
        num_workers=num_workers,
        pin_memory=True,  # Enable if using a GPU
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle data during validation
        sampler=val_sampler,
        num_workers=num_workers // 2,
        pin_memory=True,  # Enable if using a GPU
    )
    return train_loader, val_loader


def gaussian_nll(recon_x, log_sigma, x):
    first_term = 0.5 * torch.pow((x - recon_x) / log_sigma.exp(), 2)
    # print(first_term.shape)
    return first_term + log_sigma + 0.5 * np.log(2 * np.pi)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var, log_sigma_x, capacity=0.0, beta=0.00025):
    # print(recon_x.shape, x.shape)
    rec_loss = F.mse_loss(recon_x, x)
    # print(recon_x.shape, x.shape, mu.shape, log_var.shape)

    # rec_loss = gaussian_nll(recon_x, log_sigma_x, x).sum()

    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    # beta = 0.00025
    # beta =
    # Latent Capacity Control
    if capacity == 0:
        kl_loss_controlled = KLD
    else:
        kl_loss_controlled = torch.max(KLD - capacity, torch.tensor(0.0).cuda())

    loss = rec_loss + beta * kl_loss_controlled
    return loss


# Beta annealing function (cyclic)
def cyclic_beta(step, cycle_length, beta_min=0.00025, beta_max=0.01):
    """Cyclic cosine annealing schedule for beta."""
    cycle_position = step % cycle_length
    fraction = cycle_position / cycle_length
    return beta_min + (beta_max - beta_min) * (1 - math.cos(math.pi * fraction)) / 2


def get_model_attribute(model, attr):
    return getattr(model.module if isinstance(model, DDP) else model, attr)


if __name__ == "__main__":
    debug = False
    compile = False
    load_from_checkpoint = False

    # System settings
    world_size = torch.cuda.device_count()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
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
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.

    print(f"Using device: {device} with {world_size} GPUs")
    print(f"Using accelerator: {accelerator}")
    # Check if NCCL backend is available
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("NCCL backend available:", torch.distributed.is_nccl_available())

    # Data settings
    batch_size = 128
    gradient_accumulation_steps = 8 * 3  # used to simulate larger batch sizes
    img_size = 128
    graph_type = "chain"
    scm_type = "haircolor"
    num_workers = 4

    check_samples_every_n_epoch = 5

    # adamw optimizer settings
    max_epochs = 5000
    lr = 3e-4
    lr_min = 6e-5
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    # Learning rate scheduler settings
    lr_scheduler = "cosine"

    # model settings
    n_channels = 3
    out_channels = 3
    latent_dim = 48
    num_blocks_per_stage = 3

    beta_max = 1.5
    annealing_epochs = 1000  # Number of epochs for full beta

    initial_capacity = 0.0
    max_capacity = 25.0
    capacity_increment = 0.1  # Increment per epoch
    current_capacity = initial_capacity

    if debug:
        accelerator = "cpu"
        device = "cpu"
        max_epochs = 5
        batch_size = 8
        check_samples_every_n_epoch = 1
        num_workers = 2
        num_blocks_per_stage = 3

        gradient_accumulation_steps = 2
        fast_dev = True

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if debug:
        ddp = 0

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        if ddp_local_rank >= torch.cuda.device_count():
            ddp_local_rank = ddp_world_size - ddp_local_rank

        device = f"cuda:{ddp_local_rank}"

        print("Setting device to", device)
        print("DDP rank: ", ddp_rank)
        print("Local rank: ", ddp_local_rank)
        print("World size: ", ddp_world_size)

        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    print(
        f"Running training with {gradient_accumulation_steps} gradient accumulation steps per process"
    )
    print(f"Over {max_epochs} epochs, with batch size {batch_size} and {num_workers} workers")

    # set seed
    seed = 1234
    np.random.seed(seed + seed_offset)
    pl.seed_everything(seed + seed_offset, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")
        root = Path("/local/eb/adam2392/")

    ctx = (
        nullcontext() if device == "cpu" else torch.autocast(device_type=accelerator, dtype=ptdtype)
    )
    ctx = nullcontext()

    # v1: K=32
    # v2: K=8
    # v3: K=8, batch higher
    model_fname = "celeba_cyclicbeta_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1.pt"
    checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / scm_type / model_fname.split(".")[0]

    # for loaded checkpoints
    checkpoint_model_fdir = (
        "celeba_cyclicbeta_noimageaug_vaeresnetreduction_batch1024_norm01_latentdim48_img128_v1.pt"
    )
    saved_checkpoint_dir = (
        root / "CausalCelebA" / "vae_reduction" / checkpoint_model_fdir.split(".")[0]
    )
    savedcheckpoint_model_fname = "celeba_fff_resnet_batch128_gradaccum_latentdim48_beta1000_v1_.pt"
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    # model.apply(weights_init)
    model = model.to(ptdtype).to(device)
    image_dim = 3 * img_size * img_size

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))

    # configure optimizers
    optimizer = configure_optimizers(
        model,
        learning_rate=lr,
        betas=(beta1, beta2),
        weight_decay=1e-4,
    )
    # Default: create pytorch optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # compile the model
    if compile:
        model = torch.compile(model)

    # load weights from checkpoint + optimizer state
    if load_from_checkpoint:
        model, start_epoch = load_model(
            model,
            saved_checkpoint_dir / savedcheckpoint_model_fname,
            device,
            optimizer=optimizer,
        )
    else:
        start_epoch = 1

    # Wrap model for distributed training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # Cosine Annealing Scheduler (adjust the T_max for the number of epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr_min
    )  # T_max = total epochs

    top_k_saver = TopKModelSaver(checkpoint_dir, k=5)  # Initialize the top-k model saver

    train_loader, val_loader = data_loader(
        root_dir=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        img_size=img_size,
        scm_type=scm_type,
    )
    print(f"Train loader has {len(train_loader)} images and val loader {len(val_loader)} images")

    # training loop
    # - log the train and val loss every 10 epochs
    # - sample from the model every 10 epochs, and save the images
    # - save the top 5 models based on the validation loss
    # - save the model at the end of training

    t0 = time.time()
    local_iter_epoch = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    # extract the variables within the batch
    train_iterator = iter(train_loader)

    # Prefetch next batch asynchronously
    try:
        batch = next(train_iterator)
    except StopIteration:
        # Reinitialize iterator when dataset is exhausted
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

    images, distr_idx, targets, meta_labels = batch
    images = images.to(device=device, dtype=ptdtype)

    print(f"Images dtype: {images.dtype}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Initialize Capacity and Scheduler
    cycle_length = len(train_loader) * 5  # Full cycle over 5 epochs

    # XXX: remove when not doing FFF-VAE
    # loss_nll = torch.tensor(0.0)
    # surrogate_loss = torch.tensor(0.0)
    effective_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
    print(f"Effective batch size: {effective_batch_size}")

    # Training loop
    max_epochs = start_epoch + max_epochs
    annealing_epochs = annealing_epochs + start_epoch
    for step, epoch in tqdm(enumerate(range(start_epoch, max_epochs)), desc="outer", position=0):
        # Training phase
        model.train()
        train_loss = 0.0

        # Create an iterator for the DataLoader
        train_iterator = iter(train_loader)

        # Compute cyclic beta, which
        global_step = epoch * len(train_loader) + step
        beta = cyclic_beta(global_step, cycle_length)

        if master_process:
            print(f"Epoch: {epoch}, Step: {step}, Beta: {beta:.6f}")

        # forward update with optional gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # DDP training requires syncing gradients at the last micro step
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

            with ctx:
                # forward pass
                images = images.to(device)
                optimizer.zero_grad()
                reconstructed, latent_mu, latent_logvar = model(images)  # Model forward pass

                # Clamp logvar to prevent numerical instability
                latent_logvar = torch.clamp_(latent_logvar, -10, 10)

                # Compute log_sigma_x
                log_sigma_x = model.log_sigma_x

                # Learning the variance can become unstable in some cases.
                # Softly limiting log_sigma to a minimum of -6 ensures stable training.
                log_sigma_x = softclip(log_sigma_x, -6)

                loss = loss_function(
                    reconstructed,
                    images,
                    latent_mu,
                    latent_logvar,
                    log_sigma_x=log_sigma_x,
                    capacity=current_capacity,
                    beta=beta,
                )  # Custom VAE loss function

                # scale loss based on how many accumulation steps we take
                loss = loss / gradient_accumulation_steps

                # backwards pass, with gradient scaling
                scaler.scale(loss).backward()

            # Prefetch next batch asynchronously
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reinitialize iterator when dataset is exhausted
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # extract the variables within the batch
            images, distr_idx, targets, meta_labels = batch
            images = images.to(device)

            # DDP: accumulate loss terms
            train_loss += loss.item()

        # clip the gradient
        if grad_clip != 0.0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step optimizer and update
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        # flush gradients to release memory
        optimizer.zero_grad(set_to_none=True)

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lr = scheduler.get_last_lr()[0]

        print(
            f"====> Epoch: {epoch} in time {dt*1000:.2f}ms \n"
            f"Average loss: {train_loss:.4f}, LR: {lr:.6f} "
        )

        # Validation phase
        if debug or epoch % check_samples_every_n_epoch == 0 and master_process:
            print()
            print(f"Saving images - Epoch [{epoch}/{max_epochs}], Val Loss: {train_loss:.4f}")
            model.eval()

            val_loss = 0.0

            # Sample and save reconstructed images
            train_images = images[:8]
            with torch.no_grad():
                log_sigma_x = model.log_sigma_x

                print('Iterating through val loader')
                for batch_idx, (
                    val_images,
                    distr_idx,
                    targets,
                    meta_labels,
                ) in enumerate(val_loader):
                    val_images = val_images.to(device)
                    reconstructed, latent_mu, latent_logvar = model(
                        val_images
                    )  # Model forward pass

                    loss = loss_function(
                        reconstructed,
                        val_images,
                        latent_mu,
                        latent_logvar,
                        log_sigma_x=log_sigma_x,
                        capacity=current_capacity,
                        beta=beta,
                    )  # Custom VAE loss function
                    val_loss += loss.item()

                    if debug:
                        break

                # pick the first 8 images in the last val batch
                # sample_images = val_images[:8]  # Pick 8 images for sampling
                sample_images = train_images[:8]  # Pick 8 images for sampling

                # Standard VAE
                encoding = model.encode(sample_images)
                reconstructed_images = model.decode(encoding).reshape(-1, 3, img_size, img_size)
                reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

                # sample images from VAE
                # 1. Sample latent variables from standard Gaussian
                num_samples = 8  # Number of images to generate
                z = torch.randn(num_samples, latent_dim).to(device)  # Sample z ~ N(0, I)

                # 2. Pass the sampled z through the decoder
                generated_images = model.decode(z)  # Shape: [num_samples, 3, 128, 128]

                # clamp
                generated_images = torch.clamp(generated_images, 0, 1)

                # now sample training images and then reconstruct
                encoding = model.encode(train_images)
                train_reconstructed_images = model.decode(encoding).reshape(
                    -1, 3, img_size, img_size
                )
                train_reconstructed_images = torch.clamp(train_reconstructed_images, 0, 1)

            val_loss /= len(val_loader)
            print(f"====> Epoch: {epoch} Average Val loss: {val_loss:.4f}")
            sample_images = torch.cat(
                (
                    sample_images.cpu(),
                    reconstructed_images.cpu(),
                    generated_images.cpu(),
                    train_images.cpu(),
                    train_reconstructed_images.cpu(),
                ),
                dim=0,
            )
            save_image(
                sample_images,
                checkpoint_dir / f"epoch_{epoch}_samples.png",
                nrow=4,
                normalize=True,
            )

            # Track top 5 models based on validation loss
            # Optionally, remove worse models if there are more than k saved models
            top_k_saver.save_model(raw_model, optimizer, epoch, train_loss)
            delete_old_checkpoints(checkpoint_dir, keep_top_k=5)

        epoch += 1
        local_iter_epoch += 1

        # termination conditions
        if epoch > max_epochs:
            break
    if ddp:
        dist.destroy_process_group()

    # Save final model
    if not ddp or dist.get_rank() == 0:
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,  # Optional: Save the current epoch
                "loss": loss,  # Optional: Save the last loss value
            },
            checkpoint_dir / model_fname,
        )
        print(f"Training complete. Models saved in {checkpoint_dir}.")
    dist.barrier()

    # Load back the saved final model and verify that it loads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage).to(device)
    model_path = checkpoint_dir / model_fname
    vae_model = load_model(vae_model, model_path, device, optimizer=optimizer)
