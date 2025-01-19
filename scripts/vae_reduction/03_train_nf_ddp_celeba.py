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
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from albumentations import CoarseDropout, Compose
from albumentations.pytorch import ToTensorV2

from ciflows.datasets.causalceleba import CausalCelebAEmbedding
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.reduction import make_nf_model
from ciflows.reduction.resnetvae import DeepResNetVAE
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
    dataset,
    scm_type,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    img_size=128,
):
    causal_celeba_dataset = CausalCelebAEmbedding(
        root=root_dir,
        graph_type=graph_type,
        dataset=dataset,
        scm_type=scm_type,
        img_size=img_size,
        fast_dev_run=False,  # Set to True for debugging
    )

    # Calculate the number of samples for training and validation
    # total_len = len(causal_celeba_dataset)
    # val_len = int(total_len * val_split)
    # train_len = total_len - val_len

    # # Split the dataset into train and validation sets
    # train_dataset, val_dataset = random_split(causal_celeba_dataset, [train_len, val_len])

    distr_labels = [x[1] for x in causal_celeba_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(f"Batch size must be at least {unique_distrs} for stratified sampling.")
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    # Define the DataLoader
    train_loader = DataLoader(
        dataset=causal_celeba_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        # shuffle=True,  # Shuffle data during training
        num_workers=num_workers,
        pin_memory=True,  # Enable if using a GPU
        persistent_workers=True,
    )

    return train_loader


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
    gradient_accumulation_steps = 2 * 3  # used to simulate larger batch sizes
    img_size = 128
    graph_type = "chain"
    scm_type = "haircolor"
    # scm_type = "eyeglass"
    num_workers = 4

    check_samples_every_n_epoch = 5

    # adamw optimizer settings
    max_epochs = 5_000
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
    num_flows = 128

    beta_max = 1.5
    annealing_epochs = 1000  # Number of epochs for full beta

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

        if dist.is_initialized() and dist.get_rank() == 0:
            print("Distributed training initialized on rank 0")

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

    if master_process:
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
    # model_fname = "celeba_cyclicbeta_eyeglassesscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v2.pt"
    model_fname = "celeba_haircolor_nfon_128flows_alldata_cyclicresnetvaereduction_batch1024_latentdim48_hcdim4_trainableedges_sep4and8_v1.pt"
    hcdim = 4
    checkpoint_model_fname = "celeba_nfon_cyclicbetaresnetvaereduction_batch1024_latentdim48_trainableedges_sep4and8_v1.pt"
    model_checkpoint_dir = (
        root / "CausalCelebA" / "nf_on_vae_reduction" / checkpoint_model_fname.split(".")[0]
    )

    vae_model_dir = "celeba_cyclicbeta_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1cont"
    vae_model_fname = "model_epoch_9040.pt"
    scm_type = "haircolor"
    dataset = "alldata"

    # prefix within the filename of embeddings
    scm_name = "hair" if scm_type == "haircolor" else "eye"

    vae_dir = root / "CausalCelebA" / "vae_reduction" / scm_type / vae_model_dir.split(".")[0]
    # vae_model = VAE().to(device)
    vae_model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    model_path = vae_dir / vae_model_fname
    vae_model, _ = load_model(vae_model, model_path, device)
    vae_model = vae_model.to(device)

    # for loaded checkpoints
    checkpoint_dir = root / "CausalCelebA" / "nf_on_vae_reduction" / model_fname.split(".")[0]
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))

    model = make_nf_model(K=num_flows, hc_dim=hcdim, trainable_edges=False, debug=debug)
    model = model.to(device)

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
            model_checkpoint_dir / checkpoint_model_fname,
            device,
            optimizer=optimizer,
        )
        # Synchronize all processes
        if ddp:
            torch.distributed.barrier()
    else:
        start_epoch = 1

    # Wrap model for distributed training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # print the number of parameters in the model
    if master_process:
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
    if master_process:
        print(
            f"Train loader has {len(train_loader)} images and val loader {len(val_loader)} images"
        )

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

    if master_process:
        print(f"Images dtype: {images.dtype}")
        print(f"Model dtype: {next(model.parameters()).dtype}")

    effective_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
    if master_process:
        print(f"Effective batch size: {effective_batch_size}")

    # Training loop
    max_epochs = start_epoch + max_epochs
    if master_process:
        print(f"Starting training loop from epoch {start_epoch} to {max_epochs}")

    for step, epoch in tqdm(enumerate(range(start_epoch, max_epochs)), desc="outer", position=0):
        # Training phase
        model.train()
        train_loss = 0.0

        # Create an iterator for the DataLoader
        train_iterator = iter(train_loader)

        if master_process:
            print(f"Epoch: {epoch}, Step: {step}")

        # forward update with optional gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # DDP training requires syncing gradients at the last micro step
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

            with ctx:
                # forward pass
                images = images.to(device)
                optimizer.zero_grad()

                # extract data from tensor to Parameterdict
                loss = model.forward_kld(images, intervention_targets=targets, distr_idx=distr_idx)

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

        if master_process:
            print(
                f"====> Epoch: {epoch} in time {dt*1000:.2f}ms \n"
                f"Average loss: {train_loss:.4f}, LR: {lr:.6f} "
            )

        # Validation phase
        if debug or (epoch % check_samples_every_n_epoch == 0 and master_process):
            print()
            print(f"Saving images - Epoch [{epoch}/{max_epochs}], Val Loss: {train_loss:.4f}")
            model.eval()

            val_loss = 0.0

            # Sample and save reconstructed images
            train_images = images[:8]
            with torch.no_grad():
                if master_process:
                    print("Iterating through val loader")
                # sample images from normalizing flow
                for distr_idx in train_loader.dataset.distr_idx_list:
                    sample_embeddings, _ = model.sample(8, distr_idx=distr_idx)

                    # reconstruct images
                    reconstructed_images = vae_model.decode(sample_embeddings).reshape(
                        -1, 3, img_size, img_size
                    )

                    # clamp said images
                    # reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
                    save_image(
                        reconstructed_images.cpu(),
                        checkpoint_dir / f"epoch_{epoch}_distr-{distr_idx}_samples.png",
                        nrow=4,
                        normalize=True,
                    )

                # reconstruct images using VAE
                embeddings = images[:8]
                images = vae_model.decoder(embeddings).reshape(-1, 3, img_size, img_size)

                # forward/inverse of flow model
                recon_embedding = model.forward(model.inverse(embeddings))

                # reconstruct images
                reconstructed_images = vae_model.decoder(recon_embedding).reshape(
                    -1, 3, img_size, img_size
                )

                # clamp said images
                reconstructed_images = torch.cat([images, reconstructed_images], dim=0)
                # reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

                save_image(
                    reconstructed_images.cpu(),
                    checkpoint_dir / f"epoch_{epoch}_reconstructed_samples.png",
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
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    # Load back the saved final model and verify that it loads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf_model = raw_model.to(device)
    model_path = checkpoint_dir / model_fname
    nf_model, _ = load_model(nf_model, model_path, device, optimizer=optimizer)
