import os
import time
from contextlib import nullcontext
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import init_process_group
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.version
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ciflows.datasets.causalceleba import CausalCelebA
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.eval import load_model
from ciflows.flows.freeform import ResnetFreeformflow
from ciflows.loss import volume_change_surrogate
from ciflows.resnet_celeba import ResNetCelebADecoder, ResNetCelebAEncoder
from ciflows.training import TopKModelSaver


def configure_optimizers(
    model,
    learning_rate,
    betas,
    device_type,
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
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, fused=True
    )
    print(f"using fused AdamW")

    return optimizer


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_model_attribute(model, attr):
    return getattr(model.module if isinstance(model, DDP) else model, attr)


def compute_loss(model: ResnetFreeformflow, x, distr_idx, beta, hutchinson_samples=2):
    # calculate volume change surrogate loss
    surrogate_loss, v_hat, x_hat = volume_change_surrogate(
        images,
        get_model_attribute(model, "encoder"),
        get_model_attribute(model, "decoder"),
        hutchinson_samples=hutchinson_samples,
    )

    # compute reconstruction loss
    loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x)

    # get negative log likelihoood over the distributions
    embed_dim = get_model_attribute(model, "latent_dim")
    v_hat = v_hat.view(-1, embed_dim)
    loss_nll = (
        -get_model_attribute(model, "latent")
        .log_prob(v_hat, distr_idx=distr_idx)
        .mean()
        - surrogate_loss
    )

    loss = beta * loss_reconstruction + loss_nll
    return loss, loss_reconstruction, loss_nll, surrogate_loss


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    img_size=64,
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # Resize images to 128x128
            transforms.CenterCrop(img_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    causal_celeba_dataset = CausalCelebA(
        root=root_dir,
        graph_type=graph_type,
        img_size=img_size,
        transform=image_transform,
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
        raise ValueError(
            f"Batch size must be at least {unique_distrs} for stratified sampling."
        )
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


def make_fff_model(num_blocks_per_stage=5, debug=False):
    node_dimensions = {
        0: 16,
        1: 16,
        2: 16,
    }
    edge_list = [(1, 2)]
    noise_means = {
        0: torch.zeros(16),
        1: torch.zeros(16),
        2: torch.zeros(16),
    }
    noise_variances = {
        0: torch.ones(16),
        1: torch.ones(16),
        2: torch.ones(16),
    }
    intervened_node_means = [{2: torch.ones(16) + 2}, {2: torch.ones(16) + 4}]
    intervened_node_vars = [{2: torch.ones(16)}, {2: torch.ones(16) + 2}]

    latent_dim = 48

    confounded_list = []
    # independent noise with causal prior
    latent = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )

    # define the encoder and decoder
    model = ResnetFreeformflow(
        latent=latent, latent_dim=latent_dim, num_blocks_per_stage=num_blocks_per_stage
    )
    return model


if __name__ == "__main__":
    debug = False
    compile = False

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
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = 'float32'

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
    batch_size = 512
    gradient_accumulation_steps = 8 * max(
        1, world_size
    )  # used to simulate larger batch sizes
    img_size = 128
    graph_type = "chain"
    num_workers = 4

    check_samples_every_n_epoch = 5

    # adamw optimizer settings
    max_epochs = 2000
    lr = 3e-4
    lr_min = 6e-5
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    # Learning rate scheduler settings
    lr_scheduler = "cosine"

    # model settings
    num_blocks_per_stage = 5

    if debug:
        accelerator = "cpu"
        device = "cpu"
        max_epochs = 5
        batch_size = 8
        check_samples_every_n_epoch = 1
        num_workers = 2
        num_blocks_per_stage = 1

        gradient_accumulation_steps = 2
        fast_dev = True

    # for FreeformFlow's loss function
    hutchinson_samples = 2
    beta = torch.tensor(10.0).to(device=device, dtype=ptdtype)

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
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
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
    print(
        f"Over {max_epochs} epochs, with batch size {batch_size} and {num_workers} workers"
    )

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
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.autocast(device_type=accelerator, dtype=ptdtype)
    )

    # v1: K=32
    # v2: K=8
    # v3: K=8, batch higher
    model_fname = "celeba_fff_resnet_batch512_gradaccum_latentdim48_beta10_v1.pt"
    checkpoint_dir = root / "CausalCelebA" / "fff" / model_fname.split(".")[0]
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = make_fff_model(num_blocks_per_stage=num_blocks_per_stage, debug=debug)
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
        device_type=device,
    )

    # compile the model
    if compile:
        model = torch.compile(model)

    # Wrap model for distributed training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Cosine Annealing Scheduler (adjust the T_max for the number of epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr_min
    )  # T_max = total epochs

    top_k_saver = TopKModelSaver(
        checkpoint_dir, k=5
    )  # Initialize the top-k model saver

    train_loader = data_loader(
        root_dir=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        img_size=img_size,
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

    print(f"Images dtype: {images.dtype}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Training loop
    for epoch in tqdm(range(1, max_epochs + 1), desc="outer", position=0):
        # Training phase
        model.train()
        train_loss = 0.0
        train_reconstruction_loss = 0.0
        train_nll_loss = 0.0
        train_surrogate_loss = 0.0

        # Create an iterator for the DataLoader
        train_iterator = iter(train_loader)

        # forward update with optional gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # DDP training requires syncing gradients at the last micro step
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

            with ctx:
                # forward pass
                # print(f"Images dtype: {images.dtype}")
                # print(f"Model dtype: {next(model.parameters()).dtype}")
                # print(f"beta dtype: {beta.dtype}")
                # compute the loss
                loss, loss_reconstruction, loss_nll, surrogate_loss = compute_loss(
                    model, images, distr_idx, beta
                )

                # sum up the loss
                loss = loss.sum()

                # compute the average
                loss = loss / gradient_accumulation_steps

                # backwards pass, with gradient scaling
                # scaler.scale(loss).backward()

                loss_nll = loss_nll.sum() / gradient_accumulation_steps
                loss_reconstruction = (
                    loss_reconstruction.sum() / gradient_accumulation_steps
                )
                surrogate_loss = surrogate_loss.sum() / gradient_accumulation_steps

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
            train_reconstruction_loss += loss_reconstruction.item()
            train_nll_loss += loss_nll.item()
            train_surrogate_loss += surrogate_loss.item()

        # clip the gradient
        if grad_clip != 0.0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step optimizer and update
        scaler.step(optimizer)
        scaler.update()

        # flush gradients to release memory
        optimizer.zero_grad(set_to_none=True)

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Log training loss
        # train_loss /= len(train_loader)
        # train_reconstruction_loss /= len(train_loader)
        # train_nll_loss /= len(train_loader)
        # train_surrogate_loss /= len(train_loader)

        lr = scheduler.get_last_lr()[0]

        print(
            f"====> Epoch: {epoch} in time {dt*1000:.2f}ms \n"
            f"Average loss: {train_loss:.4f}, LR: {lr:.6f} "
            f"Reconstruction Loss: {train_reconstruction_loss:.4f}, NLL Loss: {train_nll_loss:.4f}, Surrogate Loss: {train_surrogate_loss:.4f}"
        )

        # Validation phase
        if debug or epoch % check_samples_every_n_epoch == 0 and master_process:
            print()
            print(
                f"Saving images - Epoch [{epoch}/{max_epochs}], Val Loss: {train_loss:.4f}"
            )
            model.eval()

            # sample images from normalizing flow
            for idx in train_loader.dataset.distr_idx_list:
                # reconstruct images
                reconstructed_images, _ = raw_model.sample(8, distr_idx=idx)

                # clamp images to show
                reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

                save_image(
                    reconstructed_images.cpu(),
                    checkpoint_dir / f"epoch_{epoch}_distr-{idx}_samples.png",
                    nrow=4,
                    normalize=True,
                )

            # Track top 5 models based on validation loss
            # Optionally, remove worse models if there are more than k saved models
            top_k_saver.save_model(raw_model, optimizer, epoch, train_loss)

        epoch += 1
        local_iter_epoch += 1

        # termination conditions
        if epoch > max_epochs:
            break
    if ddp:
        dist.destroy_process_group()

    # Save final model
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

    # Load back the saved final model and verify that it loads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fff_model = model.to(device)
    model_path = checkpoint_dir / model_fname
    fff_model = load_model(fff_model, model_path, device, optimizer=optimizer)
