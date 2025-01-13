import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
import torch.version
from torch import nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ciflows.datasets.causalceleba import CausalCelebA, CausalCelebAEyeGlasses
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.flows.glow import GlowBlock, InjectiveGlowBlock, ReshapeFlow, Squeeze
from ciflows.flows.model import CausalInjectiveFlow
from ciflows.reduction.resnetvae import DeepResNetVAE
from ciflows.training import TopKModelSaver, delete_old_checkpoints


def get_inj_model(input_shape):
    use_lu = True
    gamma = 1e-3
    activation = "linear"
    dropout_probability = 0.2

    net_actnorm = False
    # n_hidden_list = [32, 64, 128, 256, 256, 256]
    n_hidden = 128
    n_glow_blocks = 2
    n_mixing_layers = 5
    n_injective_layers = 10
    n_layers = n_mixing_layers + n_injective_layers

    # hidden layers for the AutoregressiveRationalQuadraticSpline
    net_hidden_layers = 2
    net_hidden_dim = 32

    n_channels = input_shape[0]
    img_size = input_shape[1]

    n_chs = n_channels
    flows = []

    debug = False

    n_chs = int(n_channels * 4**n_mixing_layers * (1 / 2) ** n_injective_layers)
    latent_size = int(img_size / (2**n_mixing_layers))
    init_n_chs = n_chs
    init_latent_size = latent_size
    print("Starting at latent representation: ", n_chs, "with latent size: ", latent_size)
    q0 = nf.distributions.DiagGaussian((n_chs, latent_size, latent_size), trainable=False)

    split_mode = "channel"

    for i in range(n_injective_layers):
        # n_hidden = n_hidden_list[-i]
        if i <= 1:
            split_mode = "checkerboard"
        else:
            split_mode = "channel"

        if i % 1 == 0:
            for j in range(n_glow_blocks):
                flows += [
                    GlowBlock(
                        channels=n_chs,
                        hidden_channels=n_hidden,
                        use_lu=use_lu,
                        scale=True,
                        split_mode=split_mode,
                        net_actnorm=net_actnorm,
                        dropout_probability=dropout_probability,
                    )
                ]
        else:
            flows += [
                ReshapeFlow(
                    shape_in=(n_chs, latent_size, latent_size),
                    shape_out=(n_chs * latent_size * latent_size,),
                )
            ]
            flows += [
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    num_input_channels=n_chs * latent_size * latent_size,
                    num_blocks=net_hidden_layers,
                    num_hidden_channels=net_hidden_dim,
                    permute_mask=True,
                )
            ]
            flows += [
                ReshapeFlow(
                    shape_in=(n_chs * latent_size * latent_size,),
                    shape_out=(n_chs, latent_size, latent_size),
                )
            ]

        # input to inj flow is what is at the X -> V layer
        flows += [
            InjectiveGlowBlock(
                channels=n_chs,
                hidden_channels=n_hidden,
                activation=activation,
                scale=True,
                gamma=gamma,
                debug=debug,
                split_mode=split_mode,
                net_actnorm=net_actnorm,
            )
        ]
        n_chs = n_chs * 2
        latent_size = latent_size
        if debug:
            print(f"On layer {n_layers - i}, n_chs = {n_chs//2} -> {n_chs}")

    # split_mode = "channel_inv"
    for i in range(n_mixing_layers):
        # if i > 0:# n_mixing_layers - 1:
        for j in range(n_glow_blocks):
            flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=False,
                    split_mode=split_mode,
                    dropout_probability=dropout_probability,
                )
            ]
        # else:
        #     flows += [
        #         ReshapeFlow(
        #             shape_in=(n_chs, latent_size, latent_size),
        #             shape_out=(n_chs * latent_size * latent_size,),
        #         )
        #     ]
        #     flows += [
        #         nf.flows.AutoregressiveRationalQuadraticSpline(
        #             num_input_channels=n_chs * latent_size * latent_size,
        #             num_blocks=net_hidden_layers,
        #             num_hidden_channels=net_hidden_dim,
        #             permute_mask=True,
        #         )
        #     ]
        #     flows += [
        #         ReshapeFlow(
        #             shape_in=(n_chs * latent_size * latent_size,),
        #             shape_out=(n_chs, latent_size, latent_size),
        #         )
        #     ]

        flows += [Squeeze()]
        n_chs = n_chs // 4
        latent_size *= 2
        if debug:
            print(f"On layer {n_mixing_layers - i}, n_chs = {n_chs}")

    inj_model = nf.NormalizingFlow(q0=None, flows=flows)
    inj_model.output_n_chs = init_n_chs
    inj_model.output_latent_size = init_latent_size
    return inj_model


def get_bij_model(
    n_chs,
    latent_size,
):
    use_lu = True
    net_actnorm = False
    n_hidden = 128
    n_glow_blocks = 10

    flows = []

    debug = False

    print("Starting at latent representation: ", n_chs, latent_size, latent_size)

    split_mode = "checkerboard"

    net_hidden_layers = 2
    net_hidden_dim = 32

    # flows += [
    #     ReshapeFlow(
    #         shape_in=(n_chs, latent_size, latent_size),
    #         shape_out=(n_chs * latent_size * latent_size,),
    #     )
    # ]
    # n_chs = n_chs * latent_size * latent_size

    # using glow blocks
    # n_chs = int(n_chs * 4**n_glow_blocks)
    for i in range(n_glow_blocks):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        # n_chs *= 4
        # param_map = nf.nets.MLP([n_chs, net_hidden_dim, n_chs*2], init_zeros=True)
        # # # Add flow layer
        # flows.append(nf.flows.AffineCouplingBlock(param_map, split_mode='channel'))
        # flows.append(nf.flows.Permute(n_chs, mode='swap'))

        # Autoregressive Neural Spline flow
        # Swap dimensions
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=n_chs * latent_size * latent_size,
                num_blocks=net_hidden_layers,
                num_hidden_channels=net_hidden_dim,
                permute_mask=True,
            )
        ]
        # if i  1:
        #     split_mode = "checkerboard"
        # else:
        #     split_mode = "channel"
        # flows += [
        #     GlowBlock(
        #         channels=n_chs,
        #         hidden_channels=n_hidden,
        #         use_lu=use_lu,
        #         scale=True,
        #         split_mode=split_mode,
        #         net_actnorm=net_actnorm,
        #         dropout_probability=0.2,
        #     )
        # ]
        # flows += [Squeeze()]
        # n_chs = n_chs // 4
        # print(n_chs)
        if debug:
            print(f"On layer {n_glow_blocks - i}, n_chs = {n_chs//2} -> {n_chs}")

    # maps x to (n_chs * latent_size * latent_size), while v is mapped to (n_chs, latent_size, latent_size)
    flows += [
        ReshapeFlow(
            shape_in=(n_chs * latent_size * latent_size,),
            shape_out=(n_chs, latent_size, latent_size),
        )
    ]
    model = nf.NormalizingFlow(q0=None, flows=flows)
    return model


def make_injflow_model(
    input_shape,
    n_chs,
    latent_size,
    adj_mat,
    cluster_sizes,
    intervention_targets,
    confounded_variables,
):
    inj_model = get_inj_model(input_shape)
    pytorch_total_params = sum(p.numel() for p in inj_model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    bij_model = get_bij_model(
        n_chs,
        latent_size,
        adj_mat,
        cluster_sizes,
        intervention_targets,
        confounded_variables,
    )

    pytorch_total_params = sum(p.numel() for p in bij_model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    print("Got Intervention targets for q0: ", intervention_targets)
    # q0 = nf.distributions.DiagGaussian(
    #     (n_chs, latent_size, latent_size), trainable=False
    # )
    q0 = nf.distributions.DiagGaussian((n_chs * latent_size * latent_size,), trainable=False)

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

    confounded_list = []
    # independent noise with causal prior
    q0 = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )

    # initialize the flow model
    initialize_flow(inj_model)
    initialize_flow(bij_model)
    causalinj_model = CausalInjectiveFlow(q0=q0, inj_model=inj_model, bij_model=bij_model)

    return causalinj_model


def initialize_flow(model):
    """
    Initialize a full normalizing flow model
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Layer-dependent initialization
            if "coupling" in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif "batch" in name:
                continue
            else:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 1e-5)


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
    elif scm_type == "eyeglass":
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


# Beta annealing function (cyclic)
def cyclic_beta(step, cycle_length, beta_min=0.00025, beta_max=0.001):
    """Cyclic cosine annealing schedule for beta."""
    cycle_position = step % cycle_length
    fraction = cycle_position / cycle_length
    return beta_min + (beta_max - beta_min) * (1 - math.cos(math.pi * fraction)) / 2


def get_model_attribute(model, attr):
    return getattr(model.module if isinstance(model, DDP) else model, attr)


if __name__ == "__main__":
    debug = True
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
    gradient_accumulation_steps = 3 * 3  # used to simulate larger batch sizes
    img_size = 128
    graph_type = "chain"
    scm_type = "haircolor"
    # scm_type = "eyeglass"
    num_workers = 4

    check_samples_every_n_epoch = 10

    # adamw optimizer settings
    max_epochs = 2000
    n_steps_mse = 50
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
    input_shape = (3, 128, 128)

    if debug:
        accelerator = "cpu"
        device = "cpu"
        max_epochs = 5
        batch_size = 8
        check_samples_every_n_epoch = 1
        num_workers = 2

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
    model_fname = "celeba_cyclicbeta_eyeglassesscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v2.pt"
    model_fname = "celeba_cyclicbeta_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1.pt"
    checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / scm_type / model_fname.split(".")[0]

    # for loaded checkpoints
    checkpoint_model_fdir = "celeba_cyclicbeta_eyeglassesscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1.pt"
    saved_checkpoint_dir = (
        root / "CausalCelebA" / "vae_reduction" / scm_type / checkpoint_model_fdir.split(".")[0]
    )
    savedcheckpoint_model_fname = "model_epoch_4865.pt"
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = make_injflow_model(
        input_shape,
        n_channels,
        latent_size=latent_dim,
    )
    model = model.to(ptdtype).to(device)
    image_dim = 3 * img_size * img_size

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))

    # configure optimizers
    mse_optimizer = configure_optimizers(
        model.inj_model,
        learning_rate=lr,
        betas=(beta1, beta2),
        weight_decay=1e-4,
    )
    nll_optimizer = configure_optimizers(
        model.bij_model,
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
    mse_scheduler = CosineAnnealingLR(
        mse_optimizer, T_max=max_epochs, eta_min=lr_min
    )  # T_max = total epochs
    nll_scheduler = CosineAnnealingLR(
        nll_optimizer, T_max=max_epochs, eta_min=lr_min
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

    # Initialize Capacity and Scheduler
    cycle_length = len(train_loader) * 5  # Full cycle over 5 epochs

    # XXX: remove when not doing FFF-VAE
    # loss_nll = torch.tensor(0.0)
    # surrogate_loss = torch.tensor(0.0)
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

        # Compute cyclic beta, which
        global_step = epoch * len(train_loader) + step
        beta = cyclic_beta(global_step, cycle_length)

        if master_process:
            print(f"Epoch: {epoch}, Step: {step}, cycling over {cycle_length} Beta: {beta:.6f}")

        # forward update with optional gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # DDP training requires syncing gradients at the last micro step
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

            with ctx:
                # forward pass
                images = images.to(device)
                mse_optimizer.zero_grad()
                nll_optimizer.zero_grad()
                reconstructed, latent_mu, latent_logvar = model(images)  # Model forward pass

                # Clamp logvar to prevent numerical instability
                latent_logvar = torch.clamp_(latent_logvar, -10, 10)

                if epoch <= n_steps_mse:
                    v_latent = model.inj_flows.inverse(x)
                    v_latent_recon = model.inj_flows.inverse(x_reconstructed)

                    loss = torch.nn.functional.mse_loss(
                        x_reconstructed, x
                    ) + torch.nn.functional.mse_loss(v_latent_recon, v_latent)
                else:
                    inj_v = model.inj_model.inverse(x)
                    vhat, log_q = model.bij_model.inverse_and_log_det(inj_v)
                    log_q += model.bij_model.q0.log_prob(vhat, distr_idx, targets)
                    loss = -torch.mean(log_q)

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
                scaler.unscale_(mse_optimizer)
                scaler.unscale_(nll_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step optimizer and update
        if epoch <= n_steps_mse:
            scaler.step(mse_optimizer)
        else:
            scaler.step(nll_optimizer)
        scaler.update()
        # optimizer.step()

        # flush gradients to release memory
        nll_optimizer.zero_grad(set_to_none=True)
        mse_optimizer.zero_grad(set_to_none=True)

        # Step the scheduler at the end of the epoch
        if epoch <= n_steps_mse:
            mse_scheduler.step()
            lr = mse_scheduler.get_last_lr()[0]
        else:
            nll_scheduler.step()
            lr = nll_scheduler.get_last_lr()[0]

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

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
                # for batch_idx, (
                #     val_images,
                #     distr_idx,
                #     targets,
                #     meta_labels,
                # ) in enumerate(val_loader):
                #     val_images = val_images.to(device)
                #     reconstructed, latent_mu, latent_logvar = model(
                #         val_images
                #     )  # Model forward pass

                #     loss = loss_function(
                #         reconstructed,
                #         val_images,
                #         latent_mu,
                #         latent_logvar,
                #         log_sigma_x=log_sigma_x,
                #         capacity=current_capacity,
                #         beta=beta,
                #     )  # Custom VAE loss function
                #     val_loss += loss.item()

                #     if debug:
                #         break

                # pick the first 8 images in the last val batch
                # sample_images = val_images[:8]  # Pick 8 images for sampling
                sample_images = train_images[:8]  # Pick 8 images for sampling

                # Standard VAE
                reconstructed_images, _, _ = model(sample_images)
                # reconstructed_images = model.decode(encoding).reshape(-1, 3, img_size, img_size)
                reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

                # sample images from VAE
                # 1. Sample latent variables from standard Gaussian
                num_samples = 8  # Number of images to generate
                z = torch.randn(num_samples, latent_dim).to(device)  # Sample z ~ N(0, I)

                # 2. Pass the sampled z through the decoder
                generated_images = raw_model.decode(z)  # Shape: [num_samples, 3, 128, 128]

                # clamp
                generated_images = torch.clamp(generated_images, 0, 1)

                # now sample training images and then reconstruct
                encoding = raw_model.encode(train_images)
                train_reconstructed_images = raw_model.decode(encoding).reshape(
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
            top_k_saver.save_model(raw_model, nll_optimizer, epoch, train_loss)
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
                "optimizer_state_dict": nll_optimizer.state_dict(),
                "epoch": epoch,  # Optional: Save the current epoch
                "loss": loss,  # Optional: Save the last loss value
            },
            checkpoint_dir / model_fname,
        )
        print(f"Training complete. Models saved in {checkpoint_dir}.")
    dist.barrier()
    if ddp:
        dist.destroy_process_group()

    # Load back the saved final model and verify that it loads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage).to(device)
    model_path = checkpoint_dir / model_fname
    vae_model = load_model(vae_model, model_path, device, optimizer=nll_optimizer)
