import shutil
import os
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
import lightning as pl

from ciflows.datasets.causalmnist import CausalMNISTEmbedding, CausalMNIST
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.eval import load_model
from ciflows.reduction.resnetvae_mnist import DeepResNetMNISTVAE
from ciflows.ncm import GAN_NCM, CausalGraph, MLP, GAN_NF_NCM, log
from ciflows.training import TopKModelSaver, delete_old_checkpoints


def make_gan_ncm_model(
    latent_dim=32,
) -> GAN_NCM:
    V_list = ["style", "digit", "digit-color", "bar-color"]
    default_u_size = int(latent_dim / len(V_list))
    default_v_size = int(latent_dim / len(V_list))
    hyperparams = {
        "h-layers": 2,
        "h-size": 128,
        "layer-norm": False,
        "neural-pu": False,
        "single-disc": True,
        "do-var-list": ["digit-color", "bar-color", "bar-color"],
    }

    directed_edges = [
        ("digit", "digit-color"),
        ("digit-color", "bar-color"),
        ("digit", "style"),
    ]
    bidirected_edges = [
        ("style", "digit"),
    ]
    cg = CausalGraph(V_list, directed_edges=directed_edges, bidirected_edges=bidirected_edges)

    gan_model = GAN_NCM(
        cg=cg,
        default_u_size=default_u_size,
        default_v_size=default_v_size,
        # f=
        hyperparams=hyperparams,
        disc_module=MLP,
        default_gen_module=MLP,
        gen_use_sigmoid=True,
        disc_use_sigmoid=True,
    )

    return gan_model


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
    """Loads CausalMNIST dataset with stratified sampling.

    Loads both the VAE-32-dim embeddings and the original CausalMNIST (32, 32, 3) datasets.
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


# ---------------------------
# Gradient Penalty Function for WGAN-GP
# ---------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)
    # Random weight term for interpolation between real and fake samples
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    # Create interpolated samples
    interpolates = epsilon * real_samples + ((1 - epsilon) * fake_samples)
    interpolates.requires_grad_(True)

    # Get critic output for the interpolated images
    d_interpolates = critic(interpolates)

    # For each sample, compute gradients of outputs with respect to inputs
    grad_outputs = torch.ones(d_interpolates.size(), device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    # Compute penalty: (||gradients||_2 - 1)^2
    gradient_penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gradient_penalty


# ---------------------------
# Main Training Script
# ---------------------------
def train_wgan_gp(
    gan_model: GAN_NF_NCM,
    dataloader,
    device,
    optimizer_critic,
    optimizer_generator,
    max_epochs=1000,
    n_critic=5,
    lambda_gp=10,
    save_interval=10,
    gan_mode="wgan",
    checkpoint_dir="./checkpoints",
    master_process=True,
    tensorboard_log_dir=None,
    debug=False,
):
    """
    Train the GAN_NCM model using the Wasserstein GAN loss with gradient penalty.

    Parameters:
      gan_model (GAN_NCM): The instantiated GAN model (generator & discriminator(s)).
      dataloader (DataLoader): DataLoader for the colored-MNIST dataset.
      device (torch.device): Device to train on.
      n_epochs (int): Number of training epochs.
      lr (float): Learning rate for both generator and critic optimizers.
      n_critic (int): Number of critic updates per generator update.
      lambda_gp (float): Gradient penalty coefficient.
      save_interval (int): Interval (in epochs) to save checkpoints.
      checkpoint_dir (str): Directory to save checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup TensorBoard writer if logging is enabled
    writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None

    # Assume that the discriminator is accessible via gan_model.f_disc
    # and the generator via gan_model.gens.
    # For simplicity, we assume a single critic (single_disc=True).
    critic = gan_model.f_disc

    # Set models to training mode and move to device
    gan_model.to(device).train()
    # generator.to(device).train()

    if debug:
        max_epochs = 5
        save_interval = 1

    # Begin training loop
    for epoch in tqdm(range(1, max_epochs + 1), desc="Epochs"):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            total_loss_critic = 0.0
            # -----------------------------
            #  Train Critic / Discriminator
            # -----------------------------
            for d_iter in range(n_critic):
                optimizer_critic.zero_grad()

                for i in range(len(gan_model.delta_v_list)):
                    # Generate fake images using the generator from distribution index i
                    # and obtain the mixture X
                    sample_mixture = gan_model.sample_mixture(n=batch_size, idx=[i], include_v=True)
                    fake_imgs_batch = sample_mixture[0]
                    sampled_v_latents = sample_mixture[1:]

                    # Critic outputs for real and fake images.
                    real_validity = gan_model.get_disc_outputs(real_imgs, index=i)
                    fake_validity = gan_model.get_disc_outputs(fake_imgs_batch, index=i)

                    # Compute gradient penalty.
                    gp = compute_gradient_penalty(
                        critic, real_imgs.data, fake_imgs_batch.data, device, lambda_gp
                    )

                    # Wasserstein critic loss
                    if gan_mode == "wgan" or gan_mode == "wgan-gp":
                        loss_critic = -fake_validity.mean() - real_validity.mean()
                    else:
                        loss_critic = -torch.mean(log(fake_validity) + log(1 - real_validity))
                        # loss_critic = -F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity)) + \
                        #               F.binary_cross_entropy_with_logits(real_validity, torch.zeros_like(real_validity))

                    loss_critic += gp
                    loss_critic.backward()
                    total_loss_critic += loss_critic.item()

                # update the critic with backprop
                optimizer_critic.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            total_loss_gen = 0.0
            optimizer_generator.zero_grad()
            for g_iter in range(len(gan_model.delta_v_list)):
                ncm_batch = gan_model.sample_mixture(n=batch_size, idx=[g_iter])[0]
                ncm_disc_fake_output = gan_model.get_disc_outputs(ncm_batch, index=g_iter)

                if gan_mode == "wgan" or gan_mode == "wgan-gp":
                    loss_generator = -torch.mean(ncm_disc_fake_output)
                else:
                    loss_generator = F.binary_cross_entropy_with_logits(
                        ncm_disc_fake_output, torch.ones_like(ncm_disc_fake_output)
                    )

                # normalize by the number of distributions it must fit
                loss_generator /= len(gan_model.delta_v_list)

                loss_generator.backward()
                total_loss_gen += loss_generator.item()

            # Update generator with backprop
            optimizer_generator.step()

            # Print training status every few iterations.
            if batch_idx % 50 == 0 and master_process:
                print(
                    f"[Epoch {epoch}/{max_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_critic.item():.4f}] [G loss: {loss_generator.item():.4f}]"
                )

        # TensorBoard Logging
        if writer:
            global_step = epoch * len(dataloader) + batch_idx

            # Log losses
            writer.add_scalar("Loss/Critic", total_loss_critic / n_critic, global_step)
            writer.add_scalar("Loss/Generator", total_loss_gen, global_step)

            # Log generated images at intervals
            if epoch % save_interval == 0:
                with torch.no_grad():
                    for idx, delta_v in enumerate(gan_model.delta_v_list):
                        # Generate samples for the current distribution index
                        sample_imgs = gan_model.sample_mixture(n=16, idx=[idx])[0]

                        # Create grid of images
                        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, nrow=4)

                        # Log each distribution separately
                        writer.add_image(f"Generated Samples/Distribution_{idx}", grid, global_step)

        # Save checkpoints every save_interval epochs.
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "ncm_model_state_dict": gan_model.state_dict(),
                    "optimizer_generator_state_dict": optimizer_generator.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    # Load configuration
    root = Path("/local/eb/adam2392/")
    root = Path("/Users/adam2392/pytorch_data/")
    ncm_config_path = "/home/adam2392/projects/ciflows/scripts/ncm/ncm_vae_experiment.yml"
    ncm_config_path = "/Users/adam2392/Documents/ciflows/scripts/ncm/ncm_vae_experiment.yml"
    ncm_config = load_experiment_config(ncm_config_path)

    # Load VAE Configuration
    vae_checkpoint_dir = (
        Path(ncm_config["vae"]["checkpoint_dir"]) / ncm_config["vae"]["experiment_name"]
    )
    vae_config_path = vae_checkpoint_dir / "experiment_config.yaml"
    vae_config = load_experiment_config(vae_config_path)

    # Paths and Experiment Settings
    exp_name = ncm_config["exp_name"]
    model_root = root / "CausalMNIST" / "ncm" / exp_name
    model_root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the NF config
    config_save_path = model_root / "experiment_config.yaml"
    shutil.copy(ncm_config_path, config_save_path)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(model_root / "logs"))

    # System settings
    sys_cfg = ncm_config["system"]
    seed = sys_cfg.get("seed", 1234)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Load device settings
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
    use_amp = ncm_config["training"].get("use_amp", True)  # Enable AMP by default
    grad_accum_steps = ncm_config["training"]["grad_accum_steps"]

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
        nullcontext()
        if accelerator in ("cpu", "mps")
        else torch.autocast(device_type=accelerator, dtype=ptdtype)
    )
    ctx = nullcontext()

    # Load Pretrained VAE Model
    vae_checkpoint_dir = (
        Path(ncm_config["vae"]["checkpoint_dir"]) / ncm_config["vae"]["experiment_name"]
    )
    vae_checkpoint_file = vae_checkpoint_dir / ncm_config["vae"]["vae_checkpoint_fname"]

    img_size = ncm_config["data"]["img_size"]
    vae_model = DeepResNetMNISTVAE(
        latent_dim=vae_config["model"]["latent_dim"],
        num_blocks_per_stage=vae_config["model"]["num_blocks_per_stage"],
    )
    vae_model, _ = load_model(vae_model, vae_checkpoint_file, device)
    vae_model.to(device)

    # Optimizer and Scheduler for VAE - optional
    # optimizer_vae = configure_optimizers(
    #     vae_model,
    #     learning_rate=ncm_config["optimizer"]["lr"],
    #     betas=tuple(ncm_config["optimizer"]["betas"]),
    #     weight_decay=ncm_config["optimizer"]["weight_decay"],
    # )

    if master_process:
        print(
            f"Training model with {world_size} GPUs on {device} with dtype {dtype} and AMP {use_amp}"
        )
        print(f"Batch size: {ncm_config['data']['batch_size']}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Loading VAE model from {vae_checkpoint_file}")

    # DataLoader
    train_loader, orig_train_loader = data_loader(
        root_dir=root,
        dataset=ncm_config["data"]["graph_type"],
        graph_type=ncm_config["data"]["graph_type"],
        num_workers=ncm_config["data"]["num_workers"],
        batch_size=ncm_config["data"]["batch_size"],
        img_size=ncm_config["data"]["img_size"],
    )

    # Load NF Checkpoint if Available
    ncm_checkpoint_dir = root / "CausalMNIST" / "ncm" / ncm_config["exp_name"]

    # Model Saving Utility
    top_k_saver = TopKModelSaver(ncm_checkpoint_dir, k=5)
    # top_k_saver_vae = TopKModelSaver(nf_checkpoint_dir, k=5)

    # Enable mixed precision training if AMP is enabled
    scaler = torch.GradScaler(device=device, enabled=(dtype == "float16"))

    grad_clip = ncm_config["training"]["grad_clip"]

    max_epochs = ncm_config["training"]["max_epochs"]

    #  make the GAN model
    gan_model = make_gan_ncm_model(latent_dim=32)
    lr = ncm_config["optimizer"]["lr"]
    if ncm_config["optimizer"]["gan_mode"] == "wgan":
        optim_func = torch.optim.RMSprop
    else:
        optim_func = torch.optim.Adam
    optimizer_critic = optim_func(gan_model.discriminator_parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_generator = optim_func(gan_model.generator_parameters(), lr=lr, betas=(0.0, 0.9))
    # scheduler = CosineAnnealingLR(
    #     optimizer_nf, T_max=ncm_config["training"]["max_epochs"], eta_min=1e-5
    # )

    start_epoch = 1
    # TODO: need to load the critic optimizer state dict as well
    if ncm_config.get("load_from_checkpoint", False):
        ncm_checkpoint_file = ncm_checkpoint_dir / ncm_config["checkpoint_file"]
        gan_model, start_epoch = load_model(
            gan_model, ncm_checkpoint_file, device, optimizer=optimizer_generator
        )

        print(f"Loaded model from {ncm_checkpoint_file} and starting from {start_epoch} epoch")

        # Synchronize all processes
        if ddp:
            torch.distributed.barrier()

    # Wrap with DDP if needed
    if ddp:
        gan_model = DDP(gan_model, device_ids=[ddp_local_rank])
        vae_model = DDP(vae_model, device_ids=[ddp_local_rank])

    if master_process:
        print(
            f"Training NF model with {sum(p.numel() for p in gan_model.parameters() if p.requires_grad):,} parameters"
        )

    train_wgan_gp(
        gan_model,
        orig_train_loader,
        device,
        max_epochs=max_epochs,
        optimizer_critic=optimizer_critic,
        optimizer_generator=optimizer_generator,
        n_critic=1,
        lambda_gp=10,
        save_interval=10,
        checkpoint_dir=ncm_checkpoint_dir,
        tensorboard_log_dir=ncm_checkpoint_dir / "logs",
        debug=ncm_config["debug"],
    )
    # for epoch in tqdm(range(start_epoch, max_epochs), desc="Epochs"):
    #     nf_model.train()
    #     total_loss = 0.0
    #     total_vae_loss = 0.0
    #     train_iterator = iter(orig_train_loader)

    #     for step, batch in enumerate(train_iterator):
    #         with ctx:
    #             imgs, distr_idx, targets, meta_labels = batch
    #             imgs = imgs.to(device)
    #             distr_idx = distr_idx.to(device)
    #             targets = targets.to(device)
    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 latent_embeddings = vae_model.module.encode(imgs)
    #                 ll_loss = nf_model.module.forward_kld(
    #                     latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
    #                 )
    #                 inv_latent = nf_model.module.inverse(latent_embeddings)
    #                 inv_latent += torch.randn_like(inv_latent).to(device) * 1e-5
    #                 recon_latent_embeddings = nf_model.module.forward(inv_latent)
    #                 recon_imgs = vae_model.module.decode(recon_latent_embeddings)

    #                 # Compute the reconstruction loss.
    #                 rec_loss = F.mse_loss(recon_imgs, imgs)
    #             else:
    #                 latent_embeddings = vae_model.encode(imgs)
    #                 ll_loss = nf_model.forward_kld(
    #                     latent_embeddings, intervention_targets=targets, distr_idx=distr_idx
    #                 )
    #                 inv_latent = nf_model.inverse(latent_embeddings)
    #                 inv_latent += torch.randn_like(inv_latent).to(device) * 1e-5
    #                 recon_latent_embeddings = nf_model.forward(inv_latent)
    #                 recon_imgs = vae_model.decode(recon_latent_embeddings)

    #                 # Compute the reconstruction loss.
    #                 rec_loss = F.mse_loss(recon_imgs, imgs)

    #             total_vae_loss = ll_loss + rec_loss
    #             total_vae_loss /= grad_accum_steps  # Scale loss for gradient accumulation

    #         # Backpropagation with AMP
    #         scaler.scale(total_vae_loss).backward(retain_graph=True)

    #         if (step + 1) % grad_accum_steps == 0:
    #             if grad_clip != 0.0 and scaler.is_enabled():
    #                 scaler.unscale_(optimizer_nf)
    #                 scaler.unscale_(optimizer_vae)
    #             torch.nn.utils.clip_grad_norm_(nf_model.parameters(), grad_clip)
    #             scaler.step(optimizer_nf)
    #             scaler.step(optimizer_vae)
    #             scaler.update()
    #             optimizer_nf.zero_grad()
    #             optimizer_vae.zero_grad()

    #         total_loss += total_vae_loss.item()

    #     scheduler.step()

    #     # Save model periodically
    #     if master_process and epoch % ncm_config["training"]["save_every"] == 0:
    #         # torch.save(nf_model.state_dict(), nf_checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
    #         top_k_saver.save_model(nf_model, optimizer_nf, epoch, total_loss)
    #         top_k_saver_vae.save_model(vae_model, optimizer_vae, epoch, total_loss)
    #         delete_old_checkpoints(ncm_checkpoint_dir, keep_top_k=10)
    #         print(f"Checkpoint saved at {ncm_checkpoint_dir}")

    #     # Image Generation & Latent Visualization
    #     if master_process and epoch % ncm_config["training"]["n_log_steps"] == 0:
    #         nf_model.eval()
    #         with torch.no_grad():
    #             # 1 & 2. Training images reconstructed using VAE and NF+VAE
    #             recon_vae = vae_model.decode(latent_embeddings[:8]).reshape(
    #                 -1, 3, img_size, img_size
    #             )
    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 nf_recon_latents = nf_model.module.forward(
    #                     nf_model.module.inverse(latent_embeddings[:8])
    #                 )
    #             else:
    #                 nf_recon_latents = nf_model.forward(nf_model.inverse(latent_embeddings[:8]))
    #             recon_nf_vae = vae_model.decode(nf_recon_latents).reshape(-1, 3, img_size, img_size)

    #             # Stack reconstructions vertically
    #             recon_stack = torch.cat(
    #                 [recon_vae, recon_nf_vae], dim=0
    #             )  # Stacking along batch dimension
    #             save_image(
    #                 recon_stack.cpu(),
    #                 ncm_checkpoint_dir / f"epoch_{epoch}_reconstructions.png",
    #                 nrow=8,
    #                 normalize=True,
    #             )

    #             # 3 & 4. Sample images from VAE and NF decoded by VAE
    #             sample_latents = torch.randn((8, 32), device=device)
    #             sample_vae = vae_model.decode(sample_latents).reshape(-1, 3, img_size, img_size)

    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=0)
    #             else:
    #                 sample_nf_latents, _ = nf_model.sample(8, distr_idx=0)
    #             sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
    #             # Stack generated samples vertically
    #             sample_stack = torch.cat([sample_vae, sample_nf], dim=0)

    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=1)
    #             else:
    #                 sample_nf_latents, _ = nf_model.sample(8, distr_idx=1)
    #             sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
    #             # Stack generated samples vertically
    #             sample_stack = torch.cat([sample_stack, sample_nf], dim=0)

    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 sample_nf_latents, _ = nf_model.module.sample(8, distr_idx=2)
    #             else:
    #                 sample_nf_latents, _ = nf_model.sample(8, distr_idx=2)
    #             sample_nf = vae_model.decode(sample_nf_latents).reshape(-1, 3, img_size, img_size)
    #             # Stack generated samples vertically
    #             sample_stack = torch.cat([sample_stack, sample_nf], dim=0)

    #             save_image(
    #                 sample_stack.cpu(),
    #                 ncm_checkpoint_dir / f"epoch_{epoch}_samples.png",
    #                 nrow=8,
    #                 normalize=True,
    #             )

    #             # 5. Visualizing NF Latent Distribution
    #             if isinstance(nf_model, torch.nn.parallel.DistributedDataParallel):
    #                 nf_latent_distr = nf_model.module.inverse(latent_embeddings)
    #             else:
    #                 nf_latent_distr = nf_model.inverse(latent_embeddings)
    #             for dim in range(32):
    #                 writer.add_histogram(
    #                     f"NF_Latents/dim_{dim}",
    #                     nf_latent_distr[:, dim].cpu().numpy().squeeze(),
    #                     epoch,
    #                 )

    #     # TensorBoard Logging
    #     if master_process:
    #         writer.add_scalar("Loss/Train", total_loss, epoch)

    # Save Final Model
    if master_process:
        torch.save(gan_model.state_dict(), ncm_checkpoint_dir / "final_model.pt")
        torch.save(vae_model.state_dict(), ncm_checkpoint_dir / "final_vae_model.pt")

    print("Training Complete!")
