from csv import writer
import os
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from ciflows.datasets.causalmnistv2 import CausalMNIST_v2
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.ncm.bigan import Decoder, Encoder, JointDiscriminator


def gradient_penalty(D, x_real, z_real, x_fake, z_fake, lambda_=10, device='cpu'):
    alpha = torch.rand(x_real.size(0), 1, 1, 1, device=device)
    interpolated_x = alpha * x_real + (1 - alpha) * x_fake
    alpha_z = alpha.view(-1, 1)  # broadcast for z
    interpolated_z = alpha_z * z_real + (1 - alpha_z) * z_fake
    interpolated_x.requires_grad_(True)
    interpolated_z.requires_grad_(True)
    prob = D(interpolated_x, interpolated_z)
    grads = torch.autograd.grad(
        outputs=prob, inputs=[interpolated_x, interpolated_z],
        grad_outputs=torch.ones_like(prob),
        create_graph=True, retain_graph=True
    )
    # combine grads of x and z
    grad_norm = torch.sqrt(
        grads[0].view(grads[0].size(0), -1).pow(2).sum(1) +
        grads[1].view(grads[1].size(0), -1).pow(2).sum(1)
    )
    return lambda_ * ((grad_norm - 1) ** 2).mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_bigan(
    root_dir,
    output_dir,
    distr_labels,
    epochs=100,
    batch_size=64,
    z_dim=32,
    gen_lr=1e-4,
    disc_lr=1e-5,
    log_every=10,
    val_split=0.1,
    save_top_k=5,
    multi_gpu=False,
    use_scheduler=False,
    device=None,
    num_workers=4,
    img_size=32,
    gp_lambda=10.,
    lambda_recon    = 10.0,
    blocks_per_stage=2,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # create a TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    image_transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), lambda x: x.expand(3, -1, -1)]
    )

    # dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Initialize the dataset (replace with your desired dataset settings)
    # Define image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = CausalMNIST_v2(
        root=root_dir,
        distr_labels=distr_labels,
        # graph_type=graph_type,
        transform=image_transform,
    )

    # print("The distribution index labels: ", distr_labels)
    orig_distr_labels_list = dataset.get_distribution_labels()

    val_len = int(val_split * len(dataset))
    train_len = len(dataset) - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    distr_labels_list = [orig_distr_labels_list[idx] for idx in train_set.indices]
    train_sampler = StratifiedSampler(distr_labels_list, batch_size)
    train_loader = DataLoader(
        shuffle=False,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        shuffle=False,
        dataset=dataset,
        batch_size=batch_size,
        # sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    default_channels = [32, 64, 128]
    encoder = Encoder(z_dim=z_dim, img_channels=3, img_size=32, channels=default_channels, blocks_per_stage=blocks_per_stage)
    decoder = Decoder(z_dim=z_dim, img_channels=3, img_size=32, channels=default_channels, blocks_per_stage=blocks_per_stage)
    discriminator = JointDiscriminator(z_dim=z_dim, img_channels=3, img_size=32, channels=default_channels, blocks_per_stage=blocks_per_stage)

    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    discriminator_params = count_parameters(discriminator)

    print(f"Encoder parameters:       {encoder_params:,}")
    print(f"Decoder parameters:       {decoder_params:,}")
    print(f"Discriminator parameters: {discriminator_params:,}")

    if multi_gpu and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        discriminator = nn.DataParallel(discriminator)

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    gen_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=gen_lr)
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

    if use_scheduler:
        scheduler = ReduceLROnPlateau(gen_opt, mode="min", factor=0.5, patience=5)

    best_models = []  # list of tuples: (loss, path)

    n_critic = 5  # (i.e., skip every other nth step for the critic learning)

    def add_instance_noise(x, sigma):
        return x + sigma * torch.randn_like(x)

    print(f"About to start training with {len(train_loader)} batches...")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Z dimension: {z_dim}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}...")

        encoder.train(), decoder.train(), discriminator.train()
        total_d_loss, total_g_loss = 0, 0

        for batch_idx, (x, meta_labels) in enumerate(train_loader):
            sigma = max(0.1 * (1 - epoch/epochs), 0.01)
            x = add_instance_noise(x.to(device), sigma)
            # if idx % 20 == 0:
            #     print(f"Batch {idx}/{len(train_loader)}...")
            #     print(f"Meta Labels: {len(meta_labels)}, {meta_labels.keys()}")

            # x = x.to(device)
            batch_size = x.size(0)

            # Encode
            z_enc = encoder(x)
            x_recon = decoder(z_enc)

            # Reconstruction loss (tune λ_recon)
            l2_recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # l1 loss
            recon_loss = F.l1_loss(x_recon, x, reduction='mean') + l2_recon_loss

            d_loss_accum = 0
            # update discriminator less often
            for _ in range(n_critic):
                # Random latent
                z_rand = torch.randn(batch_size, z_dim).to(device)

                # detatch from decoder to avoid backprop through it
                # to the generator's decoder/encoder
                x_rand = decoder(z_rand).detach()

                # Discriminator step
                dis_opt.zero_grad()
                real_score = discriminator(x, z_enc.detach())
                fake_score = discriminator(x_rand, z_rand)

                # 1) Wasserstein critic loss
                d_wass = fake_score.mean() - real_score.mean()

                # 2) gradient penalty
                gp = gradient_penalty(
                    discriminator,
                    x, z_enc.detach(),
                    x_rand, z_rand,
                    lambda_=gp_lambda,       # you can tune λ
                    device=device
                )

                d_loss = d_wass + gp
                d_loss.backward()
                dis_opt.step()
                d_loss_accum += d_loss.item()
            total_d_loss += d_loss_accum
            
            # Generator step
            # ——— Generator (Wasserstein) ———
            gen_opt.zero_grad()
            z_g = torch.randn(batch_size, z_dim, device=device)
            x_fake_for_g = decoder(z_g)   # DO NOT detach

            # 1) adversarial term
            g_wass = -discriminator(x_fake_for_g, z_g).mean()

            # 2) reconstruction term (unchanged)
            g_loss = g_wass + lambda_recon * recon_loss
            g_loss.backward()
            gen_opt.step()

            total_g_loss += g_loss.item()

            # Optional: log individual components
            avg_d_loss_batch = d_loss_accum / n_critic
            writer.add_scalar("Loss/Discriminator_batch", avg_d_loss_batch, global_step)
            writer.add_scalar("Loss/G_Wasserstein_batch", g_wass.item(), global_step)
            writer.add_scalar("Loss/Recon_batch", recon_loss.item(), global_step)
            writer.add_scalar("Loss/D_Wasserstein_batch", d_wass.item(), global_step)
            writer.add_scalar("Loss/GradientPenalty_batch", gp.item(), global_step)

            global_step += 1

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)

        # Validation (recon loss)
        encoder.eval(), decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                z_val = encoder(x_val)
                x_recon = decoder(z_val)
                val_loss += F.mse_loss(x_recon, x_val, reduction="sum").item()
        val_loss /= len(val_loader.dataset)

        if use_scheduler:
            scheduler.step(val_loss)

        # ——— LOG TO TENSORBOARD ———
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Log
        print(
            f"Epoch {epoch}: D Loss = {avg_d_loss:.4f}, G Loss = {avg_g_loss:.4f}, Val Loss = {val_loss:.4f}"
        )
        if epoch % log_every == 0:
            with torch.no_grad():
                print(f"Generating samples... and saving to {sample_dir}")  
                z_sample = torch.randn(64, z_dim).to(device)
                x_sample = decoder(z_sample)
                utils.save_image(
                    x_sample,
                    os.path.join(sample_dir, f"sample_epoch_{epoch}.png"),
                    nrow=8,
                    normalize=True,
                )

                # shape: (B, C, H, W), values in [0,1]
                writer.add_images("Generated/Samples", x_sample, global_step=epoch)

        # Save top-k models
        # Inside training loop after computing val_loss
        best_models.append((val_loss, epoch, None))  # no path yet
        best_models = sorted(best_models, key=lambda x: x[0])[:save_top_k]

        # If this epoch is in top-k, save it
        if any(epoch == e for _, e, _ in best_models):
            model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                model_path,
            )

            # Replace the None path in best_models with the actual path
            best_models = [(loss, e, model_path if e == epoch else p) 
                        for loss, e, p in best_models]

        # Clean up models that are no longer in top-k
        current_paths = {p for _, _, p in best_models if p is not None}
        for fname in os.listdir(output_dir):
            path = os.path.join(output_dir, fname)
            if fname.startswith("model_epoch_") and path not in current_paths:
                print(f"Removing old model: {path}")
                os.remove(path)


def data_loader(root_dir, num_workers, batch_size, img_size, debug=False):
    """Loads CausalMNIST dataset with stratified sampling.

    Loads the original CausalMNIST (32, 32, 3) datasets.
    """
    # Initialize the dataset (replace with your desired dataset settings)
    # Define image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = CausalMNIST_v2(
        root=root_dir,
        # graph_type=graph_type,
        transform=image_transform,
    )

    distr_labels_list = dataset.get_distribution_labels()

    # print("The distribution index labels: ", distr_labels)
    train_sampler = StratifiedSampler(distr_labels_list, batch_size)

    return DataLoader(
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
    # Generate a unique timestamped subfolder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "/local/eb/adam2392/bigan_causal/initial/"
    unique_output_dir = os.path.join(base_output_dir, run_id)

    # Create the directory
    os.makedirs(unique_output_dir, exist_ok=True)

    # Configuration
    config = {
        "output_dir": unique_output_dir,
        "epochs": 500,
        "batch_size": 2048,
        "z_dim": 64,
        "gen_lr": 1e-4,
        "disc_lr": 1e-5,
        "log_every": 10,
        "val_split": 0.1,
        "save_top_k": 3,
        "multi_gpu": False, #torch.cuda.device_count() > 1,
        "use_scheduler": True,
        "device": (
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        ),
        "num_workers": 10,
        "img_size": 32,
        "lambda_recon": 1.0,  # default value if not provided
        "gp_lambda": 5.0,  # default value for gradient penalty
        "blocks_per_stage": 3,  # number of residual blocks per stage in the encoder/decoder
        "graph_type": "chain",  # or whichever causal structure you want to test
        "debug": False,
    }
    # Save config to JSON inside the output directory
    config_path = os.path.join(unique_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {config_path}")

    root = Path("/local/eb/adam2392/")
    # root = Path("/Users/adam2392/pytorch_data/")
    distr_labels = ["observational", "int_colorbar_0", "int_colorbar_1", "int_colorbar_2"]

    # Pre-warm CUDA to avoid cuBLAS warning
    if torch.cuda.is_available():
        _ = torch.tensor([0.0], device="cuda")
        torch.cuda.reset_peak_memory_stats()

    print("=================== Training Configuration ===================")
    print(f"Device: {config['device']}")
    print(f"Multi-GPU: {config['multi_gpu']} ({torch.cuda.device_count()} GPUs detected)")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Latent Dimension: {config['z_dim']}")
    print(f"Generator Learning Rate: {config['gen_lr']}")
    print(f"Discriminator Learning Rate: {config['disc_lr']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Validation Split: {config['val_split']}")
    print(f"Dataset Root: {root}")
    print(f"Causal Graph Type: {config['graph_type']}")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Initial CUDA Memory Usage: {mem:.2f} MB (Peak: {peak_mem:.2f} MB)")

    print("=============================================================")

    # Load training data
    # train_loader = data_loader(
    #     root_dir=root,
    #     # dataset='CausalMNIST',
    #     # graph_type=config['graph_type'],
    #     num_workers=config['num_workers'],
    #     batch_size=config['batch_size'],
    #     img_size=config['img_size'],
    #     debug=config['debug']
    # )

    # Start training
    train_bigan(
        root_dir=root,
        output_dir=config["output_dir"],
        distr_labels=distr_labels,
        # data_loader=train_loader,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        z_dim=config["z_dim"],
        gen_lr=config["gen_lr"],
        disc_lr=config["disc_lr"],
        log_every=config["log_every"],
        val_split=config["val_split"],
        save_top_k=config["save_top_k"],
        multi_gpu=config["multi_gpu"],
        use_scheduler=config["use_scheduler"],
        device=config["device"],
        num_workers=config["num_workers"],
        img_size=config["img_size"],
        gp_lambda=config.get("gp_lambda", 10.0),
        lambda_recon=config.get("lambda_recon", 10.0),  # default value if not provided
        blocks_per_stage=config.get("blocks_per_stage", 2),
    )
