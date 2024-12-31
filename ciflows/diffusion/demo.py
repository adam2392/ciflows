from copy import copy
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt


import torch
from ciflows.diffusion.ddpm.wrapper import DDPMWrapper
from ciflows.diffusion.ddpm.ddpm_form2 import DDPMv2
from ciflows.diffusion.ddpm.ddpm import DDPM
from ciflows.diffusion.ddpm.unet_openai import SuperResModel
from ciflows.diffusion.ddpm.vae import VAE as BACKVAE


if __name__ == "__main__":
    print(torch.__version__)

    root = Path("/Users/adam2392/pytorch_data/")
    root = Path("/local/eb/adam2392/")
    seed_offset = 1

    # set seed
    seed = 1234
    np.random.seed(seed + seed_offset)
    pl.seed_everything(seed + seed_offset, workers=True)

    image_size = 128
    latent_dim = 48
    num_blocks_per_stage = 3
    debug = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"

        # Get the index of the current device
        device_index = 7
        device = torch.device(f"cuda:{device_index}")
        print(f"Current CUDA device index: {device_index}")

        # Get the name of the current device
        device_name = torch.cuda.get_device_name(device_index)
        print(f"Current CUDA device name: {device_name}")

        num_devices = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        print("NUm devices: ", num_devices)
        print(device_names)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    # 128x128
    # load DDPM and apply upscaling to image
    img_size = 128
    attn_resolutions = [16]
    dim_mults = [1, 2, 2, 3, 4]
    n_channels = 3
    dim = 128
    n_residual = 2
    dropout = 0.0
    n_heads = 1
    z_dim = 1024
    type = "form1"
    beta1 = 0.0001
    beta2 = 0.02
    n_timesteps = 1000
    variance = "fixedsmall"
    guidance_weight = 0.0
    resample_strategy = "spaced"
    skip_strategy = "uniform"
    sample_method = "ddpm"
    sample_from = "target"
    seed = 0
    n_samples = 50000
    n_steps = 1000
    # n_steps = 2
    workers = 2
    save_vae = False
    sample_prefix = ""
    temp = 1.0
    norm = True
    chkpt_path = "/local/eb/adam2392/diffvae_chq128_form1_loss=0.0066.ckpt"
    vae_chkpt_path = "/local/eb/adam2392/vae_chq128_alpha=1.0_loss=0.0000.ckpt"

    z_cond = False

    torch.cuda.empty_cache()
    ddpm_device = torch.device(f"cuda:{5}")

    vae_backup = BACKVAE.load_from_checkpoint(
        vae_chkpt_path,
        input_res=64,
        map_location=ddpm_device,
    )
    vae_backup.eval()
    print("Loaded VAE backup.")

    decoder = SuperResModel(
        in_channels=n_channels,
        model_channels=dim,
        out_channels=3,
        num_res_blocks=n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=dropout,
        num_heads=n_heads,
        z_dim=z_dim,
        use_scale_shift_norm=z_cond,
        use_z=z_cond,
    ).to(ddpm_device)
    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    print("Loaded decoder model.")

    ddpm_cls = DDPMv2 if type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=beta1,
        beta_2=beta2,
        T=n_timesteps,
        var_type=variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=beta1,
        beta_2=beta2,
        T=n_timesteps,
        var_type=variance,
    )

    online_ddpm.eval()
    target_ddpm.eval()

    print("Loading online and target DDPM.")

    # device = torch.cuda.get_device_name(5)
    n_steps = 1000
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae_backup,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="vae_sample",
        resample_strategy=resample_strategy,
        skip_strategy=skip_strategy,
        sample_method=sample_method,
        sample_from=sample_from,
        data_norm=norm,
        temp=temp,
        guidance_weight=guidance_weight,
        z_cond=z_cond,
        ddpm_latents=None,
        strict=True,
        map_location=ddpm_device,
    )
    # ddpm_wrapper.to(ddpm_device)
    ddpm_wrapper.eval()

    print("Loading DDPM wrapper.")

    torch.cuda.empty_cache()

    # load in VAE-NF images?
    # recons_inter = reconstructed_imgs.to(ddpm_device)
    # n_steps = 10
    recons_inter = torch.rand((n_steps, 3, img_size, img_size), device=ddpm_device)

    # sample random noise over the latent space to feed to diffusion
    x_t = temp * torch.randn_like(recons_inter, device=ddpm_device)

    print(n_steps, recons_inter.shape, x_t.shape)

    # ddpm_samples = apply_ddpm_wrapper(ddpm_wrapper, x_t, recons_inter=recons_inter, img_size=img_size, n_steps=n_steps)
    # print(ddpm_samples.shape)
    ddpm_latents = torch.randn((n_steps, 3, img_size, img_size), device=ddpm_device)
    # ddpm_latents = None
    ddpm_sample_do = ddpm_wrapper(
        x_t,
        cond=recons_inter,
        z=None,
        n_steps=n_steps,
        ddpm_latents=ddpm_latents,
    )[str(n_steps)].cpu()
    print(ddpm_sample_do.shape)
