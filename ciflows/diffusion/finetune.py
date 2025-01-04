import copy

import torch
from torchvision import transforms

from .ddpm.ddpm import DDPM
from .ddpm.ddpm_form2 import DDPMv2
from .ddpm.unet_openai import SuperResModel
from .ddpm.vae import VAE as BACKVAE
from .ddpm.wrapper import DDPMWrapper


def apply_ddpm_wrapper(ddpm_wrapper, x_t, recons_inter, img_size=128, n_steps=1000):
    device = x_t.device

    ddpm_latents = torch.randn(n_steps, 3, img_size, img_size).to(device)
    ddpm_sample_do = ddpm_wrapper(
        x_t,
        cond=recons_inter,
        z=None,
        n_steps=n_steps,
        ddpm_latents=ddpm_latents,
    )[str(n_steps)].cpu()
    return ddpm_sample_do


if __name__ == "__main__":
    device = "cuda"

    img_size = 128
    attn_resolutions = "16,"
    dim_mults = "1,2,2,3,4"
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
    workers = 2
    batch_size = 8
    save_vae = False
    sample_prefix = ""
    temp = 1.0
    norm = True

    z_cond = False

    chkpt_path = "out/diffusion/diffvae_celeba64_form1_loss=0.0146.ckpt"
    vae_chkpt_path = "out/diffusion/vae_celeba64_loss=0.0000.ckpt"

    vae_backup = BACKVAE.load_from_checkpoint(
        vae_chkpt_path,
        input_res=64,
    )
    vae_backup.eval()

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
    )
    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

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
    )

    trans_f = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(148),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    ddpm_wrapper.to(device)
    ddpm_wrapper.eval()

    # load in VAE-NF images?
    recons_inter = torch.load("out/diffusion/recons_inter_do.pt")

    # sample random noise over the latent space to feed to diffusion
    x_t = (temp * torch.randn_like(recons_inter)).to(device)

    # apply DDPM and obtain the new images over the entire batch
    new_batch_imgs = apply_ddpm_wrapper(
        ddpm_wrapper, x_t, recons_inter, img_size=img_size, n_steps=1000
    )
