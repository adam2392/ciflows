from pathlib import Path

import lightning as pl
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from ciflows.datasets.causalceleba_scm.pretrained import MultiTaskResNet
from ciflows.eval import load_model

if __name__ == "__main__":
    seed = 1234

    # set seed
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    debug = True
    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")
        # root = Path("/Users/adam2392/pytorch_data/")

    output_dir = root / "CausalCelebA" / "pretrained" / "generated_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the pretrained model
    generator = torch.hub.load(
        "facebookresearch/pytorch_GAN_zoo:hub",
        "PGAN",
        model_name="celebAHQ-512",
        pretrained=True,
        useGPU=True,
    )
    generator.eval()
    latent_dim = 512

    # load the pretrained classifier
    # v1: K=32
    # v2: K=8
    # v3: K=8, batch higher
    model_fname = "celeba_predictor_batch256_v1.pt"

    # checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / "latentdim24"
    checkpoint_dir = root / "CausalCelebA" / "pretrained" / model_fname.split(".")[0]
    classifier = MultiTaskResNet().to(device)
    model_path = checkpoint_dir / "model_fname"
    classifier = load_model(classifier, model_path, device)
    classifier.eval()

    num_images = 10

    n_images = 0

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Observational setting:
    # U_{g,a} = U[0, 1]
    # P(gender == male) = U_{g,a}
    # P(age == old) = U_{g,a}
    # P(hair == black | age == old) = 0.2
    # P(hair == brown | age == old) = 0.25
    # P(hair == blond | age == old) = 0.25
    # P(hair == gray | age == old) = 0.3
    # P(hair == black | age == young) = 0.3
    # P(hair == brown | age == young) = 0.25
    # P(hair == blond | age == young) = 0.25
    # P(hair == gray | age == young) = 0.2

    while n_images < num_images:
        # Function to generate images
        noise = torch.randn(num_images, latent_dim, 1, 1)

        with torch.no_grad():
            fake_images = generator(noise)

            # predict on the images
            fake_images = transform(fake_images)

            # Forward pass
            (gender_out, gender_prob), (hair_out, hair_prob), (age_out, age_prob) = (
                classifier(fake_images)
            )

            # now decide which images to save
            # we apply an algorithm to ensure that:
            # P(gender) is what we want
            # P(age) is what we want
            # P(hair | age) is what we want
            for i in range(num_images):
                save_image(fake_images[i], output_dir / f"image_{i}.png")
