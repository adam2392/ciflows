import os
import re
from copy import copy
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import (
    CoarseDropout,
    Compose,
    ElasticTransform,
    GridDistortion,
    HorizontalFlip,
    OneOf,
    RandomBrightnessContrast,
    HueSaturationValue,
    GaussianBlur,
    Perspective,
    RandomCrop,
)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CelebA


def exponential_weights(i_range, alpha=1.0):
    weights = [np.exp(alpha * i) for i in i_range]
    # Normalize weights so they sum to 1
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    return normalized_weights


def print_transforms(transforms, indent=0):
    for i, transform in enumerate(transforms):
        if isinstance(transform, OneOf):
            print("  " * indent + f"Transformation {i+1}: OneOf")
            print_transforms(transform.transforms, indent + 1)
        else:
            print(
                "  " * indent + f"Transformation {i+1}: {transform.__class__.__name__}"
            )


# Albumentations transformations pipeline
def get_random_transforms(image_size):
    return Compose(
        [
            OneOf(
                [
                    RandomCrop(
                        width=image_size - 6,
                        height=image_size - 6,
                        always_apply=False,
                        p=0.3,
                    ),
                    # GridDistortion(num_steps=10, p=0.3),
                    # ElasticTransform(alpha=1, sigma=25, alpha_affine=None, p=0.3),
                ],
                p=0.5,
            ),
            # OneOf(
            #     [
            #         GaussianBlur(blur_limit=(3, 3), p=0.3),
            #         Perspective(scale=(0.05, 0.1), p=0.3),
            #     ],
            #     p=0.5,
            # ),
            # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            # HueSaturationValue(
            #     # hue_shift_limit=10,
            #     # sat_shift_limit=5,
            #     # val_shift_limit=20,
            #     p=0.4,
            # ),
            CoarseDropout(
                max_holes=16,
                max_height=3,
                max_width=3,
                min_holes=1,
                min_height=1,
                min_width=1,
                p=0.3,
            ),
            HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )


def obs_sample_img_indices(
    male_attrs, young_attrs, hair_attrs, n_samples=1000, seed=None
):
    """Set up the observational SCM."""
    rng = np.random.default_rng(seed)

    sample_idx = np.arange(len(male_attrs))
    image_attrs = np.concatenate(
        (
            sample_idx.reshape(-1, 1),
            male_attrs.reshape(-1, 1),
            young_attrs.reshape(-1, 1),
            hair_attrs.reshape(-1, 1),
        ),
        axis=1,
    )

    # Precompute hair categories
    # hair_categories = np.unique(hair_attrs).astype(int).tolist()
    hair_categories = ["Black", "Blond", "Brown", "Gray"]

    # List to store sampled indices
    sampled_indices = []
    sampled_attrs = []
    for idx in range(n_samples):
        # shuffle image attributes
        rng.shuffle(sample_idx)
        image_attrs = image_attrs[sample_idx]

        # now, sample U_gh and use this to initialize the sampling process
        U_genderhair = rng.uniform()

        # now sample male based on bernoulli
        p_male = U_genderhair
        gender_str = "Male" if rng.uniform() < p_male else "Female"
        gender_map = {"Male": 1, "Female": 0}
        gender = gender_map[gender_str]

        p_old = U_genderhair
        age_str = "Old" if rng.uniform() < p_old else "Young"
        age_map = {"Young": 1, "Old": 0}
        age = age_map[age_str]

        hair_range = np.arange(4)
        if age == "Old":
            hair_range = hair_range[::-1]
        p_hairs = exponential_weights(hair_range, alpha=1.0)
        hair_str = rng.choice(hair_categories, p=p_hairs)

        # 0: black
        # 1: blond
        # 2: brown
        # 3: gray
        hair_map = {"Black": 0, "Blond": 1, "Brown": 2, "Gray": 3}
        hair = hair_map[hair_str]

        # now sample an individual that is Male, Old and X-Hair color
        matching_indices = image_attrs[
            (image_attrs[:, 1] == gender)
            & (image_attrs[:, 2] == age)
            & (image_attrs[:, 3] == hair)
        ][
            :, 0
        ].tolist()  # Extract indices

        # Sample a single individual if there are matches
        if matching_indices:
            sampled_index = rng.choice(matching_indices)
            sampled_indices.append(sampled_index)
            sampled_attrs.append((sampled_index, gender_str, age_str, hair_str))
    return sampled_indices, sampled_attrs


def interventional_sample_img_indices(
    male_attrs, young_attrs, hair_attrs, idx=0, n_samples=1000, seed=None
):
    """Set up the observational SCM."""
    # Precompute hair categories
    if idx == 0:
        hair_categories = ["Gray", "Brown"]
    elif idx == 1:
        hair_categories = ["Black", "Blond"]

    rng = np.random.default_rng(seed)

    sample_idx = np.arange(len(male_attrs))
    image_attrs = np.concatenate(
        (
            sample_idx.reshape(-1, 1),
            male_attrs.reshape(-1, 1),
            young_attrs.reshape(-1, 1),
            hair_attrs.reshape(-1, 1),
        ),
        axis=1,
    )

    # List to store sampled indices
    sampled_indices = []
    sampled_attrs = []
    for idx in range(n_samples):
        # shuffle image attributes
        rng.shuffle(sample_idx)
        image_attrs = image_attrs[sample_idx]

        # now, sample U_gh and use this to initialize the sampling process
        U_genderhair = rng.uniform()

        # now sample male based on bernoulli
        p_male = U_genderhair
        gender_str = "Male" if rng.uniform() < p_male else "Female"
        gender_map = {"Male": 1, "Female": 0}
        gender = gender_map[gender_str]

        p_old = U_genderhair
        age_str = "Old" if rng.uniform() < p_old else "Young"
        age_map = {"Young": 1, "Old": 0}
        age = age_map[age_str]

        hair_range = np.arange(2)
        if age == "Old":
            hair_range = hair_range[::-1]
        p_hairs = exponential_weights(hair_range, alpha=1.0)
        hair_str = rng.choice(hair_categories, p=p_hairs)

        # 0: black
        # 1: blond
        # 2: brown
        # 3: gray
        hair_map = {"Black": 0, "Blond": 1, "Brown": 2, "Gray": 3}
        hair = hair_map[hair_str]

        # now sample an individual that is Male, Old and X-Hair color
        matching_indices = image_attrs[
            (image_attrs[:, 1] == gender)
            & (image_attrs[:, 2] == age)
            & (image_attrs[:, 3] == hair)
        ][
            :, 0
        ].tolist()  # Extract indices

        # Sample a single individual if there are matches
        if matching_indices:
            sampled_index = rng.choice(matching_indices)
            sampled_indices.append(sampled_index)
            sampled_attrs.append((sampled_index, gender_str, age_str, hair_str))
    return sampled_indices, sampled_attrs


def celeba_scm(
    celeba_data,
    save_dir,
    image_size=64,
    scm_type="obs",
    interv_idx=0,
    n_samples=1000,
    seed=None,
    append=False,
):
    attr_names = copy(celeba_data.attr_names)
    gender_idx = attr_names.index("Male")
    age_idx = attr_names.index("Young")

    blackhair_idx = attr_names.index("Black_Hair")
    blondhair_idx = attr_names.index("Blond_Hair")
    brownhair_idx = attr_names.index("Brown_Hair")
    grayhair_idx = attr_names.index("Gray_Hair")

    # Use np.argwhere to get the indices of matching elements
    hair_cols = np.array([blackhair_idx, blondhair_idx, brownhair_idx, grayhair_idx])

    male_attrs = celeba_data.attr[:, gender_idx]
    young_attrs = celeba_data.attr[:, age_idx]
    hair_attrs = celeba_data.attr[:, hair_cols]

    # 0: black
    # 1: blond
    # 2: brown
    # 3: gray
    hair_attrs = np.argmax(hair_attrs, axis=1)

    if scm_type == "obs":
        sample_indices, sampled_attrs = obs_sample_img_indices(
            male_attrs, young_attrs, hair_attrs, n_samples=n_samples, seed=seed
        )
    else:
        sample_indices, sampled_attrs = interventional_sample_img_indices(
            male_attrs,
            young_attrs,
            hair_attrs,
            idx=interv_idx,
            n_samples=n_samples,
            seed=seed,
        )
    saved_attrs = []
    saved_causal_attrs = []

    if append:
        # Define the pattern to match the file names
        pattern = re.compile(r"sample_(\d+)\.jpg")
        max_idx = 0  # Start with a default value for empty directory
        for file_name in os.listdir(save_dir):
            match = pattern.match(file_name)
            if match:
                idx = int(match.group(1))  # Extract the number
                max_idx = max(max_idx, idx)  # Update the maximum
    else:
        max_idx = 0

    # now actually sample the images, apply transformation and save them to disc
    for idx, sample_idx in enumerate(sample_indices):
        image, meta_attrs = celeba_data[sample_idx]
        image = torch.permute(image, (1, 2, 0))

        # Apply transformations
        transform_pipeline = get_random_transforms(image_size=image_size)
        transformed = transform_pipeline(image=np.array(image))
        transformed_image = transformed["image"]

        # Convert to a PIL Image
        # Convert to a PIL Image
        transformed_image = (
            transformed_image * 0.5 + 0.5
        )  # Undo normalization (if applied)
        transformed_image = (transformed_image.numpy() * 255).astype(np.uint8)
        if transformed_image.shape[0] == 3:
            transformed_image = np.transpose(transformed_image, (1, 2, 0))
        image_pil = Image.fromarray(transformed_image)

        # Save the image as PNG
        save_path = save_dir / f"sample_{idx + max_idx}.jpg"
        image_pil.save(save_path)

        attrs = sampled_attrs[idx]

        saved_causal_attrs.append(attrs)
        saved_attrs.append(meta_attrs.numpy())
        # print(attrs)

        # if idx == 3:
        #     return transformed

    return saved_causal_attrs, saved_attrs


if __name__ == "__main__":
    # Root directory for the dataset
    data_root = Path("/Users/adam2392/pytorch_data/")
    # Spatial size of training images, images are resized to this size.
    image_size = 128
    n_samples = 20_000
    seed = 1234

    celeba_data = CelebA(
        data_root,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    scm_type = "int_hair_1"
    # scm_type = "obs"
    interv_idx = 1
    append = True

    save_dir = data_root / "CausalCelebA" / "dim128" / scm_type
    save_dir.mkdir(exist_ok=True, parents=True)

    saved_causal_attrs, saved_attrs = celeba_scm(
        celeba_data,
        save_dir,
        scm_type=scm_type,
        image_size=image_size,
        interv_idx=interv_idx,
        n_samples=n_samples,
        seed=seed,
        append=append,
    )

    # save the metadata csv files
    saved_causal_df = pd.DataFrame(
        saved_causal_attrs, columns=["Sampled Index", "Gender", "Age", "Haircolor"]
    )
    saved_attrs_df = pd.DataFrame(saved_attrs, columns=celeba_data.attr_names[:-1])

    if scm_type == "obs":
        saved_causal_df["Intervention"] = ""
    else:
        saved_causal_df["Intervention"] = "Haircolor"

    causal_attrs_path = save_dir / "causal_attrs.csv"
    meta_attrs_path = save_dir / "meta_attrs.csv"
    if append:
        # Check if the files already exist and append if they do
        if causal_attrs_path.exists():
            existing_causal_df = pd.read_csv(causal_attrs_path, index_col=0)
            saved_causal_df = pd.concat(
                [existing_causal_df, saved_causal_df], ignore_index=True
            )

        if meta_attrs_path.exists():
            existing_attrs_df = pd.read_csv(meta_attrs_path, index_col=0)
            saved_attrs_df = pd.concat(
                [existing_attrs_df, saved_attrs_df], ignore_index=True
            )

    saved_causal_df.to_csv(causal_attrs_path)
    saved_attrs_df.to_csv(meta_attrs_path)
