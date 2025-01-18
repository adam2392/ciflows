import os
import re
from copy import copy
from pathlib import Path
from collections import Counter

from joblib import Parallel, delayed
from tqdm import tqdm

from tqdm_joblib import tqdm_joblib

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
from albumentations import CoarseDropout, Compose, HorizontalFlip, OneOf, RandomCrop
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CelebA
from tqdm import tqdm


def exponential_weights(i_range, alpha=1.0):
    weights = [np.exp(alpha * i) for i in i_range]
    # Normalize weights so they sum to 1
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    return normalized_weights


def linear_weights(i_range):
    weights = [i for i in i_range]
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
            print("  " * indent + f"Transformation {i+1}: {transform.__class__.__name__}")


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
            CoarseDropout(
                max_holes=32,
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
    file_sample_indices,
    male_attrs,
    young_attrs,
    hair_attrs,
    hair_categories,
    n_samples=1000,
    seed=None,
):
    """Set up the observational SCM."""
    rng = np.random.default_rng(seed)

    # index over the rows in the dataframe
    assert len(file_sample_indices) == len(male_attrs) == len(young_attrs) == len(hair_attrs)
    print(
        hair_attrs.reshape(-1, 1).shape,
        male_attrs.reshape(-1, 1).shape,
        young_attrs.reshape(-1, 1).shape,
    )
    # assert False

    # the hair map encodings
    hair_map = {"Black": 1, "Blond": 2, "Gray": 3}
    # hair_map = {"Black": 1, "Gray": 2}
    # hair_map = {"Black_Hair": 1, "Blond_Hair": 2, "Gray_Hair": 3}
    # Precompute hair categories
    # hair_categories = np.unique(hair_attrs).astype(int).tolist()

    image_attrs = np.concatenate(
        (
            file_sample_indices.reshape(-1, 1),
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

        hair_range = np.arange(len(hair_categories))
        # make older ppl more likely to have gray hair
        if age_str == "Young":
            p_hair_range = hair_range[::-1]
        else:
            p_hair_range = hair_range
        p_hairs = exponential_weights(p_hair_range, alpha=1.0)
        if idx < 2:
            print(p_hairs)
            print(age_str)
            print(p_hair_range)
            print(hair_categories)
        hair_str = rng.choice(hair_categories, p=p_hairs)

        # 0: black
        # 1: blond
        # 2: brown
        # 3: gray
        # hair_map = {"Black": 1, "Blond": 2, "Gray": 3}
        hair = hair_map[hair_str]
        # if idx == 0:
        #     print(hair_categories)
        #     print(hair_str)
        #     print()

        # now sample an individual that is Male, Old and X-Hair color
        matching_indices = image_attrs[
            (image_attrs[:, 1] == gender) & (image_attrs[:, 2] == age) & (image_attrs[:, 3] == hair)
        ][
            :, 0
        ].tolist()  # Extract file-sample indices

        # Sample a single individual if there are matches
        if matching_indices:
            sampled_index = rng.choice(matching_indices)
            sampled_indices.append(sampled_index)
            sampled_attrs.append((gender_str, age_str, hair_str))
    return sampled_indices, sampled_attrs


def interventional_sample_img_indices(
    file_sample_indices,
    male_attrs,
    young_attrs,
    hair_attrs,
    interv_idx=0,
    n_samples=1000,
    seed=None,
):
    """Set up the observational SCM."""
    # Precompute hair categories
    if interv_idx == 0:
        hair_categories = ["Gray", "Black"]
        weight_func = exponential_weights
    elif interv_idx == 1:
        hair_categories = ["Black", "Blond"]
        weight_func = None
    elif interv_idx == 2:
        hair_categories = ["Gray"]
        weight_func = None
    elif interv_idx == 3:
        hair_categories = ["Black", "Brown"]
    elif interv_idx == 4:
        hair_categories = ["Blond"]
    alpha = 1.0

    rng = np.random.default_rng(seed)
    image_attrs = np.concatenate(
        (
            file_sample_indices.reshape(-1, 1),
            male_attrs.reshape(-1, 1),
            young_attrs.reshape(-1, 1),
            hair_attrs.reshape(-1, 1),
        ),
        axis=1,
    )
    hair_map = {"Black": 1, "Blond": 2, "Gray": 3}

    # List to store sampled indices
    sampled_indices = []
    sampled_attrs = []
    for idx in range(n_samples):
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

        hair_range = np.arange(len(hair_categories))
        if age_str == "Young":
            p_hair_range = hair_range[::-1]
        else:
            p_hair_range = hair_range

        if weight_func is None:
            p_hairs = None
        else:
            p_hairs = weight_func(p_hair_range, alpha=alpha)
        hair_str = rng.choice(hair_categories, p=p_hairs)

        # 0: black
        # 1: blond
        # 2: brown
        # 3: gray
        # hair_map = {"Black": 0, "Blond": 1, "Brown": 2, "Gray": 3}
        hair = hair_map[hair_str]

        # now sample an individual that is Male, Old and X-Hair color
        matching_indices = image_attrs[
            (image_attrs[:, 1] == gender) & (image_attrs[:, 2] == age) & (image_attrs[:, 3] == hair)
        ][
            :, 0
        ].tolist()  # Extract indices

        # Sample a single individual if there are matches
        if matching_indices:
            sampled_index = rng.choice(matching_indices)
            sampled_indices.append(sampled_index)
            sampled_attrs.append((gender_str, age_str, hair_str))
    return sampled_indices, sampled_attrs


def celeba_scm(
    celeba_data,
    save_dir,
    sample_indices,
    append=False,
    img_size=128,
    n_workers=-1,
):
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

    def process_sample(sample_idx, idx_offset):
        # Load image and metadata
        image, _ = celeba_data[sample_idx]
        image = torch.permute(image, (1, 2, 0))

        # Apply transformations
        # transform_pipeline = get_random_transforms(image_size=img_size)
        # transformed = transform_pipeline(image=np.array(image))
        # transformed_image = transformed["image"]
        transformed_image = image

        # Convert to a PIL Image
        transformed_image = (transformed_image.numpy() * 255).astype(np.uint8)
        if transformed_image.shape[0] == 3:
            transformed_image = np.transpose(transformed_image, (1, 2, 0))
        image_pil = Image.fromarray(transformed_image)

        # Save the image as JPG
        save_path = save_dir / f"sample_{idx_offset}.jpg"
        image_pil.save(save_path)

    if n_workers == 1:
        # Sequential execution
        for idx, sample_idx in tqdm(enumerate(sample_indices), desc="Processing Samples"):
            process_sample(sample_idx, idx + max_idx)
    else:
        # Wrap Parallel jobs with tqdm
        with tqdm_joblib(tqdm(desc="Processing Samples", total=len(sample_indices))):
            Parallel(n_jobs=n_workers)(
                delayed(process_sample)(sample_idx, idx + max_idx)
                for idx, sample_idx in enumerate(sample_indices)
            )


def get_joint_probability_table(sampled_attrs, verbose=False):
    # Step 1: Count occurrences of each combination
    counter = Counter(sampled_attrs)

    # Step 2: Create a DataFrame for analysis
    df = pd.DataFrame(counter.items(), columns=["Combination", "Count"])
    # if verbose:
    # display(df.head())
    df[["Gender", "Age", "Hair Color"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop(columns="Combination")

    # Step 3: Calculate joint probabilities
    total_count = df["Count"].sum()
    df["Joint Probability"] = df["Count"] / total_count

    # P(Gender | Age, Hair Color)
    df["P(Gender | Age, Hair Color)"] = df.groupby(["Age", "Hair Color"])["Count"].transform(
        lambda x: x / x.sum()
    )

    # P(Age | Gender, Hair Color)
    df["P(Age | Gender, Hair Color)"] = df.groupby(["Gender", "Hair Color"])["Count"].transform(
        lambda x: x / x.sum()
    )

    # P(Hair Color | Gender, Age)
    df["P(Hair Color | Gender, Age)"] = df.groupby(["Gender", "Age"])["Count"].transform(
        lambda x: x / x.sum()
    )

    # test that the output makes sense
    assert_allclose(df["Joint Probability"].sum(), 1.0)
    return df


from scipy.stats import chi2_contingency


# Function to compute Cramér's V
def cramers_v(confusion_matrix):
    # Perform the chi-squared test
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()  # Total number of observations
    r, k = confusion_matrix.shape  # Rows and columns
    # Compute and return Cramér's V as a scalar
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


# Function to compute pairwise correlations
def pairwise_cramers_v(df, columns):
    # Ensure the DataFrame has correct dimensions and numeric dtype
    results = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                results.at[col1, col2] = 1.0  # Correlation with itself
            else:
                # Create a contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])
                corr_vals = cramers_v(contingency_table)
                # print(f"Contingency table for {col1} vs {col2}:\n{contingency_table}\n")
                # print(corr_vals)
                results.at[col1, col2] = corr_vals
    return results


# Function to compute conditional Cramér's V
def conditional_cramers_v(df, x_col, y_col, z_col):
    results = {}
    for z_value, subset in df.groupby(z_col):
        contingency_table = pd.crosstab(subset[x_col], subset[y_col])
        results[z_value] = cramers_v(contingency_table)
    return results


def inspect_sampled_causal_distr(df):
    # Define the columns for pairwise analysis
    # columns = ['Male', 'Young', 'Hair_Category']
    columns = ["Gender", "Age", "Hair Color"]
    df_selected = df.loc[:, columns]

    # Compute pairwise Cramér's V
    correlation_matrix = pairwise_cramers_v(df_selected, columns)

    # Display the correlation matrix
    print("\nPairwise Cramér's V Correlation Matrix:")
    from IPython import display

    display(correlation_matrix)

    # Compute conditional Cramér's V
    conditional_results = conditional_cramers_v(
        df_selected, x_col="Gender", y_col="Age", z_col="Hair Color"
    )
    print("Conditional Cramér's V:")
    for val, v in conditional_results.items():
        print(f"Hair Color = {val}: Cramér's V = {v:.4f}")

    # Compute conditional Cramér's V
    conditional_results = conditional_cramers_v(
        df_selected, z_col="Gender", x_col="Age", y_col="Hair Color"
    )
    print("Conditional Cramér's V:")
    for val, v in conditional_results.items():
        print(f"Gender = {val}: Cramér's V = {v:.4f}")

    conditional_results = conditional_cramers_v(
        df_selected, x_col="Gender", y_col="Hair Color", z_col="Age"
    )
    print("Conditional Cramér's V:")
    for val, v in conditional_results.items():
        print(f"Age = {val}: Cramér's V = {v:.4f}")



if __name__ == "__main__":
    # Root directory for the dataset
    data_root = Path("/Users/adam2392/pytorch_data/")
    data_root = Path("/local/eb/adam2392/")

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

    df_filtered = filter_df(celeba_data)
    df = df_filtered.copy()
    hair_cat_col = "Hair_Category"
    male_col = "Male"
    young_col = "Young"
    # Precompute hair categories
    # hair_map = {"Black_Hair": 1, "Blond_Hair": 2, "Gray_Hair": 3}
    # hair_categories = ["Black", "Blond", "Gray"]
    hair_categories = ["Black", "Gray"]
    sampled_indices, sampled_attrs = obs_sample_img_indices(
        df["sample_idx"].values,
        df[male_col].values,
        df[young_col].values,
        df[hair_cat_col].values,
        hair_categories=hair_categories,
        n_samples=n_samples,
        seed=seed,
    )

    # save joint probability table
    df = get_joint_probability_table(sampled_attrs)

    # save resulting dataframe table
    df.to_csv(
        data_root / "CausalCelebA" / "chain" / "dim128" / "causalceleba_obs_joint_probability.csv",
        index=False,
    )

    scm_type = "obs"
    append = False

    save_dir = data_root / "CausalCelebA" / "chain" / "dim128" / scm_type
    save_dir.mkdir(exist_ok=True, parents=True)

    celeba_scm(
        celeba_data,
        save_dir,
        sample_indices=sampled_indices,
        append=append,
        img_size=image_size,
        n_workers=1,
    )

    # save the metadata csv files
    saved_causal_df = pd.DataFrame(sampled_attrs, columns=["Gender", "Age", "Hair Color"])
    saved_causal_df["Sample Index"] = sampled_indices

    # save the metadata csv files
    if scm_type == "obs":
        saved_causal_df["Intervention"] = "Obs"
    else:
        saved_causal_df["Intervention"] = "Haircolor"

    causal_attrs_path = save_dir / "causal_attrs.csv"
    meta_attrs_path = save_dir / "meta_attrs.csv"
    if append:
        # TODO: need to append the existing CSV
        # Define file paths

        # Check if the files already exist and append if they do
        if causal_attrs_path.exists():
            existing_causal_df = pd.read_csv(causal_attrs_path, index_col=0)
            saved_causal_df = pd.concat([existing_causal_df, saved_causal_df], ignore_index=True)

    saved_causal_df.to_csv(causal_attrs_path)


    scm_types = [
        "int_hair_0",
        "int_hair_1",
        "int_hair_2",
    ]
    interv_idxs = [
        0,
        1,
        2,
    ]
    for interv_idx, scm_type in zip(interv_idxs, scm_types):
        print(f"Computing for {interv_idx} - {scm_type}")
        # scm_type = "int_hair_2"
        # scm_type = "obs"
        # interv_idx = 2
        append = False

        save_dir = data_root / "CausalCelebA" / "chain" / "dim128" / scm_type
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
                saved_attrs_df = pd.concat([existing_attrs_df, saved_attrs_df], ignore_index=True)

        saved_causal_df.to_csv(causal_attrs_path)
        saved_attrs_df.to_csv(meta_attrs_path)

        