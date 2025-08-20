from pathlib import Path

import pandas as pd
from torchvision import transforms
from torchvision.datasets import CelebA

from ciflows.datasets.causalceleba_scm.sampling import (
    celeba_scm,
    get_joint_probability_table,
    interventional_sample_img_indices,
    obs_sample_img_indices,
)


def filter_df(celeba_data):
    # create a dataframe for the celebA attributes
    df = pd.DataFrame(celeba_data.attr, columns=celeba_data.attr_names[:-1])
    df = df.reset_index(names=["sample_idx"])

    # now filter the dataframe based on meeting exactly one of the chosen hair colors
    hair_colors = ["Black_Hair", "Blond_Hair", "Gray_Hair"]
    df_filtered = df[df[hair_colors].sum(axis=1) == 1]

    # filter out images with bald, eyeglasses, or blurry
    df_filtered = df_filtered[
        (df_filtered["Blurry"] == 0) & (df_filtered["Eyeglasses"] == 0) & (df_filtered["Bald"] == 0)
    ]
    df_filtered.reset_index(inplace=True, drop=True)

    hair_map = {"Black_Hair": 1, "Blond_Hair": 2, "Gray_Hair": 3}
    # display(df_filtered[hair_colors].idxmax(axis=1))
    df_filtered["Hair_Category"] = df_filtered[hair_colors].idxmax(axis=1).map(hair_map)

    df_filtered = df_filtered.loc[:, ["sample_idx", "Male", "Young", "Hair_Category"]]

    print(
        f"Number of total samples used: {len(df_filtered)} filtered from total of {len(df)} - {len(df_filtered) / len(df):.3f} of the total"
    )
    print(df_filtered["Hair_Category"].value_counts())
    return df_filtered


def create_obs_dataset(data_root):
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
    if append:
        # TODO: need to append the existing CSV
        # Define file paths

        # Check if the files already exist and append if they do
        if causal_attrs_path.exists():
            existing_causal_df = pd.read_csv(causal_attrs_path, index_col=0)
            saved_causal_df = pd.concat([existing_causal_df, saved_causal_df], ignore_index=True)

    saved_causal_df.to_csv(causal_attrs_path)


def create_int_datasets(data_root):
    # Spatial size of training images, images are resized to this size.
    image_size = 128
    n_samples = 20_000
    seed = 1234
    append = False

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
        df = df_filtered.copy()

        hair_cat_col = "Hair_Category"
        male_col = "Male"
        young_col = "Young"

        # sample the corresponding indices
        sampled_indices, sampled_attrs = interventional_sample_img_indices(
            df["sample_idx"].values,
            df[male_col].values,
            df[young_col].values,
            df[hair_cat_col].values,
            interv_idx=interv_idx,
            n_samples=n_samples,
            seed=seed,
        )

        # create the directory
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
        if append:
            # Check if the files already exist and append if they do
            if causal_attrs_path.exists():
                existing_causal_df = pd.read_csv(causal_attrs_path, index_col=0)
                saved_causal_df = pd.concat(
                    [existing_causal_df, saved_causal_df], ignore_index=True
                )

        saved_causal_df.to_csv(causal_attrs_path)

        # compute joint distribution of the dataset
        joint_df = get_joint_probability_table(sampled_attrs, verbose=False)

        # save resulting dataframe table
        fname = (
            data_root
            / "CausalCelebA"
            / "chain"
            / "dim128"
            / f"causalceleba_{scm_type}_joint_probability.csv"
        )
        print("Saving to ", fname)
        joint_df.to_csv(
            fname,
            index=False,
        )


if __name__ == "__main__":
    # Root directory for the dataset
    data_root = Path("/Users/adam2392/pytorch_data/")
    data_root = Path("/local/eb/adam2392/")

    create_obs_dataset(data_root)
    create_int_datasets(data_root)
