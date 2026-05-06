"""
A module containing utility functions for the Touch-Ex dataset.

Author: Gemma McLean
Date: April 2026
"""

from io import BytesIO
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import json
from data.transforms import get_transform

# Columns in the dataset that we will be using as labels for classification tasks
LABEL_COLS = ["object", "object_region", "force_level", "hardness", "material"]


def load_touch_ex_dataset():
    """
    Load the Touch-Ex Hugging Face dataset and return a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the Touch-Ex dataset.
    """

    dataset = load_dataset("gemixin/touch-ex", split="train")
    return dataset.to_pandas()


def split_by_interaction(df, split_size, stratify_label, random_state):
    """
    Split a frame-level dataframe by interaction_id using the specified split size and
    stratification label. Splits into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        split_size (float): The proportion of the dataset for val/test splits.
        stratify_label (str): The column name to use for stratification.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """

    # Get the unique interactions by grouping on the interaction column
    # Take the first frame of each interaction
    interaction_df = df.groupby("interaction_id").first().reset_index()

    # Split the interactions into train and temp (val+test) sets
    train_ids_df, temp_ids_df = train_test_split(
        interaction_df,
        test_size=split_size,
        stratify=interaction_df[stratify_label],
        random_state=random_state,
    )

    # Split the temp set into val and test sets (50% each)
    val_ids_df, test_ids_df = train_test_split(
        temp_ids_df,
        test_size=0.50,
        stratify=temp_ids_df[stratify_label],
        random_state=random_state,
    )

    # Apply the splits back to the original dataframe by filtering on the interaction IDs
    train_df = df[df["interaction_id"].isin(train_ids_df["interaction_id"])]
    val_df = df[df["interaction_id"].isin(val_ids_df["interaction_id"])]
    test_df = df[df["interaction_id"].isin(test_ids_df["interaction_id"])]

    # Print split set sizes
    print("Dataframe split into following sizes:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def split_by_interaction_unseen_objs(
    df, split_size, stratify_label, random_state, unseen_objs
):
    """
    Split a frame-level dataframe by interaction_id using the specified split size and
    stratification label, ensuring that all interactions involving the specified unseen
    objects are placed in the test set.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        split_size (float): The proportion of the dataset for the val split.
        stratify_label (str): The column name to use for stratification.
        random_state (int): The random seed for reproducibility.
        unseen_objs (list): A list of object names to be included in the test set.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """

    # Get the unique interactions by grouping on the interaction column
    # Take the first frame of each interaction
    interaction_df = df.groupby("interaction_id").first().reset_index()

    # Identify interactions that involve any of the unseen objects
    test_ids_df = interaction_df[interaction_df["object"].isin(unseen_objs)]

    # The remaining interactions are eligible for train/val splits
    remaining_ids_df = interaction_df[
        ~interaction_df["interaction_id"].isin(test_ids_df["interaction_id"])
    ]

    # Split the remaining interactions into train and val sets
    train_ids_df, val_ids_df = train_test_split(
        remaining_ids_df,
        test_size=split_size,
        stratify=remaining_ids_df[stratify_label],
        random_state=random_state,
    )

    # Apply the splits back to the original dataframe by filtering on the interaction IDs
    train_df = df[df["interaction_id"].isin(train_ids_df["interaction_id"])]
    val_df = df[df["interaction_id"].isin(val_ids_df["interaction_id"])]
    test_df = df[df["interaction_id"].isin(test_ids_df["interaction_id"])]

    # Print split set sizes
    print("Dataframe split into following sizes:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def get_label_mappings(df):
    """
    Generate required label mappings from the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        dict: A dictionary containing label-to-index and index-to-label mappings.
    """

    # Create label-to-index and index-to-label mappings for each label column
    label2idx = {}
    idx2label = {}
    for col in LABEL_COLS:
        unique_labels = df[col].unique()
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}

    # Return the mappings in a single dictionary
    return {
        "label2idx": label2idx,
        "idx2label": idx2label,
    }


def process_tactile_image(img_data, transform_name, bg_tensor=None):
    """
    Process a tactile image by loading it, optionally subtracting the background,
    and resizing it, before converting it to a tensor.

    Args:
        img_data (dict): The image data (a dictionary containing 'bytes' or 'path').
        transform_name (str): The name of the transform to apply.
        bg_tensor (torch.Tensor, optional): The background tensor to subtract from the
        image. Defaults to None, meaning no background subtraction will be applied.

    Returns:
        torch.Tensor: The processed image tensor.
    """

    # Load the image from bytes or path (Hugging Face image format)
    if img_data["bytes"] is not None:
        img = Image.open(BytesIO(img_data["bytes"]))
    else:
        img = Image.open(img_data["path"])

    # Convert to RGB and then to tensor
    img = img.convert("RGB")
    img_tensor = transforms.ToTensor()(img)

    # Optionally subtract the background image (if provided)
    if bg_tensor is not None:
        img_tensor = img_tensor - bg_tensor

    # Resize the final image using provided transform name
    resize = get_transform(transform_name)
    img_tensor = resize(img_tensor)

    # Return the processed image tensor
    return img_tensor


def get_imagenet_norm_stats():
    """
    Get the mean and standard deviation for normalisation based on ImageNet stats.

    Returns:
        tuple: A tuple containing the mean and standard deviation as torch tensors.
    """

    # ImageNet mean and std values for normalisation
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    # Return the stats as torch tensors
    return torch.tensor(imagenet_mean), torch.tensor(imagenet_std)


def calculate_dataset_norm_stats(df, data_config, use_cache=True):
    """
    Get the mean and standard deviation for normalisation based on the provided dataset,
    either by loading from cache or by calculating from the provided DataFrame.
    We use a cache to avoid expensive recalculation of these stats every time but need to
    ensure the cache is valid for the current dataset configuration.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        data_config (dict): The data configuration containing normalisation parameters.
        use_cache (bool): Whether to use cached normalisation stats.

    Returns:
        tuple: A tuple containing the mean and standard deviation as torch tensors.
    """

    # Create a norm config dictionary that captures the relevant parameters
    norm_config = {
        "random_state": data_config["random_state"],
        "split_size": data_config["split_size"],
        "stratify_label": data_config["stratify_label"],
        "unseen_objs": data_config["unseen_objs"],
        "transform_name": data_config["transform_name"],
        "bg_path": data_config["bg_path"],
    }

    # Initialise cache variable
    cache = None

    # Check if the cache file exists and if the norm config matches
    if use_cache:
        path = Path(data_config["norm_cache_path"])
        if not path.exists():
            print("No norm cache found.")
        else:
            cache = json.loads(path.read_text())
            if cache.get("norm_config") != norm_config:
                cache = None
                print("Norm cache found but config does not match.")

    # If the cache is not found or the norm config has changed, or if use_cache is False,
    # (cache would be set to None) calculate the stats
    if cache is None:
        print("Calculating normalisation stats...")

        # Get transform name from the norm config
        transform_name = norm_config.get("transform_name")

        # Get background path from norm config
        bg_path = norm_config.get("bg_path")

        # Load and preprocess the background image if a path is provided
        if bg_path is not None:
            bg_img = Image.open(bg_path).convert("RGB")
            bg_tensor = transforms.ToTensor()(bg_img)
        else:
            bg_tensor = None

        # Calculate mean and std across the dataset
        pixel_sum = torch.zeros(3)
        pixel_sum_sq = torch.zeros(3)
        pixel_count = 0

        # Loop through the dataset and process each image to update the sums
        for idx in range(len(df)):
            # Access the image data from the DataFrame row
            row = df.iloc[idx]
            img_data = row["image"]
            # Process the image (including transform and optional bg subtraction)
            img_tensor = process_tactile_image(img_data, transform_name, bg_tensor)
            # Update the pixel count and sums for mean/std calculation
            c, h, w = img_tensor.shape
            pixel_count += h * w
            pixel_sum += img_tensor.sum(dim=(1, 2))
            pixel_sum_sq += (img_tensor**2).sum(dim=(1, 2))

        # Calculate mean and std from the sums
        mean = pixel_sum / pixel_count
        std = torch.sqrt((pixel_sum_sq / pixel_count) - (mean**2))

        if use_cache:
            # Save the calculated stats to cache
            data = {
                "norm_config": norm_config,
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            # Create the configs directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            print("Normalisation stats calculated and saved to cache.")

        # Return the stats as tensors
        return mean, std

    # If the cache is found and valid, load the stats from the cache
    else:
        print("Norm cache found, loading normalisation stats.")
        return (
            torch.tensor(cache["mean"]),
            torch.tensor(cache["std"]),
        )


def get_norm_stats(df, data_config, use_cache=True):
    """
    Get the mean and standard deviation for normalisation based on normalisation type
    specified in the data configuration.
    If the norm type is None, simply return None.
    If the norm type is 'imagenet', return the standard ImageNet stats.
    If the norm type is 'dataset', calculate the stats based on the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        data_config (dict): The data configuration containing normalisation parameters.
        use_cache (bool): Whether to use cached normalisation stats (only applicable for
        'dataset' norm type).

    Returns:
        tuple: A tuple containing the mean and standard deviation as torch tensors, or
        None if normalisation is not enabled.
    """

    # If normalisation is not enabled, return None
    if data_config["norm_type"] is None:
        return None

    # If the norm type is 'imagenet', just return the ImageNet stats
    if data_config["norm_type"] == "imagenet":
        print("Loading imagenet normalisation stats.")
        return get_imagenet_norm_stats()

    # If the norm type is 'dataset', calculate the stats
    elif data_config["norm_type"] == "dataset":
        return calculate_dataset_norm_stats(df, data_config, use_cache=use_cache)

    # If the norm type is not recognised, raise an error
    else:
        raise ValueError(f"Invalid norm type: {data_config['norm_type']}.")
