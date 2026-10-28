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
from data.transforms import get_random_resized_crop_224, get_transform

# Columns in the dataset that can be used as labels for classification tasks
LABEL_COLS = ["object", "region", "object_region", "force_level", "motion"]

# Partition membership is independent of model and training randomness.
SPLIT_SEED = 129
EVALUATION_SPLITS = {
    "model_selection": ("val", "validation_unseen_matched"),
    "final_evaluation": ("test", "test_unseen_matched", "test_unseen_related"),
}


def load_touch_ex_dataset(experiment_purpose="model_selection"):
    """Load only the Hugging Face source splits needed for this experiment."""
    source_splits = ["train", *EVALUATION_SPLITS[experiment_purpose][1:]]
    return {
        split: load_dataset("gemixin/touch-ex", split=split).to_pandas()
        for split in source_splits
    }


def split_by_interaction(df, split_size, stratify_label):
    """
    Split a frame-level dataframe by interaction_id using the specified split size and
    stratification label and fixed seed 129. Splits into train, validation, and test sets.
    Classification passes its target label; regression passes the configured label.
    Different stratification labels can produce different interaction partitions.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        split_size (float): The proportion of the dataset for val/test splits.
        stratify_label (str): The column name to use for stratification.

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
        random_state=SPLIT_SEED,
    )

    # Split the temp set into val and test sets (50% each)
    val_ids_df, test_ids_df = train_test_split(
        temp_ids_df,
        test_size=0.50,
        stratify=temp_ids_df[stratify_label],
        random_state=SPLIT_SEED,
    )

    # Apply the splits back to the original dataframe by filtering on the interaction IDs
    train_df = df[df["interaction_id"].isin(train_ids_df["interaction_id"])]
    val_df = df[df["interaction_id"].isin(val_ids_df["interaction_id"])]
    test_df = df[df["interaction_id"].isin(test_ids_df["interaction_id"])]

    # Print split set sizes
    print("Dataframe split into following sizes:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def get_label_lists(df):
    """
    Get the unique class labels for each classification label column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        dict: A dictionary mapping each label column to its ordered class list.
    """

    # Preserve DataFrame order so each class list corresponds to indices 0 to n-1
    return {label: df[label].unique().tolist() for label in LABEL_COLS}


def process_tactile_image(
    img_data,
    transform_name,
    bg_tensor=None,
    color_jitter=None,
    random_resized_crop=False,
):
    """
    Process a tactile image by loading it, optionally subtracting the background,
    and resizing it, before converting it to a tensor.

    Args:
        img_data (dict): The image data (a dictionary containing 'bytes' or 'path').
        transform_name (str): The name of the transform to apply.
        bg_tensor (torch.Tensor, optional): The background tensor to subtract from the
        image. Defaults to None, meaning no background subtraction will be applied.
        color_jitter (torchvision.transforms.ColorJitter, optional): A transform applied
            to the RGB source image before background subtraction. Defaults to None.
        random_resized_crop (bool): Whether to replace the deterministic transform with
            a training-only random resized crop. Defaults to False.

    Returns:
        torch.Tensor: The processed image tensor.
    """

    # Load the image from bytes or path (Hugging Face image format)
    if img_data["bytes"] is not None:
        img = Image.open(BytesIO(img_data["bytes"]))
    else:
        img = Image.open(img_data["path"])

    # Convert to RGB and apply optional colour augmentation before tensor conversion
    img = img.convert("RGB")
    if color_jitter is not None:
        img = color_jitter(img)
    img_tensor = transforms.ToTensor()(img)

    # Optionally subtract the background image (if provided)
    if bg_tensor is not None:
        img_tensor = img_tensor - bg_tensor

    # Apply the configured deterministic transform, unless this is a training sample
    # explicitly configured to use random-resized-crop spatial augmentation.
    resize = (
        get_random_resized_crop_224()
        if random_resized_crop
        else get_transform(transform_name)
    )
    img_tensor = resize(img_tensor)

    return img_tensor


def get_imagenet_norm_stats():
    """
    Get the mean and standard deviation for normalisation based on ImageNet stats.

    Returns:
        tuple: A tuple containing the mean and standard deviation as torch tensors.
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
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
        "split_seed": SPLIT_SEED,
        "split_size": data_config["split_size"],
        "stratify_label": data_config["stratify_label"],
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

        transform_name = norm_config.get("transform_name")
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
            # Create the cache file’s parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            print("Normalisation stats calculated and saved to cache.")

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
