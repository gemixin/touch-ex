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

# Columns in the dataset that we can use as labels for classification tasks
LABEL_COLS = ["object", "region", "object_region", "force_level", "motion"]

# Mappings for expected object labels in the unseen test split
EXPECTED_OBJECTS = {
    "tin_peas": "tin_beans",
    "small_scissors": "scissors",
    "patterned_mug": "mug",
    "media_remote": "tv_remote",
    "table_knife": "scissors",
    "microfibre_cloth": "tea_towel",
    "dish_brush": "toothbrush",
    "beaker": "mug",
    "rubber_ball": "tennis_ball",
}

# Mappings for expected object_region labels in the unseen test split
EXPECTED_OBJECT_REGIONS = {
    "tin_peas_body": "tin_beans_body",
    "tin_peas_lid": "tin_beans_lid",
    "tin_peas_base": "tin_beans_base",
    "small_scissors_handle": "scissors_handle",
    "small_scissors_blades": "scissors_blades",
    "patterned_mug_body": "mug_body",
    "patterned_mug_handle": "mug_handle",
    "patterned_mug_rim": "mug_rim",
    "media_remote_body": "tv_remote_body",
    "media_remote_buttons": "tv_remote_buttons",
    "table_knife_handle": "scissors_blade",
    "table_knife_blade": "scissors_blade",
    "microfibre_cloth_surface": "tea_towel_surface",
    "microfibre_cloth_edge": "tea_towel_edge",
    "dish_brush_handle": "toothbrush_handle",
    "dish_brush_head": "toothbrush_head",
    "beaker_body": "mug_body",
    "beaker_rim": "mug_rim",
    "rubber_ball_body": "tennis_ball_body",
    "rubber_ball_ridges": "tennis_ball_seam",
}

# Categorise objects from the unseen test split
# Matched objects are those that have a close counterpart in the training set
MATCHED_OBJECTS = ["tin_peas", "small_scissors", "patterned_mug", "media_remote"]

# Related objects are different but evaluated against a related training class
RELATED_OBJECTS = [
    "table_knife",
    "microfibre_cloth",
    "dish_brush",
    "beaker",
    "rubber_ball",
]


def load_touch_ex_dataset():
    """
    Load the Touch-Ex Hugging Face dataset and return a dictionary containing the main
    train split and the test_unseen split as pandas DataFrames.

    Returns:
        dict: A dictionary containing the train and test_unseen splits as pandas DataFrames.
    """

    # Load the Touch-Ex dataset from Hugging Face
    dataset = load_dataset("gemixin/touch-ex")

    # Convert the train and test_unseen splits to pandas DataFrames
    train_df = dataset["train"].to_pandas()
    test_unseen_df = dataset["test_unseen"].to_pandas()

    # Return the DataFrames in a dictionary
    return {"train": train_df, "test_unseen": test_unseen_df}


def add_expected_labels(df):
    """
    Add expected object and object_region labels for the unseen test split.

    Args:
        df (pd.DataFrame): The unseen test split DataFrame.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with expected label columns added.
    """

    # Create a copy of the input DataFrame
    updated_df = df.copy()
    # Map the original labels to the expected labels using the predefined mappings
    updated_df["expected_object"] = updated_df["object"].map(EXPECTED_OBJECTS)
    updated_df["expected_object_region"] = updated_df["object_region"].map(
        EXPECTED_OBJECT_REGIONS
    )
    return updated_df


def filter_by_force_level(df, force_level):
    """
    Filter the DataFrame by the given force level.

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        force_level (str): The force level to filter by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with the given force level.
    """

    # Filter the DataFrame based on the given force level
    filtered_df = df[df["force_level"] == force_level]
    return filtered_df


def filter_by_motion(df, motion):
    """
    Filter the DataFrame by the given motion.

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        motion (str): The motion to filter by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with the given motion.
    """

    # Filter the DataFrame based on the specified motion
    filtered_df = df[df["motion"] == motion]
    return filtered_df


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


def split_test_unseen(df):
    """
    Split the test_unseen DataFrame into matched and related unseen-object sets.

    Args:
        df (pd.DataFrame): The input test_unseen DataFrame.

    Returns:
        tuple: A tuple containing the matched and related unseen-object DataFrames.
    """

    # Split unseen test data into matched and related unseen-object sets
    test_unseen_matched_df = df[df["object"].isin(MATCHED_OBJECTS)]
    test_unseen_related_df = df[df["object"].isin(RELATED_OBJECTS)]

    return test_unseen_matched_df, test_unseen_related_df


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
        "filtered_force_level": data_config["filtered_force_level"],
        "filtered_motion": data_config["filtered_motion"],
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
            # Create the cache file’s parent directory if it doesn't exist
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
