"""
A collection of utility functions for the Touch-FL dataset.

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
import data.transforms as my_transforms

# Constants for label columns and cache path
LABEL_COLS = ['object', 'region', 'object_region', 'motion', 'hardness']


def load_touch_fl_dataset():
    """
    Load the Touch-FL Hugging Face dataset and return a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the Touch-FL dataset.
    """
    dataset = load_dataset('gemixin/touch-fl', split='train')
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
    interaction_df = df.groupby('interaction_id').first().reset_index()

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
    train_df = df[df['interaction_id'].isin(train_ids_df['interaction_id'])]
    val_df = df[df['interaction_id'].isin(val_ids_df['interaction_id'])]
    test_df = df[df['interaction_id'].isin(test_ids_df['interaction_id'])]

    # Print split set sizes
    print('Dataframe split into following sizes:')
    print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

    return train_df, val_df, test_df


def get_label_info(df):
    """
    Generate required label information from the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        dict: A dictionary containing:
            - 'label2idx': A dictionary mapping column names to label-to-index mappings.
            - 'idx2label': A dictionary mapping column names to index-to-label mappings.
            - 'materials_list': A sorted list of unique materials in the dataset.
    """

    # Create label-to-index and index-to-label mappings for each label column
    label2idx = {}
    idx2label = {}
    for col in LABEL_COLS:
        unique_labels = df[col].unique()
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}

    # Extract unique materials from the 'materials' column, which may contain
    # comma-separated values
    materials = set()
    for value in df['materials']:
        if value is not None:
            for mat in str(value).split(', '):
                materials.add(mat.strip())
    materials_list = sorted(materials)

    # Return all the mappings and materials list in a single dictionary
    return {'label2idx': label2idx,
            'idx2label': idx2label,
            'materials_list': materials_list}


def decode_materials(materials_vector, materials_list):
    """
    Decode a multi-hot vector back into a list of material names.

    Args:
        materials_vector (list/tuple/torch.Tensor): The multi-hot encoded vector.
        materials_list (list): The list of material names corresponding to the indices.

    Returns:
        list: A list of material names that are active in the multi-hot vector.
    """

    # Decode the multi-hot vector to get the active materials
    active_materials = [
        materials_list[i]
        for i, v in enumerate(materials_vector)
        if v > 0.5
    ]
    return active_materials


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
    if img_data['bytes'] is not None:
        img = Image.open(BytesIO(img_data['bytes']))
    else:
        img = Image.open(img_data['path'])

    # Convert to RGB and then to tensor
    img = img.convert('RGB')
    img_tensor = transforms.ToTensor()(img)

    # Optionally subtract the background image (if provided)
    if bg_tensor is not None:
        img_tensor = img_tensor - bg_tensor

    # Resize the final image using provided transform name
    resize = my_transforms.get_transform(transform_name)
    img_tensor = resize(img_tensor)

    # Return the processed image tensor
    return img_tensor


def get_norm_stats(df, cache_path, dataset_config):
    """
    Get the mean and standard deviation for normalisation, either by loading from cache
    or by calculating from the provided DataFrame. We use a cache to avoid expensive
    recalculation of these stats every time but need to ensure the cache is valid for the
    current dataset configuration.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        cache_path (str): The path to the normalisation cache file.
        dataset_config (dict): The dataset configuration to check against the cache.

    Returns:
        tuple: A tuple containing the mean and standard deviation as torch tensors.
    """

    # Create a Path object for the cache file
    path = Path(cache_path)
    # Check if the cache file exists and if the dataset config matches
    if not path.exists():
        cache = None
        print('No norm cache found.')
    else:
        cache = json.loads(path.read_text())
        if cache.get('dataset_config') != dataset_config:
            cache = None
            print('Norm cache found but dataset config does not match.')

    # If the cache is not found or the dataset config has changed, calculate the stats and
    # save to cache
    if cache is None:
        print('Calculating normalisation stats...')

        # Get transform name from the dataset config
        transform_name = dataset_config.get('transform_name')

        # Get background path from dataset config
        bg_path = dataset_config.get('bg_path')

        # Load and preprocess the background image if a path is provided
        if bg_path is not None:
            bg_img = Image.open(bg_path).convert('RGB')
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
            img_data = row['image']
            # Process the image (including transform and optional background subtraction)
            img_tensor = process_tactile_image(img_data, transform_name, bg_tensor)
            # Update the pixel count and sums for mean/std calculation
            c, h, w = img_tensor.shape
            pixel_count += h * w
            pixel_sum += img_tensor.sum(dim=(1, 2))
            pixel_sum_sq += (img_tensor ** 2).sum(dim=(1, 2))

        # Calculate mean and std from the sums
        mean = pixel_sum / pixel_count
        std = torch.sqrt((pixel_sum_sq / pixel_count) - (mean ** 2))

        # Save the calculated stats to cache
        data = {
            'dataset_config': dataset_config,
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
        # Create the configs directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        print('Normalisation stats calculated and saved to cache.')

        # Return the stats as tensors
        return mean, std

    # If the cache is found and valid, load the stats from the cache
    else:
        print('Norm cache found, loading normalisation stats.')
        return (
            torch.tensor(cache['mean']),
            torch.tensor(cache['std']),
        )
