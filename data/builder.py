"""
A module containing the builder function for loading and preparing the Touch-EX dataset.

Author: Gemma McLean
Date: April 2026
"""

from torch.utils.data import DataLoader
from data.dataset import TouchEXDataset
import data.utils as utils


def get_dataloaders(data_config):
    """
    Load the Touch-EX dataset, split it into train/val/test sets, create a TouchEXDataset
    for each split, and return DataLoaders for each split plus label info (mappings and
    materials list).

    Args:
        data_config (dict): A dictionary containing configuration parameters for loading and
        preparing the dataset.

    Returns:
        tuple: A tuple containing the DataLoaders dictionary and the label info dictionary.
    """

    # --- Load and prepare the dataset --- #

    # Load the HuggingFace dataset into a DataFrame
    df = utils.load_touch_ex_dataset()

    # Split the dataset by interaction_id into train, val, and test sets
    # If unseen_objs is provided, split the dataset with unseen objects in the test set
    if data_config["unseen_objs"] is not None:
        # Split size is just for the val split so we only want half of our usual split size
        train_df, val_df, test_df = utils.split_by_interaction_unseen_objs(
            df,
            split_size=data_config["split_size"] / 2,
            stratify_label=data_config["stratify_label"],
            random_state=data_config["random_state"],
            unseen_objs=data_config["unseen_objs"],
        )
    # Otherwise, do a regular split without unseen objects in the test set
    else:
        train_df, val_df, test_df = utils.split_by_interaction(
            df,
            split_size=data_config["split_size"],
            stratify_label=data_config["stratify_label"],
            random_state=data_config["random_state"],
        )

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # Get label info (mappings and materials list) from the training set
    label_info = utils.get_label_info(train_df)

    # Get normalisation stats (mean and std) or None if not enabled
    norm_stats = utils.get_norm_stats(train_df, data_config)

    # Create a TouchEXDataset for each split, passing in the appropriate parameters
    datasets = {
        split: TouchEXDataset(
            dataframe=df_split,
            label_info=label_info,
            transform_name=data_config["transform_name"],
            norm_stats=norm_stats,
            bg_path=data_config["bg_path"],
        )
        for split, df_split in split_dfs.items()
    }

    # Create a DataLoader for each dataset split, using the specified settings
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=data_config["batch_size"],
            shuffle=data_config["shuffle_map"][split],
            num_workers=data_config["num_workers"],
        )
        for split in split_dfs
    }

    # Return the DataLoaders and label info
    return dataloaders, label_info
