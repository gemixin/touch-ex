"""
A module containing the builder function for loading and preparing the Touch-Ex dataset.

Author: Gemma McLean
Date: April 2026
"""

from torch.utils.data import DataLoader
from data.dataset import TouchExDataset
import data.utils as utils


def get_dataloaders(data_config):
    """
    Load the Touch-Ex dataset, split the main train split into train/val/test sets,
    create a TouchEXDataset for each split, and return DataLoaders for each split plus
    label mappings. The separate test_unseen split is included as an additional dataloader
    along with its own label mappings.

    Args:
        data_config (dict): A dictionary containing configuration parameters for loading and
            preparing the dataset.

    Returns:
        tuple: A tuple containing the DataLoaders dictionary, the training label mappings
            dictionary, and the unseen test label mappings dictionary.
    """

    # --- Load and prepare the dataset --- #

    # Load the HuggingFace dataset splits into DataFrames
    dataframes = utils.load_touch_ex_dataset()

    train_source_df = dataframes["train"]  # Main training split for train/val/test
    test_unseen_df = dataframes["test_unseen"]  # Separate test split with unseen labels

    # Add new columns to test_unseen_df for expected labels
    test_unseen_df = utils.add_expected_labels(test_unseen_df)

    # Filter dataframes if needed
    # If a specific force level is specified in the data_config
    if data_config["filtered_force_level"]:
        # Get the force level string value
        force_val = data_config["filtered_force_level"]
        # Filter the train and test_unseen DataFrames by the specified force level
        train_source_df = utils.filter_by_force_level(train_source_df, force_val)
        test_unseen_df = utils.filter_by_force_level(test_unseen_df, force_val)
    # If a specific motion type is specified in the data_config
    if data_config["filtered_motion"]:
        # Get the motion type string value
        motion_val = data_config["filtered_motion"]
        # Filter the train and test_unseen DataFrames by the specified motion type
        train_source_df = utils.filter_by_motion(train_source_df, motion_val)
        test_unseen_df = utils.filter_by_motion(test_unseen_df, motion_val)

    # Split the main dataset by interaction_id into train, val, and test sets
    train_df, val_df, test_df = utils.split_by_interaction(
        train_source_df,
        split_size=data_config["split_size"],
        stratify_label=data_config["stratify_label"],
        random_state=data_config["random_state"],
    )

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "test_unseen": test_unseen_df,
    }

    # Get label mappings from the training set
    train_label_mappings = utils.get_label_mappings(train_df)
    # Get label mappings from the unseen test set
    test_unseen_label_mappings = utils.get_label_mappings(test_unseen_df)

    # Get normalisation stats (mean and std) or None if not enabled
    norm_stats = utils.get_norm_stats(train_df, data_config)

    # Create a TouchEXDataset for each split, passing in the appropriate parameters
    datasets = {
        split: TouchExDataset(
            dataframe=df_split,
            train_label_mappings=train_label_mappings,
            test_unseen_label_mappings=test_unseen_label_mappings,
            transform_name=data_config["transform_name"],
            norm_stats=norm_stats,
            bg_path=data_config["bg_path"],
            test_unseen=(split == "test_unseen"),
        )
        for split, df_split in split_dfs.items()
    }

    # Create a DataLoader for each dataset split, using the specified settings
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=data_config["batch_size"],
            shuffle=data_config["shuffle_map"].get(split, False),
            num_workers=data_config["num_workers"],
        )
        for split in split_dfs
    }

    # Return the DataLoaders and label mappings
    return dataloaders, train_label_mappings, test_unseen_label_mappings
