"""
A module containing functions to build the Touch-Ex datasets and DataLoaders.

Author: Gemma McLean
Date: April 2026
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import TouchExDataset
from data.validate import validate_data_config
import data.utils as utils


def seed_worker(worker_id):
    """
    Seed Python and NumPy random number generators in a DataLoader worker.

    Args:
        worker_id (int): The ID of the DataLoader worker.
    """

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_datasets(data_config):
    """
    Load the Touch-Ex dataset and split the main train split into train/val/test sets.
    Split the unseen test data into two sets: matched unseen objects and related unseen
    objects. Create a TouchExDataset for each split and return them with the class-label
    lists.

    Args:
        data_config (dict): A dictionary containing configuration parameters for loading and
            preparing the dataset.

    Returns:
        tuple: A tuple containing the Datasets dictionary and the class-label lists
            dictionary.
    """

    # If the data config is invalid, raise an error before proceeding
    validate_data_config(data_config)

    # Load the HuggingFace dataset splits into DataFrames
    dataframes = utils.load_touch_ex_dataset()
    train_source_df = dataframes["train"]  # Main training split for train/val/test
    test_unseen_df = dataframes["test_unseen"]  # Separate test split with unseen labels

    # Add new columns to test_unseen_df for expected labels
    test_unseen_df = utils.add_expected_labels(test_unseen_df)

    # Filter dataframes by force if specified in the data_config
    if data_config["filtered_force_level"]:
        force_val = data_config["filtered_force_level"]
        train_source_df = utils.filter_by_force_level(train_source_df, force_val)
        test_unseen_df = utils.filter_by_force_level(test_unseen_df, force_val)

    # Filter dataframes by motion if specified in the data_config
    if data_config["filtered_motion"]:
        motion_val = data_config["filtered_motion"]
        train_source_df = utils.filter_by_motion(train_source_df, motion_val)
        test_unseen_df = utils.filter_by_motion(test_unseen_df, motion_val)

    # Split the main dataset by interaction_id into train, val, and test sets
    train_df, val_df, test_df = utils.split_by_interaction(
        train_source_df,
        split_size=data_config["split_size"],
        stratify_label=data_config["stratify_label"],
        random_state=data_config["random_state"],
    )

    # Split the unseen test data into matched and related unseen-object sets
    test_unseen_matched_df, test_unseen_related_df = utils.split_test_unseen(test_unseen_df)

    # Create a dictionary of DataFrames for each split
    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "test_unseen_matched": test_unseen_matched_df,
        "test_unseen_related": test_unseen_related_df,
    }

    # Get class-label lists
    label_lists = {
        "train": utils.get_label_lists(train_df),
        "test_unseen_matched": utils.get_label_lists(test_unseen_matched_df),
        "test_unseen_related": utils.get_label_lists(test_unseen_related_df),
    }

    # Get normalisation stats (mean and std) or None if not enabled
    norm_stats = utils.get_norm_stats(train_df, data_config)

    # Create a TouchEXDataset for each split, passing in the appropriate parameters
    datasets = {
        split: TouchExDataset(
            dataframe=df_split,
            label_lists=label_lists,
            transform_name=data_config["transform_name"],
            norm_stats=norm_stats,
            bg_path=data_config["bg_path"],
            split=split,
        )
        for split, df_split in split_dfs.items()
    }

    return datasets, label_lists


def create_dataloaders(datasets, data_config):
    """
    Create a fresh dictionary of seeded DataLoaders from prepared datasets.

    Args:
        datasets (dict): A dictionary of TouchExDataset objects keyed by split name.
        data_config (dict): A dictionary containing DataLoader settings.

    Returns:
        dict: A dictionary of DataLoaders keyed by split name.
    """

    # If the data config is invalid, raise an error before proceeding
    validate_data_config(data_config)

    # Create a DataLoader for each dataset split using seeded generators
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=data_config["batch_size"],
            shuffle=data_config["shuffle_map"].get(split, False),
            num_workers=data_config["num_workers"],
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(data_config["random_state"]),
        )
        for split in datasets
    }

    return dataloaders


def get_dataloaders(data_config):
    """
    Build the Touch-Ex datasets and DataLoaders from a data configuration using a single
    function call.

    Args:
        data_config (dict): A dictionary containing dataset and DataLoader settings.

    Returns:
        tuple: A tuple containing the DataLoaders dictionary and the class-label lists
            dictionary.
    """

    datasets, label_lists = get_datasets(data_config)
    return create_dataloaders(datasets, data_config), label_lists
