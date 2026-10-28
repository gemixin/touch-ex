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


def get_datasets(data_config, experiment_purpose="model_selection"):
    """Prepare training, validation, and the purpose's evaluation datasets."""
    validate_data_config(data_config)
    dataframes = utils.load_touch_ex_dataset(experiment_purpose)
    for split, df in dataframes.items():
        if data_config["filtered_force_level"]:
            df = utils.filter_by_force_level(df, data_config["filtered_force_level"])
        if data_config["filtered_motion"]:
            df = utils.filter_by_motion(df, data_config["filtered_motion"])
        dataframes[split] = df

    train_df, val_df, test_df = utils.split_by_interaction(
        dataframes.pop("train"),
        split_size=data_config["split_size"],
        stratify_label=data_config["stratify_label"],
    )
    split_dfs = {"train": train_df, "val": val_df, **dataframes}
    if experiment_purpose == "final_evaluation":
        split_dfs["test"] = test_df

    label_lists = {"train": utils.get_label_lists(train_df)}
    norm_stats = utils.get_norm_stats(train_df, data_config)
    datasets = {
        split: TouchExDataset(
            dataframe=df,
            split=split,
            label_lists=label_lists,
            transform_name=data_config["transform_name"],
            train_augmentations=data_config["train_augmentations"],
            norm_stats=norm_stats,
            bg_path=data_config["bg_path"],
        )
        for split, df in split_dfs.items()
    }
    return datasets, label_lists


def create_dataloaders(datasets, data_config, seed=129):
    """
    Create a fresh dictionary of seeded DataLoaders from prepared datasets.

    Args:
        datasets (dict): A dictionary of TouchExDataset objects keyed by split name.
        data_config (dict): A dictionary containing DataLoader settings.
        seed (int): Experiment seed for shuffling and worker randomness.

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
            generator=torch.Generator().manual_seed(seed),
        )
        for split in datasets
    }

    return dataloaders


def get_dataloaders(data_config, experiment_purpose="model_selection", seed=129):
    """
    Build the Touch-Ex datasets and DataLoaders from a data configuration using a single
    function call.

    Args:
        data_config (dict): A dictionary containing dataset and DataLoader settings.
        experiment_purpose (str): model_selection (default) or final_evaluation.
        seed (int): Experiment seed for shuffling and worker randomness.

    Returns:
        tuple: A tuple containing the DataLoaders dictionary and the class-label lists
            dictionary.
    """

    datasets, label_lists = get_datasets(data_config, experiment_purpose)
    return create_dataloaders(datasets, data_config, seed=seed), label_lists
