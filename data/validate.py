"""
A module for validating config dictionaries.

Author: Gemma McLean
Date: July 2026
"""

from pathlib import Path


def validate_data_config(data_config):
    """
    Validate the data configuration used to build dataloaders.

    Args:
        data_config (dict): Data configuration dictionary.

    Raises:
        ValueError: If the first invalid field is found.
    """

    # Define valid values
    label_cols = ["object", "region", "object_region", "force_level", "motion"]
    motions = ["sliding", "rotation"]
    force_levels = ["1", "2", "3"]
    transform_names = ["pad_224", "center_crop_224", "random_crop_224"]
    norm_types = ["dataset", "imagenet"]

    # Define the required keys
    required_keys = [
        "random_state",
        "split_size",
        "stratify_label",
        "filtered_force_level",
        "filtered_motion",
        "transform_name",
        "bg_path",
        "norm_type",
        "norm_cache_path",
        "batch_size",
        "num_workers",
        "shuffle_map",
    ]

    # First check it's actually a dictionary
    if not isinstance(data_config, dict):
        raise ValueError("data_config must be a dictionary.")

    # Check that all required keys are present
    for key in required_keys:
        if key not in data_config:
            raise ValueError(f"data_config is missing required key: {key}.")

    # Validate random_state
    if type(data_config["random_state"]) is not int or data_config["random_state"] < 0:
        raise ValueError("random_state must be an integer greater than or equal to 0.")

    # Validate split_size
    if (
        type(data_config["split_size"]) is not float
        or not 0 < data_config["split_size"] < 1
    ):
        raise ValueError("split_size must be a float between 0 and 1.")

    # Validate stratify_label
    if data_config["stratify_label"] not in label_cols:
        raise ValueError(f"stratify_label must be one of: {label_cols}.")

    # Validate filtered_force_level
    filtered_force_level = data_config["filtered_force_level"]
    if filtered_force_level is not None and filtered_force_level not in force_levels:
        raise ValueError(f"filtered_force_level must be one of {force_levels} or None.")

    # Validate filtered_motion
    filtered_motion = data_config["filtered_motion"]
    if filtered_motion is not None and filtered_motion not in motions:
        raise ValueError(f"filtered_motion must be one of {motions} or None.")

    # Validate transform_name
    if data_config["transform_name"] not in transform_names:
        raise ValueError(f"transform_name must be one of: {transform_names}.")

    # Validate bg_path
    bg_path = data_config["bg_path"]
    if bg_path is not None:
        if not isinstance(bg_path, (str, Path)):
            raise ValueError("bg_path must be an existing .jpg file or None.")
        bg_path = Path(bg_path)
        if bg_path.suffix.lower() != ".jpg" or not bg_path.is_file():
            raise ValueError("bg_path must be an existing .jpg file or None.")

    # Validate norm_type
    norm_type = data_config["norm_type"]
    if norm_type is not None and norm_type not in norm_types:
        raise ValueError(f"norm_type must be one of {norm_types} or None.")

    # Validate norm_cache_path
    norm_cache_path = data_config["norm_cache_path"]
    if norm_cache_path is None:
        if norm_type is not None:
            raise ValueError(
                "norm_cache_path can only be None when norm_type is also None."
            )
    else:
        if not isinstance(norm_cache_path, (str, Path)):
            raise ValueError("norm_cache_path must be a .json path or None.")
        norm_cache_path = Path(norm_cache_path)
        if norm_cache_path.suffix.lower() != ".json":
            raise ValueError("norm_cache_path must have a .json extension.")

    # Validate batch_size
    if type(data_config["batch_size"]) is not int or data_config["batch_size"] < 1:
        raise ValueError("batch_size must be an integer greater than or equal to 1.")

    # Validate num_workers
    if type(data_config["num_workers"]) is not int or data_config["num_workers"] < 0:
        raise ValueError("num_workers must be an integer greater than or equal to 0.")

    # Validate shuffle_map
    shuffle_map = data_config["shuffle_map"]
    if not isinstance(shuffle_map, dict):
        raise ValueError("shuffle_map must be a dictionary.")
    for split in ["train", "val", "test"]:
        if split not in shuffle_map:
            raise ValueError(f"shuffle_map is missing required key: {split}.")
        if type(shuffle_map[split]) is not bool:
            raise ValueError(f"shuffle_map['{split}'] must be a boolean.")


def validate_train_config(train_config):
    """
    Validate the train configuration used for training.

    Args:
        train_config (dict): Training configuration dictionary.

    Raises:
        ValueError: If the first invalid field is found.
    """

    # Define valid values
    optimizers = ["adam", "adamw", "sgd"]

    # Define the required keys
    required_keys = [
        "optimizer",
        "learning_rate",
        "warmup_epochs",
        "warmup_start_factor",
        "min_learning_rate",
        "momentum",
        "weight_decay",
        "num_epochs",
        "checkpoint_dir",
        "model_title",
    ]

    # First check it's actually a dictionary
    if not isinstance(train_config, dict):
        raise ValueError("train_config must be a dictionary.")

    # Check that all required keys are present
    for key in required_keys:
        if key not in train_config:
            raise ValueError(f"train_config is missing required key: {key}.")

    # Validate optimizer
    if train_config["optimizer"] not in optimizers:
        raise ValueError(f"optimizer must be one of: {optimizers}.")

    # Validate learning_rate
    if (
        type(train_config["learning_rate"]) not in [int, float]
        or train_config["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be a number greater than 0.")

    # Validate num_epochs
    if type(train_config["num_epochs"]) is not int or train_config["num_epochs"] < 1:
        raise ValueError("num_epochs must be an integer greater than or equal to 1.")

    # Validate warmup_epochs
    if (
        type(train_config["warmup_epochs"]) is not int
        or train_config["warmup_epochs"] < 0
        or train_config["warmup_epochs"] > train_config["num_epochs"]
    ):
        raise ValueError(
            "warmup_epochs must be an integer between 0 and num_epochs."
        )

    # Validate warmup_start_factor
    if (
        type(train_config["warmup_start_factor"]) not in [int, float]
        or not 0 < train_config["warmup_start_factor"] <= 1
    ):
        raise ValueError(
            "warmup_start_factor must be a number greater than 0 and at most 1."
        )

    # Validate min_learning_rate
    if (
        type(train_config["min_learning_rate"]) not in [int, float]
        or not 0 < train_config["min_learning_rate"] <= train_config["learning_rate"]
    ):
        raise ValueError(
            "min_learning_rate must be a number greater than 0 and no greater than "
            "learning_rate."
        )

    # Validate momentum
    if (
        type(train_config["momentum"]) not in [int, float]
        or not 0 <= train_config["momentum"] <= 1
    ):
        raise ValueError("momentum must be a number between 0 and 1.")

    # Validate weight_decay
    if (
        type(train_config["weight_decay"]) not in [int, float]
        or train_config["weight_decay"] < 0
    ):
        raise ValueError("weight_decay must be a number greater than or equal to 0.")

    # Validate checkpoint_dir
    checkpoint_dir = train_config["checkpoint_dir"]
    if checkpoint_dir is not None and (
        not isinstance(checkpoint_dir, str) or checkpoint_dir == ""
    ):
        raise ValueError("checkpoint_dir must be a non-empty string or None.")

    # Validate model_title
    if (
        not isinstance(train_config["model_title"], str)
        or train_config["model_title"] == ""
    ):
        raise ValueError("model_title must be a non-empty string.")
