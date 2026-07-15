"""
A script to compare the performance of different models on a classification task.
Results are saved in the results directory.

Author: Gemma McLean
Date: April 2026
"""

from models.experiments import classify

# --- CONFIGURABLE PARAMETERS --- #

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', or 'motion'
TARGET_LABEL = "object"
# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_quick"
# Set random seed for reproducibility
SEED = 146
# Model types to compare
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'sparsh, 'anytouch'
MODEL_TYPES = ["resnet18"]

# --- CONFIGURATION OVERRIDES --- #

# Values here override keys in configs/default_data_config.json.
DATA_CONFIG_OVERRIDES = {}

# Values here override keys in configs/default_train_config.json.
TRAIN_CONFIG_OVERRIDES = {
    # "optimizer": "adamw",
    "weight_decay": 0.02,
    "learning_rate": 0.0002,
    "num_epochs": 1,
}

# --- Train and evaluate models --- #

classify(
    model_types=MODEL_TYPES,
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
)
