"""
A script to compare the performance of different models on a classification task.
Results are saved in the results directory.

Author: Gemma McLean
Date: April 2026
"""

from models.experiments import classify

# --- Configurable parameters --- #

# Target label for classification
# Choose from 'object', 'region', 'object_region', 'force_level', or 'motion'
TARGET_LABEL = "object"
# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_quick"
# Set random seed for reproducibility
SEED = 129
# Set deterministic behavior for PyTorch
DETERMINISTIC = True
# Model types to compare
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'sparsh, 'anytouch'
MODEL_TYPES = ["resnet18"]
# Set to False to skip t-SNE feature plots.
PLOT_TSNE = True
# Maximum number of class-balanced samples used in each t-SNE plot.
TSNE_MAX_SAMPLES = 2_000

# --- Config overrides --- #

# Values here override keys in configs/default_data_config.json.
DATA_CONFIG_OVERRIDES = {}

# Values here override keys in configs/default_train_config.json.
TRAIN_CONFIG_OVERRIDES = {
    "num_epochs": 1,
}

# --- Train and evaluate models --- #

classify(
    model_types=MODEL_TYPES,
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    deterministic=DETERMINISTIC,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
    plot_tsne=PLOT_TSNE,
    tsne_max_samples=TSNE_MAX_SAMPLES,
)
