"""
A script to compare the performance of different models (or a single model) on a
classification task. Results are saved in the results directory.

Author: Gemma McLean
Date: April 2026
"""

from ml.experiments import classify


# --- Configurable parameters --- #

# Model types to compare
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'deit_tiny', or
# 't3_tiny'
MODEL_TYPES = [
    "baseline",
    "resnet18",
    "efficientnet_b0",
    "vit_b_16",
    "deit_tiny",
    "t3_tiny",
]

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', or 'motion'
TARGET_LABEL = "object"

# Experiment name for tracking results
EXPERIMENT_NAME = "finetuned_comparison"

# Randomisation settings
SEED = 129
DETERMINISTIC = True

# Set to True to train only the classifier of pretrained models
# Baseline models are always trained end-to-end
FREEZE_BACKBONE = False

# Values here override keys in provided data_config and train_config files
DATA_CONFIG_OVERRIDES = {}
TRAIN_CONFIG_OVERRIDES = {}

# t-SNE feature plot settings
PLOT_TSNE = True
TSNE_MAX_SAMPLES = -1

# Paths for files and directories
DATA_CONFIG_PATH = "configs/default_data_config.json"
TRAIN_CONFIG_PATH = (
    "configs/frozen_train_config.json"
    if FREEZE_BACKBONE
    else "configs/finetuned_train_config.json"
)
BASELINE_TRAIN_CONFIG_PATH = "configs/baseline_train_config.json"
# RESULTS_DIR = "/home/gemma/development/python/touch-ex-results/results"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

# --- Train and evaluate models --- #

classify(
    model_types=MODEL_TYPES,
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    deterministic=DETERMINISTIC,
    freeze_backbone=FREEZE_BACKBONE,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
    plot_tsne=PLOT_TSNE,
    tsne_max_samples=TSNE_MAX_SAMPLES,
    data_config_path=DATA_CONFIG_PATH,
    train_config_path=TRAIN_CONFIG_PATH,
    baseline_train_config_path=BASELINE_TRAIN_CONFIG_PATH,
    results_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)
