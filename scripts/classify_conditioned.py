"""
A script to train and evaluate a conditioned ResNet-18 model on an object or
object-region classification task. Force level is used as the conditioning input.
Results are saved in the results directory.

Author: Gemma McLean
Date: September 2026
"""

from ml.experiments import classify_conditioned


# --- Configurable parameters --- #

# Target label for classification
# Choose from 'object' or 'object_region'
TARGET_LABEL = "object_region"

# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_force_level_conditioned"

# Randomisation settings
SEED = 129
DETERMINISTIC = True

# Set to True to train only the fusion and classification layers
FREEZE_BACKBONE = False

# Fusion multilayer perceptron settings
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.2

# Values here override keys in provided data_config and train_config files
DATA_CONFIG_OVERRIDES = {}
TRAIN_CONFIG_OVERRIDES = {}

# t-SNE feature plot settings
PLOT_TSNE = True
TSNE_MAX_SAMPLES = -1

# Paths for files and directories
DATA_CONFIG_PATH = "configs/pad_jitter_data_config.json"
TRAIN_CONFIG_PATH = (
    "configs/frozen_train_config.json"
    if FREEZE_BACKBONE
    else "configs/finetuned_train_config.json"
)
RESULTS_DIR = "/home/gemma/development/python/touch-ex-results/results"
CHECKPOINT_DIR = "checkpoints"

# --- Train and evaluate model --- #

classify_conditioned(
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    deterministic=DETERMINISTIC,
    freeze_backbone=FREEZE_BACKBONE,
    fusion_hidden_dim=FUSION_HIDDEN_DIM,
    fusion_dropout=FUSION_DROPOUT,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
    plot_tsne=PLOT_TSNE,
    tsne_max_samples=TSNE_MAX_SAMPLES,
    data_config_path=DATA_CONFIG_PATH,
    train_config_path=TRAIN_CONFIG_PATH,
    results_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)
