"""
A script to compare one model across every combination of named data and training
configuration variants. Results are saved in the results directory.

Author: Gemma McLean
Date: July 2026
"""

from ml.experiments import classify_sweep


# --- Configurable parameters --- #

# Chosen model type for sweep experiments
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'deit_tiny', or
# 't3_tiny'
MODEL_TYPE = "resnet18"

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', or 'motion'
TARGET_LABEL = "object"

# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_filter_sweep"

# Randomisation settings
SEED = 129
DETERMINISTIC = True


# Set to True to train only the classifier of pretrained models
# Baseline models are always trained end-to-end
FREEZE_BACKBONE = False

# Each data variant is combined with every training variant below
# Values override keys in the chosen data config file
DATA_CONFIG_VARIANTS = {
    "force_level_1": {
        "filtered_force_level": "1",
        "filtered_motion": None,
    },
    "force_level_2": {
        "filtered_force_level": "2",
        "filtered_motion": None,
    },
    "force_level_3": {
        "filtered_force_level": "3",
        "filtered_motion": None,
    },
    "sliding": {
        "filtered_force_level": None,
        "filtered_motion": "sliding",
    },
    "rotation": {
        "filtered_force_level": None,
        "filtered_motion": "rotation",
    },
}

# Each training variant is combined with every data variant above
# Values override keys in the chosen training config file
TRAIN_CONFIG_VARIANTS = {
    "default": {},
}

# t-SNE feature plot settings
PLOT_TSNE = False
TSNE_MAX_SAMPLES = -1

# Paths for files and directories
DATA_CONFIG_PATH = "configs/pad_jitter_data_config.json"
TRAIN_CONFIG_PATH = (
    "configs/frozen_train_config.json"
    if FREEZE_BACKBONE
    else "configs/finetuned_train_config.json"
)
BASELINE_TRAIN_CONFIG_PATH = "configs/baseline_train_config.json"
RESULTS_DIR = "/home/gemma/development/python/touch-ex-results/results"
CHECKPOINT_DIR = "checkpoints"

# --- Train and evaluate the configuration sweep --- #

classify_sweep(
    model_type=MODEL_TYPE,
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    deterministic=DETERMINISTIC,
    freeze_backbone=FREEZE_BACKBONE,
    data_config_variants=DATA_CONFIG_VARIANTS,
    train_config_variants=TRAIN_CONFIG_VARIANTS,
    plot_tsne=PLOT_TSNE,
    tsne_max_samples=TSNE_MAX_SAMPLES,
    data_config_path=DATA_CONFIG_PATH,
    train_config_path=TRAIN_CONFIG_PATH,
    baseline_train_config_path=BASELINE_TRAIN_CONFIG_PATH,
    results_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)
