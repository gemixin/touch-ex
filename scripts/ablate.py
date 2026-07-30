"""
A script to compare the performance of a single model across force-level and motion data
ablations. Results are saved in the results directory.

Author: Gemma McLean
Date: July 2026
"""

from models.experiments import classify_ablation


# --- Configurable parameters --- #

# Chosen model type for ablation experiments
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'deit_tiny', or
# 't3_tiny'
MODEL_TYPE = "resnet18"

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', or 'motion'
TARGET_LABEL = "object"

# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_ablation"

# Randomisation settings
SEED = 129
DETERMINISTIC = True

# Set to True to train only the classifier of pretrained models
# Baseline models are always trained end-to-end
FREEZE_BACKBONE = False

# Each run trains the selected model using its corresponding data filter
DATA_CONFIG_OVERRIDES = {
    "all_data": {},
    "force_level_1": {"filtered_force_level": "1"},
    "force_level_2": {"filtered_force_level": "2"},
    "force_level_3": {"filtered_force_level": "3"},
    "sliding": {"filtered_motion": "sliding"},
    "rotation": {"filtered_motion": "rotation"},
}

# Values here override keys in the data config for every ablation run
# Run-specific values in DATA_CONFIG_OVERRIDES take precedence
SHARED_DATA_CONFIG_OVERRIDES = {}

# Values here override keys in provided train_config file
TRAIN_CONFIG_OVERRIDES = {}

# t-SNE feature plot settings
PLOT_TSNE = False
TSNE_MAX_SAMPLES = -1

# Paths for files and directories
DATA_CONFIG_PATH = "configs/default_data_config.json"
TRAIN_CONFIG_PATH = (
    "configs/frozen_train_config.json"
    if FREEZE_BACKBONE
    else "configs/finetuned_train_config.json"
)
BASELINE_TRAIN_CONFIG_PATH = "configs/baseline_train_config.json"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

# --- Train and evaluate ablations --- #

classify_ablation(
    model_type=MODEL_TYPE,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    shared_data_config_overrides=SHARED_DATA_CONFIG_OVERRIDES,
    target_label=TARGET_LABEL,
    experiment_name=EXPERIMENT_NAME,
    seed=SEED,
    deterministic=DETERMINISTIC,
    freeze_backbone=FREEZE_BACKBONE,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
    plot_tsne=PLOT_TSNE,
    tsne_max_samples=TSNE_MAX_SAMPLES,
    data_config_path=DATA_CONFIG_PATH,
    train_config_path=TRAIN_CONFIG_PATH,
    baseline_train_config_path=BASELINE_TRAIN_CONFIG_PATH,
    results_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)
