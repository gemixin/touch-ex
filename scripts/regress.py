"""
A script to train and evaluate a pretrained model on a regression task. Results are saved
in the results directory.

Author: Gemma McLean
Date: August 2026
"""

from ml.experiments import regress


# --- Configurable parameters --- #

# Pretrained backbone: resnet18, resnet50, efficientnet_b0, vit_b_16, deit_tiny, t3_tiny
MODEL_TYPE = "resnet18"

# Target for regression
# Choose from 'force_n' or 'fsr_voltage'
REGRESSION_TARGET = "fsr_voltage"

# Experiment name for tracking results
EXPERIMENT_NAME = f"{MODEL_TYPE}_main"

# Use model_selection for configuration/model choice; final_evaluation for testing
EXPERIMENT_PURPOSE = "model_selection"

# Randomisation settings (dataset partitioning always uses seed 129)
SEED = 129
DETERMINISTIC = True

# Set to True to train only the regression head
FREEZE_BACKBONE = False

# Values here override keys in provided data_config and train_config files
DATA_CONFIG_OVERRIDES = {}
TRAIN_CONFIG_OVERRIDES = {"early_stopping_min_delta": 0.001}

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

regress(
    model_type=MODEL_TYPE,
    regression_target=REGRESSION_TARGET,
    experiment_name=EXPERIMENT_NAME,
    experiment_purpose=EXPERIMENT_PURPOSE,
    seed=SEED,
    deterministic=DETERMINISTIC,
    freeze_backbone=FREEZE_BACKBONE,
    data_config_overrides=DATA_CONFIG_OVERRIDES,
    train_config_overrides=TRAIN_CONFIG_OVERRIDES,
    data_config_path=DATA_CONFIG_PATH,
    train_config_path=TRAIN_CONFIG_PATH,
    results_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)
