"""
A script to train and evaluate a ResNet-18 model on a regression task. Results are saved
in the results directory.

Author: Gemma McLean
Date: August 2026
"""

from ml.experiments import regress


# --- Configurable parameters --- #

# Target for regression
# Choose from 'force_n' or 'fsr_voltage'
REGRESSION_TARGET = "force_n"

# Experiment name for tracking results
EXPERIMENT_NAME = "resnet18_force_n_regression"

# Randomisation settings
SEED = 129
DETERMINISTIC = True

# Set to True to train only the regression head
FREEZE_BACKBONE = False

# Values here override keys in provided data_config and train_config files
DATA_CONFIG_OVERRIDES = {}
TRAIN_CONFIG_OVERRIDES = {}

# Paths for files and directories
DATA_CONFIG_PATH = "configs/default_data_config.json"
TRAIN_CONFIG_PATH = (
    "configs/frozen_train_config.json"
    if FREEZE_BACKBONE
    else "configs/finetuned_train_config.json"
)
RESULTS_DIR = "/home/gemma/development/python/touch-ex-results/results"
CHECKPOINT_DIR = "checkpoints"

# --- Train and evaluate model --- #

regress(
    regression_target=REGRESSION_TARGET,
    experiment_name=EXPERIMENT_NAME,
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
