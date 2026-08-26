"""Train a ResNet-18 tactile regressor for force_n or fsr_voltage."""

from ml.experiments import regress


# --- Configurable parameters --- #

# Choose either "force_n" or "fsr_voltage"
REGRESSION_TARGET = "force_n"
EXPERIMENT_NAME = "resnet18_force_n_regression"
SEED = 129
DETERMINISTIC = True

# Set to True to train only the final regression head
FREEZE_BACKBONE = False

# Values here override keys in the provided data and training configuration files
DATA_CONFIG_OVERRIDES = {}
TRAIN_CONFIG_OVERRIDES = {"num_epochs": 3}

DATA_CONFIG_PATH = "configs/default_data_config.json"
TRAIN_CONFIG_PATH = "configs/finetuned_train_config.json"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"


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
