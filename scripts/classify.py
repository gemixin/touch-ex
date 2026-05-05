"""
A script to compare the performance of different models on a classification task.
Results are saved in the results directory.

Author: Gemma McLean
Date: April 2026
"""

import json
import pandas as pd
import os
import torch
from data.builder import get_dataloaders
from models.baseline import BaselineCNNModel
from models.pretrained import PretrainedModel
from models.train_eval import train_classifier, eval_classifier
from models.torch_functions import get_device

# --- CONFIGURABLE PARAMETERS --- #

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', 'hardness'
# All these are categorical labels ('materials' is multiclass and is handled separately)
TARGET_LABEL = "object"
# Experiment name for tracking results
EXPERIMENT_NAME = "adam_5_epochs"
# Set random seed for reproducibility
SEED = 275
# Model types to compare
# Choose from 'baseline', 'resnet18', 'resnet50', 'vit_b_16'
MODEL_TYPES = ["baseline"]

# --- Setup --- #

# Get device
DEVICE = get_device()

# Folder name for saving results and checkpoints
# We will use a new folder for each target label
FOLDER_NAME = f"{TARGET_LABEL}_classify"

# Paths for saving outputs and loading configs
OUTPUTS_PATH = f"results/{FOLDER_NAME}/outputs.parquet"
DATA_CONFIG_PATH = "configs/default_data_config.json"
MODEL_CONFIG_PATH = "configs/default_model_config.json"

# Set torch random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Get experiment number --- #

# Check if an outputs parquet file already exists for this folder
if os.path.exists(OUTPUTS_PATH):
    # If it exists, load the existing outputs into a DataFrame
    existing_results_df = pd.read_parquet(OUTPUTS_PATH)
    # Get max experiment number and increment it
    experiment_number = existing_results_df["experiment_number"].max() + 1
    concatenate_outputs = True
# Otherwise
else:
    # Start a new experiment number at 1
    experiment_number = 1
    concatenate_outputs = False

# --- Load and prepare the dataset --- #

# Get default data config from json file
with open(DATA_CONFIG_PATH, "r", encoding="utf-8") as f:
    data_config = json.load(f)

# Update default data config with custom settings
data_config["target_label"] = TARGET_LABEL
data_config["random_state"] = SEED

# Create a copy of the data config for each model type
data_configs = [data_config.copy() for _ in MODEL_TYPES]

# Get (dataloaders, label_info) tuples for each fold using the different data configs
data = [get_dataloaders(cfg) for cfg in data_configs]
# Get dataloaders list from the data tuples
dataloaders = [item[0] for item in data]
# Label_info is the same for all configs
label_info = data[0][1]

# Extract number of classes from the label info for the target label
list_classes = list(label_info["label2idx"][TARGET_LABEL].keys())
num_classes = len(list_classes)

# --- Prepare models --- #

# Get default model config from json file
with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
    model_config = json.load(f)


# Update default model config with custom settings
model_config["checkpoint_dir"] = (
    f"checkpoints/{FOLDER_NAME}/{str(experiment_number).zfill(3)}"
)
model_config["learning_rate"] = 0.004
model_config["num_epochs"] = 1

# Create a copy of the model config for each model type
model_configs = [model_config.copy() for _ in MODEL_TYPES]

# Set model title in the model config for each model type
for config, model_type in zip(model_configs, MODEL_TYPES):
    config["model_title"] = model_type

# --- Train models --- #

# Create empty lists to store models and histories
models = []
histories = []

# Loop through each model type
for i in range(len(dataloaders)):
    # Create the model based on the model type
    if MODEL_TYPES[i] == "baseline":
        model = BaselineCNNModel(num_classes=num_classes)
    else:
        model = PretrainedModel(model_type=MODEL_TYPES[i], num_classes=num_classes)

    # Train the model and save the history
    model, history = train_classifier(
        model=model,
        device=DEVICE,
        train_loader=dataloaders[i]["train"],
        val_loader=dataloaders[i]["val"],
        target_label=TARGET_LABEL,
        config=model_configs[i],
    )

    # Append the model and history to the respective lists
    models.append(model)
    histories.append(history)

# --- Evaluate models --- #

# Create empty list to store results
results = []

# Loop through each model
for i in range(len(dataloaders)):
    # Evaluate the current model on the test set
    result = eval_classifier(
        model=models[i],
        model_title=MODEL_TYPES[i],
        test_loader=dataloaders[i]["test"],
        target_label="object",
        device=DEVICE,
    )

    # Append the result to the results list
    results.append(result)

# --- Save outputs --- #

# Create a new DataFrame for the new outputs
df = pd.DataFrame(
    {
        "experiment_number": experiment_number,
        "experiment_name": FOLDER_NAME,
        "model_config": model_configs,
        "data_config": data_configs,
        "model_type": MODEL_TYPES,
        "list_classes": [list_classes for _ in range(len(MODEL_TYPES))],
        "history": histories,
        "test_acc": [result["test_acc"] for result in results],
        "test_loss": [result["test_loss"] for result in results],
        "weighted_f1_avg": [result["weighted_f1_avg"] for result in results],
        "y_pred": [result["y_pred"] for result in results],
        "y_true": [result["y_true"] for result in results],
    }
)

# If there are existing outputs
if concatenate_outputs:
    # Concatenate the existing and new dataframes
    new_df = pd.concat([existing_results_df, df], ignore_index=True)
# Otherwise
else:
    # Create folder if it doesn't exist in the results directory
    os.makedirs(f"results/{FOLDER_NAME}", exist_ok=True)
    # Just use the new dataframe
    new_df = df

# Save DataFrame to parquet file
new_df.to_parquet(OUTPUTS_PATH, index=False)
