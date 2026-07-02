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
import models.visualise as mv

# --- CONFIGURABLE PARAMETERS --- #

# Target label for classification
# Choose from 'object', 'object_region', 'force_level'
TARGET_LABEL = "object"
# Experiment name for tracking results
EXPERIMENT_NAME = "efficientnet_10"
# Set random seed for reproducibility
SEED = 146
# Model types to compare
# Choose from 'baseline', 'resnet18', 'efficientnet_b0', 'vit_b_16', 'sparsh, 'anytouch'
MODEL_TYPES = ["efficientnet_b0"]

# --- Setup --- #

# Get device
DEVICE = get_device()

# Folder name for saving results and checkpoints
# We will use a new folder for each target label
FOLDER_NAME = f"{TARGET_LABEL}_classify"

# Paths for saving and loading checkpoints and results and configs
CHECKPOINTS_PATH = f"checkpoints/{FOLDER_NAME}"
RESULTS_PATH = f"results/{FOLDER_NAME}"
EXPERIMENTS_DF_PATH = f"{RESULTS_PATH}/experiments.parquet"
DATA_CONFIG_PATH = "configs/default_data_config.json"
MODEL_CONFIG_PATH = "configs/default_model_config.json"

# Set torch random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Get experiment number --- #

# Check if an experiments dataframe parquet file already exists for this folder
if os.path.exists(EXPERIMENTS_DF_PATH):
    # If it exists, load the existing file into a DataFrame
    existing_results_df = pd.read_parquet(EXPERIMENTS_DF_PATH)
    # Get max experiment number and increment it
    experiment_number = existing_results_df["experiment_number"].max() + 1
    concat = True
# Otherwise
else:
    # Start a new experiment number at 1
    experiment_number = 1
    concat = False

# --- Load and prepare the dataset --- #

# Get default data config from json file
with open(DATA_CONFIG_PATH, "r", encoding="utf-8") as f:
    data_config = json.load(f)

# Update default data config with custom settings
data_config["stratify_label"] = TARGET_LABEL
data_config["random_state"] = SEED
data_config["batch_size"] = 64

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

# Set checkpoint directory
# (new folder for each experiment number within the target label folder)
model_config["checkpoint_dir"] = (
    f"checkpoints/{FOLDER_NAME}/{str(experiment_number).zfill(3)}"
)

# Update default model config with custom settings
# model_config["optimizer"] = "adamw"
model_config["weight_decay"] = 0.02
model_config["learning_rate"] = 0.0002

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
        target_label=TARGET_LABEL,
        device=DEVICE,
    )

    # Append the test result to the list
    results.append(result)

# --- Save experiment data --- #

# Create a new DataFrame with relevant information for this experiment
df = pd.DataFrame(
    {
        "experiment_number": experiment_number,
        "experiment_name": EXPERIMENT_NAME,
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

# If there is an existing experiments dataframe
if concat:
    # Concatenate the existing and new dataframes
    new_df = pd.concat([existing_results_df, df], ignore_index=True)
# Otherwise
else:
    # Create folder if it doesn't exist in the results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    # Just use the new dataframe
    new_df = df

# Save DataFrame to parquet file
new_df.to_parquet(EXPERIMENTS_DF_PATH, index=False)

# --- Generate plots --- #

# Path for saving plots
plots_path = f"{RESULTS_PATH}/plots/{str(experiment_number).zfill(3)}"

# If there are multiple models, plot the model comparison
if len(MODEL_TYPES) > 1:
    mv.plot_model_comparison(results, MODEL_TYPES, plots_path)

# Plot training curves for each model
mv.plot_training_curves(histories, MODEL_TYPES, plots_path)

# Plot confusion matrices for each model
mv.plot_confusion_matrices(results, MODEL_TYPES, list_classes, plots_path)
