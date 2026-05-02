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
from models.train_eval import train_classifier
from models.torch_functions import get_device

# --- Setup --- #

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', 'hardness'
# All these are categorical labels ('materials' is multiclass and is handled separately)
TARGET_LABEL = "object"
# Folder name for saving results and checkpoints
FOLDER_NAME = f"{TARGET_LABEL}_classification_comparison"
# Set random seed for reproducibility
SEED = 27
# Model types to compare
MODEL_TYPES = ["baseline", "resnet", "vit"]
# Get device
DEVICE = get_device()

# Set torch random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Load and prepare the dataset --- #

# Get default data config from json file
with open("configs/default_data_config.json", "r", encoding="utf-8") as f:
    data_config = json.load(f)

# Update default data config with custom settings
data_config["target_label"] = TARGET_LABEL
data_config["random_state"] = SEED

# Create a copy of the data config for each model type
data_configs = [data_config.copy() for _ in MODEL_TYPES]

# Set normalisation type to imagenet for resnet and vit models
# for config in data_configs[1:]:
#     config["norm_type"] = "imagenet"

# Get (dataloaders, label_info) tuples for each fold using the different data configs
data = [get_dataloaders(cfg) for cfg in data_configs]
# Get dataloaders list from the data tuples
dataloaders = [item[0] for item in data]
# Label_info is the same for all configs
label_info = data[0][1]

# Extract number of classes from the label info for the target label
num_classes = len(label_info["label2idx"][TARGET_LABEL].keys())

# --- Prepare models --- #

# Get default model config from json file
with open("configs/default_model_config.json", "r", encoding="utf-8") as f:
    model_config = json.load(f)

# Update default model config with custom settings
model_config["checkpoint_dir"] = f"checkpoints/{FOLDER_NAME}"
model_config["num_epochs"] = 1  # Set to 1 for testing

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

# Convert histories to Pandas DataFrame
# Columns should start from 1 instead of 0 to match epoch numbers
histories_df = pd.DataFrame(
    histories,
    index=MODEL_TYPES,
    columns=[f"epoch_{i + 1}" for i in range(len(histories[0]))],
)
# Create folder if it doesn't exist in the results directory
os.makedirs(f"results/{FOLDER_NAME}", exist_ok=True)
# Save Dataframe to parquet file
histories_df.to_parquet(f"results/{FOLDER_NAME}/histories.parquet", index=True)
# Save model and data configs
with open(f"results/{FOLDER_NAME}/model_configs.json", "w", encoding="utf-8") as f:
    json.dump(model_configs, f, indent=4)
with open(f"results/{FOLDER_NAME}/data_configs.json", "w", encoding="utf-8") as f:
    json.dump(data_configs, f, indent=4)
