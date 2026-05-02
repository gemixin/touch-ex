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
FOLDER_NAME = f"{TARGET_LABEL}_classification"
# Set random seed for reproducibility
SEED = 25
# Model type
# Choose from 'baseline', 'resnet', 'vit'
MODEL_TYPE = "baseline"
# Set a title for the model (defaults to model type but can be customised here)
# e.g. 'vit_adamw' if using the AdamW optimizer with the ViT model
MODEL_TITLE = MODEL_TYPE
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

# Get (dataloaders, label_info) tuple
dataloaders, label_info = get_dataloaders(data_config)

# Extract number of classes from the label info for the target label
num_classes = len(label_info["label2idx"][TARGET_LABEL].keys())

# --- Prepare model --- #

# Get default model config from json file
with open("configs/default_model_config.json", "r", encoding="utf-8") as f:
    model_config = json.load(f)

# Update default model config with custom settings
model_config["checkpoint_dir"] = f"checkpoints/{FOLDER_NAME}"
model_config["model_title"] = MODEL_TITLE
model_config["num_epochs"] = 3

# --- Train model --- #

# Create the model
if MODEL_TYPE == "baseline":
    model = BaselineCNNModel(num_classes=num_classes)
else:
    model = PretrainedModel(model_type=MODEL_TYPE, num_classes=num_classes)

# Train the model and save the history
model, history = train_classifier(
    model=model,
    device=DEVICE,
    train_loader=dataloaders["train"],
    val_loader=dataloaders["val"],
    target_label=TARGET_LABEL,
    config=model_config,
)

# Convert history to Pandas DataFrame with epoch as index
history_df = pd.DataFrame(history).set_index("epoch")
# Create folder if it doesn't exist in the results directory
os.makedirs(f"results/{FOLDER_NAME}", exist_ok=True)
# Save Dataframe to parquet file
history_df.to_parquet(f"results/{FOLDER_NAME}/{MODEL_TITLE}_history.parquet", index=True)
# Save model and data configs
with open(
    f"results/{FOLDER_NAME}/{MODEL_TITLE}_model_config.json", "w", encoding="utf-8"
) as f:
    json.dump(model_config, f, indent=4)
with open(
    f"results/{FOLDER_NAME}/{MODEL_TITLE}_data_config.json", "w", encoding="utf-8"
) as f:
    json.dump(data_config, f, indent=4)
