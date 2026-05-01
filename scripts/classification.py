import json
from data.builder import get_dataloaders

# --- Setup --- #

# Target label for classification
# Choose from 'object', 'object_region', 'force_level', 'hardness'
# All these are categorical labels ('materials' is multiclass and is handled separately)
TARGET_LABEL = "object"

# Get default data config from json file
with open("configs/default_data_config.json", "r", encoding="utf-8") as f:
    data_config = json.load(f)

# Override any config parameters here if needed
# data_config['split_size'] = 0.20
# data_config['norm_type'] = None

# --- Load and prepare the dataset --- #
dataloaders, label_info = get_dataloaders(data_config)
print(dataloaders["train"].dataset.df.shape)
print(label_info["materials_list"])

# Create folder if it doesn't exist in the results directory for this target label
# (inc plots subfolder)
# Train model, evaluate, and print results, inc classification report
# Save results to 'results/{TARGET_LABEL}_classification' folder in parquet format
# Use visualise.py to plot comparisons (baseline, resnet, vit) for test data and training
# curves and confusion matrices
# Save plots to 'plots' subfolder

# Example
# results
# --object_classification
# ----history.parquet
# ----results.csv
# ----results.parquet
# ----plots
# ------train_curves.png
# ------test_acc.png
# ------test_loss.png
# ------confusion_matrices.png
