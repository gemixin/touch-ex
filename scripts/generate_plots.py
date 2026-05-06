"""
A script to generate plots for a given experiment number and target label, using the results
stored in the associated experiments dataframe.

Author: Gemma McLean
Date: April 2026
"""

import pandas as pd
import models.visualise as mv

# --- CONFIGURABLE PARAMETERS --- #

# Experiment we want to generate plots for
TARGET_LABEL = "object"
EXPERIMENT_NUMBER = 1

# --- Setup --- #

# Paths for saving and loading
FOLDER_NAME = f"{TARGET_LABEL}_classify"
RESULTS_PATH = f"results/{TARGET_LABEL}_classify"
EXPERIMENTS_DF_PATH = f"{RESULTS_PATH}/experiments.parquet"
PLOTS_FOLDER = f"{RESULTS_PATH}/plots/{str(EXPERIMENT_NUMBER).zfill(3)}"

# --- Load experiment data --- #

# Import experiments dataframe
df = pd.read_parquet(EXPERIMENTS_DF_PATH)

# Load the rows for the chosen experiment number
df_experiment = df[df["experiment_number"] == EXPERIMENT_NUMBER]
if df_experiment.empty:
    raise ValueError(f"Experiment number {EXPERIMENT_NUMBER} not found in DataFrame.")

# --- Extract required info from the dataframe --- #

# Get list of result dictionaries for each row (model type)
# (test_acc, test_loss, weighted_f1_avg, y_true, y_pred)
results = df_experiment[
    ["model_type", "test_acc", "test_loss", "weighted_f1_avg", "y_true", "y_pred"]
].to_dict(orient="records")

# Get list of training histories for each row (model type)
histories = df_experiment["history"].tolist()

# Get list of model types
model_types = df_experiment["model_type"].tolist()

# Get list of classes from list_classes column (same for all rows)
list_classes = df_experiment.iloc[0]["list_classes"].tolist()

# --- Generate plots --- #

# If there are multiple models, plot the model comparison
if len(model_types) > 1:
    mv.plot_model_comparison(results, model_types, PLOTS_FOLDER)

# Plot training curves for each model
mv.plot_training_curves(histories, model_types, PLOTS_FOLDER)

# Plot confusion matrices for each model
mv.plot_confusion_matrices(results, model_types, list_classes, PLOTS_FOLDER)
