"""
A module containing functions for preparing and running experiments.

Author: Gemma McLean
Date: July 2026
"""

import json
import os
import pandas as pd
from data.builder import get_dataloaders
from models.baseline import BaselineCNNModel
from models.pretrained import DEIT_TINY_CHECKPOINT, PretrainedModel
from models.t3 import T3_REPOSITORY, T3_REVISION
from models.train_eval import eval_classifier, train_classifier
from models.torch_functions import get_device, set_random_seed
import models.visualise as mv


# Map each target label to its unseen-evaluation label and whether its true and
# expected labels use different class spaces
UNSEEN_EVALUATION_TARGETS = {
    "object": {"target_label": "expected_object", "cross_space": True},
    "object_region": {"target_label": "expected_object_region", "cross_space": True},
    "force_level": {"target_label": "force_level", "cross_space": False},
    "motion": {"target_label": "motion", "cross_space": False},
}


def classify(
    model_types,
    target_label,
    experiment_name,
    seed,
    deterministic,
    freeze_backbone,
    data_config_overrides,
    train_config_overrides,
    plot_tsne,
    tsne_max_samples,
    data_config_path,
    train_config_path,
    results_dir,
    checkpoint_dir,
):
    """
    Prepare, train, evaluate, save, and plot a classification experiment.

    Args:
        model_types (list): A list of model types to be used in the experiment.
        target_label (str): The target label for classification.
        experiment_name (str): A name for the experiment, used for saving results.
        seed (int): Random seed for reproducibility.
        deterministic (bool): Whether to require deterministic PyTorch algorithms.
        freeze_backbone (bool): Whether to freeze pretrained model backbones. Baseline
            models are always trained end-to-end.
        data_config_overrides (dict): A dictionary containing overrides for the default data
            configuration.
        train_config_overrides (dict): A dictionary containing overrides for the default
            training configuration.
        plot_tsne (bool): Whether to generate t-SNE plots.
        tsne_max_samples (int): Maximum samples per t-SNE plot. -1 indicates that all test
            samples should be used.
        data_config_path (str): Path to the default data configuration JSON file.
        train_config_path (str): Path to the default training configuration JSON file.
        results_dir (str): Root directory for experiment results.
        checkpoint_dir (str): Root directory for model checkpoints.

    Returns:
        tuple: A tuple containing the trained models, training histories, and evaluation
            results.
    """

    # Check that model_types is not empty
    if not model_types:
        raise ValueError("model_types must contain at least one model type.")
    # Check that target_label is valid
    if target_label not in UNSEEN_EVALUATION_TARGETS:
        raise ValueError(
            "target_label must be 'object', 'object_region', 'force_level', or 'motion'."
        )
    # Check that the configuration files exist
    if not os.path.isfile(data_config_path):
        raise FileNotFoundError(f"Data configuration file not found: {data_config_path}")
    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(
            f"Training configuration file not found: {train_config_path}"
        )

    # Prepare the experiment context
    experiment = prepare_experiment(
        model_types=model_types,
        target_label=target_label,
        seed=seed,
        deterministic=deterministic,
        freeze_backbone=freeze_backbone,
        data_config_overrides=data_config_overrides,
        train_config_overrides=train_config_overrides,
        plot_tsne=plot_tsne,
        tsne_max_samples=tsne_max_samples,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
    )

    # Train, evaluate, save, and plot the experiment
    train_models(experiment)
    evaluate_models(experiment)
    save_experiment_results(experiment, experiment_name)
    generate_experiment_plots(experiment)

    # Return the outputs as a tuple of (models, histories, test_results)
    return experiment["models"], experiment["histories"], experiment["test_results"]


def prepare_experiment(
    model_types,
    target_label,
    seed,
    deterministic,
    freeze_backbone,
    data_config_overrides,
    train_config_overrides,
    plot_tsne,
    tsne_max_samples,
    data_config_path,
    train_config_path,
    results_dir,
    checkpoint_dir,
):
    """
    Load configurations and prepare the shared experiment context.

    Args:
        model_types (list): A list of model types to be used in the experiment.
        target_label (str): The target label for classification.
        seed (int): Random seed for reproducibility.
        deterministic (bool): Whether to require deterministic PyTorch algorithms.
        freeze_backbone (bool): Whether to freeze pretrained model backbones.
        data_config_overrides (dict): A dictionary containing overrides for the default
            data configuration.
        train_config_overrides (dict): A dictionary containing overrides for the default
            training configuration.
        plot_tsne (bool): Whether to generate t-SNE plots.
        tsne_max_samples (int): Maximum samples per t-SNE plot. -1 indicates that all test
            samples should be used.
        data_config_path (str): Path to the default data configuration JSON file.
        train_config_path (str): Path to the default training configuration JSON file.
        results_dir (str): Root directory for experiment results.
        checkpoint_dir (str): Root directory for model checkpoints.

    Returns:
        dict: A dictionary containing the prepared experiment context, including
            dataloaders, device, configurations, and paths for saving results.
    """

    # --- Setup --- #

    # Set random seeds and deterministic CUDA behavior
    set_random_seed(seed, deterministic=deterministic)

    # Get device
    device = get_device()

    # Folder name for saving results and checkpoints
    # Use a new folder for each target label
    folder_name = f"{target_label}_classify"

    # Paths for checkpoints, results, and configuration files
    checkpoints_path = os.path.join(checkpoint_dir, folder_name)
    results_path = os.path.join(results_dir, folder_name)

    # --- Get experiment number --- #

    # Path for saving the experiments dataframe parquet file
    experiments_df_path = os.path.join(results_path, "experiments.parquet")
    # Check if an experiments dataframe parquet file already exists for this folder
    if os.path.exists(experiments_df_path):
        # If it exists, load the existing file into a DataFrame
        existing_results_df = pd.read_parquet(experiments_df_path)
        # Get max experiment number and increment it
        experiment_number = existing_results_df["experiment_number"].max() + 1
    # Otherwise
    else:
        # Start a new experiment number at 1
        existing_results_df = None
        experiment_number = 1

    # --- Load and prepare the dataset --- #

    # Get default data config from json file
    with open(data_config_path, "r", encoding="utf-8") as file:
        data_config = json.load(file)

    # Update default data config with custom settings
    data_config.update(data_config_overrides)
    # Set the stratify label and random state
    data_config["stratify_label"] = target_label
    data_config["random_state"] = seed

    # Create a copy of the data config for each model type
    data_configs = [data_config.copy() for _ in model_types]

    # Get (dataloaders, label_lists) tuples for each model type using the data configs
    data = [get_dataloaders(config) for config in data_configs]
    # Get dataloaders and class-label lists from the data tuples
    dataloaders = [item[0] for item in data]
    label_lists = [item[1] for item in data]

    # Extract relevant class-label lists for target label
    train_labels = label_lists[0]["train"][target_label]
    test_unseen_matched_labels = list(label_lists[0]["test_unseen_matched"][target_label])
    test_unseen_related_labels = list(label_lists[0]["test_unseen_related"][target_label])

    # --- Prepare models --- #

    # Get default train config from json file
    with open(train_config_path, "r", encoding="utf-8") as file:
        train_config = json.load(file)

    # Update default train config with custom settings
    train_config.update(train_config_overrides)
    # Set checkpoint directory
    # Use a new folder for each experiment number within the target label folder
    train_config["checkpoint_dir"] = os.path.join(
        checkpoints_path, str(experiment_number).zfill(3)
    )

    # Create a copy of the model config for each model type
    train_configs = [train_config.copy() for _ in model_types]
    freeze_backbones = [
        model_type != "baseline" and freeze_backbone for model_type in model_types
    ]

    # Set model title in the model config for each model type
    for config, model_type in zip(train_configs, model_types):
        config["model_title"] = model_type

    # Return the experiment context as a dictionary
    return {
        "data_configs": data_configs,
        "dataloaders": dataloaders,
        "deterministic": deterministic,
        "device": device,
        "existing_results_df": existing_results_df,
        "experiment_number": experiment_number,
        "experiments_df_path": experiments_df_path,
        "model_types": model_types,
        "pretrained_checkpoints": [
            (
                f"{T3_REPOSITORY}@{T3_REVISION}"
                if model_type == "t3_tiny"
                else DEIT_TINY_CHECKPOINT
                if model_type == "deit_tiny"
                else None
            )
            for model_type in model_types
        ],
        "freeze_backbones": freeze_backbones,
        "plot_tsne": plot_tsne,
        "tsne_max_samples": tsne_max_samples,
        "target_label": target_label,
        "seed": seed,
        "results_path": results_path,
        "train_labels": train_labels,
        "test_unseen_matched_labels": test_unseen_matched_labels,
        "test_unseen_related_labels": test_unseen_related_labels,
        "train_configs": train_configs,
    }


def train_models(experiment):
    """
    Create and train the models defined by an experiment context. Modifies the experiment
    context in place to add trained models and training histories.

    Args:
        experiment (dict): A dictionary containing the prepared experiment context,
            including dataloaders, device, configurations, and paths for saving results.
    """

    # --- Train models --- #

    # Create empty lists to store models and histories
    experiment["models"] = []
    experiment["histories"] = []

    # Get the number of training classes for the target label
    num_train_classes = len(experiment["train_labels"])

    # Loop through each model type
    for i in range(len(experiment["dataloaders"])):
        # Create the model based on the model type
        if experiment["model_types"][i] == "baseline":
            model = BaselineCNNModel(num_classes=num_train_classes)
        else:
            model = PretrainedModel(
                model_type=experiment["model_types"][i],
                num_classes=num_train_classes,
                freeze_backbone=experiment["freeze_backbones"][i],
            )

        # Train the model and save the history
        model, history = train_classifier(
            model=model,
            device=experiment["device"],
            train_loader=experiment["dataloaders"][i]["train"],
            val_loader=experiment["dataloaders"][i]["val"],
            target_label=experiment["target_label"],
            train_config=experiment["train_configs"][i],
        )

        # Append the model and history to the respective lists
        experiment["models"].append(model)
        experiment["histories"].append(history)


def evaluate_models(experiment):
    """
    Evaluate the trained models in an experiment context. Modifies the experiment context in
    place to add evaluation results.

    Args:
        experiment (dict): A dictionary containing the prepared experiment context,
            including dataloaders, device, configurations, and paths for saving results.
    """

    # --- Evaluate using standard test set --- #

    # Create empty list to store results
    experiment["test_results"] = []

    # Loop through each model
    for i in range(len(experiment["dataloaders"])):
        # Evaluate the current model on the standard test set
        result = eval_classifier(
            model=experiment["models"][i],
            model_title=experiment["model_types"][i],
            device=experiment["device"],
            test_loader=experiment["dataloaders"][i]["test"],
            target_label=experiment["target_label"],
            evaluation_name="test",
        )
        # Append the test result to the list
        experiment["test_results"].append(result)

    # --- Evaluate using unseen test sets --- #

    # Get the unseen evaluation target for the experiment's target label
    unseen_evaluation = UNSEEN_EVALUATION_TARGETS[experiment["target_label"]]

    # Create empty lists to store results
    experiment["test_unseen_matched_results"] = []
    experiment["test_unseen_related_results"] = []

    # Loop through each model
    for i in range(len(experiment["dataloaders"])):
        # Evaluate the current model on the test_unseen_matched set
        result = eval_classifier(
            model=experiment["models"][i],
            model_title=experiment["model_types"][i],
            device=experiment["device"],
            test_loader=experiment["dataloaders"][i]["test_unseen_matched"],
            target_label=unseen_evaluation["target_label"],
            true_label=experiment["target_label"],
            evaluation_name="test_unseen_matched",
        )
        experiment["test_unseen_matched_results"].append(result)

    # Loop through each model
    for i in range(len(experiment["dataloaders"])):
        # Evaluate the current model on the test_unseen_related set
        result = eval_classifier(
            model=experiment["models"][i],
            model_title=experiment["model_types"][i],
            device=experiment["device"],
            test_loader=experiment["dataloaders"][i]["test_unseen_related"],
            target_label=unseen_evaluation["target_label"],
            true_label=experiment["target_label"],
            evaluation_name="test_unseen_related",
        )
        experiment["test_unseen_related_results"].append(result)


def save_experiment_results(experiment, experiment_name):
    """
    Save an experiment context's configurations, metrics, and test results to a parquet
    file.

    Args:
        experiment (dict): A dictionary containing the prepared experiment context,
            including dataloaders, device, configurations, and paths for saving results.
        experiment_name (str): A name for the experiment, used for saving results.
    """

    # --- Save experiment data --- #

    # Create a new DataFrame with relevant information for this experiment
    experiment_df = pd.DataFrame(
        {
            "experiment_number": experiment["experiment_number"],
            "experiment_name": experiment_name,
            "seed": experiment["seed"],
            "deterministic": experiment["deterministic"],
            "freeze_backbone": experiment["freeze_backbones"],
            "pretrained_checkpoint": experiment["pretrained_checkpoints"],
            "train_config": experiment["train_configs"],
            "data_config": experiment["data_configs"],
            "model_type": experiment["model_types"],
            "train_labels": [
                experiment["train_labels"] for _ in range(len(experiment["model_types"]))
            ],
            "test_unseen_matched_labels": [
                experiment["test_unseen_matched_labels"]
                for _ in range(len(experiment["model_types"]))
            ],
            "test_unseen_related_labels": [
                experiment["test_unseen_related_labels"]
                for _ in range(len(experiment["model_types"]))
            ],
            "history": experiment["histories"],
            "test_acc": [result["test_acc"] for result in experiment["test_results"]],
            "test_loss": [result["test_loss"] for result in experiment["test_results"]],
            "test_weighted_f1_avg": [
                result["weighted_f1_avg"] for result in experiment["test_results"]
            ],
            "test_y_pred": [result["y_pred"] for result in experiment["test_results"]],
            "test_y_true": [result["y_true"] for result in experiment["test_results"]],
            "test_unseen_matched_acc": [
                result["test_acc"] for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_matched_loss": [
                result["test_loss"] for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_matched_weighted_f1_avg": [
                result["weighted_f1_avg"]
                for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_matched_y_true": [
                result["y_true"] for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_matched_y_expected": [
                result["y_expected"] for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_matched_y_pred": [
                result["y_pred"] for result in experiment["test_unseen_matched_results"]
            ],
            "test_unseen_related_acc": [
                result["test_acc"] for result in experiment["test_unseen_related_results"]
            ],
            "test_unseen_related_loss": [
                result["test_loss"] for result in experiment["test_unseen_related_results"]
            ],
            "test_unseen_related_weighted_f1_avg": [
                result["weighted_f1_avg"]
                for result in experiment["test_unseen_related_results"]
            ],
            "test_unseen_related_y_true": [
                result["y_true"] for result in experiment["test_unseen_related_results"]
            ],
            "test_unseen_related_y_expected": [
                result["y_expected"] for result in experiment["test_unseen_related_results"]
            ],
            "test_unseen_related_y_pred": [
                result["y_pred"] for result in experiment["test_unseen_related_results"]
            ],
        }
    )

    # If there is an existing experiments dataframe
    if experiment["existing_results_df"] is not None:
        # Concatenate the existing and new dataframes
        experiment_df = pd.concat(
            [experiment["existing_results_df"], experiment_df], ignore_index=True
        )
    # Otherwise
    else:
        # Create folder if it doesn't exist in the results directory
        os.makedirs(experiment["results_path"], exist_ok=True)

    # Save DataFrame to parquet file
    experiment_df.to_parquet(experiment["experiments_df_path"], index=False)


def generate_experiment_plots(experiment):
    """
    Generate plots for an experiment context.

    Args:
        experiment (dict): A dictionary containing the prepared experiment context,
            including dataloaders, device, configurations, and paths for saving results.
    """

    # --- Generate plots --- #

    # Path for saving plots
    plots_path = os.path.join(
        experiment["results_path"],
        "plots",
        str(experiment["experiment_number"]).zfill(3),
    )

    # --- Training plots --- #

    # Plot training curves for each model
    for history, model_type in zip(experiment["histories"], experiment["model_types"]):
        model_plots_path = os.path.join(plots_path, model_type)
        mv.plot_training_curves(
            history=history,
            model_type=model_type,
            plots_path=model_plots_path,
        )

    # --- Standard test set plots --- #

    # If there are multiple models, plot the model comparison
    if len(experiment["model_types"]) > 1:
        mv.plot_model_comparison(
            results=experiment["test_results"],
            model_types=experiment["model_types"],
            plots_path=plots_path,
            test_set_name="test",
        )

    # Plot confusion matrices for each model
    for results, model_type in zip(experiment["test_results"], experiment["model_types"]):
        model_plots_path = os.path.join(plots_path, model_type)
        mv.plot_confusion_matrix(
            results=results,
            model_type=model_type,
            row_labels=experiment["train_labels"],
            column_labels=experiment["train_labels"],
            plots_path=model_plots_path,
            test_set_name="test",
            rotate_x_labels=experiment["target_label"] != "force_level",
        )

    # --- Unseen matched test set plots --- #

    # Use unseen labels for rows only when the evaluation uses different label spaces
    cross_space = UNSEEN_EVALUATION_TARGETS[experiment["target_label"]]["cross_space"]

    # If there are multiple models, plot the model comparison
    if len(experiment["model_types"]) > 1:
        mv.plot_model_comparison(
            results=experiment["test_unseen_matched_results"],
            model_types=experiment["model_types"],
            plots_path=plots_path,
            test_set_name="test_unseen_matched",
        )
    # Get the row labels for the confusion matrix based on whether the evaluation uses
    # different label spaces
    matched_row_labels = (
        experiment["test_unseen_matched_labels"]
        if cross_space
        else experiment["train_labels"]
    )
    # Plot confusion matrices for each model
    for results, model_type in zip(
        experiment["test_unseen_matched_results"], experiment["model_types"]
    ):
        model_plots_path = os.path.join(plots_path, model_type)
        mv.plot_confusion_matrix(
            results=results,
            model_type=model_type,
            row_labels=matched_row_labels,
            column_labels=experiment["train_labels"],
            plots_path=model_plots_path,
            test_set_name="test_unseen_matched",
            cross_space=cross_space,
            rotate_x_labels=experiment["target_label"] != "force_level",
        )

    # --- Unseen related test set plots --- #

    # If there are multiple models, plot the model comparison
    if len(experiment["model_types"]) > 1:
        mv.plot_model_comparison(
            results=experiment["test_unseen_related_results"],
            model_types=experiment["model_types"],
            plots_path=plots_path,
            test_set_name="test_unseen_related",
        )
    # Get the row labels for the confusion matrix based on whether the evaluation uses
    # different label spaces
    related_row_labels = (
        experiment["test_unseen_related_labels"]
        if cross_space
        else experiment["train_labels"]
    )
    # Plot confusion matrices for each model
    for results, model_type in zip(
        experiment["test_unseen_related_results"], experiment["model_types"]
    ):
        model_plots_path = os.path.join(plots_path, model_type)
        mv.plot_confusion_matrix(
            results=results,
            model_type=model_type,
            row_labels=related_row_labels,
            column_labels=experiment["train_labels"],
            plots_path=model_plots_path,
            test_set_name="test_unseen_related",
            cross_space=cross_space,
            rotate_x_labels=experiment["target_label"] != "force_level",
        )

    # --- t-SNE feature plots ---

    # If t-SNE feature plots are enabled, generate a test-set plot for each model
    if experiment["plot_tsne"]:
        for model, model_type, dataloaders in zip(
            experiment["models"], experiment["model_types"], experiment["dataloaders"]
        ):
            model_plots_path = os.path.join(plots_path, model_type)
            # Use true test labels to show how held-out samples group in feature space
            mv.plot_tsne_features(
                model=model,
                dataloader=dataloaders["test"],
                target_label=experiment["target_label"],
                seed=experiment["seed"],
                label_names=experiment["train_labels"],
                model_type=model_type,
                device=experiment["device"],
                plots_path=model_plots_path,
                max_samples=experiment["tsne_max_samples"],
            )
