"""
A module containing functions for plotting and saving training and evaluation results for
Pytorch models, and performing dimensionality reduction (PCA, t-SNE) for visualising
feature representations.

Author: Gemma McLean
Date: April 2026
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set default Seaborn styles
sns.set_style("darkgrid")
sns.set_palette("hls")


TEST_SET_TITLES = {
    "test": "Standard Test",
    "test_unseen_matched": "Unseen Matched Objects",
    "test_unseen_related": "Unseen Related Objects",
}


def plot_training_curves(histories, model_types, plots_path):
    """
    Plot training and validation loss and accuracy curves for each model.

    Args:
        histories (list): A list of training histories for each model.
        model_types (list): A list of model type names corresponding to each history.
        plots_path (str): The folder path where the plots will be saved.
    """

    # Loop through each model and plot curves
    for i in range(len(model_types)):
        # Get the history for the current model
        history = histories[i]

        # Create a figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Extract epochs, training and validation losses from the history
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        # Plot training and validation loss curves
        axs[0].plot(epochs, train_losses, label="Train Loss")
        axs[0].plot(epochs, val_losses, label="Val Loss")
        axs[0].set_title(
            f"Loss Curves for {model_types[i]}", fontsize=14, fontweight="bold"
        )
        axs[0].legend()

        # Extract training and validation accuracies from the history
        train_accs = [h["train_acc"] for h in history]
        val_accs = [h["val_acc"] for h in history]

        # Plot training and validation accuracy curves
        axs[1].plot(epochs, train_accs, label="Train Acc")
        axs[1].plot(epochs, val_accs, label="Val Acc")
        axs[1].set_title(
            f"Accuracy Curves for {model_types[i]}", fontsize=14, fontweight="bold"
        )
        axs[1].legend()

        # Save the plots
        plt.tight_layout()
        save_path = f"{plots_path}/curves_{model_types[i]}.png"
        os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
        plt.savefig(save_path)
        print(f"Saved training curves for {model_types[i]} at {save_path}")


def plot_model_comparison(results, model_types, plots_path, test_set_name="test"):
    """
    Plot a comparison of test accuracy, test loss and weighted F1 average for each model.

    Args:
        results (list): A list of result dictionaries for each model.
        model_types (list): A list of model type names corresponding to each result.
        plots_path (str): The folder path where the plot will be saved.
        test_set_name (str): Name of the evaluated test set, used in plot titles and paths.
    """

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = sns.color_palette("hls", len(model_types))

    test_set_title = TEST_SET_TITLES[test_set_name]

    # Plot test accuracy for each model
    axs[0].bar(model_types, [result["test_acc"] for result in results], color=colors)
    axs[0].set_title(f"{test_set_title} Accuracy", fontsize=14, fontweight="bold")
    axs[0].set_ylabel("Accuracy")

    # Plot test loss for each model
    axs[1].bar(model_types, [result["test_loss"] for result in results], color=colors)
    axs[1].set_title(f"{test_set_title} Loss", fontsize=14, fontweight="bold")
    axs[1].set_ylabel("Loss")

    # Plot weighted F1 average for each model
    axs[2].bar(model_types, [result["weighted_f1_avg"] for result in results], color=colors)
    axs[2].set_title(f"{test_set_title} Weighted F1", fontsize=14, fontweight="bold")
    axs[2].set_ylabel("F1 Score")

    # Save plot
    plt.tight_layout()
    save_path = f"{plots_path}/model_comparison_{test_set_name}.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    plt.savefig(save_path)
    print(f"Saved model comparison at {save_path}")


def plot_confusion_matrices(
    results, model_types, list_classes, plots_path, test_set_name="test"
):
    """
    Plot confusion matrices for each model.

    Args:
        results (list): A list of result dictionaries for each model.
        model_types (list): A list of model type names corresponding to each result.
        list_classes (list): A list of class names.
        plots_path (str): The folder path where the plots will be saved.
        test_set_name (str): Name of the evaluated test set, used in plot titles and paths.
    """

    # Set Seaborn style for confusion matrix plots (no grid)
    sns.set_style("white")

    # Loop through each model and plot confusion matrix
    for i in range(len(model_types)):
        # Get the true and predicted labels for the current model
        y_true = results[i]["y_true"]
        y_pred = results[i]["y_pred"]

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(list_classes)))

        # Make figure much larger
        fig, ax = plt.subplots(figsize=(12, 12))

        # Create confusion matrix display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_classes)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        # Rotate x-axis labels if there are more than 3 classes for better readability
        rot = 45 if len(list_classes) > 3 else 0
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha="right")
        ax.set_yticklabels(ax.get_yticklabels())
        test_set_title = TEST_SET_TITLES[test_set_name]
        ax.set_title(f"Confusion Matrix — {test_set_title} ({model_types[i]})")

        # Save plot
        save_path = f"{plots_path}/confusion_matrix_{test_set_name}_{model_types[i]}.png"
        os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved confusion matrix for {model_types[i]} at {save_path}")


def plot_cross_space_confusion_matrices(
    results,
    model_types,
    source_labels,
    predicted_labels,
    plots_path,
    test_set_name,
):
    """
    Plot confusion matrices whose true and predicted labels use different class spaces.

    Args:
        results (list): Evaluation result dictionaries for each model.
        model_types (list): Model type names corresponding to each result.
        source_labels (list): Original unseen-label names for the matrix rows.
        predicted_labels (list): Training-label names for the matrix columns.
        plots_path (str): Folder path where plots will be saved.
        test_set_name (str): Name of the evaluated unseen test set.
    """

    sns.set_style("white")
    test_set_title = TEST_SET_TITLES[test_set_name]

    for i in range(len(model_types)):
        cm = np.zeros((len(source_labels), len(predicted_labels)), dtype=int)
        for source_index, predicted_index in zip(
            results[i]["y_true_source"], results[i]["y_pred"]
        ):
            if not (
                0 <= source_index < len(source_labels)
                and 0 <= predicted_index < len(predicted_labels)
            ):
                raise ValueError(
                    "Confusion-matrix indices must be valid for their source and "
                    "predicted label spaces."
                )
            cm[source_index, predicted_index] += 1

        fig, ax = plt.subplots(figsize=(max(12, len(predicted_labels)), 8))
        sns.heatmap(
            cm,
            cmap="Blues",
            annot=True,
            fmt="d",
            xticklabels=predicted_labels,
            yticklabels=source_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Training Class")
        ax.set_ylabel("Source Unseen Object")
        ax.set_title(f"Confusion Matrix — {test_set_title} ({model_types[i]})")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        save_path = f"{plots_path}/confusion_matrix_{test_set_name}_{model_types[i]}.png"
        os.makedirs(plots_path, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved confusion matrix for {model_types[i]} at {save_path}")
