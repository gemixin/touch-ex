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

# Set default Seaborn styles
sns.set_style("darkgrid")
sns.set_palette("hls")

# Define a mapping of test set names to their corresponding titles for plots
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
    results,
    model_types,
    row_labels,
    column_labels,
    plots_path,
    test_set_name,
    cross_space=False,
    rotate_x_labels=True,
):
    """
    Plot confusion matrices for each model.

    Args:
        results (list): A list of result dictionaries for each model.
        model_types (list): A list of model type names corresponding to each result.
        row_labels (list): Class names for the matrix rows.
        column_labels (list): Class names for the matrix columns.
        plots_path (str): The folder path where the plots will be saved.
        test_set_name (str): Name of the evaluated test set, used in plot titles and paths.
        cross_space (bool, optional): Whether the rows and columns use different class
            spaces. Defaults to False.
        rotate_x_labels (bool, optional): Whether to rotate x-axis labels.
            Defaults to True.
    """

    # Set Seaborn style for confusion matrix plots (no grid)
    sns.set_style("white")

    # Loop through each model and plot confusion matrix
    for i in range(len(model_types)):
        # Create confusion matrix
        cm = np.zeros((len(row_labels), len(column_labels)), dtype=int)
        # Count each true-label and predicted-label pair
        for true_index, predicted_index in zip(results[i]["y_true"], results[i]["y_pred"]):
            cm[true_index, predicted_index] += 1

        # Size standard matrices according to their number of class labels
        if cross_space:
            figure_size = (max(12, len(column_labels)), 8)
        else:
            figure_size = (
                max(12, len(column_labels) * 0.6),
                max(12, len(row_labels) * 0.45),
            )
        fig, ax = plt.subplots(figsize=figure_size)

        # Create confusion matrix display
        sns.heatmap(
            cm,
            cmap="Blues",
            annot=True,
            fmt="d",
            xticklabels=column_labels,
            yticklabels=row_labels,
            ax=ax,
        )

        # Set axis labels and title based on whether the confusion matrix is cross-space
        if cross_space:
            ax.set_xlabel("Predicted Training Class")
            ax.set_ylabel("Unseen Source Class")
        else:
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")

        # Rotate x-axis labels for better readability
        if rotate_x_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        test_set_title = TEST_SET_TITLES[test_set_name]
        ax.set_title(f"Confusion Matrix — {test_set_title} ({model_types[i]})")

        # Save plot
        save_path = f"{plots_path}/confusion_matrix_{test_set_name}_{model_types[i]}.png"
        os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved confusion matrix for {model_types[i]} at {save_path}")
