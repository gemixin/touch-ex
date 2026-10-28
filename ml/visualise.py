"""
A module containing functions for plotting and saving training and evaluation results for
Pytorch models, including t-SNE visualisations of feature representations.

Author: Gemma McLean
Date: April 2026
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from sklearn.manifold import TSNE

# Shared typography for static plots, including colour-bar labels and annotations.
PLOT_FONT = "DejaVu Sans"
PLOT_FONT_SIZE = 11
AXIS_LABEL_SIZE = 12
sns.set_theme(
    style="darkgrid",
    palette="deep",
    font=PLOT_FONT,
    rc={
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelweight": "normal",
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "legend.title_fontsize": AXIS_LABEL_SIZE,
        "savefig.bbox": "tight",
    },
)


def plot_tsne_features(
    model,
    dataloader,
    target_label,
    seed,
    label_names,
    model_type,
    device,
    plots_path,
    max_samples=-1,
    conditioning_label=None,
    split_name="test",
):
    """
    Extract model features from the evaluation split and save a labelled 2-D t-SNE plot.

    Args:
        model (torch.nn.Module): The trained model used to extract features.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation split.
        target_label (str): The label key used to colour points in the plot.
        seed (int): Random seed used for sampling and t-SNE initialisation.
        label_names (list): Ordered class names corresponding to encoded label indices.
        model_type (str): The model/run name used in progress output.
        device (torch.device): The device on which to run feature extraction.
        plots_path (str): The folder path where the t-SNE plot will be saved.
        max_samples (int): Maximum number of class-balanced samples used for t-SNE.
            Defaults to -1 (use all test samples).
        conditioning_label (str, optional): An additional label key supplied during
            conditioned feature extraction. Defaults to None.
        split_name (str): Split used in the filenames.
    """

    print(f"Starting t-SNE features for {model_type}...")

    # Create lists to collect feature vectors and labels from every batch
    all_features = []
    all_labels = []
    # Set the model to evaluation mode before extracting features
    model.eval()

    # Extract the features before the classification layer without computing gradients
    with torch.no_grad():
        for batch in dataloader:
            if conditioning_label is None:
                features = model(batch["image"].to(device), return_features=True)
            else:
                features = model(
                    batch["image"].to(device),
                    batch[conditioning_label].to(device),
                    return_features=True,
                )
            all_features.append(features.cpu().numpy())
            all_labels.append(batch[target_label].numpy())

    # Combine the feature vectors and labels collected from every test batch
    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    # Use every test sample when the split is within the configured t-SNE sample limit
    # or if max_samples is set to -1
    if len(features) > max_samples and max_samples > 0:
        # Use a fixed random generator so the sampled test frames are reproducible
        rng = np.random.default_rng(seed)
        unique_labels = np.unique(labels)
        # Allocate an equal initial number of samples to every class
        samples_per_label = max_samples // len(unique_labels)
        selected_indices = []

        # Sample from each class without replacement, taking all examples from rare classes
        for label_index in unique_labels:
            label_indices = np.where(labels == label_index)[0]
            selected_indices.extend(
                rng.choice(
                    label_indices,
                    size=min(samples_per_label, len(label_indices)),
                    replace=False,
                )
            )

        # Fill any unused places with randomly selected examples from the remaining data
        remaining = max_samples - len(selected_indices)
        if remaining:
            available_indices = np.setdiff1d(np.arange(len(features)), selected_indices)
            selected_indices.extend(
                rng.choice(available_indices, size=remaining, replace=False)
            )

        # Keep only the selected feature vectors and their matching labels for t-SNE
        features = features[selected_indices]
        labels = labels[selected_indices]

    # Perform t-SNE dimensionality reduction to 2D
    embedding = TSNE(
        n_components=2,
        perplexity=min(30, len(features) - 1),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    # Create a scatter plot with one colour and legend entry per class
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colours = sns.color_palette("husl", len(unique_labels)).as_hex()
    for label_index, colour in zip(unique_labels, colours):
        points = labels == label_index
        ax.scatter(
            embedding[points, 0],
            embedding[points, 1],
            s=15,
            alpha=0.7,
            color=colour,
            label=label_names[label_index],
        )

    # Set axis labels and legend; captions supply the plot context.
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Save the plots
    fig.tight_layout()
    save_path = f"{plots_path}/tsne_{split_name}.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved t-SNE features for {model_type} at {save_path}")

    # Save an interactive version of the same embedding with hoverable class labels
    interactive_fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=[label_names[label_index] for label_index in labels],
        color_discrete_map={
            label_names[label_index]: colour
            for label_index, colour in zip(unique_labels, colours)
        },
        labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2", "color": "Label"},
        title=f"t-SNE: {model_type} | {target_label} | {split_name}",
        template="seaborn",
    )
    interactive_fig.update_traces(marker=dict(size=5, opacity=0.7))
    interactive_path = f"{plots_path}/tsne_{split_name}_interactive.html"
    interactive_fig.write_html(interactive_path)
    print(f"Saved interactive t-SNE features at {interactive_path}")


def plot_classification_training_curves(history, model_type, plots_path):
    """
    Plot training and validation loss and accuracy curves for one classification model.

    Args:
        history (list): Training history for the model.
        model_type (str): The model/run name used in progress output.
        plots_path (str): The folder path where the plots will be saved.
    """

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Extract epochs, training and validation losses from the history
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    # Plot training and validation loss curves
    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Extract training and validation accuracies from the history
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]

    # Plot training and validation accuracy curves
    axs[1].plot(epochs, train_accs, label="Train Acc")
    axs[1].plot(epochs, val_accs, label="Val Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    # Save the plot
    fig.tight_layout()
    save_path = f"{plots_path}/curves.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training curves for {model_type} at {save_path}")


def plot_model_comparison(results, model_types, plots_path, split_name):
    """
    Plot a comparison of accuracy and macro F1 average for each model.

    Args:
        results (list): A list of result dictionaries for each model.
        model_types (list): Run names corresponding to each result, used as axis labels.
        plots_path (str): The folder path where the plot will be saved.
        split_name (str): Evaluation split used in filenames.
    """

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("deep", len(model_types))

    # Plot accuracy for each model
    axs[0].bar(model_types, [result["accuracy"] for result in results], color=colors)
    axs[0].set_xlabel("Run")
    axs[0].set_ylabel("Accuracy (%)")

    # Plot macro F1 average for each model
    axs[1].bar(model_types, [result["macro_f1_avg"] for result in results], color=colors)
    axs[1].set_xlabel("Run")
    axs[1].set_ylabel("Macro F1")

    # Save plot
    fig.tight_layout()
    save_path = f"{plots_path}/comparison_{split_name}.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved model comparison at {save_path}")


def plot_confusion_matrix(
    results,
    model_type,
    row_labels,
    column_labels,
    plots_path,
    split_name,
    cross_space=False,
    rotate_x_labels=True,
):
    """
    Plot a confusion matrix for one model.

    Args:
        results (dict): Evaluation results for the model.
        model_type (str): The model/run name used in progress output.
        row_labels (list): Class names for the matrix rows.
        column_labels (list): Class names for the matrix columns.
        plots_path (str): The folder path where the plots will be saved.
        split_name (str): Evaluation split used in filenames.
        cross_space (bool, optional): Whether the rows and columns use different class
            spaces. Defaults to False.
        rotate_x_labels (bool, optional): Whether to rotate x-axis labels.
            Defaults to True.
    """

    # Create confusion matrix
    cm = np.zeros((len(row_labels), len(column_labels)), dtype=int)
    # Count each true-label and predicted-label pair
    for true_index, predicted_index in zip(results["y_true"], results["y_pred"]):
        cm[true_index, predicted_index] += 1

    # Size standard matrices according to their number of class labels
    if cross_space:
        figure_size = (max(12, len(column_labels)), 8)
    # Keep small matrices, such as force levels, proportionate to their text
    elif max(len(row_labels), len(column_labels)) <= 3:
        figure_size = (6, 5)
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
        annot_kws={"fontsize": PLOT_FONT_SIZE},
        fmt="d",
        xticklabels=column_labels,
        yticklabels=row_labels,
        ax=ax,
    )

    # Set axis labels based on whether the confusion matrix is cross-space
    if cross_space:
        ax.set_xlabel("Predicted Training Class")
        ax.set_ylabel("Unseen Source Class")
    else:
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    # Rotate x-axis labels for better readability
    if rotate_x_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.tick_params(axis="y", labelrotation=0)

    # Save plot
    fig.tight_layout()
    save_path = f"{plots_path}/confusion_matrix_{split_name}.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix for {model_type} at {save_path}")


def plot_regression_training_curves(history, model_type, plots_path):
    """
    Plot training and validation loss and MAE curves for one regression model.

    Args:
        history (list): Training history for the model.
        model_type (str): The model/run name used in progress output.
        plots_path (str): The folder path where the plots will be saved.
    """

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Extract epochs, training and validation losses from the history
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    # Plot training and validation loss curves
    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Extract training and validation MAEs from the history
    train_maes = [h["train_mae"] for h in history]
    val_maes = [h["val_mae"] for h in history]

    # Plot training and validation MAE curves
    axs[1].plot(epochs, train_maes, label="Train MAE")
    axs[1].plot(epochs, val_maes, label="Val MAE")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MAE")
    axs[1].legend()

    # Save the plot
    fig.tight_layout()
    save_path = f"{plots_path}/curves.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training curves for {model_type} at {save_path}")


def plot_regression_predictions(results, regression_target, plots_path, split_name):
    """
    Plot and save predicted-versus-true and residual plots in raw target units.

    Args:
        results (dict): Evaluation results for the model.
        regression_target (str): The continuous target displayed on the plot axes.
        plots_path (str): The folder path where the plot will be saved.
        split_name (str): Evaluation split used in filenames.
    """

    # Convert saved predictions and targets to NumPy arrays for plotting
    y_true = np.asarray(results["y_true"])
    y_pred = np.asarray(results["y_pred"])

    # Get shared axis limits for the true and predicted values
    minimum = min(y_true.min(), y_pred.min())
    maximum = max(y_true.max(), y_pred.max())

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot predicted values against true values
    axs[0].scatter(y_true, y_pred, alpha=0.65, s=18)
    axs[0].plot([minimum, maximum], [minimum, maximum], "k--", label="Ideal")
    axs[0].set_xlabel(f"True {regression_target}")
    axs[0].set_ylabel(f"Predicted {regression_target}")
    axs[0].legend()

    # Plot residuals against true values
    axs[1].scatter(y_true, y_pred - y_true, alpha=0.65, s=18)
    axs[1].axhline(0, color="black", linestyle="--")
    axs[1].set_xlabel(f"True {regression_target}")
    axs[1].set_ylabel("Prediction Error")

    # Save the plot
    fig.tight_layout()
    save_path = f"{plots_path}/predictions_{split_name}.png"
    os.makedirs(plots_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved predictions at {save_path}")
