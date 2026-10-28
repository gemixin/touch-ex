"""
A module containing functions for training and evaluating Pytorch models.

Author: Gemma McLean
Date: April 2026
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from data.validate import validate_train_config


def train_classifier(
    model,
    device,
    train_loader,
    val_loader,
    target_label,
    train_config,
    conditioning_label=None,
):
    """
    Train the given classifier model.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to use for training.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        target_label (str): The label key for the target variable.
        train_config (dict): Configuration dictionary containing training settings.
        conditioning_label (str, optional): An additional label key supplied to a
            conditioned classifier. Defaults to None.

    Returns:
        nn.Module: The trained model.
        list: Training history.
    """

    # If the train config is invalid, raise an error before proceeding
    validate_train_config(train_config)

    # Get the title of the model from the config for logging and checkpointing
    model_title = train_config["model_title"]
    num_epochs = train_config["num_epochs"]
    early_stopping_patience = train_config["early_stopping_patience"]
    early_stopping_min_delta = train_config["early_stopping_min_delta"]
    # Move model to the specified device
    model = model.to(device)
    # Get the optimizer and criterion from the config
    optimizer = get_optimizer(model, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config, len(train_loader))
    # Criterion is cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss().to(device)

    # If checkpointing is enabled, get path
    if train_config["checkpoint_dir"]:
        os.makedirs(train_config["checkpoint_dir"], exist_ok=True)
        checkpoint_path = os.path.join(train_config["checkpoint_dir"], f"{model_title}.pth")

    # Track the best validation accuracy for checkpointing
    best_val_acc = float("-inf")
    # Track meaningful validation improvements for early stopping separately so the
    # checkpoint remains the best model even when its improvement is below min_delta
    best_early_stopping_val_acc = float("-inf")
    epochs_without_improvement = 0
    # List to store training history (loss and accuracy for each epoch)
    history = []

    print(f"Starting training of {model_title} on device: {device}")

    for epoch in range(num_epochs):
        # -- Training Phase -- #

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Iterate over training batches
        for features in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Train",
            leave=False,
            total=len(train_loader),
        ):
            # Get images and labels from features and move to device
            imgs = features["image"].to(device)
            labels = features[target_label].to(device)

            # Perform forward pass, compute loss, and backpropagate
            optimizer.zero_grad()
            if conditioning_label is None:
                outputs = model(imgs)
            else:
                conditioning_values = features[conditioning_label].to(device)
                outputs = model(imgs, conditioning_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update overall training loss
            train_loss += loss.item()
            # Get predicted classes and update overall correct/total counts for accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)  # Total number of samples in this batch
            train_correct += (predicted == labels).sum().item()  # Number of correct

        # -- Validation Phase -- #

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        # Iterate over validation batches without computing gradients
        with torch.no_grad():
            for features in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Val",
                leave=False,
                total=len(val_loader),
            ):
                # Get images and labels from features and move to device
                imgs = features["image"].to(device)
                labels = features[target_label].to(device)

                # Perform forward pass and compute loss
                if conditioning_label is None:
                    outputs = model(imgs)
                else:
                    conditioning_values = features[conditioning_label].to(device)
                    outputs = model(imgs, conditioning_values)
                loss = criterion(outputs, labels)

                # Update overall validation loss
                val_loss += loss.item()
                # Get predicted classes and update overall correct/total counts for accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)  # Total number of samples in this batch
                val_correct += (predicted == labels).sum().item()  # Number of correct

        # -- Metrics -- #

        # Compute average loss and accuracy for this epoch
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Store metrics in dictionary and append to history
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)

        # Print metrics for this epoch (use tqdm.write to avoid corrupting bars)
        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # -- Checkpointing -- #

        # If checkpointing is enabled
        if train_config["checkpoint_dir"]:
            # If this is the best validation accuracy so far, save the model checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )

        # -- Early stopping -- #

        # If early stopping is enabled
        if early_stopping_patience is not None:
            # Reset the patience counter when validation accuracy improves by min_delta
            if val_acc > best_early_stopping_val_acc + early_stopping_min_delta:
                best_early_stopping_val_acc = val_acc
                epochs_without_improvement = 0
            # Otherwise, count this epoch toward the early-stopping patience limit
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    tqdm.write(
                        "Early stopping after "
                        f"{epoch + 1} epochs: validation accuracy did not improve "
                        f"by at least {early_stopping_min_delta} for "
                        f"{early_stopping_patience} consecutive epochs."
                    )
                    break

    # After training is complete, if checkpointing is enabled, load the best checkpoint
    if train_config["checkpoint_dir"]:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model checkpoint from epoch {checkpoint['epoch']}.")

    # Return the trained model and the history of metrics
    return model, history


def get_optimizer(model, train_config):
    """
    Get the Torch optimizer based on the configuration.

    Args:
        model (nn.Module): The model for which to create the optimizer.
        train_config (dict): Configuration dictionary containing optimizer settings.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """

    # Get class based on string name in config
    if train_config["optimizer"].lower() == "adam":
        optimizer = optim.Adam
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer type: {train_config['optimizer']}")

    # Optimise only parameters that are not frozen
    trainable_parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )

    return optimizer(
        trainable_parameters,
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )


def get_lr_scheduler(optimizer, train_config, steps_per_epoch):
    """
    Create a batch-level linear warmup and cosine learning-rate decay scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
        train_config (dict): Configuration dictionary containing scheduler settings.
        steps_per_epoch (int): Number of training batches in each epoch.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The configured learning-rate scheduler.
    """

    # Calculate the number of warmup steps and total steps for the scheduler
    warmup_steps = train_config["warmup_epochs"] * steps_per_epoch
    total_steps = train_config["num_epochs"] * steps_per_epoch
    min_lr_factor = train_config["min_learning_rate"] / train_config["learning_rate"]
    warmup_start_factor = train_config["warmup_start_factor"]

    def lr_lambda(step):
        """
        Compute the learning rate factor for the given step.

        Args:
            step (int): The current training step.

        Returns:
            float: The learning rate factor to apply to the optimizer's base learning rate.
        """

        # Increase linearly from the warmup start factor to the configured rate
        if step < warmup_steps:
            return warmup_start_factor + (1 - warmup_start_factor) * step / max(
                1, warmup_steps
            )

        # Decay from the configured rate to the minimum rate using a cosine curve
        decay_progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * min(1, decay_progress)))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay

    # Return the LambdaLR scheduler that applies the lr_lambda function to the optimizer
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def eval_classifier(
    model,
    model_title,
    device,
    data_loader,
    target_label,
    evaluation_name="evaluation",
    conditioning_label=None,
    compute_metrics=True,
):
    """Evaluate a labelled split, or collect predictions for unknown identities."""
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device) if compute_metrics else None
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred, source_labels = [], [], []

    print(f"Starting {evaluation_name} evaluation of {model_title} on device: {device}")
    with torch.no_grad():
        for features in tqdm(data_loader, desc=evaluation_name, total=len(data_loader)):
            imgs = features["image"].to(device)
            labels = features[target_label].to(device)
            if conditioning_label is None:
                outputs = model(imgs)
            else:
                outputs = model(imgs, features[conditioning_label].to(device))
            predicted = outputs.argmax(dim=1)
            if compute_metrics:
                total_loss += criterion(outputs, labels).item()
                correct += (predicted == labels).sum().item()
            else:
                source_labels.extend(features[f"source_{target_label}"])
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    results = {"y_true": y_true, "y_pred": y_pred}
    if compute_metrics:
        results.update(
            loss=total_loss / len(data_loader),
            accuracy=100 * correct / total,
            macro_f1_avg=f1_score(
                y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0
            ),
        )
        tqdm.write(
            f"Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.2f}%, "
            f"Macro F1: {results['macro_f1_avg']:.4f}"
        )
    else:
        results["source_labels"] = source_labels
    return results


def get_target_normalizer(dataset, regression_target):
    """
    Calculate training-split normalisation statistics for a regression target.

    Args:
        dataset (Dataset): The training dataset containing the regression target.
        regression_target (str): The continuous target column to normalise.

    Returns:
        dict: Mean and standard deviation calculated from the training split only.
    """

    # Extract the continuous training targets as a NumPy array
    values = dataset.df[regression_target].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{regression_target} contains non-finite values.")

    # Calculate mean and standard deviation for normalisation
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        raise ValueError(f"{regression_target} has zero variance in the training split.")
    return {"mean": mean, "std": std}


def _normalise_targets(targets, target_normalizer):
    """Normalise target tensors using training-split statistics."""

    return (targets - target_normalizer["mean"]) / target_normalizer["std"]


def _denormalise_targets(targets, target_normalizer):
    """Convert normalised target tensors back to their original units."""

    return targets * target_normalizer["std"] + target_normalizer["mean"]


def _regression_metrics(y_true, y_pred):
    """Calculate regression metrics in the original target units."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_regressor(
    model,
    device,
    train_loader,
    val_loader,
    regression_target,
    target_normalizer,
    train_config,
):
    """
    Train the given regressor model.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to use for training.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        regression_target (str): The label key for the continuous target variable.
        target_normalizer (dict): The training-split target mean and standard deviation.
        train_config (dict): Configuration dictionary containing training settings.

    Returns:
        nn.Module: The trained model.
        list: Training history.
    """

    # If the train config is invalid, raise an error before proceeding
    validate_train_config(train_config)

    # Get the title of the model from the config for logging and checkpointing
    model_title = train_config["model_title"]
    num_epochs = train_config["num_epochs"]
    early_stopping_patience = train_config["early_stopping_patience"]
    early_stopping_min_delta = train_config["early_stopping_min_delta"]
    # Move model to the specified device
    model = model.to(device)
    # Get the optimizer and criterion from the config
    optimizer = get_optimizer(model, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config, len(train_loader))
    # Criterion is Huber loss for regression
    criterion = nn.HuberLoss().to(device)

    # If checkpointing is enabled, get path
    if train_config["checkpoint_dir"]:
        os.makedirs(train_config["checkpoint_dir"], exist_ok=True)
        checkpoint_path = os.path.join(train_config["checkpoint_dir"], f"{model_title}.pth")

    # Track the best validation loss for checkpointing
    best_val_loss = float("inf")
    # Track meaningful validation improvements for early stopping separately so the
    # checkpoint remains the best model even when its improvement is below min_delta
    best_early_stopping_val_loss = float("inf")
    epochs_without_improvement = 0
    # List to store training history (loss and metrics for each epoch)
    history = []

    print(f"Starting training of {model_title} on device: {device}")

    for epoch in range(num_epochs):
        # -- Training Phase -- #

        model.train()
        train_loss = 0.0
        train_y_true, train_y_pred = [], []

        # Iterate over training batches
        for features in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Train",
            leave=False,
            total=len(train_loader),
        ):
            # Get images and targets from features and move to device
            imgs = features["image"].to(device)
            targets = features[regression_target].to(device)
            normalised_targets = _normalise_targets(targets, target_normalizer)

            # Perform forward pass, compute loss, and backpropagate
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, normalised_targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update overall training loss
            train_loss += loss.item()
            # Append true and predicted targets to lists for metric calculation
            train_y_true.extend(targets.detach().cpu().numpy())
            train_y_pred.extend(
                _denormalise_targets(outputs.detach(), target_normalizer).cpu().numpy()
            )

        # -- Validation Phase -- #

        model.eval()
        val_loss = 0.0
        val_y_true, val_y_pred = [], []

        # Iterate over validation batches without computing gradients
        with torch.no_grad():
            for features in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Val",
                leave=False,
                total=len(val_loader),
            ):
                # Get images and targets from features and move to device
                imgs = features["image"].to(device)
                targets = features[regression_target].to(device)
                normalised_targets = _normalise_targets(targets, target_normalizer)

                # Perform forward pass and compute loss
                outputs = model(imgs)
                loss = criterion(outputs, normalised_targets)

                # Update overall validation loss
                val_loss += loss.item()
                # Append true and predicted targets to lists for metric calculation
                val_y_true.extend(targets.cpu().numpy())
                val_y_pred.extend(
                    _denormalise_targets(outputs, target_normalizer).cpu().numpy()
                )

        # -- Metrics -- #

        # Compute average normalised loss and raw-unit metrics for this epoch
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_metrics = _regression_metrics(train_y_true, train_y_pred)
        val_metrics = _regression_metrics(val_y_true, val_y_pred)

        # Store metrics in dictionary and append to history
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)

        # Print metrics for this epoch (use tqdm.write to avoid corrupting bars)
        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Train MAE: {train_metrics['mae']:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE: {val_metrics['mae']:.4f}, Val R2: {val_metrics['r2']:.4f}"
        )

        # -- Checkpointing -- #

        # If checkpointing is enabled
        if train_config["checkpoint_dir"]:
            # If this is the best validation loss so far, save the model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )

        # -- Early stopping -- #

        # If early stopping is enabled
        if early_stopping_patience is not None:
            # Reset the patience counter when validation loss improves by min_delta
            if val_loss < best_early_stopping_val_loss - early_stopping_min_delta:
                best_early_stopping_val_loss = val_loss
                epochs_without_improvement = 0
            # Otherwise, count this epoch toward the early-stopping patience limit
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    tqdm.write(
                        "Early stopping after "
                        f"{epoch + 1} epochs: validation loss did not improve "
                        f"by at least {early_stopping_min_delta} for "
                        f"{early_stopping_patience} consecutive epochs."
                    )
                    break

    # After training is complete, if checkpointing is enabled, load the best checkpoint
    if train_config["checkpoint_dir"]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model checkpoint from epoch {checkpoint['epoch']}.")

    # Return the trained model and the history of metrics
    return model, history


def eval_regressor(
    model,
    model_title,
    device,
    data_loader,
    regression_target,
    target_normalizer,
    evaluation_name="evaluation",
):
    """
    Evaluate the given regressor model on an evaluation split.

    Args:
        model (nn.Module): The model to evaluate.
        model_title (str): The title of the model for logging purposes.
        device (torch.device): The device to use for evaluation.
        data_loader (DataLoader): The evaluation data loader.
        regression_target (str): The label key for the continuous target variable.
        target_normalizer (dict): The training-split target mean and standard deviation.
        evaluation_name (str): Name used in evaluation progress output.

    Returns:
        dict: A dictionary containing loss, raw-unit regression metrics, true
            targets, and predicted targets.
    """

    model = model.to(device)
    # Criterion is Huber loss for regression
    criterion = nn.HuberLoss().to(device)

    # -- Evaluation Phase -- #

    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    print(f"Starting {evaluation_name} evaluation of {model_title} on device: {device}")

    # Iterate over evaluation batches without computing gradients
    with torch.no_grad():
        for features in tqdm(data_loader, desc=evaluation_name, total=len(data_loader)):
            # Get images and targets from features and move to device
            imgs = features["image"].to(device)
            targets = features[regression_target].to(device)
            normalised_targets = _normalise_targets(targets, target_normalizer)

            # Perform forward pass and compute loss
            outputs = model(imgs)
            loss = criterion(outputs, normalised_targets)

            # Update overall evaluation loss
            total_loss += loss.item()
            # Append true and predicted targets to lists for later analysis
            # Move to CPU and convert to numpy array before appending
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(_denormalise_targets(outputs, target_normalizer).cpu().numpy())

    # Compute average evaluation loss
    total_loss = total_loss / len(data_loader)

    # Compute raw-unit regression metrics
    metrics = _regression_metrics(y_true, y_pred)

    tqdm.write(
        f"Loss: {total_loss:.4f}, MAE: {metrics['mae']:.4f}, "
        f"RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}"
    )

    # Return a dictionary containing the results
    return {"loss": total_loss, **metrics, "y_true": y_true, "y_pred": y_pred}
