"""
A module containing functions for training and evaluating Pytorch models.

Author: Gemma McLean
Date: April 2026
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import os
from data.validate import validate_train_config


def train_classifier(model, device, train_loader, val_loader, target_label, train_config):
    """
    Train the given classifier model.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to use for training.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        target_label (str): The label key for the target variable.
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
    best_val_acc = 0.0
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
            outputs = model(imgs)
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
                outputs = model(imgs)
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
    if train_config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD
    elif train_config["optimizer"].lower() == "adam":
        optimizer = optim.Adam
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer type: {train_config['optimizer']}")

    # Optimise only parameters that are not frozen
    trainable_parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )

    # Return the optimizer with the appropriate parameters based on the config
    # If using SGD, include momentum
    if optimizer == optim.SGD:
        return optimizer(
            trainable_parameters,
            lr=train_config["learning_rate"],
            momentum=train_config["momentum"],
            weight_decay=train_config["weight_decay"],
        )
    else:
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
    test_loader,
    target_label,
    true_label=None,
    evaluation_name="test",
):
    """
    Evaluate the given classifier model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        model_title (str): The title of the model for logging purposes.
        device (torch.device): The device to use for evaluation.
        test_loader (DataLoader): The test data loader.
        target_label (str): The expected-label key used to calculate loss and metrics.
        true_label (str, optional): The original-label key used for confusion-matrix rows.
            Defaults to target_label.
        evaluation_name (str): Name used in evaluation progress output.

    Returns:
        dict: A dictionary containing test loss, test accuracy, true labels, expected
            labels, and predicted labels.
    """

    model = model.to(device)
    # Criterion is cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Evaluation Phase -- #

    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    y_true, y_expected, y_pred = [], [], []

    print(f"Starting {evaluation_name} evaluation of {model_title} on device: {device}")

    # Iterate over test batches without computing gradients
    with torch.no_grad():
        for features in tqdm(test_loader, desc=evaluation_name, total=len(test_loader)):
            # Get images and labels from features and move to device
            imgs = features["image"].to(device)
            expected_labels = features[target_label].to(device)
            true_labels = features[true_label or target_label].to(device)

            # Perform forward pass and compute loss
            outputs = model(imgs)
            loss = criterion(outputs, expected_labels)

            # Update overall test loss
            test_loss += loss.item()
            # Get predicted classes and update overall correct/total counts for accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += expected_labels.size(0)
            test_correct += (predicted == expected_labels).sum().item()
            # Append true, expected, and predicted labels to lists for later analysis
            # Move to CPU and convert to numpy array before appending
            y_true.extend(true_labels.cpu().numpy())
            y_expected.extend(expected_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute average test loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total

    # Compute weighted F1 average using sklearn's f1_score function
    weighted_f1_avg = f1_score(y_expected, y_pred, average="weighted")

    tqdm.write(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    tqdm.write(f"Weighted F1 Average: {weighted_f1_avg:.4f}")

    # Return a dictionary containing the results
    # For anything except object/object_region in the unseen test sets, y_expected will be
    # the same as y_true
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weighted_f1_avg": weighted_f1_avg,
        "y_true": y_true,
        "y_expected": y_expected,
        "y_pred": y_pred,
    }
