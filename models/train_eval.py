"""
A module containing functions for training and evaluating Pytorch models.

Author: Gemma McLean
Date: April 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import os


def get_optimizer(model, config):
    """
    Get the Torch optimizer based on the configuration.

    Args:
        model (nn.Module): The model for which to create the optimizer.
        config (dict): Configuration dictionary containing optimizer settings.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """

    # Get class based on string name in config
    if config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD
    elif config["optimizer"].lower() == "adam":
        optimizer = optim.Adam
    elif config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

    # Return the optimizer with the appropriate parameters based on the config
    # If using SGD, include momentum
    if optimizer == optim.SGD:
        return optimizer(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        return optimizer(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )


def train_classifier(model, device, train_loader, val_loader, target_label, config):
    """
    Train the given classifier model.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to use for training.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        target_label (str): The label key for the target variable.
        config (dict): Configuration dictionary containing training settings.

    Returns:
        nn.Module: The trained model.
        list: Training history.
    """

    # Get the title of the model from the config for logging and checkpointing
    model_title = config["model_title"]
    num_epochs = config["num_epochs"]
    # Move model to the specified device
    model = model.to(device)
    # Get the optimizer and criterion from the config
    optimizer = get_optimizer(model, config)
    # Criterion is cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # If checkpointing is enabled (checkpoint_dir in the config is not None)
    if config["checkpoint_dir"]:
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        # Get the path to save the model checkpoint based on the model title
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"{model_title}.pth")

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
        for features in train_loader:
            # Get images and labels from features and move to device
            imgs = features["image"].to(device)
            labels = features[target_label].to(device)

            # Perform forward pass, compute loss, and backpropagate
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            for features in val_loader:
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
        }
        history.append(metrics)

        # Print metrics for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # -- Checkpointing -- #

        # If checkpointing is enabled (checkpoint_dir in the config is not None)
        if config["checkpoint_dir"]:
            # If this is the best validation accuracy so far, save the model checkpoint
            if val_acc > best_val_acc:
                # Update the best validation accuracy
                best_val_acc = val_acc
                # Save the model checkpoint to the specified path
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )

    # If checkpointing is enabled (checkpoint_dir in the config is not None)
    if config["checkpoint_dir"]:
        # Load the best model checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model checkpoint from epoch {checkpoint['epoch']}.")

    # After training is complete, return the trained model and the history of metrics
    return model, history


def eval_classifier(model, device, test_loader, target_label):
    """
    Evaluate the given classifier model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to use for evaluation.
        test_loader (DataLoader): The test data loader.
        target_label (str): The label key for the target variable.

    Returns:
        dict: A dictionary containing test loss, test accuracy, true labels, and
        predicted labels.
    """

    # Move model to the specified device
    model = model.to(device)
    # Criterion is cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # -- Evaluation Phase -- #

    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    y_true, y_pred = [], []

    # Iterate over test batches without computing gradients
    with torch.no_grad():
        for features in test_loader:
            # Get images and labels from features and move to device
            imgs = features["image"].to(device)
            labels = features[target_label].to(device)

            # Perform forward pass and compute loss
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # Update overall test loss
            test_loss += loss.item()
            # Get predicted classes and update overall correct/total counts for accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            # Append true and predicted labels to lists for later analysis
            # Move to CPU and convert to numpy array before appending
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute average test loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total

    # Compute weighted F1 average using sklearn's f1_score function
    weighted_f1_avg = f1_score(y_true, y_pred, average="weighted")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    print(f"Weighted F1 Average: {weighted_f1_avg:.4f}")

    # Return a dictionary containing the results
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weighted_f1_avg": weighted_f1_avg,
        "y_true": y_true,
        "y_pred": y_pred,
    }
