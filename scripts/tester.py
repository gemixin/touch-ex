import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import parquet dataframe
df = pd.read_parquet("results/object_classify_cnn_compare/outputs.parquet")

EXPERIMENT_NUMBER = 5

PLOTS_FOLDER = (
    f"results/object_classify_cnn_compare/plots/{str(EXPERIMENT_NUMBER).zfill(3)}"
)
# Create if it doesn't exist
import os

os.makedirs(PLOTS_FOLDER, exist_ok=True)
# Load the rows for the chosen experiment number
df_experiment = df[df["experiment_number"] == EXPERIMENT_NUMBER]

# Extract required info from the dataframe
MODEL_TYPES = df_experiment["model_type"].tolist()
results = df_experiment[
    ["model_type", "test_acc", "test_loss", "weighted_f1_avg", "y_true", "y_pred"]
].to_dict(orient="records")
# Get list of classes from list_classes column (same for all rows)
list_classes = ["beans", "pringles", "hammer", "thingy", "other", "lol"]

histories = df_experiment["history"].tolist()

if len(df_experiment) > 1:
    # Plot to show comparison between models for test_acc, test_loss and weighted_f1_avg
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = sns.color_palette("husl", len(df_experiment))

    axs[0].bar(MODEL_TYPES, [result["test_acc"] for result in results], color=colors)
    axs[0].set_title("Test Accuracy", fontsize=14, fontweight="bold")
    axs[0].set_ylabel("Accuracy")

    axs[1].bar(MODEL_TYPES, [result["test_loss"] for result in results], color=colors)
    axs[1].set_title("Test Loss", fontsize=14, fontweight="bold")
    axs[1].set_ylabel("Loss")

    axs[2].bar(MODEL_TYPES, [result["weighted_f1_avg"] for result in results], color=colors)
    axs[2].set_title("Weighted F1 Average", fontsize=14, fontweight="bold")
    axs[2].set_ylabel("F1 Score")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/model_comparison.png")
    plt.show()

# Plot to show confusion matrix for each model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set_style("white")

# Loop through each model and plot confusion matrix
for i in range(len(MODEL_TYPES)):
    # Get the true and predicted labels for the current model
    y_true = results[i]["y_true"]
    y_pred = results[i]["y_pred"]
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create confusion matrix display (set x axis labels to 90 degrees)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_classes)
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix for {MODEL_TYPES[i]}")
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_FOLDER}/confusion_matrix_{MODEL_TYPES[i]}.png", bbox_inches="tight"
    )
    plt.show()

# Plot to show training and validation loss and accuracy curves for each model
# Loop through each model and plot curves
for i in range(len(MODEL_TYPES)):
    # Get the history for the current model
    history = histories[i]
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Plot training and validation loss curves
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Val Loss")
    axs[0].set_title(f"Loss Curves for {MODEL_TYPES[i]}", fontsize=14, fontweight="bold")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    # Plot training and validation accuracy curves
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    axs[1].plot(epochs, train_accs, label="Train Acc")
    axs[1].plot(epochs, val_accs, label="Val Acc")
    axs[1].set_title(
        f"Accuracy Curves for {MODEL_TYPES[i]}", fontsize=14, fontweight="bold"
    )
    axs[1].legend()
    # Show and save the plots
    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/curves_{MODEL_TYPES[i]}.png")
    plt.show()
