from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def calculate_metrics(labels, preds, num_classes):
    """Calculates all relevant metrics from predictions and labels."""

    # Ensure preds and labels are numpy arrays
    preds = np.array(preds)
    labels = np.array(labels)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro', zero_division=0),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(labels, preds),
        'cm': confusion_matrix(labels, preds, labels=list(range(num_classes)))
    }
    return metrics


def compute_loss_per_class(model, data_loader, device, num_classes, dataset_type):
    """
    Computes the average loss for each class over a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to run on.
        num_classes (int): The total number of classes.
        dataset_type (str): The type of dataset ('Single', 'Dual', etc.) to handle inputs correctly.

    Returns:
        list: A list of floats containing the average loss for each class.
    """
    model.eval()
    class_losses = torch.zeros(num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Calculating Class Losses"):
            labels = labels.to(device)
            # Handle different input structures
            if dataset_type in ["Dual", "SequencedDual"]:
                inputs = tuple(d.type(torch.FloatTensor).to(device) for d in data)
                outputs = model(*inputs)
            else:  # Single input
                inputs = data.type(torch.FloatTensor).to(device)
                outputs = model(inputs)

            # Calculate loss for each sample in the batch without reduction
            losses = F.cross_entropy(outputs, labels, reduction='none')
            if dataset_type == "SequencedDual":
                labels = labels.view(-1)
                losses = losses.view(-1)

            # Accumulate loss for each class
            for i in range(num_classes):
                class_mask = (labels == i)
                class_losses[i] += losses[class_mask].sum()
                class_counts[i] += class_mask.sum()

    # Calculate average loss, avoiding division by zero for classes not present
    avg_class_losses = class_losses / (class_counts + 1e-8)

    return avg_class_losses.cpu().tolist()