"""Single-image and batch prediction, with crack as the positive class (1)."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data import evaluation_transform


@torch.inference_mode()
def predict_dataset(model, loader, device):
    model.eval()
    total_loss = 0.0
    labels, probabilities, names = [], [], []
    for images, batch_labels, batch_names in loader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, batch_labels)
        total_loss += loss.item() * len(batch_labels)
        labels.extend(batch_labels.cpu().numpy())
        probabilities.extend(logits.softmax(1)[:, 1].float().cpu().numpy())
        names.extend(batch_names)
    return total_loss / len(loader.dataset), np.asarray(labels), np.asarray(probabilities), names


@torch.inference_mode()
def predict_image(model, image_path, threshold=0.5):
    """Return p(crack) and the thresholded label for an unlabelled image."""
    device = next(model.parameters()).device
    model.eval()
    with Image.open(image_path) as image:
        tensor = evaluation_transform()(image).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(tensor)
    probability = float(logits.softmax(1)[0, 1])
    label = int(probability >= threshold)
    return {
        "path": str(image_path),
        "p_crack": probability,
        "label": label,
        "class_name": {0: "corrosion", 1: "crack"}[label],
        "threshold": threshold,
    }


def metric_summary(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if len(labels) == 0 or labels.shape != probabilities.shape:
        raise ValueError("Expected equally sized, nonempty label and probability arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (corrosion) or 1 (crack)")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    pr_precision, pr_recall, _ = precision_recall_curve(labels, probabilities)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=[0, 1], average="macro"),
        "roc_auc": roc_auc_score(labels, probabilities)
        if len(np.unique(labels)) == 2
        else float("nan"),
        "pr_auc_crack": auc(pr_recall, pr_precision),
        "corrosion_precision": precision[0],
        "corrosion_recall": recall[0],
        "corrosion_f1": class_f1[0],
        "corrosion_support": int(support[0]),
        "crack_precision": precision[1],
        "crack_recall": recall[1],
        "crack_f1": class_f1[1],
        "crack_support": int(support[1]),
    }
    return metrics, matrix, predictions
