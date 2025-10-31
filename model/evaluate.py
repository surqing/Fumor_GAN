# evaluate.py

import torch
import numpy as np


def evaluateDis(prediction, y_true):
    """
    Evaluate the discriminator's performance for 2-class (real vs fake) task.
    Args:
        prediction: list or np.array of shape (N, 1, 2) — model outputs (softmax scores)
        y_true: list or np.array of shape (N, 2) — true one-hot labels
    Returns:
        (precision, acc, recall, f1)
    """
    results = evaluation_2class(prediction, y_true)
    # 解析返回的结果
    metrics = dict(zip(results[::2], results[1::2]))  # {'acc':..., 'nPrec':..., ...}
    return metrics['nPrec'], metrics['acc'], metrics['nRec'], metrics['nF1']



def evaluation_2class(predictions, targets):
    """
    Evaluate binary (2-class) classification performance.

    Args:
        predictions: torch.Tensor or np.ndarray, shape (N, 2)
                     predicted class probabilities or logits
        targets: torch.Tensor or np.ndarray, shape (N,)
                 true class indices (0 or 1)

    Returns:
        dict: accuracy, precision, recall, F1 per class
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Convert logits → predicted labels
    if predictions.ndim > 1:
        pred_labels = predictions.argmax(dim=1)
    else:
        pred_labels = (predictions > 0.5).long()

    true_labels = targets.long()

    # Compute confusion matrix elements
    TP = ((pred_labels == 1) & (true_labels == 1)).sum().item()
    TN = ((pred_labels == 0) & (true_labels == 0)).sum().item()
    FP = ((pred_labels == 1) & (true_labels == 0)).sum().item()
    FN = ((pred_labels == 0) & (true_labels == 1)).sum().item()

    eps = 1e-8
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision_pos = TP / (TP + FP + eps)
    recall_pos = TP / (TP + FN + eps)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + eps)

    precision_neg = TN / (TN + FN + eps)
    recall_neg = TN / (TN + FP + eps)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + eps)

    return {
        "accuracy": round(accuracy, 4),
        "positive": {
            "precision": round(precision_pos, 4),
            "recall": round(recall_pos, 4),
            "f1": round(f1_pos, 4),
        },
        "negative": {
            "precision": round(precision_neg, 4),
            "recall": round(recall_neg, 4),
            "f1": round(f1_neg, 4),
        },
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate a PyTorch model on a given dataloader.

    Args:
        model: torch.nn.Module
        dataloader: torch.utils.data.DataLoader
        device: 'cpu' or 'cuda'

    Returns:
        dict: evaluation metrics
    """
    model.eval()
    all_preds, all_labels = [], []

    for x_batch, _, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model.discriminator(x_batch) if hasattr(model, "discriminator") else model(x_batch)
        if outputs.ndim == 1:
            probs = torch.sigmoid(outputs)
            preds = torch.stack([1 - probs, probs], dim=1)
        else:
            preds = torch.softmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_labels)

    return evaluation_2class(predictions, targets)
