"""
Utility functions for Cucumber Leaf Disease and Nutrient Deficiency Detection.

Includes:
  - Accuracy / metric computation
  - Checkpoint save / load helpers
  - Visualisation helpers (confusion matrix, GradCAM)
  - Seed initialisation
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = cfg.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a float in [0, 1]."""
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        return (preds == targets).float().mean().item()


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy and per-class precision/recall/F1 over a dataloader.

    Args:
        model: Model to evaluate (put in eval mode before calling).
        loader: DataLoader yielding (images, labels).
        device: Device to run inference on.
        class_names: Class names for per-class reporting. Defaults to
            ``config.CLASS_NAMES``.

    Returns:
        Dictionary with keys ``"accuracy"``, ``"macro_f1"`` and per-class
        metrics ``"{class_name}_precision"``, ``"{class_name}_recall"``,
        ``"{class_name}_f1"``.
    """
    class_names = class_names or cfg.CLASS_NAMES
    num_classes = len(class_names)

    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    all_preds_t = torch.tensor(all_preds)
    all_targets_t = torch.tensor(all_targets)

    correct = (all_preds_t == all_targets_t).sum().item()
    total = len(all_targets_t)
    overall_acc = correct / total if total > 0 else 0.0

    # Per-class TP / FP / FN
    metrics: Dict[str, float] = {"accuracy": overall_acc}
    f1_scores = []
    for c in range(num_classes):
        tp = ((all_preds_t == c) & (all_targets_t == c)).sum().item()
        fp = ((all_preds_t == c) & (all_targets_t != c)).sum().item()
        fn = ((all_preds_t != c) & (all_targets_t == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)
        name = class_names[c]
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1

    metrics["macro_f1"] = float(np.mean(f1_scores))
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    path: str | Path,
    is_best: bool = False,
    best_path: Optional[str | Path] = None,
) -> None:
    """
    Save a training checkpoint to *path*.

    If *is_best* is ``True``, also copy it to *best_path* (defaults to
    ``path.parent / "best_model.pth"``).

    Args:
        state: Anything ``torch.save`` can serialise (typically a dict with
               ``model_state_dict``, ``optimizer_state_dict``, ``epoch``, etc.).
        path: Where to save the checkpoint.
        is_best: Whether this is the best validation checkpoint seen so far.
        best_path: Destination for the best checkpoint copy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = Path(best_path) if best_path else path.parent / "best_model.pth"
        torch.save(state, best_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a checkpoint from *path* into *model* (and optionally *optimizer*).

    Args:
        path: Path to the ``.pth`` checkpoint file.
        model: Model whose weights should be restored.
        optimizer: Optional optimizer to restore state into.
        device: Device to map tensors to (defaults to ``"cpu"``).

    Returns:
        The full checkpoint dictionary (contains ``epoch``, ``best_val_acc``, etc.).
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# GradCAM – gradient-weighted class activation mapping
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) for MobileNetV4.

    Attach it to any convolutional layer to visualise where the model
    focuses when making a prediction.

    Example::

        cam = GradCAM(model, target_layer=model.head_conv[0])
        heatmap = cam(image_tensor, class_idx=2)

    Args:
        model: The neural network.
        target_layer: The convolutional layer to hook.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove the registered hooks (call when done)."""
        for h in self._hook_handles:
            h.remove()

    def __call__(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the GradCAM heatmap for *image*.

        Args:
            image: Input tensor of shape ``(1, C, H, W)``.
            class_idx: Class index to explain. If ``None``, uses the
                predicted class.

        Returns:
            Normalised heatmap as a ``numpy`` array of shape ``(H, W)``
            with values in ``[0, 1]``.
        """
        self.model.eval()
        image = image.requires_grad_(True)
        logits = self.model(image)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[:, class_idx].backward()

        # Global-average-pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.clamp(cam, min=0)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


# ---------------------------------------------------------------------------
# Confusion matrix (text representation)
# ---------------------------------------------------------------------------

def print_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Print a confusion matrix to stdout and return it as a numpy array.

    Args:
        model: Trained model (set to eval mode before calling).
        loader: DataLoader yielding (images, labels).
        device: Inference device.
        class_names: Class names. Defaults to ``config.CLASS_NAMES``.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` as an
        ``int64`` numpy array.
    """
    class_names = class_names or cfg.CLASS_NAMES
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for t, p in zip(labels_np, preds):
                matrix[t, p] += 1

    # Print header
    col_w = max(len(n) for n in class_names) + 2
    header = " " * col_w + "".join(f"{n[:col_w - 1]:>{col_w}}" for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"{row_name:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in matrix[i])
        print(row)

    return matrix
