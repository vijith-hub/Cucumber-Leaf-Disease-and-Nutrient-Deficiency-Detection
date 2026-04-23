"""
Training script for Cucumber Leaf Disease and Nutrient Deficiency Detection.

Usage
-----
Train with default settings (data must exist at ``data/``):

    python train.py

Train with custom options:

    python train.py \\
        --data-dir /path/to/dataset \\
        --variant medium \\
        --epochs 100 \\
        --batch-size 64 \\
        --lr 5e-4 \\
        --output-dir checkpoints/

Resume from a checkpoint:

    python train.py --resume checkpoints/checkpoint_epoch_20.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import config as cfg
from dataset import build_dataloaders, build_train_transforms, build_eval_transforms, CucumberLeafDataset
from model import build_mobilenetv4
from utils import accuracy, compute_metrics, load_checkpoint, save_checkpoint, set_seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV4 for cucumber leaf disease detection"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=cfg.DATA_DIR,
        help="Root directory of the dataset (expects train/ val/ sub-dirs or flat layout)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Dataset has a flat layout (class sub-dirs, no train/val split dirs). "
             "Splits are created automatically.",
    )
    parser.add_argument(
        "--variant",
        choices=["small", "medium", "large"],
        default=cfg.MODEL_VARIANT,
        help="MobileNetV4 variant to train",
    )
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "step"],
        default=cfg.LR_SCHEDULER,
    )
    parser.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.CHECKPOINT_DIR,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument(
        "--num-workers", type=int, default=cfg.NUM_WORKERS, help="DataLoader workers"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=cfg.EARLY_STOPPING_PATIENCE,
        help="Early-stopping patience (epochs without val-acc improvement)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch.  Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(outputs, labels)
        total_loss += loss.item()
        total_acc += batch_acc

        if batch_idx % 20 == 0:
            logger.info(
                "Epoch %d [%d/%d]  loss=%.4f  acc=%.4f",
                epoch,
                batch_idx,
                num_batches,
                loss.item(),
                batch_acc,
            )

    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on *loader*.  Returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        total_acc += accuracy(outputs, labels)

    return total_loss / num_batches, total_acc / num_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Datasets -------------------------------------------------------
    data_dir = args.data_dir
    if args.flat:
        logger.info("Using flat dataset layout with automatic splitting …")
        train_ds, val_ds, test_ds = CucumberLeafDataset.from_flat_dir(
            data_dir,
            train_transform=build_train_transforms(),
            eval_transform=build_eval_transforms(),
        )
    else:
        train_ds = CucumberLeafDataset(
            data_dir / "train", transform=build_train_transforms()
        )
        val_ds = CucumberLeafDataset(
            data_dir / "val", transform=build_eval_transforms()
        )
        test_dir = data_dir / "test"
        test_ds = (
            CucumberLeafDataset(test_dir, transform=build_eval_transforms())
            if test_dir.is_dir()
            else None
        )

    logger.info(
        "Dataset sizes – train: %d | val: %d | test: %s",
        len(train_ds),
        len(val_ds),
        len(test_ds) if test_ds else "N/A",
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- Model ----------------------------------------------------------
    model = build_mobilenetv4(
        variant=args.variant,
        num_classes=cfg.NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MobileNetV4-%s | trainable params: %s", args.variant, f"{n_params:,}")

    # ---- Optimiser & scheduler ------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = StepLR(
            optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
        )

    # ---- Resume ---------------------------------------------------------
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and args.resume.exists():
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        logger.info(
            "Resumed from checkpoint %s (epoch %d, best_val_acc=%.4f)",
            args.resume,
            start_epoch - 1,
            best_val_acc,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    # ---- Training loop --------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info("  ↑ New best val_acc=%.4f", best_val_acc)
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "variant": args.variant,
                "num_classes": cfg.NUM_CLASSES,
                "class_names": cfg.CLASS_NAMES,
            },
            path=args.output_dir / f"checkpoint_epoch_{epoch:03d}.pth",
            is_best=is_best,
            best_path=args.output_dir / "best_model.pth",
        )

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                args.patience,
            )
            break

    # ---- Final evaluation on test set -----------------------------------
    if test_loader is not None:
        logger.info("Loading best model for test evaluation …")
        best_ckpt = args.output_dir / "best_model.pth"
        if best_ckpt.exists():
            load_checkpoint(best_ckpt, model, device=device)
        metrics = compute_metrics(model, test_loader, device)
        logger.info(
            "Test results – accuracy=%.4f  macro_f1=%.4f",
            metrics["accuracy"],
            metrics["macro_f1"],
        )


if __name__ == "__main__":
    main()
