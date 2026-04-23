"""
Dataset utilities for Cucumber Leaf Disease and Nutrient Deficiency Detection.

Expected directory layout
-------------------------
data/
├── train/
│   ├── Healthy/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Angular_Leaf_Spot/
│   └── ...
├── val/
│   └── <same class sub-dirs>
└── test/
    └── <same class sub-dirs>

If you have a flat structure (all images in one folder with class sub-dirs
but no train/val/test split), use ``CucumberLeafDataset.from_flat_dir``
which automatically creates stratified splits.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import config as cfg


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transforms(image_size: int = cfg.IMAGE_SIZE) -> T.Compose:
    """Return augmentation + normalisation transforms for training."""
    return T.Compose(
        [
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=cfg.RANDOM_HORIZONTAL_FLIP_PROB),
            T.RandomVerticalFlip(p=cfg.RANDOM_VERTICAL_FLIP_PROB),
            T.RandomRotation(degrees=cfg.RANDOM_ROTATION_DEGREES),
            T.ColorJitter(
                brightness=cfg.COLOR_JITTER_BRIGHTNESS,
                contrast=cfg.COLOR_JITTER_CONTRAST,
                saturation=cfg.COLOR_JITTER_SATURATION,
                hue=cfg.COLOR_JITTER_HUE,
            ),
            T.ToTensor(),
            T.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD),
        ]
    )


def build_eval_transforms(image_size: int = cfg.IMAGE_SIZE) -> T.Compose:
    """Return deterministic transforms for validation / test / inference."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CucumberLeafDataset(Dataset):
    """
    PyTorch Dataset for cucumber leaf disease and nutrient-deficiency images.

    Parameters
    ----------
    root : str | Path
        Root directory that contains one sub-directory per class.
    class_names : list[str], optional
        Ordered list of class names.  Sub-directories that do not appear in
        this list are ignored.  Defaults to ``config.CLASS_NAMES``.
    transform : callable, optional
        Transform applied to every ``PIL.Image`` before returning it.
    extensions : sequence[str]
        File extensions that count as images.
    """

    EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    def __init__(
        self,
        root: str | Path,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        extensions: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.class_names: List[str] = list(class_names or cfg.CLASS_NAMES)
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.transform = transform
        self.extensions = tuple(extensions or self.EXTENSIONS)

        self.samples: List[Tuple[Path, int]] = self._make_dataset()

    # ------------------------------------------------------------------ #
    def _make_dataset(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]
            for fpath in sorted(class_dir.iterdir()):
                if fpath.suffix.lower() in self.extensions:
                    samples.append((fpath, label))
        return samples

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------ #
    @classmethod
    def from_flat_dir(
        cls,
        root: str | Path,
        class_names: Optional[List[str]] = None,
        train_ratio: float = cfg.TRAIN_SPLIT,
        val_ratio: float = cfg.VAL_SPLIT,
        seed: int = cfg.RANDOM_SEED,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
    ) -> Tuple["CucumberLeafDataset", "CucumberLeafDataset", "CucumberLeafDataset"]:
        """
        Create stratified train / val / test splits from a flat directory.

        Returns three datasets whose underlying sample lists share no overlap.

        Args:
            root: Directory with one sub-directory per class (no split folders).
            class_names: Optional ordered class list (uses ``config.CLASS_NAMES``).
            train_ratio: Fraction of data to use for training.
            val_ratio: Fraction of data to use for validation.
            seed: Random seed for reproducibility.
            train_transform: Transform applied to training images.
            eval_transform: Transform applied to val/test images.

        Returns:
            ``(train_ds, val_ds, test_ds)`` tuple of :class:`CucumberLeafDataset`.
        """
        rng = random.Random(seed)

        full_ds = cls(root, class_names=class_names)
        # Build per-class index lists
        class_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(full_ds.samples):
            class_indices.setdefault(label, []).append(idx)

        train_idx, val_idx, test_idx = [], [], []
        for label_indices in class_indices.values():
            shuffled = label_indices[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train : n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val :])

        def _subset(indices: List[int], transform: Optional[Callable]) -> "CucumberLeafDataset":
            ds = cls(root, class_names=class_names, transform=transform)
            ds.samples = [full_ds.samples[i] for i in indices]
            return ds

        return (
            _subset(train_idx, train_transform or build_train_transforms()),
            _subset(val_idx, eval_transform or build_eval_transforms()),
            _subset(test_idx, eval_transform or build_eval_transforms()),
        )


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Wrap datasets in DataLoaders with sensible defaults.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Optional test dataset.
        batch_size: Mini-batch size.
        num_workers: Number of data-loading worker processes.

    Returns:
        ``(train_loader, val_loader, test_loader)`` where ``test_loader``
        is ``None`` when ``test_ds`` is ``None``.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader
