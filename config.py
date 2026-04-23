"""
Project-wide configuration for the Cucumber Leaf Disease and
Nutrient Deficiency Detection system.

Disease / nutrient-deficiency categories are based on the publicly available
PlantVillage dataset and domain literature for cucumber plants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Healthy",
    "Angular_Leaf_Spot",
    "Anthracnose",
    "Bacterial_Wilt",
    "Downy_Mildew",
    "Mosaic_Virus",
    "Powdery_Mildew",
    "Nitrogen_Deficiency",
    "Iron_Deficiency",
    "Magnesium_Deficiency",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_VARIANT = "small"   # "small" | "medium" | "large"
DROPOUT = 0.2

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"   # "cosine" | "step"
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 10

# ---------------------------------------------------------------------------
# Input image dimensions (MobileNetV4 default)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224    # pixels (square)

# ---------------------------------------------------------------------------
# Data split ratios
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15   # must sum to 1.0

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

RANDOM_HORIZONTAL_FLIP_PROB = 0.5
RANDOM_VERTICAL_FLIP_PROB = 0.3
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.1
RANDOM_ROTATION_DEGREES = 30

# ---------------------------------------------------------------------------
# ImageNet normalisation statistics (used when training from scratch or
# fine-tuning from an ImageNet-pretrained checkpoint).
# ---------------------------------------------------------------------------

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

NUM_WORKERS = 4
RANDOM_SEED = 42
