# Cucumber Leaf Disease and Nutrient Deficiency Detection

Early and accurate detection of plant diseases and nutrient deficiencies is crucial for crop yield and quality. This project uses a **MobileNetV4** architecture implemented in **PyTorch** to classify images of cucumber leaves into 10 health categories.

MobileNetV4 was specifically chosen for its optimal balance between accuracy and computational efficiency, making it ideal for on-device or edge-based mobile inference.

---

## Classes

| Label | Description |
|---|---|
| `Healthy` | No disease or deficiency detected |
| `Angular_Leaf_Spot` | Bacterial infection causing angular water-soaked lesions |
| `Anthracnose` | Fungal disease causing sunken dark lesions |
| `Bacterial_Wilt` | Bacterial infection spread by cucumber beetles |
| `Downy_Mildew` | Water-mould causing yellow patches on upper leaf surface |
| `Mosaic_Virus` | Viral disease causing mottled, mosaic discolouration |
| `Powdery_Mildew` | Fungal disease causing white powdery coating |
| `Nitrogen_Deficiency` | Yellowing (chlorosis) starting from older leaves |
| `Iron_Deficiency` | Interveinal chlorosis on young leaves |
| `Magnesium_Deficiency` | Interveinal yellowing on older leaves |

---

## Model Architecture — MobileNetV4

The project provides three Conv-only variants (no attention layers required):

| Variant | Factory function | Parameters | Recommended use |
|---|---|---|---|
| **Conv-Small** | `MobileNetV4ConvSmall` | ~2.4 M | Edge devices / mobile inference |
| **Conv-Medium** | `MobileNetV4ConvMedium` | ~11 M | Server / on-device with GPU |
| **Conv-Large** | `MobileNetV4ConvLarge` | ~49 M | Highest accuracy |

### Key building blocks

- **ConvBNAct** — Conv2d → BatchNorm2d → ReLU6
- **FusedInvertedBottleneck (FIB)** — Fused 3×3 conv + 1×1 projection; used in early stages
- **UniversalInvertedBottleneck (UIB)** — Generalises IB, ConvNext, FFN, and ExtraDW via two boolean flags:
  - `start_dw`: optional depthwise conv *before* the expand pointwise
  - `middle_dw`: optional depthwise conv *between* the two pointwise convs
- **SqueezeExcitation** — Channel attention (SE) block, used where `se_ratio > 0`

---

## Project Structure

```
.
├── model/
│   ├── __init__.py          # Public exports
│   └── mobilenetv4.py       # MobileNetV4 architecture (all 3 variants)
├── tests/
│   └── test_model.py        # Unit + integration tests (pytest)
├── config.py                # Project-wide settings and class names
├── dataset.py               # CucumberLeafDataset + transform builders
├── utils.py                 # Metrics, checkpointing, GradCAM
├── train.py                 # Training script (CLI)
├── predict.py               # Inference script (CLI)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Layout

```
data/
├── train/
│   ├── Healthy/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Angular_Leaf_Spot/
│   └── ...        (one sub-dir per class)
├── val/
└── test/
```

If you have a flat layout (all images under a single root, one sub-dir per class but no train/val split), pass `--flat` to the training script and splits will be created automatically.

---

## Training

```bash
# Train with default settings (data in ./data/)
python train.py

# Custom options
python train.py \
    --data-dir /path/to/dataset \
    --variant medium \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-4

# Flat dataset layout (auto-split)
python train.py --flat --data-dir /path/to/flat/dataset

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_020.pth
```

Key CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/` | Root dataset directory |
| `--flat` | off | Use flat directory layout with auto-split |
| `--variant` | `small` | `small` / `medium` / `large` |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--scheduler` | `cosine` | `cosine` or `step` |
| `--patience` | 10 | Early-stopping patience |
| `--output-dir` | `checkpoints/` | Checkpoint output directory |

---

## Inference

```bash
# Single image
python predict.py \
    --image leaf.jpg \
    --checkpoint checkpoints/best_model.pth

# Directory of images
python predict.py \
    --image-dir test_images/ \
    --checkpoint checkpoints/best_model.pth \
    --top-k 3

# Save predictions to JSON
python predict.py \
    --image leaf.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output-json results.json

# GradCAM visualisation
python predict.py \
    --image leaf.jpg \
    --checkpoint checkpoints/best_model.pth \
    --gradcam
```

---

## Python API

```python
from model import build_mobilenetv4

# Build a model
model = build_mobilenetv4("small", num_classes=10)

# Or use the explicit factory functions
from model import MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge

model = MobileNetV4ConvSmall(num_classes=10, dropout=0.2)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)
logits = model(x)           # (1, 10)
features = model.forward_features(x)  # (1, 960, H', W')
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## References

- Howard, A. et al. *MobileNetV4 — Universal Models for the Mobile Ecosystem*. arXiv:2404.10518, 2024.
- Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
- Hu, J. et al. *Squeeze-and-Excitation Networks*. CVPR 2018.
