"""
Inference script for Cucumber Leaf Disease and Nutrient Deficiency Detection.

Usage
-----
Predict a single image:

    python predict.py --image leaf.jpg --checkpoint checkpoints/best_model.pth

Predict all images in a directory:

    python predict.py --image-dir test_images/ --checkpoint checkpoints/best_model.pth

Generate a GradCAM visualisation:

    python predict.py --image leaf.jpg --checkpoint checkpoints/best_model.pth --gradcam
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import config as cfg
from dataset import build_eval_transforms
from model import build_mobilenetv4
from utils import GradCAM, load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MobileNetV4 inference on cucumber leaf images"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image file")
    group.add_argument(
        "--image-dir", type=Path, help="Directory containing images to predict"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--variant",
        choices=["small", "medium", "large"],
        default=cfg.MODEL_VARIANT,
        help="MobileNetV4 variant (must match the checkpoint)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate a GradCAM visualisation and save it alongside the input",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write predictions to a JSON file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    variant: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load MobileNetV4 model from *checkpoint_path*."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names: List[str] = checkpoint.get("class_names", cfg.CLASS_NAMES)
    num_classes = checkpoint.get("num_classes", len(class_names))
    variant = checkpoint.get("variant", variant)

    model = build_mobilenetv4(variant=variant, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: List[str],
    device: torch.device,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """
    Run inference on a single image.

    Args:
        model: Trained MobileNetV4 model.
        image_path: Path to the image file.
        class_names: Ordered list of class names.
        device: Inference device.
        top_k: Number of top predictions to return.

    Returns:
        List of dicts with ``"rank"``, ``"class"``, and ``"confidence"`` keys,
        sorted by descending confidence.
    """
    transform = build_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(top_k)

    return [
        {
            "rank": rank + 1,
            "class": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]


def save_gradcam(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    class_idx: Optional[int] = None,
) -> Path:
    """
    Compute and save a GradCAM overlay for *image_path*.

    The resulting file is saved as ``<image_stem>_gradcam.png`` next to the
    input image.

    Args:
        model: Trained MobileNetV4 model.
        image_path: Path to the input image.
        device: Inference device.
        class_idx: Class to visualise. Uses predicted class when ``None``.

    Returns:
        Path to the saved GradCAM image.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        print("numpy is required for GradCAM visualisation.", file=sys.stderr)
        raise

    transform = build_eval_transforms()
    pil_image = PILImage.open(image_path).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # Attach GradCAM to the last conv in head_conv
    target_layer = model.head_conv[0]  # Conv2d inside the ConvBNAct Sequential
    cam = GradCAM(model, target_layer)
    heatmap = cam(tensor, class_idx=class_idx)
    cam.remove_hooks()

    # Resize heatmap to image dimensions
    h, w = pil_image.size[1], pil_image.size[0]
    heatmap_resized = np.array(
        PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
    )

    # Blend heatmap with original image
    img_array = np.array(pil_image).astype(np.float32)
    colormap = np.zeros((h, w, 3), dtype=np.float32)
    colormap[:, :, 0] = heatmap_resized.astype(np.float32)  # Red channel
    colormap[:, :, 2] = (255 - heatmap_resized).astype(np.float32)  # Blue channel

    blended = (0.6 * img_array + 0.4 * colormap).clip(0, 255).astype(np.uint8)
    output_path = image_path.parent / f"{image_path.stem}_gradcam.png"
    PILImage.fromarray(blended).save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model -----------------------------------------------------
    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    model, class_names = load_model(args.checkpoint, args.variant, device)
    print(
        f"Loaded MobileNetV4 checkpoint from {args.checkpoint} "
        f"({len(class_names)} classes)"
    )

    # ---- Collect image paths --------------------------------------------
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = [
            p for p in sorted(args.image_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            print(f"No images found in {args.image_dir}", file=sys.stderr)
            sys.exit(1)

    # ---- Run inference --------------------------------------------------
    all_results: Dict[str, List[Dict]] = {}

    for image_path in image_paths:
        predictions = predict_image(
            model, image_path, class_names, device, top_k=args.top_k
        )
        all_results[str(image_path)] = predictions

        print(f"\nImage: {image_path.name}")
        for pred in predictions:
            bar = "█" * int(pred["confidence"] * 20)
            print(
                f"  #{pred['rank']} {pred['class']:<25}  "
                f"{pred['confidence']:.1%}  {bar}"
            )

        if args.gradcam:
            cam_path = save_gradcam(model, image_path, device)
            print(f"  GradCAM saved → {cam_path}")

    # ---- Write JSON output if requested ---------------------------------
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nPredictions written to {args.output_json}")


if __name__ == "__main__":
    main()
