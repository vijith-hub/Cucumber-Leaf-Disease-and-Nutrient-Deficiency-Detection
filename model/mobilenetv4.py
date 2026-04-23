"""
MobileNetV4 architecture for cucumber leaf disease and nutrient deficiency detection.

Reference: "MobileNetV4 -- Universal Models for the Mobile Ecosystem"
           Howard et al., 2024 (arXiv:2404.10518)

This module provides three variants:
  - MobileNetV4ConvSmall  (MNV4CS)  ~2.4 M params
  - MobileNetV4ConvMedium (MNV4CM)  ~11 M params
  - MobileNetV4ConvLarge  (MNV4CL)  ~49 M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Helper: make channel counts divisible by a given divisor (avoids
# odd channel numbers that hurt hardware efficiency).
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) block
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel-attention block."""

    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 0.25,
        act_layer: nn.Module = nn.ReLU,
        gate_layer: nn.Module = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        rd_channels = _make_divisible(in_channels * rd_ratio, 8)
        self.fc1 = nn.Conv2d(in_channels, rd_channels, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, in_channels, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


# ---------------------------------------------------------------------------
# ConvBNAct: Conv2d -> BatchNorm2d -> Activation
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Sequential):
    """Convolution + BatchNorm + Activation helper block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act_layer: Optional[type] = nn.ReLU6,
        bias: bool = False,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act_layer is not None:
            layers.append(act_layer())
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# FusedInvertedBottleneck (FIB) block
# Used in early layers where fusing the depthwise and pointwise convolution
# into a single 3×3 conv is more efficient on hardware.
#
# Architecture:
#   input -> Conv3×3 (expand, stride) -> [SE] -> Conv1×1 (project) -> output
# with skip connection when in_channels == out_channels and stride == 1.
# ---------------------------------------------------------------------------

class FusedInvertedBottleneck(nn.Module):
    """Fused Inverted Bottleneck block for early layers of MobileNetV4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.0,
        act_layer: type = nn.ReLU6,
    ) -> None:
        super().__init__()
        self.use_skip = stride == 1 and in_channels == out_channels
        mid_channels = _make_divisible(int(in_channels * expand_ratio))

        self.conv_exp = ConvBNAct(
            in_channels, mid_channels, kernel_size=3, stride=stride, act_layer=act_layer
        )
        self.se = (
            SqueezeExcitation(mid_channels, rd_ratio=se_ratio)
            if se_ratio > 0.0
            else nn.Identity()
        )
        self.conv_proj = ConvBNAct(
            mid_channels, out_channels, kernel_size=1, stride=1, act_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_exp(x)
        x = self.se(x)
        x = self.conv_proj(x)
        if self.use_skip:
            x = x + shortcut
        return x


# ---------------------------------------------------------------------------
# UniversalInvertedBottleneck (UIB) block
# The central building block of MobileNetV4. It generalises the classic
# Inverted Bottleneck, ConvNext-style block, and FFN via two boolean flags:
#   start_dw    – insert a depthwise conv BEFORE the first pointwise
#   middle_dw   – insert a depthwise conv BETWEEN the two pointwise convs
#
# Variants (Table 2 from the MobileNetV4 paper):
#   start_dw=F, middle_dw=F  → FFN / pointwise-only
#   start_dw=F, middle_dw=T  → IB (classic Inverted Bottleneck, MobileNetV2)
#   start_dw=T, middle_dw=F  → ConvNext-style
#   start_dw=T, middle_dw=T  → ExtraDW
# ---------------------------------------------------------------------------

class UniversalInvertedBottleneck(nn.Module):
    """Universal Inverted Bottleneck block – the core of MobileNetV4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        start_dw_kernel_size: int = 0,
        middle_dw_kernel_size: int = 3,
        middle_dw_downsample: bool = True,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.0,
        act_layer: type = nn.ReLU6,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the spatial downsampling step.
            start_dw_kernel_size: Kernel size of the optional start depthwise
                conv. 0 = disabled (no start_dw).
            middle_dw_kernel_size: Kernel size of the optional middle depthwise
                conv. 0 = disabled (no middle_dw).
            middle_dw_downsample: When True, apply the stride in the middle DW
                conv; otherwise apply it in the start DW conv.
            expand_ratio: Channel expansion ratio for the bottleneck.
            se_ratio: SE reduction ratio (0 = no SE).
            act_layer: Activation function class.
        """
        super().__init__()
        self.use_skip = stride == 1 and in_channels == out_channels
        mid_channels = _make_divisible(int(in_channels * expand_ratio))

        # ---- optional start depthwise --------------------------------
        if start_dw_kernel_size:
            start_dw_stride = 1 if middle_dw_downsample else stride
            self.start_dw = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=start_dw_kernel_size,
                stride=start_dw_stride,
                groups=in_channels,
                act_layer=None,
            )
        else:
            self.start_dw = None

        # ---- expand pointwise ----------------------------------------
        self.expand_conv = ConvBNAct(
            in_channels, mid_channels, kernel_size=1, act_layer=act_layer
        )

        # ---- optional middle depthwise -------------------------------
        if middle_dw_kernel_size:
            middle_dw_stride = stride if middle_dw_downsample else 1
            self.middle_dw = ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=middle_dw_kernel_size,
                stride=middle_dw_stride,
                groups=mid_channels,
                act_layer=act_layer,
            )
        else:
            self.middle_dw = None

        # ---- squeeze-and-excitation ----------------------------------
        self.se = (
            SqueezeExcitation(mid_channels, rd_ratio=se_ratio)
            if se_ratio > 0.0
            else nn.Identity()
        )

        # ---- project pointwise ---------------------------------------
        self.project_conv = ConvBNAct(
            mid_channels, out_channels, kernel_size=1, act_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        if self.start_dw is not None:
            x = self.start_dw(x)

        x = self.expand_conv(x)

        if self.middle_dw is not None:
            x = self.middle_dw(x)

        x = self.se(x)
        x = self.project_conv(x)

        if self.use_skip:
            x = x + shortcut
        return x


# ---------------------------------------------------------------------------
# Architecture specifications
# Each entry in a stage list is a tuple:
#   (block_type, out_ch, stride, start_dw_ks, middle_dw_ks, expand_ratio, se)
# block_type: "fib" = FusedInvertedBottleneck, "uib" = UniversalInvertedBottleneck
# ---------------------------------------------------------------------------

MNV4CS_SPEC: List[List[Tuple]] = [
    # Stage 0 – initial stem (handled separately as a plain conv)
    # Stage 1
    [
        ("fib", 32, 1, 0, 0, 2.0, 0.0),
        ("fib", 32, 1, 0, 0, 2.0, 0.0),
    ],
    # Stage 2
    [
        ("fib", 64, 2, 0, 0, 4.0, 0.0),
        ("fib", 64, 1, 0, 0, 2.0, 0.0),
        ("fib", 64, 1, 0, 0, 2.0, 0.0),
        ("fib", 64, 1, 0, 0, 2.0, 0.0),
    ],
    # Stage 3
    [
        ("uib", 96, 2, 3, 3, 4.0, 0.0),   # ExtraDW – stride in middle_dw
        ("uib", 96, 1, 0, 3, 4.0, 0.0),   # IB style
        ("uib", 96, 1, 0, 3, 4.0, 0.0),
        ("uib", 96, 1, 0, 3, 4.0, 0.0),
    ],
    # Stage 4
    [
        ("uib", 128, 2, 3, 3, 4.0, 0.0),  # ExtraDW – stride in middle_dw
        ("uib", 128, 1, 0, 3, 4.0, 0.0),
        ("uib", 128, 1, 0, 3, 4.0, 0.0),
        ("uib", 128, 1, 0, 3, 4.0, 0.0),
    ],
    # Stage 5
    [
        ("uib", 128, 1, 3, 5, 6.0, 0.0),
        ("uib", 128, 1, 5, 0, 6.0, 0.0),
        ("uib", 128, 1, 3, 5, 6.0, 0.0),
        ("uib", 128, 1, 0, 5, 6.0, 0.0),
        ("uib", 128, 1, 0, 5, 6.0, 0.0),
        ("uib", 128, 1, 0, 0, 2.0, 0.0),
    ],
]
MNV4CS_STEM_CH = 32
MNV4CS_HEAD_CH = 960

MNV4CM_SPEC: List[List[Tuple]] = [
    # Stage 1
    [
        ("fib", 48, 2, 0, 0, 4.0, 0.0),
    ],
    # Stage 2
    [
        ("uib", 80, 2, 3, 5, 4.0, 0.0),
        ("uib", 80, 1, 3, 3, 4.0, 0.0),
    ],
    # Stage 3
    [
        ("uib", 160, 2, 3, 5, 4.0, 0.0),
        ("uib", 160, 1, 3, 3, 4.0, 0.0),
        ("uib", 160, 1, 3, 3, 4.0, 0.0),
        ("uib", 160, 1, 3, 5, 4.0, 0.0),
        ("uib", 160, 1, 3, 3, 4.0, 0.0),
        ("uib", 160, 1, 3, 0, 4.0, 0.0),
        ("uib", 160, 1, 0, 0, 2.0, 0.0),
    ],
    # Stage 4
    [
        ("uib", 256, 2, 5, 5, 4.0, 0.0),
        ("uib", 256, 1, 5, 5, 4.0, 0.0),
        ("uib", 256, 1, 3, 5, 4.0, 0.0),
        ("uib", 256, 1, 3, 5, 4.0, 0.0),
        ("uib", 256, 1, 0, 0, 4.0, 0.0),
        ("uib", 256, 1, 3, 0, 4.0, 0.0),
        ("uib", 256, 1, 3, 5, 4.0, 0.0),
        ("uib", 256, 1, 0, 0, 4.0, 0.0),
        ("uib", 256, 1, 3, 0, 4.0, 0.0),
        ("uib", 256, 1, 5, 0, 4.0, 0.0),
    ],
    # Stage 5
    [
        ("uib", 256, 1, 5, 0, 6.0, 0.0),
        ("uib", 256, 1, 5, 0, 6.0, 0.0),
        ("uib", 256, 1, 5, 0, 6.0, 0.0),
        ("uib", 256, 1, 5, 0, 6.0, 0.0),
        ("uib", 256, 1, 5, 0, 6.0, 0.0),
        ("uib", 256, 1, 0, 0, 2.0, 0.0),
    ],
]
MNV4CM_STEM_CH = 32
MNV4CM_HEAD_CH = 960

MNV4CL_SPEC: List[List[Tuple]] = [
    # Stage 1
    [
        ("fib", 48, 2, 0, 0, 4.0, 0.0),
    ],
    # Stage 2
    [
        ("uib", 96, 2, 3, 5, 4.0, 0.0),
        ("uib", 96, 1, 3, 3, 4.0, 0.0),
    ],
    # Stage 3
    [
        ("uib", 192, 2, 3, 5, 4.0, 0.0),
        ("uib", 192, 1, 3, 3, 4.0, 0.0),
        ("uib", 192, 1, 3, 3, 4.0, 0.0),
        ("uib", 192, 1, 3, 3, 4.0, 0.0),
        ("uib", 192, 1, 3, 5, 4.0, 0.0),
        ("uib", 192, 1, 5, 3, 4.0, 0.0),
        ("uib", 192, 1, 5, 3, 4.0, 0.0),
        ("uib", 192, 1, 5, 3, 4.0, 0.0),
        ("uib", 192, 1, 5, 3, 4.0, 0.0),
        ("uib", 192, 1, 5, 5, 4.0, 0.0),
        ("uib", 192, 1, 3, 0, 4.0, 0.0),
        ("uib", 192, 1, 0, 0, 2.0, 0.0),
    ],
    # Stage 4
    [
        ("uib", 512, 2, 5, 5, 4.0, 0.0),
        ("uib", 512, 1, 5, 5, 4.0, 0.0),
        ("uib", 512, 1, 5, 5, 4.0, 0.0),
        ("uib", 512, 1, 5, 5, 4.0, 0.0),
        ("uib", 512, 1, 5, 0, 4.0, 0.0),
        ("uib", 512, 1, 5, 3, 4.0, 0.0),
        ("uib", 512, 1, 5, 0, 4.0, 0.0),
        ("uib", 512, 1, 5, 0, 4.0, 0.0),
        ("uib", 512, 1, 5, 0, 4.0, 0.0),
        ("uib", 512, 1, 5, 0, 4.0, 0.0),
        ("uib", 512, 1, 5, 3, 4.0, 0.0),
        ("uib", 512, 1, 5, 5, 4.0, 0.0),
        ("uib", 512, 1, 0, 5, 4.0, 0.0),
        ("uib", 512, 1, 0, 5, 4.0, 0.0),
    ],
    # Stage 5
    [
        ("uib", 512, 1, 5, 0, 6.0, 0.0),
        ("uib", 512, 1, 5, 0, 6.0, 0.0),
        ("uib", 512, 1, 5, 0, 6.0, 0.0),
        ("uib", 512, 1, 5, 0, 6.0, 0.0),
        ("uib", 512, 1, 5, 0, 6.0, 0.0),
        ("uib", 512, 1, 0, 0, 2.0, 0.0),
    ],
]
MNV4CL_STEM_CH = 24
MNV4CL_HEAD_CH = 960


# ---------------------------------------------------------------------------
# Core MobileNetV4 class
# ---------------------------------------------------------------------------

class MobileNetV4(nn.Module):
    """
    MobileNetV4 backbone for image classification.

    Supports Conv-Small, Conv-Medium, and Conv-Large variants.
    """

    def __init__(
        self,
        spec: List[List[Tuple]],
        stem_channels: int,
        head_channels: int,
        num_classes: int = 10,
        dropout: float = 0.2,
        act_layer: type = nn.ReLU6,
    ) -> None:
        """
        Args:
            spec: Stage specifications list (see MNV4CS_SPEC etc.).
            stem_channels: Number of output channels for the initial stem conv.
            head_channels: Number of channels in the pre-classifier head conv.
            num_classes: Number of output classes.
            dropout: Dropout probability before the final FC layer.
            act_layer: Activation function class used throughout the network.
        """
        super().__init__()

        # ---- Stem -------------------------------------------------------
        self.stem = ConvBNAct(
            3, stem_channels, kernel_size=3, stride=2, act_layer=act_layer
        )

        # ---- Body (stages) ----------------------------------------------
        stages: List[nn.Module] = []
        in_ch = stem_channels
        for stage_spec in spec:
            blocks: List[nn.Module] = []
            for block_spec in stage_spec:
                (btype, out_ch, stride, sdw_ks, mdw_ks, exp_r, se_r) = block_spec
                if btype == "fib":
                    blocks.append(
                        FusedInvertedBottleneck(
                            in_ch, out_ch,
                            stride=stride,
                            expand_ratio=exp_r,
                            se_ratio=se_r,
                            act_layer=act_layer,
                        )
                    )
                elif btype == "uib":
                    blocks.append(
                        UniversalInvertedBottleneck(
                            in_ch, out_ch,
                            stride=stride,
                            start_dw_kernel_size=sdw_ks,
                            middle_dw_kernel_size=mdw_ks,
                            expand_ratio=exp_r,
                            se_ratio=se_r,
                            act_layer=act_layer,
                        )
                    )
                else:
                    raise ValueError(f"Unknown block type: {btype!r}")
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # ---- Head -------------------------------------------------------
        self.head_conv = ConvBNAct(
            in_ch, head_channels, kernel_size=1, act_layer=act_layer
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = nn.Linear(head_channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature map after the head conv (before pooling)."""
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def MobileNetV4ConvSmall(num_classes: int = 10, dropout: float = 0.2, **kwargs) -> MobileNetV4:
    """MobileNetV4-Conv-Small (~2.4 M params) – best for resource-constrained inference."""
    return MobileNetV4(
        spec=MNV4CS_SPEC,
        stem_channels=MNV4CS_STEM_CH,
        head_channels=MNV4CS_HEAD_CH,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )


def MobileNetV4ConvMedium(num_classes: int = 10, dropout: float = 0.2, **kwargs) -> MobileNetV4:
    """MobileNetV4-Conv-Medium (~11 M params) – balanced accuracy / efficiency."""
    return MobileNetV4(
        spec=MNV4CM_SPEC,
        stem_channels=MNV4CM_STEM_CH,
        head_channels=MNV4CM_HEAD_CH,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )


def MobileNetV4ConvLarge(num_classes: int = 10, dropout: float = 0.2, **kwargs) -> MobileNetV4:
    """MobileNetV4-Conv-Large (~49 M params) – highest accuracy."""
    return MobileNetV4(
        spec=MNV4CL_SPEC,
        stem_channels=MNV4CL_STEM_CH,
        head_channels=MNV4CL_HEAD_CH,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )


def build_mobilenetv4(
    variant: str = "small",
    num_classes: int = 10,
    dropout: float = 0.2,
    **kwargs,
) -> MobileNetV4:
    """
    Factory helper to build a MobileNetV4 model by name.

    Args:
        variant: One of ``"small"``, ``"medium"``, or ``"large"``.
        num_classes: Number of output classes.
        dropout: Dropout probability before the final FC layer.
        **kwargs: Forwarded to :class:`MobileNetV4`.

    Returns:
        A :class:`MobileNetV4` instance.

    Example::

        model = build_mobilenetv4("small", num_classes=10)
    """
    variants = {
        "small": MobileNetV4ConvSmall,
        "medium": MobileNetV4ConvMedium,
        "large": MobileNetV4ConvLarge,
    }
    if variant not in variants:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {list(variants.keys())}."
        )
    return variants[variant](num_classes=num_classes, dropout=dropout, **kwargs)
