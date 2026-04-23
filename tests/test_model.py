"""
Tests for the MobileNetV4 model architecture and related utilities.

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure project root is on the path so imports work when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mobilenetv4 import (
    MobileNetV4,
    MobileNetV4ConvSmall,
    MobileNetV4ConvMedium,
    MobileNetV4ConvLarge,
    build_mobilenetv4,
    ConvBNAct,
    FusedInvertedBottleneck,
    UniversalInvertedBottleneck,
    SqueezeExcitation,
    _make_divisible,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
NUM_CLASSES = 10
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def small_model() -> MobileNetV4:
    return MobileNetV4ConvSmall(num_classes=NUM_CLASSES)


@pytest.fixture(scope="module")
def medium_model() -> MobileNetV4:
    return MobileNetV4ConvMedium(num_classes=NUM_CLASSES)


@pytest.fixture(scope="module")
def large_model() -> MobileNetV4:
    return MobileNetV4ConvLarge(num_classes=NUM_CLASSES)


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# _make_divisible
# ---------------------------------------------------------------------------

class TestMakeDivisible:
    def test_already_divisible(self):
        assert _make_divisible(64, 8) == 64

    def test_nearest_multiple(self):
        # _make_divisible finds the nearest multiple of the divisor.
        # 65 → nearest multiple of 8 is 64 (within the 10% tolerance).
        result = _make_divisible(65, 8)
        assert result % 8 == 0
        # Result should be within 10% of the original value
        assert abs(result - 65) / 65 < 0.1

    def test_min_value(self):
        assert _make_divisible(1, 8) >= 8

    def test_near_divisible(self):
        # 63 → rounds to nearest multiple of 8 ≥ 63, but ≥ min_value (8)
        result = _make_divisible(63, 8)
        assert result % 8 == 0


# ---------------------------------------------------------------------------
# ConvBNAct
# ---------------------------------------------------------------------------

class TestConvBNAct:
    def test_output_shape(self):
        block = ConvBNAct(32, 64, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 56, 56)
        out = block(x)
        assert out.shape == (1, 64, 28, 28)

    def test_no_activation(self):
        block = ConvBNAct(16, 32, kernel_size=1, act_layer=None)
        # Should not contain any activation module
        act_types = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)
        has_act = any(isinstance(m, act_types) for m in block.modules())
        assert not has_act

    def test_depthwise_conv(self):
        block = ConvBNAct(32, 32, kernel_size=3, groups=32)
        x = torch.randn(1, 32, 14, 14)
        out = block(x)
        assert out.shape == (1, 32, 14, 14)


# ---------------------------------------------------------------------------
# SqueezeExcitation
# ---------------------------------------------------------------------------

class TestSqueezeExcitation:
    def test_output_shape_preserved(self):
        se = SqueezeExcitation(64, rd_ratio=0.25)
        x = torch.randn(2, 64, 7, 7)
        out = se(x)
        assert out.shape == x.shape

    def test_scaling_range(self):
        se = SqueezeExcitation(32)
        x = torch.ones(1, 32, 4, 4)
        out = se(x)
        # Output should be non-negative (identity × positive scale)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# FusedInvertedBottleneck
# ---------------------------------------------------------------------------

class TestFusedInvertedBottleneck:
    def test_same_spatial_size(self):
        block = FusedInvertedBottleneck(32, 32, stride=1, expand_ratio=2.0)
        x = torch.randn(1, 32, 28, 28)
        out = block(x)
        assert out.shape == (1, 32, 28, 28)

    def test_downsampling(self):
        block = FusedInvertedBottleneck(32, 64, stride=2, expand_ratio=4.0)
        x = torch.randn(1, 32, 28, 28)
        out = block(x)
        assert out.shape == (1, 64, 14, 14)

    def test_skip_connection_used_when_same_shape(self):
        block = FusedInvertedBottleneck(32, 32, stride=1)
        assert block.use_skip is True

    def test_no_skip_on_channel_change(self):
        block = FusedInvertedBottleneck(32, 64, stride=1)
        assert block.use_skip is False

    def test_no_skip_on_stride_change(self):
        block = FusedInvertedBottleneck(32, 32, stride=2)
        assert block.use_skip is False


# ---------------------------------------------------------------------------
# UniversalInvertedBottleneck
# ---------------------------------------------------------------------------

class TestUniversalInvertedBottleneck:
    def _make(self, start_dw=0, middle_dw=3, stride=1, in_ch=32, out_ch=32):
        return UniversalInvertedBottleneck(
            in_channels=in_ch,
            out_channels=out_ch,
            stride=stride,
            start_dw_kernel_size=start_dw,
            middle_dw_kernel_size=middle_dw,
            expand_ratio=4.0,
        )

    def test_ib_style(self):
        """IB style: no start_dw, middle_dw=3."""
        block = self._make(start_dw=0, middle_dw=3, stride=1)
        x = torch.randn(1, 32, 14, 14)
        assert block(x).shape == (1, 32, 14, 14)

    def test_convnext_style(self):
        """ConvNext style: start_dw=3, no middle_dw."""
        block = self._make(start_dw=3, middle_dw=0, stride=1)
        x = torch.randn(1, 32, 14, 14)
        assert block(x).shape == (1, 32, 14, 14)

    def test_extradw_style(self):
        """ExtraDW style: start_dw=3, middle_dw=3."""
        block = self._make(start_dw=3, middle_dw=3, stride=1)
        x = torch.randn(1, 32, 14, 14)
        assert block(x).shape == (1, 32, 14, 14)

    def test_ffn_style(self):
        """FFN style: no start_dw, no middle_dw."""
        block = self._make(start_dw=0, middle_dw=0, stride=1)
        x = torch.randn(1, 32, 14, 14)
        assert block(x).shape == (1, 32, 14, 14)

    def test_downsampling(self):
        block = self._make(start_dw=3, middle_dw=3, stride=2, in_ch=64, out_ch=128)
        x = torch.randn(1, 64, 28, 28)
        assert block(x).shape == (1, 128, 14, 14)

    def test_skip_enabled(self):
        block = self._make(stride=1)
        assert block.use_skip is True

    def test_skip_disabled_on_stride(self):
        block = self._make(stride=2, out_ch=64)
        assert block.use_skip is False


# ---------------------------------------------------------------------------
# Full model forward pass – output shape
# ---------------------------------------------------------------------------

class TestMobileNetV4ForwardPass:
    """Verify that each variant produces correct output shapes."""

    def test_small_output_shape(self, small_model, dummy_batch):
        small_model.eval()
        with torch.no_grad():
            out = small_model(dummy_batch)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_medium_output_shape(self, medium_model, dummy_batch):
        medium_model.eval()
        with torch.no_grad():
            out = medium_model(dummy_batch)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_large_output_shape(self, large_model, dummy_batch):
        large_model.eval()
        with torch.no_grad():
            out = large_model(dummy_batch)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_small_accepts_non_standard_resolution(self, small_model):
        """Model should work with any input divisible by 32 due to adaptive pooling."""
        small_model.eval()
        for h, w in [(192, 192), (256, 256), (320, 320)]:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                out = small_model(x)
            assert out.shape == (1, NUM_CLASSES), f"Failed for resolution {h}×{w}"

    def test_forward_features_shape(self, small_model, dummy_batch):
        small_model.eval()
        with torch.no_grad():
            feat = small_model.forward_features(dummy_batch)
        # Feature map should have 960 channels (head_channels for small variant)
        assert feat.shape[0] == BATCH_SIZE
        assert feat.shape[1] == 960

    def test_custom_num_classes(self, dummy_batch):
        for n_cls in [3, 5, 15]:
            model = MobileNetV4ConvSmall(num_classes=n_cls)
            model.eval()
            with torch.no_grad():
                out = model(dummy_batch)
            assert out.shape == (BATCH_SIZE, n_cls)


# ---------------------------------------------------------------------------
# build_mobilenetv4 factory
# ---------------------------------------------------------------------------

class TestBuildFactory:
    def test_small_variant(self):
        model = build_mobilenetv4("small", num_classes=NUM_CLASSES)
        assert isinstance(model, MobileNetV4)

    def test_medium_variant(self):
        model = build_mobilenetv4("medium", num_classes=NUM_CLASSES)
        assert isinstance(model, MobileNetV4)

    def test_large_variant(self):
        model = build_mobilenetv4("large", num_classes=NUM_CLASSES)
        assert isinstance(model, MobileNetV4)

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            build_mobilenetv4("xlarge", num_classes=NUM_CLASSES)


# ---------------------------------------------------------------------------
# Parameter count sanity checks
# ---------------------------------------------------------------------------

class TestParameterCounts:
    """Verify that model sizes are in the expected ballpark."""

    def _count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def test_small_param_count(self, small_model):
        n = self._count_params(small_model)
        # MNV4CS should be roughly 3–6 M parameters
        assert 1_000_000 < n < 10_000_000, f"Small model has {n:,} params"

    def test_medium_param_count(self, medium_model):
        n = self._count_params(medium_model)
        # MNV4CM should be roughly 8–15 M parameters
        assert 5_000_000 < n < 20_000_000, f"Medium model has {n:,} params"

    def test_large_param_count(self, large_model):
        n = self._count_params(large_model)
        # MNV4CL should be roughly 25–50 M parameters
        assert 15_000_000 < n < 100_000_000, f"Large model has {n:,} params"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through_small_model(self, dummy_batch):
        model = MobileNetV4ConvSmall(num_classes=NUM_CLASSES)
        model.train()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ---------------------------------------------------------------------------
# Train / eval mode behaviour
# ---------------------------------------------------------------------------

class TestTrainEvalMode:
    def test_dropout_inactive_in_eval(self, small_model, dummy_batch):
        """Inference should be deterministic in eval mode."""
        small_model.eval()
        with torch.no_grad():
            out1 = small_model(dummy_batch)
            out2 = small_model(dummy_batch)
        assert torch.allclose(out1, out2), "Eval mode output is not deterministic"

    def test_bn_running_stats_update_in_train(self, dummy_batch):
        model = MobileNetV4ConvSmall(num_classes=NUM_CLASSES)
        model.train()
        bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        mean_before = bn.running_mean.clone()
        model(dummy_batch)
        # Running mean should have changed after a forward pass
        assert not torch.equal(bn.running_mean, mean_before)
