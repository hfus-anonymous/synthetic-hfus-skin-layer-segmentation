from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNormalize(nn.Module):
    """Normalize 0..1 RGB tensors with ImageNet statistics.

    The synthetic/real dataset loaders return RGB tensors in [0, 1]. Open-source
    ImageNet/ADE-pretrained segmentation models expect normalized RGB input.
    SegUNet B-2/B-3 is not pretrained and is therefore not wrapped by this module.
    """

    def __init__(self, mean: Tuple[float, float, float] = IMAGENET_MEAN, std: Tuple[float, float, float] = IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"ImageNetNormalize expects 3-channel RGB input, got shape={tuple(x.shape)}")
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, enabled: bool = True):
        super().__init__()
        self.normalize = ImageNetNormalize() if enabled else nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReleasedStyleSegUNet(nn.Module):
    """PyTorch reimplementation of the released 64x128 SegUNet architecture.

    This is a block-level reimplementation for B-2/B-3, not the original MATLAB
    model object. B-1 uses the released .mat DAGNetwork weights directly.
    """

    model_source = {
        "library": "custom_pytorch_reimplementation",
        "architecture": "Mendeley released SegUNet block-level reimplementation",
        "reference": "Ultras_SegUnetFull_Mendeley_v0/v1 DAGNetwork inspection",
        "pretrained_weights": "none",
    }

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base: int = 64):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(in_channels, base), ConvBNReLU(base, base))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNReLU(base, base * 2), ConvBNReLU(base * 2, base * 2))
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(ConvBNReLU(base * 2, base * 4), ConvBNReLU(base * 4, base * 4))
        self.p3 = nn.MaxPool2d(2)
        self.e4 = nn.Sequential(ConvBNReLU(base * 4, base * 8), ConvBNReLU(base * 8, base * 8))
        self.drop_e4 = nn.Dropout2d(0.5)
        self.p4 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 16, base * 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2), nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(ConvBNReLU(base * 16, base * 8), ConvBNReLU(base * 8, base * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2), nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(ConvBNReLU(base * 8, base * 4), ConvBNReLU(base * 4, base * 4))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2), nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(ConvBNReLU(base * 4, base * 2), ConvBNReLU(base * 2, base * 2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 2, stride=2), nn.ReLU(inplace=True))
        self.d4 = nn.Sequential(ConvBNReLU(base * 2, base), ConvBNReLU(base, base))
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))
        b = self.bridge(self.p4(self.drop_e4(e4)))
        x = self.up1(b)
        x = self.d1(torch.cat([x, e4], dim=1))
        x = self.up2(x)
        x = self.d2(torch.cat([x, e3], dim=1))
        x = self.up3(x)
        x = self.d3(torch.cat([x, e2], dim=1))
        x = self.up4(x)
        x = self.d4(torch.cat([x, e1], dim=1))
        return self.out(x)


class HFSegFormerWrapper(nn.Module):
    """Official Hugging Face SegFormer wrapper.

    This uses transformers.SegformerForSemanticSegmentation.from_pretrained,
    replaces/resizes the segmentation head for the requested number of labels,
    and upsamples logits to the input spatial size so the shared training loop can
    compute pixel-wise loss against the resized masks.
    """

    def __init__(
        self,
        checkpoint: str,
        num_classes: int,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except Exception as e:
            raise ImportError(
                "transformers is required for the official SegFormer model. "
                "Install it with: pip install transformers"
            ) from e
        if id2label is None:
            id2label = {i: str(i) for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
        self.model_source = {
            "library": "transformers",
            "architecture": "SegformerForSemanticSegmentation",
            "checkpoint": checkpoint,
            "num_labels": num_classes,
            "ignore_mismatched_sizes": True,
            "local_files_only": local_files_only,
            "input_normalization": "ImageNet mean/std wrapper",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits


def _attach_model_source(model: nn.Module, source: Dict[str, Any]) -> nn.Module:
    setattr(model, "model_source", source)
    return model


def get_model_source(model: nn.Module) -> Dict[str, Any]:
    if hasattr(model, "model_source"):
        return getattr(model, "model_source")
    if hasattr(model, "model") and hasattr(model.model, "model_source"):
        return getattr(model.model, "model_source")
    return {"library": "unknown"}


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    encoder: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    segformer_local_files_only: bool = False,
    id2label: Optional[Dict[int, str]] = None,
    normalize_pretrained: bool = True,
    **legacy_kwargs,
) -> nn.Module:
    """Build research-valid, traceable segmentation models.

    U-Net and DeepLabV3+ are loaded from segmentation_models_pytorch.
    SegFormer is loaded from Hugging Face transformers, not a custom/timm clone.
    SegUNet is only used for the Mendeley architecture-controlled B-2/B-3 tests.
    """
    model_name = model_name.lower()
    if "segformer_backbone" in legacy_kwargs and legacy_kwargs["segformer_backbone"] and segformer_checkpoint == "nvidia/segformer-b0-finetuned-ade-512-512":
        # Backward compatibility: old argument is accepted but deliberately not used
        # as a timm backbone. Official SegFormer always comes from HF checkpoint.
        pass
    if model_name == "segunet":
        return ReleasedStyleSegUNet(in_channels=in_channels, num_classes=num_classes)

    if in_channels != 3 and model_name in {"unet", "deeplabv3plus", "fpn", "segformer"}:
        raise ValueError(f"{model_name} open-source pretrained setup expects RGB 3-channel input, got in_channels={in_channels}")

    if encoder_weights is not None and str(encoder_weights).lower() in {"none", "null", "false", "0"}:
        encoder_weights = None

    if model_name == "segformer":
        label2id = None if id2label is None else {v: k for k, v in id2label.items()}
        model = HFSegFormerWrapper(
            checkpoint=segformer_checkpoint,
            num_classes=num_classes,
            id2label=id2label,
            label2id=label2id,
            local_files_only=segformer_local_files_only,
        )
        return NormalizedModel(model, enabled=normalize_pretrained)

    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise ImportError(
            "segmentation_models_pytorch is required for U-Net/DeepLabV3+/FPN. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from e

    if model_name == "unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes)
        source = {
            "library": "segmentation_models_pytorch",
            "architecture": "Unet",
            "encoder": encoder,
            "encoder_weights": encoder_weights,
            "input_normalization": "ImageNet mean/std wrapper" if normalize_pretrained and encoder_weights is not None else "none",
        }
        return NormalizedModel(_attach_model_source(model, source), enabled=normalize_pretrained and encoder_weights is not None)

    if model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes)
        source = {
            "library": "segmentation_models_pytorch",
            "architecture": "DeepLabV3Plus",
            "encoder": encoder,
            "encoder_weights": encoder_weights,
            "input_normalization": "ImageNet mean/std wrapper" if normalize_pretrained and encoder_weights is not None else "none",
        }
        return NormalizedModel(_attach_model_source(model, source), enabled=normalize_pretrained and encoder_weights is not None)

    if model_name == "fpn":
        model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes)
        source = {
            "library": "segmentation_models_pytorch",
            "architecture": "FPN",
            "encoder": encoder,
            "encoder_weights": encoder_weights,
            "input_normalization": "ImageNet mean/std wrapper" if normalize_pretrained and encoder_weights is not None else "none",
        }
        return NormalizedModel(_attach_model_source(model, source), enabled=normalize_pretrained and encoder_weights is not None)

    raise ValueError(f"Unknown model: {model_name}")
