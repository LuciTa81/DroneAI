"""Modern PyTorch implementation of the published CSRNet architecture."""

from __future__ import annotations

import torch
from torch import nn

FRONTEND_FEATURES = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512)
BACKEND_FEATURES = (512, 512, 512, 256, 128, 64)
OUTPUT_STRIDE = 8


def _make_layers(
    config: tuple[int | str, ...], *, in_channels: int = 3, dilation: int = 1
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        layers.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """VGG-style frontend, dilation-2 backend and one-channel density output."""

    output_stride = OUTPUT_STRIDE

    def __init__(self, *, pretrained_frontend: bool = False) -> None:
        super().__init__()
        self.frontend = _make_layers(FRONTEND_FEATURES)
        self.backend = _make_layers(BACKEND_FEATURES, in_channels=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        if pretrained_frontend:
            self._load_vgg16_frontend()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(inputs)))

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _load_vgg16_frontend(self) -> None:
        from torchvision.models import VGG16_Weights, vgg16

        source = vgg16(weights=VGG16_Weights.DEFAULT).features.state_dict()
        target = self.frontend.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        if compatible.keys() != target.keys():
            missing = sorted(set(target) - set(compatible))
            raise RuntimeError(f"VGG16 frontend mapping incomplete: {missing}")
        self.frontend.load_state_dict(compatible)


def architecture_summary(model: CSRNet) -> dict[str, int]:
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    dilated = [module for module in convolutions if module.dilation == (2, 2)]
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "convolution_layers": len(convolutions),
        "dilated_convolution_layers": len(dilated),
        "output_stride": model.output_stride,
    }
