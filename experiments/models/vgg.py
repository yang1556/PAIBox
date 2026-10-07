"""VGG, in the configurations from the original paper.

Depths 11/13/16/19 are the published channel layouts, with the CIFAR-sized
classifier that the literature uses on 32x32 inputs: five max-pools take the
feature map to 1x1, so the head is a single ``Linear(512, classes)`` rather
than the two 4096-wide layers of the ImageNet version.

Whether any of these fit a 63-core chip is a separate question from whether
they are the right baseline, and it is measured rather than assumed — see
``experiments/core_probe.py``.
"""

from torch import nn

from .datasets import Dataset

__all__ = ["CONFIGS", "DEFAULTS", "build", "descriptor"]

#: Convolution channel counts per depth, ``"M"`` marking a max-pool.
CONFIGS = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M",
    ],
    19: [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M",
    ],
}

DEFAULTS = {
    "cifar10": {"depth": 16, "width": 1.0, "batch_norm": True, "fc_bias": True},
}


def build(
    spec: Dataset,
    depth: int,
    width: float,
    batch_norm: bool,
    fc_bias: bool,
) -> nn.Sequential:
    """Args:
        depth: One of :data:`CONFIGS`.
        width: Channel multiplier. Below 1.0 this is no longer the published
            network, but it is the one knob that trades the shape of VGG
            against a core budget without changing its depth.
    """
    if depth not in CONFIGS:
        raise ValueError(f"no VGG-{depth} configuration; have {sorted(CONFIGS)}")

    layers: list[nn.Module] = []
    c = spec.shape[0]
    for entry in CONFIGS[depth]:
        if entry == "M":
            layers.append(nn.MaxPool2d(2))
            continue
        out_c = max(1, int(entry * width))
        layers.append(nn.Conv2d(c, out_c, 3, padding=1, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU())
        c = out_c

    layers += [nn.Flatten(), nn.Linear(c, spec.classes, bias=fc_bias)]
    return nn.Sequential(*layers)


def descriptor(depth: int, width: float, **_) -> str:
    return f"vgg{depth}" if width == 1.0 else f"vgg{depth}w{width:g}"
