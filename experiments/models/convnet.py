"""One convolution per resolution, followed by two fully connected layers.

This is the shape every MNIST measurement so far used, kept unchanged so that
CIFAR-10 results differ from MNIST results by the task and not by the network
family. It is deliberately not a strong CIFAR-10 baseline; see :mod:`.vgg` for
that.

Activations are plain ``nn.ReLU`` so the export stage can find each activation
site by module type, and convolutions carry no bias when a BatchNorm follows,
since folding produces the bias instead.
"""

from torch import nn

from .datasets import Dataset

__all__ = ["DEFAULTS", "build", "descriptor"]

DEFAULTS = {
    "mnist": {
        "channels": (16, 32),
        "fc": 128,
        "batch_norm": False,
        "fc_bias": False,
    },
    "cifar10": {
        "channels": (32, 64, 128),
        "fc": 256,
        "batch_norm": True,
        "fc_bias": True,
    },
}


def build(
    spec: Dataset,
    channels: tuple[int, ...],
    fc: int,
    batch_norm: bool,
    fc_bias: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    c, h, w = spec.shape

    for out_c in channels:
        layers.append(nn.Conv2d(c, out_c, 3, padding=1, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        c, h, w = out_c, h // 2, w // 2

    layers += [
        nn.Flatten(),
        nn.Linear(c * h * w, fc, bias=fc_bias),
        nn.ReLU(),
        nn.Linear(fc, spec.classes, bias=fc_bias),
    ]
    return nn.Sequential(*layers)


def descriptor(channels: tuple[int, ...], fc: int, **_) -> str:
    """Checkpoint stem. Unchanged from before the models package existed, so
    the MNIST checkpoints already on disk keep being found."""
    return f"c{'-'.join(map(str, channels))}_fc{fc}"
