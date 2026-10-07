"""Registry adapter for the externally trained :mod:`models.simplecnn`.

``simplecnn.py`` is kept exactly as it was trained, so the state dict in
``ckpt/cifar10_cnn.pt`` still loads. The registry interface lives here instead
of in that file.

There is no ``batch_norm`` switch: this is a fixed external architecture, not a
family being explored, and removing its BatchNorm is what ``tool.paicore.fuse``
does to any trained model on the way to deployment.
"""

from .datasets import Dataset
from .simplecnn import SimpleCNN

__all__ = ["DEFAULTS", "build", "descriptor"]

DEFAULTS = {
    "cifar10": {"width": 64},
}


def build(spec: Dataset, width: int) -> SimpleCNN:
    return SimpleCNN(num_classes=spec.classes, width=width)


def descriptor(width: int, **_) -> str:
    return f"cnn_w{width}"
