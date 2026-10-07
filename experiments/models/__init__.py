"""Float reference networks and the datasets they are trained on.

A model module supplies ``DEFAULTS`` (per-dataset keyword arguments), ``build``
and ``descriptor``. Keeping the shape out of the quantization pipeline means a
new network — or a natively spiking one — is a file here rather than another
branch in the middle of the pipeline.

This package depends on ``torch`` alone, so it can be copied to a training
machine unchanged and the network trained there is the same object the
deployment path reconstructs. Turning a trained model into something a core can
run — folding BatchNorm and the input normalization, dropping inference no-ops —
is target-specific and belongs to the quantization entry point instead, in
``tool.paicore.fuse``.
"""

from torch import nn

from . import convnet, simplecnn_spec, vgg
from .datasets import DATASETS, Dataset

__all__ = [
    "DATASETS",
    "DEFAULT_MODEL",
    "Dataset",
    "REGISTRY",
    "build",
    "checkpoint_stem",
    "resolve",
]

REGISTRY = {"convnet": convnet, "simplecnn": simplecnn_spec, "vgg": vgg}

#: `convnet` on CIFAR-10 too, so the first CIFAR-10 numbers differ from the
#: MNIST ones by the task alone. `vgg` is the opt-in stronger baseline.
DEFAULT_MODEL = {"mnist": "convnet", "cifar10": "convnet"}


def resolve(model: str, dataset: str, **overrides) -> dict:
    """Keyword arguments for a model on a dataset, with overrides applied.

    Raises:
        KeyError: On an unknown model, or one with no defaults for this dataset.
        TypeError: On an override the model does not accept, which would
            otherwise be silently ignored.
    """
    if model not in REGISTRY:
        raise KeyError(f"unknown model {model!r}; have {sorted(REGISTRY)}")

    defaults = REGISTRY[model].DEFAULTS
    if dataset not in defaults:
        raise KeyError(
            f"model {model!r} has no defaults for {dataset!r}; "
            f"it defines {sorted(defaults)}"
        )

    given = {k: v for k, v in overrides.items() if v is not None}
    unknown = set(given) - set(defaults[dataset])
    if unknown:
        raise TypeError(
            f"model {model!r} does not take {sorted(unknown)}; "
            f"it takes {sorted(defaults[dataset])}"
        )
    return {**defaults[dataset], **given}


def build(model: str, dataset: str, **overrides) -> nn.Sequential:
    return REGISTRY[model].build(
        DATASETS[dataset], **resolve(model, dataset, **overrides)
    )


def checkpoint_stem(model: str, dataset: str, **overrides) -> str:
    """Filename stem identifying a trained model.

    Each model's ``descriptor`` names itself, which lets ``convnet`` keep the
    pre-existing ``<dataset>_c<channels>_fc<fc>`` form — the MNIST checkpoints
    already on disk are still found, and retraining them would silently move
    every accuracy number reported from them.
    """
    kwargs = resolve(model, dataset, **overrides)
    return f"{dataset}_{REGISTRY[model].descriptor(**kwargs)}"
