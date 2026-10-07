"""Classification tasks, described only by what a model needs to know.

Pixels reach the chip as uint8 codes, so the only input transform the hardware
can express is ``code * s_in``. That rules out a mean/std normalizing transform,
which would move the input off the code grid; whitening is left to the first
BatchNorm instead, and folding it later turns it into weights and a bias.
"""

from dataclasses import dataclass

__all__ = ["DATASETS", "Dataset"]


@dataclass(frozen=True)
class Dataset:
    """A task and the shape of one sample."""

    name: str
    shape: tuple[int, int, int]
    classes: int

    @property
    def torchvision_name(self) -> str:
        return {"mnist": "MNIST", "cifar10": "CIFAR10"}[self.name]

    @property
    def augmentable(self) -> bool:
        """Whether random crop and horizontal flip are label-preserving here.

        False for MNIST: digits are not mirror-symmetric, so a flip relabels the
        sample.
        """
        return self.name == "cifar10"


DATASETS = {
    "mnist": Dataset("mnist", (1, 28, 28), 10),
    "cifar10": Dataset("cifar10", (3, 32, 32), 10),
}
