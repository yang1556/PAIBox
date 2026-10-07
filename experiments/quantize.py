"""Integer-only quantization pipeline targeting a PAICORE 2.5 offline core.

Numerics
--------
A core computes ``acc = sum(W_q * x_code)`` in a 32-bit accumulator, then a
256-entry LUT maps ``acc`` to the output code. There is no per-channel
multiplier after the accumulator: the only rescaling hardware is ``leak_tau``
(a shift, so powers of two only) and the LUT itself, and the LUT is a *per-core*
resource shared by every neuron on that core. So the LUT has to absorb
requantization, the activation, and output quantization in one table.

With per-tensor symmetric weights:

    s_w     = max|W| / (2**(b_w - 1) - 1)          weight scale
    x_code  in [0, 2**b_in - 1]                    input codes, real = x_code * s_in
    acc     = sum(W_q * x_code)                    integer, real = acc * s_w * s_in
    s_out   = y_max / (2**b_o - 1)                 output scale from calibration
    r       = s_w * s_in / s_out
    code    = clamp(round(acc * r), 0, 2**b_o - 1)

``code`` steps at ``acc = (k - 0.5) / r``, which is exactly where the LUT
thresholds go. The chip's SAR lookup then reproduces software round-to-nearest,
matching the <=0.5 LSB claim in the activation-function design doc.

The final layer uses a signed linear LUT instead of ReLU: clamping negative
logits to zero would destroy ``argmax``.

Scope
-----
SNN layers are out of scope here. A spiking layer emits 1-bit events and PAIBox
wires them straight into the next layer's axons with no rate decoding, so
accuracy for mixed ANN/SNN patterns needs threshold balancing and a timestep
choice that this module does not implement.

Usage
-----
    uv run python experiments/quantize.py --train
    uv run python experiments/quantize.py --weight-bits 4 --act-bits 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from paicorelib import DataSign, DataWidth
from torch import Tensor, nn
from torch.utils.data import DataLoader

import models
from models import DATASETS

from paibox.paiir import ANNNodeV25, LutCustom, compile_to_paiir

LUT_TABLE_SIZE = 256
DATA_ROOT = Path(__file__).resolve().parent / "data"
CKPT_DIR = Path(__file__).resolve().parent / "ckpt"

# Pixels enter the chip as uint8 codes 0..255, so the only input transform the
# hardware can express is ``code * s_in``. Training therefore uses a bare
# ``ToTensor`` and leaves whitening to the first BatchNorm rather than to a
# mean/std transform, which would put the input off the code grid.
INPUT_SCALE = 1.0 / 255.0
INPUT_BITS = 8

#: Seed for the shuffled training loader. Calibration draws its batches from it,
#: so an unseeded shuffle makes the calibrated ranges — and therefore every
#: accuracy number downstream — drift between runs.
SHUFFLE_SEED = 0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


#: Datasets supplied as raw ``.npy`` arrays rather than through torchvision.
#: Checked before the torchvision path, since torchvision would re-download.
NPY_ROOTS = {"cifar10": Path(__file__).resolve().parent / "ckpt" / "cifar10_data"}


class NpyImages(torch.utils.data.Dataset):
    """uint8 ``HWC`` arrays served as ``ToTensor`` would serve them.

    That means ``CHW`` float in ``[0, 1]``, i.e. the pixel code divided by 255,
    which is the only input transform the chip can express.
    """

    def __init__(self, root: Path, split: str, normalize: tuple | None = None) -> None:
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.array(self.images[index]))
        image = image.permute(2, 0, 1).float().div_(255.0)
        if self.normalize is not None:
            mean, std = self.normalize
            image = (image - torch.as_tensor(mean).reshape(-1, 1, 1)) / torch.as_tensor(
                std
            ).reshape(-1, 1, 1)
        return image, int(self.labels[index])


def loaders(
    dataset: str = "mnist",
    batch: int = 256,
    augment: bool = False,
    normalize: tuple | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Training and test loaders.

    Args:
        augment: Random crop and horizontal flip, which CIFAR-10 needs to train
            to a useful accuracy. Left off for calibration, so the ranges come
            from the data the model will actually see.
        normalize: ``(mean, std)`` to apply after scaling to ``[0, 1]``. Only for
            reproducing an external training recipe; a model being deployed
            should have this folded into its first convolution instead.
    """
    spec = DATASETS[dataset]
    root = NPY_ROOTS.get(dataset)
    if root is not None and root.is_dir():
        train_set = NpyImages(root, "train", normalize)
        test_set = NpyImages(root, "test", normalize)
        if augment:
            raise NotImplementedError(
                f"{dataset} is served from raw arrays in {root}, which carry no "
                "augmentation pipeline; train on a machine with torchvision data"
            )
    else:
        from torchvision import datasets, transforms

        train_tf: list = []
        if augment and spec.augmentable:
            h, w = spec.shape[1:]
            train_tf += [
                transforms.RandomCrop((h, w), padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        train_tf.append(transforms.ToTensor())  # scales uint8 -> [0, 1]
        if normalize is not None:
            train_tf.append(transforms.Normalize(*normalize))

        test_tf = [transforms.ToTensor()]
        if normalize is not None:
            test_tf.append(transforms.Normalize(*normalize))

        factory = getattr(datasets, spec.torchvision_name)
        train_set = factory(
            DATA_ROOT, train=True, download=True, transform=transforms.Compose(train_tf)
        )
        test_set = factory(
            DATA_ROOT, train=False, download=True, transform=transforms.Compose(test_tf)
        )

    return (
        DataLoader(
            train_set,
            batch_size=batch,
            shuffle=True,
            generator=torch.Generator().manual_seed(SHUFFLE_SEED),
        ),
        DataLoader(test_set, batch_size=batch),
    )


def mnist_loaders(batch: int = 256) -> tuple[DataLoader, DataLoader]:
    """Backwards-compatible alias."""
    return loaders("mnist", batch)


# --------------------------------------------------------------------------
# float reference model
# --------------------------------------------------------------------------


def training_device() -> torch.device:
    """Where to train. Everything downstream of training stays on the CPU:
    calibration, export, mapping and the PAIIR simulator are CPU paths."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_epochs(dataset: str) -> int:
    return 30 if dataset == "cifar10" else 2


def load_or_train(
    channels: tuple[int, ...] | None = None,
    fc: int | None = None,
    epochs: int | None = None,
    retrain: bool = False,
    dataset: str = "mnist",
    model: str | None = None,
    **overrides,
) -> tuple[nn.Sequential, list["LayerCalib"], float]:
    """Return a trained float model, its calibration, and its float accuracy.

    Any BatchNorm is already folded into the returned model, so what the caller
    sees is what gets quantized and deployed, and the reported accuracy is
    measured after folding — the baseline the chip is actually compared to.

    Calibration is cached with the weights so that sweep workers, which each run
    in a fresh process, do not repeat training or the calibration passes.

    Args:
        channels: ``convnet`` override, kept positional for existing callers.
        fc: Hidden fully connected width.
        model: Model name from :data:`models.REGISTRY`. Defaults per dataset.
        **overrides: Further model arguments, e.g. ``blocks`` for ``vgg``.
    """
    model = models.DEFAULT_MODEL[dataset] if model is None else model
    if channels is not None:
        overrides["channels"] = channels
    if fc is not None:
        overrides["fc"] = fc
    epochs = default_epochs(dataset) if epochs is None else epochs

    resolved = models.resolve(model, dataset, **overrides)
    folded = resolved.get("batch_norm", False)
    path = CKPT_DIR / f"{models.checkpoint_stem(model, dataset, **overrides)}.pt"

    def shell() -> nn.Sequential:
        net = models.build(model, dataset, **overrides)
        return models.fold_batchnorm(net) if folded else net.eval()

    if path.exists() and not retrain:
        blob = torch.load(path, weights_only=False)
        # Checkpoints predating BatchNorm support carry no "folded" key, and
        # were all trained without it.
        net = shell() if blob.get("folded", False) else models.build(model, dataset, **overrides)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net, [LayerCalib(**c) for c in blob["calibs"]], blob["float_acc"]

    net = models.build(model, dataset, **overrides)
    train(net, epochs, dataset)
    net = models.fold_batchnorm(net) if folded else net.eval()

    train_loader, test_loader = loaders(dataset)
    acc = float_accuracy(net, test_loader)
    calibs = calibrate(net, train_loader)
    print(f"  float test acc after folding = {acc:.4f}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "calibs": [c.__dict__ for c in calibs],
            "float_acc": acc,
            "folded": folded,
            "model": model,
            "resolved": resolved,
        },
        path,
    )
    return net, calibs, acc


def train(model: nn.Module, epochs: int = 2, dataset: str = "mnist") -> None:
    device = training_device()
    model.to(device)
    train_loader, test_loader = loaders(dataset, augment=True)

    if dataset == "cifar10":
        opt = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = None

    print(f"  training on {device}", flush=True)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if sched is not None:
            sched.step()
        print(
            f"  epoch {epoch + 1}/{epochs}: float test acc = "
            f"{float_accuracy(model, test_loader, device):.4f}",
            flush=True,
        )

    model.to("cpu")


@torch.no_grad()
def float_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> float:
    model.eval()
    device = next(model.parameters()).device if device is None else device
    correct = total = 0
    for x, y in loader:
        correct += int((model(x.to(device)).argmax(1).cpu() == y).sum())
        total += y.numel()
    return correct / total


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


@dataclass
class LayerCalib:
    """Calibrated dynamic range at one activation site of the float model."""

    act_max: float  # max of ReLU output (or max |pre-activation| for the last layer)
    is_final: bool


@torch.no_grad()
def calibrate(
    model: nn.Sequential, loader: DataLoader, n_batches: int = 8, percentile: float = 99.9
) -> list[LayerCalib]:
    """Record the dynamic range after each ReLU, plus the final linear output.

    A high percentile rather than the absolute max keeps a single outlier from
    stretching the LUT range and wasting all 256 bins.
    """
    sites: list[int] = [i for i, m in enumerate(model) if isinstance(m, nn.ReLU)]
    samples: dict[int, list[Tensor]] = {i: [] for i in (*sites, len(model))}

    model.eval()
    for batch, (x, _) in enumerate(loader):
        if batch >= n_batches:
            break
        h = x
        for i, mod in enumerate(model):
            h = mod(h)
            if i in samples:
                samples[i].append(h.detach().flatten().abs())
        samples[len(model)].append(h.detach().flatten().abs())

    # torch.quantile caps its input size, and one activation site easily exceeds
    # it, so take the percentile over a bounded random subsample.
    max_samples = 1 << 20

    def pct(values: list[Tensor]) -> float:
        flat = torch.cat(values).float()
        if flat.numel() > max_samples:
            idx = torch.randint(flat.numel(), (max_samples,), generator=torch.Generator().manual_seed(0))
            flat = flat[idx]
        return float(torch.quantile(flat, percentile / 100.0))

    calibs = [LayerCalib(pct(samples[i]), is_final=False) for i in sites]
    calibs.append(LayerCalib(pct(samples[len(model)]), is_final=True))
    return calibs


# --------------------------------------------------------------------------
# LUT construction
# --------------------------------------------------------------------------


def _pad_to_table(thresholds: list[int], values: list[int]) -> tuple[Tensor, Tensor]:
    """Pad to 256 entries keeping thresholds strictly increasing."""
    last_t, last_v = thresholds[-1], values[-1]
    while len(thresholds) < LUT_TABLE_SIZE:
        last_t += 1
        thresholds.append(last_t)
        values.append(last_v)
    return (
        torch.tensor(thresholds[:LUT_TABLE_SIZE], dtype=torch.int32),
        torch.tensor(values[:LUT_TABLE_SIZE], dtype=torch.int32),
    )


def build_lut(ratio: float, bits: int, signed: bool) -> LutCustom:
    """LUT implementing ``clamp(round(acc * ratio), lo, hi)`` over the accumulator.

    ``ratio = s_w * s_in / s_out``. Thresholds sit at ``(k - 0.5) / ratio``, the
    accumulator values where the output code increments, so the table reproduces
    software round-to-nearest exactly.
    """
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError(f"bad scale ratio {ratio}")

    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
    # The first bucket catches everything below the lowest step.
    thresholds = [-(2**30)]
    values = [lo]
    for k in range(lo + 1, hi + 1):
        edge = math.ceil((k - 0.5) / ratio)
        if edge <= thresholds[-1]:
            # Ratio so large that several codes share one accumulator value; the
            # surplus codes are unreachable and must not break monotonicity.
            edge = thresholds[-1] + 1
        thresholds.append(edge)
        values.append(k)

    thr, val = _pad_to_table(thresholds, values)
    return LutCustom(thr, val, output_signed=signed)


# --------------------------------------------------------------------------
# deployment model
# --------------------------------------------------------------------------


@dataclass
class QuantPlan:
    weight_bits: int
    act_bits: tuple[int, ...]


def quantize(
    model: nn.Sequential, calibs: list[LayerCalib], plan: QuantPlan
) -> nn.Sequential:
    """Build a PAIBox-lowerable model: integer weights plus calibrated LUTs.

    Weights are materialised as integers in a standard ``nn.Conv2d``/``nn.Linear``
    so ``fuse_to_offline_cores`` still sees a bare ``comp -> act`` pair. Wrapping
    the conv in a quantizer module would block that fusion and cost an extra core.
    """
    bw = plan.weight_bits
    w_limit = 2 ** (bw - 1) - 1 if bw > 1 else 1

    out: list[nn.Module] = []
    site = 0
    s_in = INPUT_SCALE
    for mod in model:
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            weight = mod.weight.detach()
            s_w = float(weight.abs().max()) / w_limit
            w_q = (weight / s_w).round().clamp(-w_limit - 1, w_limit)
            has_bias = mod.bias is not None

            new: nn.Module
            if isinstance(mod, nn.Conv2d):
                new = nn.Conv2d(
                    mod.in_channels, mod.out_channels, mod.kernel_size,
                    stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                    groups=mod.groups, bias=has_bias,
                )
            else:
                new = nn.Linear(mod.in_features, mod.out_features, bias=has_bias)
            with torch.no_grad():
                new.weight.copy_(w_q)
                if has_bias:
                    # The bias joins the accumulator, so it is expressed in
                    # accumulator steps rather than as a real value.
                    new.bias.copy_((mod.bias.detach() / (s_w * s_in)).round())
            out.append(new)

            calib = calibs[site]
            bits = plan.act_bits[site]
            span = (2 ** (bits - 1) - 1) if calib.is_final else (2**bits - 1)
            s_out = calib.act_max / max(span, 1)
            ratio = s_w * s_in / s_out
            out.append(ANNNodeV25(build_lut(ratio, bits, signed=calib.is_final)))
            s_in = s_out
            site += 1
        elif isinstance(mod, nn.ReLU):
            continue  # already emitted as the LUT activation above
        else:
            out.append(mod)  # MaxPool / Flatten are scale-preserving

    return nn.Sequential(*out).eval()


# --------------------------------------------------------------------------
# chip-accurate evaluation
# --------------------------------------------------------------------------


def pipeline_depth(graph) -> int:
    """Longest chain of offline cores from an input to an output.

    ``assign_tick_params`` gives each core a ``tick_start`` from its DAG depth,
    so a value needs one timestep per cascaded core to reach the output. Pooling
    occupies a core of its own and therefore costs a timestep too. Simulating
    with fewer timesteps than this returns zeros, and with ``auto_reset`` the
    cores also reset every ``timesteps`` ticks, wiping a half-filled pipeline.
    """
    from paibox.paiir.ir.op_node import OfflineCoreOp

    longest: dict[str, int] = {}
    for name in graph.topo_sort():
        preds = graph.predecessors(name)
        base = max((longest.get(p, 0) for p in preds), default=0)
        cost = 1 if isinstance(graph.nodes[name], OfflineCoreOp) else 0
        longest[name] = base + cost
    return max(longest.values(), default=0)


@torch.no_grad()
def chip_accuracy(graph, loader: DataLoader, timesteps: int, n_batches: int) -> float:
    """Evaluate with ``PAIIRGraph.run``, which reproduces core neuron semantics.

    ``run`` wants ``[T, N, ...]``, so the same image is held on the input for
    every timestep and the answer is read from the final step, by which point
    the pipeline has filled.

    Args:
        n_batches: Batches to evaluate, or 0 for the whole loader. A few hundred
            samples leave roughly a point of noise, which is more than the gaps
            between neighbouring points on the accuracy/core front.
    """
    correct = total = 0
    for batch, (x, y) in enumerate(loader):
        if n_batches > 0 and batch >= n_batches:
            break
        for i in range(x.shape[0]):  # PAIIR simulates batch size 1
            codes = (x[i : i + 1] / INPUT_SCALE).round().clamp(0, 2**INPUT_BITS - 1)
            held = codes.unsqueeze(0).expand(timesteps, *codes.shape)
            logits = torch.as_tensor(graph.run(held, T=timesteps))
            correct += int(logits.reshape(timesteps, -1)[-1].argmax() == y[i])
            total += 1
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="retrain even if a checkpoint exists")
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, help="default depends on the dataset")
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument(
        "--timesteps", type=int, default=0,
        help="0 = use the measured pipeline depth (minimum for a valid output)",
    )
    parser.add_argument("--eval-batches", type=int, default=1, help="batches of 256 for chip eval")
    args = parser.parse_args()

    model, calibs, float_acc = load_or_train(
        retrain=args.train, epochs=args.epochs, dataset=args.dataset
    )
    _, test_loader = loaders(args.dataset)
    print(f"float test acc = {float_acc:.4f}")

    for i, c in enumerate(calibs):
        print(f"  site {i}: act_max={c.act_max:.4f} final={c.is_final}")

    n_sites = len(calibs)
    plan = QuantPlan(args.weight_bits, (args.act_bits,) * n_sites)
    deploy = quantize(model, calibs, plan)

    # Depth is only known after fusion, so compile with a generous window first
    # and then simulate for exactly as many timesteps as the pipeline needs.
    compile_t = args.timesteps if args.timesteps > 0 else 64
    with contextlib.redirect_stdout(io.StringIO()):
        graph = compile_to_paiir(
            deploy,
            torch.zeros(1, *DATASETS[args.dataset].shape),
            input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
            timesteps=compile_t,
            strict=True,
        )
    depth = pipeline_depth(graph)
    run_t = args.timesteps if args.timesteps > 0 else depth
    print(f"compiled: {len(graph.nodes)} PAIIR nodes, pipeline depth {depth} cores")
    if run_t < depth:
        print(f"  WARNING: T={run_t} < depth {depth}; output will still be empty")

    acc = chip_accuracy(graph, test_loader, run_t, args.eval_batches)
    print(f"chip-accurate acc (w{args.weight_bits}/a{args.act_bits}, T={run_t}) = {acc:.4f}")


if __name__ == "__main__":
    main()
