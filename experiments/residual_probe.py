"""Does the lowering path accept a residual connection?

``converter.py`` lists ``operator.add`` and ``torch.add`` among the ops it maps,
and the IR has a ``PotentialAddOp``, so on paper a residual add is expressible.
On paper is not the same as working, so this compiles the two shapes a residual
can take and reports what actually happens to each.

The two differ in what is being added, which matters because the chip's only
addition mechanism is accumulation into a membrane potential:

**Potential add** — two convolutions feed one activation. Both operands are
32-bit accumulators, so the add is in the domain the hardware adds in.

**Identity skip** — the classic ``act(conv(act(conv(x))) + x)``. Here one
operand is an accumulator and the other is the *codes* that entered the block,
which are a different unit. Something has to bring the skip path into
accumulator units before the add is meaningful.

Usage
-----
    uv run python experiments/residual_probe.py
"""

import argparse
import contextlib
import io
import subprocess
import sys
import tempfile

import torch
from paicorelib import DataSign, DataWidth
from torch import nn

from paibox.backendv2 import Mapper
from paibox.paiir import ANNNodeV25, compile_to_paiir
from paibox.quant.lut import fit_affine_lut

CHANNELS = 8
SIDE = 8
ACC_STEPS_PER_CODE = 32.0


def integer_conv(in_c: int, out_c: int, kernel: int = 3) -> nn.Conv2d:
    conv = nn.Conv2d(in_c, out_c, kernel, padding=kernel // 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.randint(-7, 8, conv.weight.shape).float())
    return conv


def activation(signed: bool = False) -> ANNNodeV25:
    return ANNNodeV25(fit_affine_lut(1.0 / ACC_STEPS_PER_CODE, 8, signed=signed))


class PotentialAdd(nn.Module):
    """Two convolutions accumulating into one activation."""

    def __init__(self) -> None:
        super().__init__()
        self.left = integer_conv(CHANNELS, CHANNELS)
        self.right = integer_conv(CHANNELS, CHANNELS)
        self.act = activation()

    def forward(self, x):
        return self.act(self.left(x) + self.right(x))


class IdentitySkip(nn.Module):
    """The textbook residual block, skip connection carrying codes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = integer_conv(CHANNELS, CHANNELS)
        self.act1 = activation()
        self.conv2 = integer_conv(CHANNELS, CHANNELS)
        self.act2 = activation()

    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        return self.act2(h + x)


class ProjectedSkip(nn.Module):
    """Skip through a convolution, so both operands are accumulators."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = integer_conv(CHANNELS, CHANNELS)
        self.act1 = activation()
        self.conv2 = integer_conv(CHANNELS, CHANNELS)
        self.project = integer_conv(CHANNELS, CHANNELS, kernel=1)
        self.act2 = activation()

    def forward(self, x):
        h = self.act1(self.conv1(x))
        return self.act2(self.conv2(h) + self.project(x))


def stem(block: nn.Module) -> nn.Sequential:
    """A block needs a producer in front of it so its input is a code stream."""
    return nn.Sequential(integer_conv(CHANNELS, CHANNELS), activation(), block).eval()


def try_compile(name: str, block: nn.Module, timesteps: int = 16) -> None:
    model = stem(block)
    example = torch.zeros(1, CHANNELS, SIDE, SIDE)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            graph = compile_to_paiir(
                model,
                example,
                input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
                timesteps=timesteps,
                strict=True,
            )
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        print(f"{name}: lowering 失败")
        print(f"  {type(exc).__name__}: {exc}")
        return

    kinds = sorted({type(node).__name__ for node in graph.nodes.values()})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mapper = Mapper()
            with tempfile.TemporaryDirectory() as out_dir:
                mapper.compile(graph, out_dir, target_platform="x86", debug=False)
        cores = sum(rg.n_core_required for rg in mapper.routing_groups)
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: lowering 通过({len(graph.nodes)} 节点),布放失败")
        print(f"  节点类型: {kinds}")
        print(f"  {type(exc).__name__}: {exc}")
        return

    # Numerics: does the compiled graph agree with the eager model it came from?
    codes = torch.randint(0, 256, (1, CHANNELS, SIDE, SIDE)).float()
    with contextlib.redirect_stdout(io.StringIO()):
        held = codes.unsqueeze(0).expand(timesteps, *codes.shape)
        got = torch.as_tensor(graph.run(held, T=timesteps)).reshape(timesteps, -1)[-1]
    want = model(codes).detach().flatten().float()
    agree = got.numel() == want.numel() and bool(torch.equal(got.float(), want))

    print(
        f"{name}: 编译通过  核数={cores}  节点={len(graph.nodes)}  "
        f"与 eager 一致={'是' if agree else '否'}"
    )
    print(f"  节点类型: {kinds}")
    if not agree and got.numel() == want.numel():
        diff = (got.float() - want).abs()
        print(f"  最大偏差={int(diff.max())} 不一致元素={int((diff > 0).sum())}/{diff.numel()}")


CASES = {
    "potential": ("膜电位相加", PotentialAdd),
    "identity": ("恒等跳连", IdentitySkip),
    "projected": ("投影跳连", ProjectedSkip),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.case:
        torch.manual_seed(0)
        label, factory = CASES[args.case]
        try_compile(label, factory())
        return

    # One case per process: IR node names come from a process-global counter, so
    # a second compile produces InputNode_1 and `input_formats` silently misses.
    print("残差三种形态在 PAIBox 上的表现")
    for case in ("potential", "identity", "projected"):
        print()
        subprocess.run([sys.executable, __file__, "--case", case], check=False)


if __name__ == "__main__":
    main()
