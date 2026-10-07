"""Core cost of an architecture, without training it.

Core count is mostly a function of layer shapes and bit widths, so an
architecture can be priced against the 63-core budget before anyone spends an
hour training it. This matters because the budget is the binding constraint on
which baselines are even available: a published network that needs several
hundred cores is not a baseline this chip can run.

**Mostly, not entirely.** The weights matter too, by a route that is easy to
miss: the backend chooses between dense and CSC weight storage from the
sparsity, and how many weight codes land on zero depends on the scale, which
comes from calibration. The same architecture at w8a8 measured 627 cores with
four calibration batches and 643 with two. So treat this as an estimate with a
few percent of slack, not the number a trained model will produce.

The spec handed to ``export`` is synthetic but well-formed — a consistent scale
chain and a uniform LUT per site — so the measurement goes through the real
export, lowering and mapping path rather than a model of it.

Usage
-----
    uv run python experiments/core_probe.py --dataset cifar10 --model vgg --depth 16
    uv run python experiments/core_probe.py --dataset cifar10 --model vgg --sweep-depth
"""

import argparse
import contextlib
import io
import sys
import tempfile
import time
from pathlib import Path

import torch
from paicorelib import DataSign, DataWidth, HwConfigV2
from torch import nn

QUANT_TOOL = Path.home() / "quant_tool"
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))

import models  # noqa: E402
from models import DATASETS  # noqa: E402
from tool.paicore import prepare  # noqa: E402

from paibox.backendv2 import Mapper
from paibox.paiir import compile_to_paiir
from paibox.quant import (
    AnnActivation,
    LayerQuant,
    QuantSpec,
    UnsupportedConfig,
    WeightQuant,
    export,
)

CORE_BUDGET = HwConfigV2.N_CORE_OFFLINE
INPUT_SCALE = 1.0 / 255.0

#: Accumulator steps per output code in the synthetic spec. Any value in this
#: range gives a well-formed LUT; it does not affect placement.
ACC_STEPS_PER_CODE = 64.0


def synthetic_spec(
    model: nn.Sequential, weight_bits: int, act_bits: int, timesteps: int
) -> QuantSpec:
    """A well-formed spec with plausible scales, for shape-only measurement."""
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE, timesteps=timesteps)
    sites = [
        name for name, mod in model.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]

    s_in = INPUT_SCALE
    for i, name in enumerate(sites):
        final = i == len(sites) - 1
        s_w = 0.01
        s_out = s_w * s_in * ACC_STEPS_PER_CODE
        span = 2 ** (act_bits - 1) - 1 if final else 2**act_bits - 1
        spec.layers[name] = LayerQuant(
            input_scale=s_in,
            output_scale=s_out,
            weight=WeightQuant(bits=weight_bits, scale=s_w),
            activation=AnnActivation(bits=act_bits, signed=final, max_val=s_out * span),
        )
        s_in = s_out

    spec.check_scale_chain(sites)
    return spec


def probe(
    dataset: str, model: str, weight_bits: int, act_bits: int, timesteps: int, **overrides
) -> dict:
    """Compile one architecture and report what it costs."""
    started = time.perf_counter()
    # Untrained, but through the real preparation step: folding an untrained
    # BatchNorm gives meaningless weights and placement does not read weights,
    # while skipping it would leave a BatchNorm the lowering has no node for.
    net = prepare(models.build(model, dataset, **overrides))
    spec = synthetic_spec(net, weight_bits, act_bits, timesteps)

    with contextlib.redirect_stdout(io.StringIO()):
        deployed = export(net, spec)
        graph = compile_to_paiir(
            deployed,
            torch.zeros(1, *DATASETS[dataset].shape),
            input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
            timesteps=timesteps,
            strict=True,
        )
        mapper = Mapper()
        with tempfile.TemporaryDirectory() as out_dir:
            mapper.compile(graph, out_dir, target_platform="x86", debug=False)

    cores = sum(rg.n_core_required for rg in mapper.routing_groups)
    return {
        "cores": cores,
        "budget": 100.0 * cores / CORE_BUDGET,
        "fits": cores <= CORE_BUDGET,
        "sites": len(spec.layers),
        "params": sum(p.numel() for p in net.parameters()),
        "max_lcn": max((int(rg.lcn.value) for rg in mapper.routing_groups), default=0),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--model", default="vgg", choices=sorted(models.REGISTRY))
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--depth", type=int, help="VGG depth")
    parser.add_argument("--width", type=float, help="VGG channel multiplier")
    parser.add_argument("--channels", help="convnet channels, comma separated")
    parser.add_argument("--fc", type=int, help="convnet hidden FC width")
    parser.add_argument(
        "--sweep-depth", action="store_true", help="price every VGG depth and width"
    )
    args = parser.parse_args()

    overrides = {}
    if args.depth is not None:
        overrides["depth"] = args.depth
    if args.width is not None:
        overrides["width"] = args.width
    if args.channels:
        overrides["channels"] = tuple(int(c) for c in args.channels.split(","))
    if args.fc is not None:
        overrides["fc"] = args.fc

    print(f"budget: {CORE_BUDGET} offline cores, w{args.weight_bits}a{args.act_bits}")
    print()
    header = f"{'config':<20}{'sites':>7}{'params':>12}{'cores':>8}{'budget':>9}{'maxLCN':>8}{'sec':>7}"
    print(header)
    print("-" * len(header))

    from models import vgg

    if args.sweep_depth:
        cases = [
            ({"depth": d, "width": w}, f"vgg{d} w={w:g}")
            for w in (1.0, 0.5, 0.25)
            for d in sorted(vgg.CONFIGS)
        ]
    else:
        cases = [(overrides, f"{args.model} {overrides or 'default'}")]

    for case, label in cases:
        try:
            got = probe(
                args.dataset, args.model, args.weight_bits, args.act_bits,
                args.timesteps, **case,
            )
        except UnsupportedConfig as exc:
            print(f"{label:<20}{'REJECTED':>7}  {exc}"[:110])
            continue
        except Exception as exc:  # noqa: BLE001 - a failure is a data point
            print(f"{label:<20}{'FAIL':>7}  {type(exc).__name__}: {exc}"[:110])
            continue

        flag = "" if got["fits"] else "  OVER BUDGET"
        print(
            f"{label:<20}{got['sites']:>7}{got['params']:>12,}{got['cores']:>8}"
            f"{got['budget']:>8.0f}%{got['max_lcn']:>8}{got['seconds']:>7.0f}{flag}"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
