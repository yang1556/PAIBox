"""Accuracy against core count over the weight/activation bit-width grid.

This is the figure the resource-aware argument rests on. Earlier sweeps measured
cores with random weights and accuracy with a crude in-file quantizer, so the
two axes came from different models and the low-bit end was unusable. Both now
come from one pipeline:

    float model
      -> quant_tool (modelopt + tool/paicore adapter, MSE-clipped calibration)
      -> QuantSpec
      -> paibox.quant.export
      -> compile_to_paiir -> backendv2   (cores)
      -> PAIIRGraph.run                  (accuracy)

MSE-clipped calibration is what makes the low-bit end meaningful. Max
calibration sets the scale from the largest value it ever sees, which collapses
at 2-bit — an earlier "accuracy cliff" between 4 and 2 bits turned out to be an
artefact of it rather than a property of the hardware.

Weight granularity is an axis of its own (``--granularities``). Per-channel
scales are expressible only as power-of-two channel ratios realised by each
neuron's shift, so there are two ways to reach them: calibrate freely and round
the channel ratios afterwards (``pc-round``), or constrain the search to the
power-of-two lattice from the start (``pc-pow2``). Both are measured against
``per-tensor``.

Each configuration runs in a subprocess: IR node names come from a process-global
counter, and conv unrolling materialises large dense matrices.

Usage
-----
    uv run python experiments/pareto.py
    uv run python experiments/pareto.py --channels 32,64 --fc 256 --json pareto.json
"""

import argparse
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

QUANT_TOOL = Path.home() / "quant_tool"
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))

from paicorelib import DataSign, DataWidth, HwConfigV2  # noqa: E402
from models import DATASETS  # noqa: E402
from quantize import (  # noqa: E402
    INPUT_SCALE,
    chip_accuracy,
    load_or_train,
    loaders,
    pipeline_depth,
)
from tool.mytorch.opt import apply_mode  # noqa: E402
from tool.mytorch.quantization.mode import QuantizeModeRegistry  # noqa: E402
from tool.paicore import (  # noqa: E402
    build_spec,
    calibrate_with,
    paicore_cfg,
)

from paibox.backendv2 import Mapper  # noqa: E402
from paibox.paiir import compile_to_paiir  # noqa: E402
from paibox.quant import export  # noqa: E402

CORE_BUDGET = HwConfigV2.N_CORE_OFFLINE
CALIB_BATCHES = 4

# The vendored quantization package prints to stdout while being imported, and a
# first run also trains, so a worker's stdout is not pure JSON. The result is
# tagged instead of assuming it stands alone.
RESULT_MARKER = "__PARETO_RESULT__"

# Owned by deploy.py, which is the stage that pairs a calibrator with a policy.
from deploy import GRANULARITIES  # noqa: E402


@dataclass
class Point:
    weight_bits: int
    act_bits: int
    granularity: str
    ok: bool
    cores: int = 0
    accuracy: float = 0.0
    depth: int = 0
    max_lcn: int = 0
    neuron_sram: int = 0
    folds: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def label(self) -> str:
        return f"w{self.weight_bits}a{self.act_bits}"


def measure(
    channels: tuple[int, ...] | None,
    fc: int | None,
    weight_bits: int,
    act_bits: int,
    granularity: str,
    timesteps: int,
    eval_batches: int,
    dataset: str,
    model: str | None,
) -> dict:
    started = time.perf_counter()
    per_channel, calibrator, policy = GRANULARITIES[granularity]
    float_model, _, _ = load_or_train(channels, fc, dataset=dataset, model=model)
    train_loader, test_loader = loaders(dataset)

    def forward_loop(model):
        for i, (x, _) in enumerate(train_loader):
            if i >= CALIB_BATCHES:
                break
            model(x)

    # The backend prints progress bars; stdout carries the JSON result. Its fold
    # decisions are only reported through those prints, so they are kept.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        quantized = copy.deepcopy(float_model)
        cfg = paicore_cfg(
            weight_bits=weight_bits, act_bits=act_bits, per_channel=per_channel
        )
        quantized = apply_mode(
            quantized, mode=[("quantize", cfg)], registry=QuantizeModeRegistry
        )
        calibrate_with(quantized, calibrator, forward_loop)

        spec = build_spec(
            quantized,
            input_scale=INPUT_SCALE,
            timesteps=timesteps,
            per_channel_policy=policy,
            # Topology from the float model: quantization replaced modules in
            # place so the paths agree, and without it the scale chain would be
            # threaded by declaration order.
            topology=float_model,
        )
        deployed = export(float_model, spec)

        graph = compile_to_paiir(
            deployed,
            torch.zeros(1, *DATASETS[dataset].shape),
            input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
            timesteps=timesteps,
            strict=True,
        )
        depth = pipeline_depth(graph)

        mapper = Mapper()
        with tempfile.TemporaryDirectory() as out_dir:
            mapper.compile(graph, out_dir, target_platform="x86", debug=False)
        cores = sum(rg.n_core_required for rg in mapper.routing_groups)
        max_lcn = max((int(rg.lcn.value) for rg in mapper.routing_groups), default=0)

        # A per-neuron shift is what makes per-channel expressible, and the
        # router folds neurons only when their Part2 attributes all agree, so
        # per-channel can cost neuron SRAM by breaking folds. Both are recorded
        # to tell a real penalty from an assumed one.
        neuron_sram = sum(
            placement.n_sram_required
            for rg in mapper.routing_groups
            for cp in (getattr(rg, "core_placements", None) or [])
            for placement in (getattr(cp, "neus", None) or [])
        )

        accuracy = chip_accuracy(graph, test_loader, max(timesteps, depth), eval_batches)

    return {
        "ok": True,
        "cores": cores,
        "accuracy": accuracy,
        "depth": depth,
        "max_lcn": max_lcn,
        "neuron_sram": neuron_sram,
        "folds": log.getvalue().count("Find fold"),
        "seconds": time.perf_counter() - started,
    }


def run_worker(args) -> None:
    try:
        result = measure(
            _channels(args),
            args.fc,
            args.weight_bits,
            args.act_bits,
            args.granularity,
            args.timesteps,
            args.eval_batches,
            args.dataset,
            args.model,
        )
    except Exception as exc:  # noqa: BLE001 - a failed configuration is a data point
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + RESULT_MARKER + json.dumps(result))


def _channels(args) -> tuple[int, ...] | None:
    """``None`` leaves the model's own default in place, which is what a model
    other than ``convnet`` needs — it does not take a channel list."""
    return tuple(int(c) for c in args.channels.split(",")) if args.channels else None


def spawn(args, weight_bits: int, act_bits: int, granularity: str) -> Point:
    cmd = [
        sys.executable, __file__, "--worker",
        "--weight-bits", str(weight_bits),
        "--act-bits", str(act_bits),
        "--granularity", granularity,
        "--dataset", args.dataset,
        "--timesteps", str(args.timesteps),
        "--eval-batches", str(args.eval_batches),
    ]
    if args.channels:
        cmd += ["--channels", args.channels]
    if args.fc:
        cmd += ["--fc", str(args.fc)]
    if args.model:
        cmd += ["--model", args.model]
    failed = lambda error: Point(weight_bits, act_bits, granularity, ok=False, error=error)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return failed(f"timeout {args.timeout:.0f}s")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return failed(tail[-1] if tail else "nonzero exit")
    _, _, tagged = (proc.stdout or "").rpartition(RESULT_MARKER)
    try:
        payload = json.loads(tagged.strip())
    except json.JSONDecodeError:
        tail = (proc.stdout or "").strip().splitlines()
        return failed(f"no tagged result; last line: {tail[-1][:60] if tail else '<empty>'}")
    if not payload.get("ok"):
        return failed(payload.get("error", "?"))

    return Point(
        weight_bits, act_bits, granularity, ok=True,
        cores=payload["cores"],
        accuracy=payload["accuracy"],
        depth=payload["depth"],
        max_lcn=payload["max_lcn"],
        neuron_sram=payload["neuron_sram"],
        folds=payload["folds"],
        seconds=payload["seconds"],
    )


def pareto_front(points: list[Point]) -> list[Point]:
    """Points no other beats on both axes at once."""
    return [
        p
        for p in points
        if not any(
            (o.cores < p.cores and o.accuracy >= p.accuracy)
            or (o.cores <= p.cores and o.accuracy > p.accuracy)
            for o in points
        )
    ]


def report_granularity(ok: list[Point]) -> None:
    """What per-channel buys and what it costs, against per-tensor."""
    grans = [g for g in GRANULARITIES if any(p.granularity == g for p in ok)]
    if len(grans) < 2:
        return

    baseline = {
        (p.weight_bits, p.act_bits): p for p in ok if p.granularity == "per-tensor"
    }
    print()
    print("Per-channel against per-tensor (same bit widths):")
    print(f"  {'config':<8}{'granularity':<13}{'d_acc':>9}{'d_cores':>9}{'d_neuSRAM':>11}")
    for p in sorted(ok, key=lambda p: (-p.weight_bits, -p.act_bits, p.granularity)):
        base = baseline.get((p.weight_bits, p.act_bits))
        if base is None or p.granularity == "per-tensor":
            continue
        print(
            f"  {p.label:<8}{p.granularity:<13}{p.accuracy - base.accuracy:>+9.4f}"
            f"{p.cores - base.cores:>+9d}"
            f"{100.0 * (p.neuron_sram / max(base.neuron_sram, 1) - 1.0):>+10.1f}%"
        )


def report(args, points: list[Point], float_acc: float) -> None:
    print()
    print(
        f"dataset: {args.dataset}, model: {args.model or 'default'}"
        f"{f' channels={args.channels}' if args.channels else ''}"
        f"{f' fc={args.fc}' if args.fc else ''}, float accuracy {float_acc:.4f}"
    )
    print(f"budget: {CORE_BUDGET} offline cores per chip")
    print()
    header = (
        f"{'config':<8}{'granularity':<13}{'cores':>7}{'budget':>8}{'acc':>9}"
        f"{'maxLCN':>8}{'depth':>7}{'neuSRAM':>9}{'folds':>7}{'sec':>7}"
    )
    print(header)
    print("-" * len(header))
    for p in sorted(points, key=lambda p: (-p.weight_bits, -p.act_bits, p.granularity)):
        if not p.ok:
            print(f"{p.label:<8}{p.granularity:<13}{'FAIL':>7}  {p.error[:44]}")
            continue
        print(
            f"{p.label:<8}{p.granularity:<13}{p.cores:>7}"
            f"{100.0 * p.cores / CORE_BUDGET:>7.0f}%{p.accuracy:>9.4f}"
            f"{p.max_lcn:>8}{p.depth:>7}{p.neuron_sram:>9}{p.folds:>7}{p.seconds:>7.0f}"
        )

    ok = [p for p in points if p.ok]
    if not ok:
        return

    report_granularity(ok)

    print()
    print("Pareto front (accuracy vs cores):")
    for p in sorted(pareto_front(ok), key=lambda p: p.cores):
        print(
            f"  {p.label:<8}{p.granularity:<13}cores={p.cores:<4}"
            f"({100.0 * p.cores / CORE_BUDGET:.0f}% budget)  acc={p.accuracy:.4f}"
        )

    cheapest = min(ok, key=lambda p: p.cores)
    best = max(ok, key=lambda p: p.accuracy)
    print()
    print(
        f"span: {best.label} needs {best.cores} cores for {best.accuracy:.4f}; "
        f"{cheapest.label} needs {cheapest.cores} for {cheapest.accuracy:.4f} "
        f"({best.cores / max(cheapest.cores, 1):.2f}x cores)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--weight-bits", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--act-bits", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--granularity", default="per-tensor", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument(
        "--model", help="model name; defaults per dataset (see models.DEFAULT_MODEL)"
    )
    parser.add_argument("--channels", help="convnet conv output channels")
    parser.add_argument("--fc", type=int, help="hidden FC width")
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument(
        "--eval-batches", type=int, default=0,
        help="batches of 256 to evaluate; 0 for the whole test set",
    )
    parser.add_argument("--grid", default="8,4,2", help="weight bit widths to sweep")
    parser.add_argument(
        "--act-grid", help="activation bit widths to sweep; defaults to --grid"
    )
    parser.add_argument(
        "--granularities", default="per-tensor",
        help=f"weight granularities to sweep, from {','.join(GRANULARITIES)}",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--json", help="write raw points here")
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    channels = _channels(args)
    weight_widths = [int(b) for b in args.grid.split(",")]
    act_widths = [int(b) for b in (args.act_grid or args.grid).split(",")]
    grans = [g.strip() for g in args.granularities.split(",")]
    unknown = [g for g in grans if g not in GRANULARITIES]
    if unknown:
        parser.error(f"unknown granularity {unknown}; pick from {list(GRANULARITIES)}")

    # Train once in the driver so every worker only loads the checkpoint.
    _, _, float_acc = load_or_train(
        channels, args.fc, dataset=args.dataset, model=args.model
    )
    print(f"float accuracy {float_acc:.4f}", flush=True)

    configs = [
        (w, a, g) for w in weight_widths for a in act_widths for g in grans
    ]
    points: list[Point] = []
    for n, (weight_bits, act_bits, gran) in enumerate(configs, start=1):
        print(f"[{n}/{len(configs)}] w{weight_bits}a{act_bits} {gran} ...", flush=True)
        point = spawn(args, weight_bits, act_bits, gran)
        print(
            f"    {'ok' if point.ok else 'FAIL'} "
            f"cores={point.cores} acc={point.accuracy:.4f} {point.error}",
            flush=True,
        )
        points.append(point)

    report(args, points, float_acc)

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "channels": channels,
                    "fc": args.fc,
                    "eval_batches": args.eval_batches,
                    "float_accuracy": float_acc,
                    "points": [p.__dict__ for p in points],
                },
                fp,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
