"""Sweep PAICORE 2.5 resource usage over per-layer mode and bit-width choices.

What this measures
------------------
A v2.5 chip has 63 offline cores. Core count is not chosen by an optimizer:
``RoutingGroup.try_store_neuron`` packs neurons and weights greedily under a
4096-line SRAM budget, and ``route_solver.py`` only *places* groups whose sizes
are already fixed. Two front-end quantization knobs reach that packing through
different paths, and a third choice changes the compute paradigm entirely:

* **weight bit width** -> weight SRAM lines per core -> cores
* **activation bit width** -> successor ``input_bit_num`` -> ``max_axon_addr``
  -> LCN -> fanout per core -> cores, *and* squeezes the timestep budget
  because the work frame splits 17 bits between Time Step and Axon address
  (``T_max = 2 ** (8 - lcn)``)
* **per-layer ANN/SNN mode** -> ``snn_mode`` is a per-core field, and an SNN
  layer emits 1-bit spikes instead of up to 8-bit values

None of these is visible to the front end today: ``propagate_data_format``
picks the narrowest width that is *correct*, with no notion of silicon cost.

Mixed ANN/SNN compiles: patterns AAAA, SSSS, AASS, SSAA, ASAS and SASA all pass
the front end and the backend. Note that compiling is not the same as being
numerically faithful -- PAIBox wires a spike train straight into a following ANN
layer's axons without rate decoding, so threshold balancing and timestep choice
remain the quantization tool's job.

Each configuration runs in its own subprocess: IR node names come from a
process-global counter (so ``input_formats`` lookups break on reuse) and conv
unrolling materialises large dense matrices.

Usage
-----
    uv run python experiments/sweep.py --plan weight,act,mode
    uv run python experiments/sweep.py --model cnn --channels 32,64 --fc 256
    uv run python experiments/sweep.py --plan all --json sweep.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from paicorelib import DataSign, DataWidth, HwConfigV2, aer_packet_area
from paibox.backendv2 import Mapper
from paibox.paiir import ANNNodeV25, LIFNodeV25, LutCustom, compile_to_paiir
from paibox.paiir.ir.ir_base import InputNode
from paibox.paiir.ir.op_node import OfflineCoreOp

from quantize import (
    QuantPlan,
    chip_accuracy,
    load_or_train,
    mnist_loaders,
    pipeline_depth,
    quantize,
)

LUT_TABLE_SIZE = 256
OFFLINE_CORE_BUDGET = HwConfigV2.N_CORE_OFFLINE
SRAM_LINES_PER_CORE = 4096
TIMESTEP_ADDR_BITS = 8  # Time Step field is 8 bits at LCN 1x, shrinking with LCN.

# A core is treated as weight-SRAM bound once it sits this close to the budget.
SRAM_BOUND_RATIO = 0.9

# Reachable AER multicast areas are nearly dense in [1, 63]; only these are not.
AER_UNREACHABLE = frozenset({11, 17, 31, 41, 53, 55, 59})


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------


def quantized_relu_lut(bits: int, max_val: float = 127.0) -> LutCustom:
    """Build a ReLU LUT whose integer outputs span ``[0, 2**bits - 1]``.

    ``propagate_data_format`` narrows a core's output width to the smallest one
    that represents the LUT's value range, so the value range is the knob that
    controls output bit width from the front end.

    Thresholds follow ``LutReLU``: one bin covers the whole negative side and
    the rest subdivide ``(0, max_val]``, so resolution is not spent on values
    ReLU maps to zero.
    """
    negative_and_zero = torch.tensor([-(2**20), 0.0], dtype=torch.float64)
    positive = torch.linspace(0.0, max_val, LUT_TABLE_SIZE, dtype=torch.float64)[1:]
    thresholds = torch.cat([negative_and_zero, positive[: LUT_TABLE_SIZE - 2]])

    edges = torch.cat([thresholds, torch.tensor([max_val], dtype=torch.float64)])
    low, high = edges[:-1], edges[1:]
    midpoints = torch.where(low < high, (low + high) / 2, low)

    hi = 2**bits - 1
    values = (midpoints.clamp(min=0) * hi / max_val).to(torch.int32).clamp(0, hi)
    return LutCustom(thresholds.round().to(torch.int32), values, output_signed=False)


def spiking_neuron(threshold: float = 8.0) -> nn.Module:
    """LIF neuron; ``tau`` must be > 1 and a power of two."""
    return LIFNodeV25(tau=2.0, v_threshold=threshold, v_reset=0.0)


def weight_bounds(bits: int) -> tuple[int, int]:
    """Two's-complement range for *bits*-wide weights.

    ``get_weight_width`` infers the narrowest hardware weight width from the
    observed value range, so filling weights over this range is how the front
    end selects weight precision.
    """
    if bits <= 1:
        return 0, 2
    return -(2 ** (bits - 1)), 2 ** (bits - 1)


# --------------------------------------------------------------------------
# model under test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    kind: str = "cnn"  # "cnn" or "mlp"
    in_shape: tuple[int, ...] = (1, 28, 28)
    channels: tuple[int, ...] = (16, 32)
    fc: int = 128
    classes: int = 10
    mlp_sizes: tuple[int, ...] = (256, 128, 64)

    @property
    def n_activations(self) -> int:
        if self.kind == "mlp":
            return len(self.mlp_sizes) + 1
        return len(self.channels) + 2

    def sample_input(self) -> Tensor:
        if self.kind == "mlp":
            return torch.zeros(1, self.in_shape[0])
        return torch.zeros(1, *self.in_shape)


@dataclass(frozen=True)
class Config:
    """One point in the {mode} x {weight bits} x {activation bits} space."""

    modes: str  # per activation: 'A' = ANN/LUT, 'S' = SNN/LIF
    weight_bits: int
    act_bits: tuple[int, ...]  # ignored where modes[i] == 'S' (spikes are 1-bit)
    timesteps: int
    label: str = ""

    def activation(self, i: int) -> nn.Module:
        if self.modes[i] == "S":
            return spiking_neuron()
        return ANNNodeV25(quantized_relu_lut(self.act_bits[i]))


def build_model(spec: ModelSpec, config: Config) -> nn.Module:
    if len(config.modes) != spec.n_activations:
        raise ValueError(f"expected {spec.n_activations} modes, got {len(config.modes)}")

    layers: list[nn.Module] = []
    if spec.kind == "mlp":
        sizes = (spec.in_shape[0], *spec.mlp_sizes, spec.classes)
        for i in range(spec.n_activations):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                config.activation(i),
            ]
    else:
        c, h, w = spec.in_shape
        for i, out_c in enumerate(spec.channels):
            layers += [
                nn.Conv2d(c, out_c, 3, padding=1, bias=False),
                config.activation(i),
                nn.MaxPool2d(2),
            ]
            c, h, w = out_c, h // 2, w // 2
        layers += [
            nn.Flatten(),
            nn.Linear(c * h * w, spec.fc, bias=False),
            config.activation(spec.n_activations - 2),
            nn.Linear(spec.fc, spec.classes, bias=False),
            config.activation(spec.n_activations - 1),
        ]

    model = nn.Sequential(*layers).eval()
    low, high = weight_bounds(config.weight_bits)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randint(low, high, param.shape, generator=generator).float()
            )
    return model


# --------------------------------------------------------------------------
# one measurement, inside a worker process
# --------------------------------------------------------------------------


def can_evaluate(spec: ModelSpec, config: Config) -> bool:
    """Accuracy needs trained weights and a value-domain readout.

    ``quantize`` only emits ANN layers: a spiking layer would need rate decoding
    and threshold balancing, which is the quantization tool's job rather than
    this harness's. So accuracy is limited to all-ANN patterns on the MNIST CNN.
    """
    return spec.kind == "cnn" and spec.in_shape == (1, 28, 28) and "S" not in config.modes


def measure_once(
    spec: ModelSpec, config: Config, accuracy: bool = False, eval_batches: int = 2
) -> dict:
    started = time.perf_counter()
    evaluating = accuracy and can_evaluate(spec, config)
    float_acc = None

    # The backend writes progress bars and per-core diagnostics to stdout, and
    # this process reserves stdout for the JSON result.
    with contextlib.redirect_stdout(io.StringIO()):
        if evaluating:
            trained, calibs, float_acc = load_or_train(spec.channels, spec.fc)
            net = quantize(trained, calibs, QuantPlan(config.weight_bits, config.act_bits))
        else:
            net = build_model(spec, config)

        # Node names come from a process-global counter, so in a fresh worker
        # the single input is always InputNode_0.
        graph = compile_to_paiir(
            net,
            spec.sample_input(),
            input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
            timesteps=config.timesteps,
            strict=True,
        )
        names = [n for n, v in graph.nodes.items() if isinstance(v, InputNode)]
        if names != ["InputNode_0"]:
            raise RuntimeError(f"unexpected input nodes: {names}")

        def width_of(enum_name: str) -> int:
            return int(enum_name.removeprefix("WIDTH_").removesuffix("BIT"))

        cores = []
        for name in graph.topo_sort():
            node = graph.nodes[name]
            if not isinstance(node, OfflineCoreOp):
                continue
            cp = node.core_params
            cores.append(
                {
                    "name": name,
                    "mode": cp.snn_mode.name,
                    "in_bits": width_of(cp.input_width.name),
                    "out_bits": width_of(cp.output_width.name),
                    "wt_bits": width_of(cp.weight_width.name),
                    "has_lut": node.hw_lut_data is not None,
                }
            )

        mapper = Mapper()
        with tempfile.TemporaryDirectory() as out_dir:
            mapper.compile(graph, out_dir, target_platform="x86", debug=False)

        groups = []
        for rg in mapper.routing_groups:
            copy_config = rg._multicast_config
            area = (
                int(aer_packet_area(copy_config.to_tuple()))
                if copy_config is not None
                else rg.n_core_required
            )
            lcn = int(rg.lcn.value)
            groups.append(
                {
                    "name": rg.name,
                    "n_core": rg.n_core_required,
                    "lcn": lcn,
                    "aer_area": area,
                    "t_max": 2 ** (TIMESTEP_ADDR_BITS - lcn),
                    "n_elems": len(rg.raw_elems),
                    "n_inputs": len(rg.input_list),
                }
            )
        sram = [int(cp.n_sram_required) for cp in mapper.coreplacements]
        depth = pipeline_depth(graph)

        # The 8-bit timestep field is shared with the slot index, so a wide
        # output LCN leaves too few bits to tag each step. That only downgrades
        # readout from STREAM to STEP (one host round-trip per timestep); it is
        # not a limit on computation. See compile_artifacts.py:_thread_decode_mode.
        out_lcn = max((int(g.lcn.value) for g in mapper.output_groups), default=0)
        stream_cap = 1 << (TIMESTEP_ADDR_BITS - out_lcn)

        # A value needs one timestep per cascaded core to reach the output, and
        # ``tick_initial`` resets the cores every ``timesteps`` ticks, so a short
        # window returns zeros rather than a wrong answer.
        chip_acc = None
        acc_note = ""
        if evaluating:
            if config.timesteps < depth:
                acc_note = f"T={config.timesteps} < depth {depth}"
            else:
                _, test_loader = mnist_loaders()
                chip_acc = chip_accuracy(graph, test_loader, config.timesteps, eval_batches)

    return {
        "ok": True,
        "seconds": time.perf_counter() - started,
        "cores": cores,
        "groups": groups,
        "sram": sram,
        "depth": depth,
        "out_lcn": out_lcn,
        "stream_cap": stream_cap,
        "chip_acc": chip_acc,
        "float_acc": float_acc,
        "acc_note": acc_note,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


@dataclass
class Result:
    config: Config
    ok: bool
    seconds: float = 0.0
    cores: list[dict] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)
    sram: list[int] = field(default_factory=list)
    depth: int = 0
    out_lcn: int = 0
    stream_cap: int = 0
    chip_acc: float | None = None
    float_acc: float | None = None
    acc_note: str = ""
    error: str = ""

    @property
    def cores_needed(self) -> int:
        return sum(g["n_core"] for g in self.groups)

    @property
    def cores_occupied(self) -> int:
        return sum(g["aer_area"] for g in self.groups)

    @property
    def waste(self) -> int:
        return self.cores_occupied - self.cores_needed

    @property
    def max_lcn(self) -> int:
        return max((g["lcn"] for g in self.groups), default=0)

    @property
    def readout(self) -> str:
        """STREAM tags every timestep in the frame; STEP needs a host round-trip.

        Decided by the *output* group's LCN alone, not by the widest compute LCN.
        """
        cap = self.stream_cap or 2**TIMESTEP_ADDR_BITS
        return "STREAM" if cap >= self.config.timesteps else "STEP"

    @property
    def sram_bound_cores(self) -> int:
        return sum(1 for s in self.sram if s >= SRAM_BOUND_RATIO * SRAM_LINES_PER_CORE)


def run_in_subprocess(
    spec: ModelSpec, config: Config, timeout: float, accuracy: bool, eval_batches: int
) -> Result:
    cmd = [
        sys.executable, __file__, "--worker",
        "--modes", config.modes,
        "--weight-bits", str(config.weight_bits),
        "--act-bits", ",".join(map(str, config.act_bits)),
        "--timesteps", str(config.timesteps),
        "--model", spec.kind,
        "--input", ",".join(map(str, spec.in_shape)),
        "--channels", ",".join(map(str, spec.channels)),
        "--fc", str(spec.fc),
        "--classes", str(spec.classes),
        "--mlp-sizes", ",".join(map(str, spec.mlp_sizes)),
        "--eval-batches", str(eval_batches),
    ]
    if accuracy:
        cmd.append("--accuracy")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(config, ok=False, error=f"timeout {timeout:.0f}s")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return Result(config, ok=False, error=tail[-1] if tail else "nonzero exit")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return Result(config, ok=False, error="worker produced no JSON")
    if not payload.get("ok"):
        return Result(config, ok=False, error=payload.get("error", "?"))

    return Result(
        config, ok=True,
        seconds=payload["seconds"],
        cores=payload["cores"],
        groups=payload["groups"],
        sram=payload["sram"],
        depth=payload.get("depth", 0),
        out_lcn=payload.get("out_lcn", 0),
        stream_cap=payload.get("stream_cap", 0),
        chip_acc=payload.get("chip_acc"),
        float_acc=payload.get("float_acc"),
        acc_note=payload.get("acc_note", ""),
    )


# --------------------------------------------------------------------------
# experiment plans
# --------------------------------------------------------------------------


def plan_configs(spec: ModelSpec, plans: list[str], timesteps: int) -> list[Config]:
    n = spec.n_activations
    all_ann = "A" * n
    out: list[Config] = []

    def add(modes: str, wb: int, ab: int | tuple[int, ...], label: str) -> None:
        bits = (ab,) * n if isinstance(ab, int) else ab
        out.append(Config(modes, wb, bits, timesteps, label))

    if "weight" in plans:
        for wb in (8, 4, 2, 1):
            add(all_ann, wb, 8, f"w{wb}bit")

    if "act" in plans:
        for ab in (8, 4, 2, 1):
            add(all_ann, 8, ab, f"act{ab}bit")

    if "mode" in plans:
        # Uniform, both boundary directions, and alternating.
        patterns = [all_ann, "S" * n]
        half = n // 2
        patterns += ["A" * half + "S" * (n - half), "S" * half + "A" * (n - half)]
        patterns += ["".join("AS"[i % 2] for i in range(n))]
        patterns += ["".join("SA"[i % 2] for i in range(n))]
        for p in dict.fromkeys(patterns):
            add(p, 8, 8, f"mode-{p}")

    if "layer-act" in plans:
        # One layer at a time: a skewed response is what justifies a search.
        for i in range(n):
            for ab in (4, 1):
                bits = [8] * n
                bits[i] = ab
                add(all_ann, 8, tuple(bits), f"actL{i}-{ab}bit")

    if "layer-mode" in plans:
        for i in range(n):
            modes = list(all_ann)
            modes[i] = "S"
            add("".join(modes), 8, 8, f"modeL{i}-S")

    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(spec: ModelSpec, results: list[Result]) -> None:
    print()
    if spec.kind == "mlp":
        print(f"model: mlp in={spec.in_shape[0]} hidden={spec.mlp_sizes} out={spec.classes}")
    else:
        print(f"model: cnn in={spec.in_shape} channels={spec.channels} fc={spec.fc}")
    print(
        f"budget: {OFFLINE_CORE_BUDGET} offline cores/chip, "
        f"{SRAM_LINES_PER_CORE} SRAM lines/core"
    )
    print()
    baseline = next((r.float_acc for r in results if r.float_acc is not None), None)
    if baseline is not None:
        print(f"float reference accuracy: {baseline:.4f}")
        print()

    header = (
        f"{'config':<16}{'modes':<8}{'wbit':>5}{'act bits':<14}"
        f"{'need':>6}{'occ':>5}{'budget':>8}{'maxLCN':>7}{'outLCN':>7}"
        f"{'readout':>8}{'depth':>6}{'acc':>9}{'sramBnd':>9}{'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        c = r.config
        if not r.ok:
            print(f"{c.label:<16}{c.modes:<8}{c.weight_bits:>5}{'':<14}{'FAIL':>6}  {r.error[:40]}")
            continue
        pct = 100.0 * r.cores_occupied / OFFLINE_CORE_BUDGET
        act = ",".join(str(b) if m == "A" else "-" for b, m in zip(c.act_bits, c.modes))
        acc = f"{r.chip_acc:.4f}" if r.chip_acc is not None else (r.acc_note or "-")
        print(
            f"{c.label:<16}{c.modes:<8}{c.weight_bits:>5}{act:<14}"
            f"{r.cores_needed:>6}{r.cores_occupied:>5}{pct:>7.0f}%{r.max_lcn:>7}"
            f"{r.out_lcn:>7}{r.readout:>8}{r.depth:>6}{acc:>9}"
            f"{r.sram_bound_cores:>5}/{len(r.sram):<3}{r.seconds:>6.0f}"
        )

    ok = [r for r in results if r.ok]
    if not ok:
        return

    scored = [r for r in ok if r.chip_acc is not None]
    if scored:
        print()
        print("accuracy / core Pareto front (no config beats these on both):")
        def dominated(r: Result) -> bool:
            """Strict domination: strictly better on one axis, no worse on the other."""
            return any(
                (o.cores_occupied < r.cores_occupied and o.chip_acc >= r.chip_acc)
                or (o.cores_occupied <= r.cores_occupied and o.chip_acc > r.chip_acc)
                for o in scored
            )

        seen: set[tuple[int, float]] = set()
        for r in sorted(scored, key=lambda r: (r.cores_occupied, -r.chip_acc)):
            key = (r.cores_occupied, r.chip_acc)
            if dominated(r) or key in seen:
                continue
            seen.add(key)
            print(f"  {r.config.label:<16}cores={r.cores_occupied:<4}acc={r.chip_acc:.4f}")

    print()
    print("per-core mode / in-out-weight bits:")
    for r in ok:
        detail = "  ".join(
            f"{c['mode'][0]}:{c['in_bits']}>{c['out_bits']}w{c['wt_bits']}" for c in r.cores
        )
        print(f"  {r.config.label:<16}{detail}")

    print()
    print("per-routing-group (need/aer/lcn):")
    for r in ok:
        detail = "  ".join(
            f"{g['name']}:{g['n_core']}/{g['aer_area']}/L{g['lcn']}" for g in r.groups
        )
        print(f"  {r.config.label:<16}{detail}")

    def spread(prefix: str, what: str) -> None:
        subset = [r for r in ok if r.config.label.startswith(prefix)]
        if len(subset) < 2:
            return
        best = min(subset, key=lambda r: r.cores_occupied)
        worst = max(subset, key=lambda r: r.cores_occupied)
        ratio = worst.cores_occupied / max(best.cores_occupied, 1)
        print(
            f"  {what:<16}{worst.config.label}={worst.cores_occupied} -> "
            f"{best.config.label}={best.cores_occupied}  ({ratio:.2f}x)"
        )

    print()
    print("core-count spread by knob:")
    spread("w", "weight width")
    spread("act", "act width")
    spread("mode-", "mode pattern")
    spread("actL", "per-layer act")
    spread("modeL", "per-layer mode")

    occupied = sum(r.cores_occupied for r in ok)
    wasted = sum(r.waste for r in ok)
    print()
    print(
        f"AER rounding waste over all runs: {wasted}/{occupied} "
        f"({100.0 * wasted / max(occupied, 1):.0f}%)  "
        f"[unreachable areas: {sorted(AER_UNREACHABLE)}]"
    )


ALL_PLANS = ["weight", "act", "mode", "layer-act", "layer-mode"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--modes", default="", help=argparse.SUPPRESS)
    parser.add_argument("--weight-bits", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--act-bits", default="", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--input", default="1,28,28", help="C,H,W for cnn or N for mlp")
    parser.add_argument("--channels", default="16,32", help="conv output channels")
    parser.add_argument("--fc", type=int, default=128, help="hidden FC width")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--mlp-sizes", default="256,128,64", help="mlp hidden sizes")
    parser.add_argument(
        "--timesteps", type=int, default=8,
        help="must be >= pipeline depth for a valid simulated output",
    )
    parser.add_argument(
        "--accuracy", action="store_true",
        help="use the trained MNIST model and report chip-accurate accuracy",
    )
    parser.add_argument("--eval-batches", type=int, default=2, help="batches of 256 for eval")
    parser.add_argument(
        "--plan", default="weight,act,mode",
        help=f"comma list from {ALL_PLANS}, or 'all'",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", help="write raw results here")
    args = parser.parse_args()

    spec = ModelSpec(
        kind=args.model,
        in_shape=tuple(int(x) for x in args.input.split(",")),
        channels=tuple(int(x) for x in args.channels.split(",")),
        fc=args.fc,
        classes=args.classes,
        mlp_sizes=tuple(int(x) for x in args.mlp_sizes.split(",")),
    )

    if args.worker:
        config = Config(
            args.modes,
            args.weight_bits,
            tuple(int(x) for x in args.act_bits.split(",")),
            args.timesteps,
        )
        try:
            result = measure_once(spec, config, args.accuracy, args.eval_batches)
        except Exception as exc:  # noqa: BLE001 - a failed config is a data point
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(result))
        return

    plans = ALL_PLANS if args.plan == "all" else args.plan.split(",")
    unknown = set(plans) - set(ALL_PLANS)
    if unknown:
        parser.error(f"unknown plan(s) {sorted(unknown)}; choose from {ALL_PLANS}")

    configs = plan_configs(spec, plans, args.timesteps)

    if args.accuracy:
        # Train once in the driver so the workers only load the checkpoint.
        _, _, float_acc = load_or_train(spec.channels, spec.fc)
        print(f"float reference accuracy: {float_acc:.4f}", flush=True)

    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config.label} ...", flush=True)
        results.append(
            run_in_subprocess(spec, config, args.timeout, args.accuracy, args.eval_batches)
        )

    print_report(spec, results)

    if args.json:
        payload = [
            {
                "label": r.config.label, "modes": r.config.modes,
                "weight_bits": r.config.weight_bits, "act_bits": list(r.config.act_bits),
                "timesteps": r.config.timesteps,
                "ok": r.ok, "error": r.error, "seconds": r.seconds,
                "cores": r.cores, "groups": r.groups, "sram": r.sram,
                "cores_needed": r.cores_needed, "cores_occupied": r.cores_occupied,
                "readout": r.readout, "out_lcn": r.out_lcn, "depth": r.depth,
                "chip_acc": r.chip_acc, "float_acc": r.float_acc,
            }
            for r in results
        ]
        with open(args.json, "w") as fp:
            json.dump({"spec": spec.__dict__, "results": payload}, fp, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
