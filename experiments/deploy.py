"""Deployment chain for the externally trained CIFAR-10 model.

    1. prepare    fold BatchNorm and the input normalization, drop no-ops
    2. calibrate  attach quantizers, measure on real data, emit a QuantSpec
    3. export     materialize integer weights and fit the activation LUTs
    4. compile    lower to PAIIR, place onto cores, report the count
    5. evaluate   optional, run the mapped graph and score it

Preparation has to come first. A weight scale is measured from the weights, and
folding BatchNorm changes them — on this network ``max|W|`` moves by up to 4.4x,
relocating 62-94% of the per-tensor weight codes. Per-channel scales absorb the
change exactly, which is why GPU flows can calibrate first and fuse later, but
PAICORE's per-channel ratios have to be powers of two and the BatchNorm factors
are not. See ``cost_model.md`` §7.5.

Run one configuration per process. IR node names come from a process-global
counter, so a second compile produces ``InputNode_1`` and the ``input_formats``
entry silently misses — which does not fail the compile, it just changes the
input width and with it the LCN and the core count. That mismatch is raised
here rather than warned about.

Usage
-----
    uv run python experiments/deploy.py
    uv run python experiments/deploy.py --weight-bits 2 --act-bits 2 --eval-batches 2
    uv run python experiments/deploy.py --granularity pc-pow2 --verbose

    for w in 8 4 2; do uv run python experiments/deploy.py --weight-bits $w; done
"""

import argparse
import contextlib
import copy
import io
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch
from paicorelib import DataSign, DataWidth, HwConfigV2
from torch import nn

from paibox.backendv2 import Mapper
from paibox.paiir import compile_to_paiir
from paibox.quant import export

QUANT_TOOL = Path.home() / "quant_tool"
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))

import tool.mytorch.opt  # noqa: E402
import tool.mytorch.quantization.mode  # noqa: E402
import tool.paicore  # noqa: E402
from models import DATASETS  # noqa: E402
from models.simplecnn import SimpleCNN  # noqa: E402
from quantize import (  # noqa: E402
    CKPT_DIR,
    INPUT_SCALE,
    chip_accuracy,
    float_accuracy,
    loaders,
)

DATASET = "cifar10"
DEFAULT_CKPT = CKPT_DIR / "cifar10_cnn.pt"
BUDGET = HwConfigV2.N_CORE_OFFLINE


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

GRANULARITIES = {
    "per-tensor": (False, tool.paicore.MseCalibrator, tool.paicore.PerChannelPolicy.PER_TENSOR),
    "pc-round": (True, tool.paicore.MseCalibrator, tool.paicore.PerChannelPolicy.APPROXIMATE),
    "pc-pow2": (True, tool.paicore.Pow2MseCalibrator, tool.paicore.PerChannelPolicy.STRICT),
}


def load_checkpoint(path: Path) -> tuple[nn.Module, float]:
    """``strict=True`` on purpose: a silent key mismatch would deploy a
    different network than the one whose accuracy is being quoted."""
    blob = torch.load(path, weights_only=False, map_location="cpu")
    model = SimpleCNN(num_classes=10, width=blob["args"]["width"])
    model.load_state_dict(blob["model"], strict=True)
    return model.eval(), blob["acc"]


def calibrated_spec(model, weight_bits, act_bits, granularity, calib_batches, timesteps):
   
    per_channel, calibrator, policy = GRANULARITIES[granularity]
    train_loader, _ = loaders(DATASET)

    def forward_loop(net):
        for index, (images, _) in enumerate(train_loader):
            if index >= calib_batches:
                break
            net(images)

    config = tool.paicore.paicore_cfg(
        weight_bits=weight_bits, act_bits=act_bits, per_channel=per_channel
    )
    notes: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):  # vendored package prints
        quantized = tool.mytorch.opt.apply_mode(
            copy.deepcopy(model),
            mode=[("quantize", config)],
            registry=tool.mytorch.quantization.mode.QuantizeModeRegistry,
        )
        tool.paicore.calibrate_with(quantized, calibrator, forward_loop)
        # Topology comes from the float model: quantization replaced modules in
        # place so the paths agree, but tracing the quantized one would descend
        # into a quantizer's internals.
        spec = tool.paicore.build_spec(
            quantized,
            input_scale=INPUT_SCALE,
            timesteps=timesteps,
            per_channel_policy=policy,
            report=notes,
            topology=model,
        )
    return spec, notes


def lower(model, spec, timesteps):
    with contextlib.redirect_stdout(io.StringIO()):
        deployed = export(model, spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(
                deployed,
                torch.zeros(1, *DATASETS[DATASET].shape),
                input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
                timesteps=timesteps,
                strict=True,
            )

    if any("input_formats" in str(w.message) for w in caught):
        raise RuntimeError(
            "input_formats did not reach an input node: IR names come from a "
            "process-global counter, so compile once per process"
        )
    return graph


def place(graph):
    """Map onto cores, returning the routing groups and the runtime metadata.

    Over budget is a measurement, not a crash: the budget check runs after
    routing groups are built, so their counts are readable either way, and that
    is the number needed to decide what to change.

    ``tick_depth`` is read back from the exported ``proto/config.pb`` rather
    than recomputed. That file is what the runtime itself consumes, and the
    backend derives the depth from each node's own ``__tick_depth__``
    ([PAIBox] ``compile_plan.py``), which a front-end guess of one tick per
    offline core does not reproduce for node types that declare anything else.
    Nothing is exported when placement fails, so it comes back ``None``.
    """
    mapper = Mapper()
    runtime = None
    with contextlib.redirect_stdout(io.StringIO()):
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                mapper.compile(graph, out_dir, target_platform="x86", debug=False)
            except ValueError as exc:
                if "exceeds the" not in str(exc):
                    raise
            else:
                runtime = _runtime_params(Path(out_dir) / "proto" / "config.pb")

    groups = [(rg.name, rg.n_core_required, int(rg.lcn.value)) for rg in mapper.routing_groups]
    return groups, runtime


def _runtime_params(pb_path: Path):
    """The first thread's ``RuntimeParams`` from the exported artifact."""
    from paibox.backendv2.generated.proto import compile_artifacts_pb2 as pb

    artifacts = pb.CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())
    threads = artifacts.io_mapping.threads
    return threads[0].runtime if threads else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--granularity", default="per-tensor", choices=sorted(GRANULARITIES))
    parser.add_argument("--calib-batches", type=int, default=4, help="batches of 256")
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument(
        "--eval-batches", type=int, default=-1,
        help="batches of 256 for chip accuracy; 0 for the whole test set, -1 to skip",
    )
    parser.add_argument("--float-eval", action="store_true", help="10k-image float baseline")
    parser.add_argument("--verbose", action="store_true", help="dump the spec")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model, reported = load_checkpoint(args.ckpt)
    prepared = tool.paicore.prepare(model, mean=MEAN, std=STD)
    print(f"1. prepare: 检查点报告 {reported:.4f}, {sum(1 for m in prepared.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))} 个计算层")

    if args.float_eval:
        # ToTensor only: the normalization is in the first convolution now, so
        # this is the chip's real input and the baseline it is judged on.
        _, test_loader = loaders(DATASET)
        print(f"   芯片真实输入下的 float 精度 = {float_accuracy(prepared, test_loader):.4f}")

    spec, notes = calibrated_spec(
        prepared, args.weight_bits, args.act_bits, args.granularity,
        args.calib_batches, args.timesteps,
    )
    print(f"2. calibrate: w{args.weight_bits}a{args.act_bits} {args.granularity}")
    # if args.verbose:
    print(tool.paicore.describe(spec))
    # print(spec)
    for note in notes:
        print(f"   {note}")

    graph = lower(prepared, spec, args.timesteps)
    # print(f"3. export + lower: {len(graph.nodes)} 个 PAIIR 节点")

    groups, runtime = place(graph)
    cores = sum(count for _, count, _ in groups)
    max_lcn = max((lcn for _, _, lcn in groups), default=0)
    fits = cores <= BUDGET
    print(
        f"4. map: {cores} 核 / {BUDGET} ({100.0 * cores / BUDGET:.0f}% 预算)  "
        f"maxLCN={max_lcn}  {'装得下' if fits else '超出预算'}"
    )
    if not fits:
        print(f"   {len(groups)} 个路由组, 占核最多的 5 个:")
        for name, count, lcn in sorted(groups, key=lambda row: -row[1])[:5]:
            print(f"     {name:<10} {count:>4} 核  LCN={lcn}")
    if runtime is not None:
        print(
            f"   运行时: tick_depth={runtime.tick_depth} "
            f"sync_steps={runtime.sync_steps} "
            f"读出模式={'STREAM' if runtime.decode_mode == 0 else 'STEP'}"
        )

    if fits and args.eval_batches >= 0:
        # A value needs one timestep per cascaded core to reach the output, so
        # simulating for fewer than tick_depth returns zeros.
        _, test_loader = loaders(DATASET)
        steps = max(args.timesteps, runtime.tick_depth)
        accuracy = chip_accuracy(graph, test_loader, steps, args.eval_batches)
        print(f"5. evaluate: 芯片仿真精度 = {accuracy:.4f} (T={steps})")

    print(f"用时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
