"""Does quant_tool quantize a natively trained SNN correctly? Measured, not assumed.

The network is the DVS-Gesture model in ``DVSGesture_sw_and_rt``: five
``Conv2d -> BatchNorm2d -> IFNode -> AvgPool2d`` blocks, two spiking fully
connected layers, a voting layer, twenty timesteps. It comes with a reference to
compare against — ``infer.py`` reports **85.42%** for the float model and
**81.94%** for its own int8, which multiplies every weight by a fixed 127 and
clips, visibly saturating the checkpoint at ±127. Per-layer calibration should
recover part of that 3.5-point drop.

What "quantized accuracy" means here
------------------------------------
Integer weights and an integer threshold, run through SpikingJelly's own
neurons. That isolates the question being asked — whether the *decisions*
quant_tool makes are right — from the separate question of whether PAIBox's
export and lowering reproduce them. It is also what the reference does, so the
numbers are comparable.

What a spike is worth
---------------------
One. Two readings compete and they differ by a factor of ``T``: under rate
coding a count of ``c`` over ``T`` steps stands for ``c * theta / T`` of real
value, which is what a value-domain consumer needs to turn counts back into
numbers — but what the scale chain feeds is the *next layer's accumulator*, and
a synapse computes ``weight * spike`` with the spike as the number one. The
trained network's threshold was calibrated against potentials accumulated that
way.

Getting this wrong is not subtle in its effect and is very subtle in its cause.
Recording ``theta / T`` on a spiking edge leaves every downstream accumulator
unit off by that factor, so each threshold and each integer bias is off by it
too, and the model scores at chance while every individual scale looks
plausible. Measured here before it was believed.

Usage
-----
    uv run python experiments/snn_quantize.py
    uv run python experiments/snn_quantize.py --weight-bits 4 --eval-batches 4
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
from torch import nn

SOURCE = Path(__file__).resolve().parent.parent / "DVSGesture_sw_and_rt"
QUANT_TOOL = Path.home() / "quant_tool"
for extra in (SOURCE, QUANT_TOOL):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import tool.mytorch.opt  # noqa: E402
import tool.mytorch.quantization.mode  # noqa: E402
import tool.paicore  # noqa: E402
from paicorelib import DataSign, DataWidth, HwConfigV2  # noqa: E402
from spikingjelly.activation_based import functional, neuron, surrogate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from paibox.backendv2 import Mapper  # noqa: E402
from paibox.paiir import compile_to_paiir  # noqa: E402
from paibox.quant import (  # noqa: E402
    LayerQuant,
    QuantSpec,
    SnnActivation,
    WeightQuant,
    export,
)

CKPT = SOURCE / "checkpoints" / "dropout_0.1_sum_qat.pt"
DATA_ROOT = SOURCE / "data" / "DVS128Gesture"

#: DVS frames are event counts, so one code is one event.
INPUT_SCALE = 1.0

#: `VotingLayer(10)` at the tail: 11 classes, each read off a committee of 10.
N_CLASSES = 11
VOTES_PER_CLASS = 10

#: `infer.py`'s own numbers on this checkpoint, for comparison.
REFERENCE_FLOAT = 0.8542
REFERENCE_INT8 = 0.8194


def build():
    """The float model the QAT checkpoint belongs to.

    ``infer.py`` carries its own copy of the architecture — ``Dropout(0.1)``
    rather than ``model.py``'s 0.5 — and that is the one the checkpoint matches.
    """
    from infer import DVSGestureNet

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = DVSGestureNet(
        channels=blob["channels"],
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
    )
    model.load_state_dict(blob["state_dict"], strict=False)
    return model.eval(), blob


class IntegratedFrames(torch.utils.data.Dataset):
    """Pre-integrated DVS frames, read straight from the ``.npz`` files.

    SpikingJelly's dataset class insists the raw recordings are present even
    when the integrated frames already are, and they are not in this tree.
    The layout is simple enough to read directly: ``<split>/<label>/*.npz``,
    each holding ``frames`` of shape ``[T, 2, H, W]`` whose values are event
    counts — small integers, not spikes, so one code is one event.
    """

    def __init__(self, root: Path, split: str) -> None:
        # Label indices follow torchvision's DatasetFolder convention, which is
        # what the model was trained against: class directories sorted as
        # *strings*, so "10" lands at index 2, between "1" and "2". Reading the
        # directory name as a number instead permutes nine of the eleven
        # classes and takes accuracy to roughly 2/11 — which looks like a broken
        # quantization rather than a broken label map.
        folders = sorted(
            (child for child in (root / split).iterdir() if child.is_dir()),
            key=lambda child: child.name,
        )
        self.classes = [child.name for child in folders]
        self.samples: list[tuple[Path, int]] = [
            (path, label)
            for label, child in enumerate(folders)
            for path in sorted(child.glob("*.npz"))
        ]
        if not self.samples:
            raise FileNotFoundError(f"no .npz frames under {root / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import numpy as np

        path, label = self.samples[index]
        with np.load(path) as blob:
            frames = torch.from_numpy(blob["frames"].astype("float32"))
        return frames, label


def loaders(timesteps: int, batch: int, split_by: str = "number"):
    root = DATA_ROOT / f"frames_number_{timesteps}_split_by_{split_by}"
    return (
        DataLoader(IntegratedFrames(root, "train"), batch_size=batch, shuffle=False),
        DataLoader(IntegratedFrames(root, "test"), batch_size=batch),
    )


@torch.no_grad()
def accuracy(model: nn.Module, loader, timesteps: int, batches: int) -> float:
    """Score over spike counts, summed across timesteps.

    A spiking network's answer is a rate, so reading one timestep is not an
    answer at all. ``sum`` rather than ``mean`` matches the reference; for an
    ``argmax`` the two agree anyway.
    """
    functional.set_step_mode(model, "m")
    model.eval()
    correct = total = 0
    for index, (frames, label) in enumerate(loader):
        if batches and index >= batches:
            break
        frames = frames.float().transpose(0, 1)  # [N,T,...] -> [T,N,...]
        logits = model(frames).sum(0)
        correct += int((logits.argmax(1) == label.long()).sum())
        total += label.numel()
        functional.reset_net(model)
    return correct / max(total, 1)


def calibrated_spec(prepared: nn.Module, train_loader, args):
    """Attach quantizers to a copy, calibrate over timesteps, emit the spec."""
    spiking_sites = tool.paicore.detect_spiking_sites(prepared)
    config = tool.paicore.paicore_cfg(
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        per_channel=False,
        spiking_sites=spiking_sites,
    )

    def forward_loop(net: nn.Module) -> None:
        functional.set_step_mode(net, "m")
        for index, (frames, _) in enumerate(train_loader):
            if index >= args.calib_batches:
                break
            net(frames.float().transpose(0, 1))
            # Membrane potential persists, so without this the next sample
            # starts charged and the observed ranges drift upward.
            functional.reset_net(net)

    notes: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):
        quantized = tool.mytorch.opt.apply_mode(
            copy.deepcopy(prepared),
            mode=[("quantize", config)],
            registry=tool.mytorch.quantization.mode.QuantizeModeRegistry,
        )
        tool.paicore.calibrate_with(quantized, tool.paicore.MseCalibrator, forward_loop)
        spec = tool.paicore.build_spec(
            quantized,
            input_scale=INPUT_SCALE,
            timesteps=args.timesteps,
            per_channel_policy=tool.paicore.PerChannelPolicy.PER_TENSOR,
            topology=prepared,
            report=notes,
        )
    return spec, spiking_sites, notes


def materialize(prepared: nn.Module, spec, round_threshold: bool = True) -> nn.Module:
    """Write the spec's decisions into a runnable integer model.

    Weights become codes, the integer bias comes straight from the spec, and
    each threshold moves into the accumulator's units. Everything stays a
    SpikingJelly module, so multi-step execution still works — this is the
    software reference, not the chip.
    """
    model = copy.deepcopy(prepared)
    modules = dict(model.named_modules())

    for path, layer in spec.layers.items():
        compute = modules[path]
        scale = float(layer.weight.scale)
        limit = 2 ** (layer.weight.bits - 1)
        accumulator = scale * layer.input_scale

        with torch.no_grad():
            codes = (compute.weight.detach() / scale).round().clamp(-limit, limit - 1)
            compute.weight.copy_(codes)
            if layer.bias is not None:
                compute.bias.copy_(layer.bias.to(compute.bias.dtype))

        if layer.activation_path is None or not layer.is_spiking:
            continue
        # The chip compares against an integer, so rounding here is the real
        # semantics. Keeping the exact ratio is a diagnostic, because an SNN
        # threshold lives in accumulator units and so falls with `1/s_w`: at
        # two-bit weights it lands around 2, where rounding is a quarter of the
        # value. Measured, that is not where low-bit SNN fails — w2 scores the
        # same either way — so the loss belongs to the ternary weights.
        exact = layer.activation.v_threshold / accumulator
        modules[layer.activation_path].v_threshold = float(
            round(exact) if round_threshold else exact
        )

    return model.eval()


def reference_int8(timesteps: int):
    """The reference's own int8 checkpoint, wrapped in a spec we can deploy.

    Their quantization is a single global factor: every float weight times 127,
    clipped to int8, and the threshold moved from 1.0 to 127 to match. So the
    integers *are* the codes and one code is worth one unit — ``s_w = 1`` — which
    puts the threshold at 127 accumulator steps with no further conversion.

    Deploying this through the same path as our own spec isolates where a chip
    gap comes from. If their int8 also drops, the gap is in the chip path rather
    than in the scales.
    """
    from infer import DVSGestureNet_NoBN

    model = DVSGestureNet_NoBN(
        channels=8, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan()
    )
    integers = torch.load(
        SOURCE / "checkpoints" / "dropout_0.1_sum_qat_int8.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict({k: v.float() for k, v in integers.items()}, strict=False)
    # No BatchNorm to fold; this only drops the dropouts, whose presence would
    # otherwise leave the lowering with a module it has no node for.
    prepared = tool.paicore.prepare(model.eval())

    graph = tool.paicore.site_graph(
        prepared,
        frozenset(
            path
            for path, module in prepared.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ),
    )

    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE, timesteps=timesteps)
    for path in graph.order:
        compute = dict(prepared.named_modules())[path]
        spec.layers[path] = LayerQuant(
            input_scale=1.0,
            output_scale=1.0,
            weight=WeightQuant(bits=8, scale=1.0),
            activation=SnnActivation(v_threshold=127.0, v_reset=0.0, tau=1.0),
            bias=compute.bias.detach().round() if compute.bias is not None else None,
            activation_path=graph.activation_of[path],
        )

    for neuron_path in graph.activation_of.values():
        if neuron_path is not None:
            dict(prepared.named_modules())[neuron_path].v_threshold = 127.0

    return prepared, spec


def deploy(prepared: nn.Module, spec, timesteps: int):
    """Export to chip neurons, lower, and place. Returns the graph and groups.

    Unlike the ANN path there is nothing to hold constant across timesteps: DVS
    frames already arrive as ``[T, ...]``, which is what the graph wants.

    ``output_approx="sum_approx_if_avgpool"`` is what makes the voting layer
    survive. ``VotingLayer`` lowers to an ``AvgPool1d``, and at the output
    boundary the default policy exports such a node as ``SumPool + IFNodeV25``
    — one *majority spike* per group rather than a count. That discards exactly
    the quantity the vote is made of, and no amount of host-side summing brings
    it back. Exporting counts instead leaves the division to the host, which is
    what the board runtime does anyway (``(counts / VOTING_SIZE).sum(axis=0)``).
    """
    # Compiled once, with a window wide enough for the run rather than one
    # matched to it. `auto_reset` clears every core once per compiled window,
    # so a run longer than the window gets wiped partway through — and the run
    # must exceed the rate window, since the pipeline needs `depth - 1` extra
    # ticks to fill. Measuring the depth first would need a second compile, and
    # a second compile in one process renames the input node, at which point
    # `input_formats` silently misses and the 0–31 event counts are read at the
    # wrong width. A generous window avoids needing to know the depth in
    # advance; anything at least as long as the run behaves the same.
    window = 4 * timesteps

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(
                export(prepared, spec),
                torch.zeros(1, 2, 128, 128),
                input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
                timesteps=window,
                strict=True,
                output_approx="sum_approx_if_avgpool",
            )
        mapper = Mapper()
        over_budget = None
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                mapper.compile(graph, out_dir, target_platform="x86", debug=False)
        except ValueError as exc:
            if "exceeds the" not in str(exc):
                raise
            over_budget = str(exc)

    if any("input_formats" in str(warning.message) for warning in caught):
        raise RuntimeError(
            "input_formats did not reach an input node, so the 0-31 event counts "
            "were read at the default width; IR names come from a process-global "
            "counter, so compile once per process"
        )

    groups = [(rg.name, rg.n_core_required, int(rg.lcn.value)) for rg in mapper.routing_groups]
    return graph, groups, over_budget


def pipeline_depth(graph) -> int:
    """Ticks before a value reaches the output, one per cascaded core."""
    depth: dict[str, int] = {}
    for name in graph.topo_sort():
        node = graph.nodes[name]
        base = max((depth.get(p, 0) for p in graph.predecessors(name)), default=0)
        depth[name] = base + getattr(node, "__tick_depth__", 1)
    return max(depth.values(), default=0)


def warn_if_staggered(graph, aligned: bool) -> None:
    """Say so when the run is about to score a staggered graph.

    Without ``--align-ticks`` the number that comes out is not the accuracy of
    the quantization, it is the accuracy of a network that lost its first
    ``tick_start`` frames at every layer. That is worth reporting on purpose but
    never by accident.
    """
    if aligned:
        return
    delay = max(
        (
            node.core_params.tick_start
            for node in graph.nodes.values()
            if getattr(node, "core_params", None) is not None
            and node.core_params.tick_start is not None
        ),
        default=0,
    )
    if delay:
        print(
            f"   注意: 活动窗口错开到第 {delay} 拍, 而仿真器在同一拍内穿透整张图, "
            f"最深的核只能积分 {max(0, 20 - delay)}/20 帧 (cost_model.md §7.8)。"
            f"加 --align-ticks 才是可比的数。"
        )


def align_activity(graph) -> None:
    """Open every core at tick zero.

    The compiler staggers ``tick_start`` by pipeline depth, which is right for
    the hardware: a value physically takes one tick per hop, so a core at depth
    k first sees data at tick k. But ``PAIIRGraph.step`` resolves the whole
    graph within one tick — predecessors are read from the same step, not the
    previous one — so there is no delay line for the stagger to line up with.
    A core opening at tick k does not receive frame 0 late; it never receives
    it at all, along with frames 1 through k-1.

    With the windows opened together the simulator matches the synchronous
    software model exactly, which is what makes the dropped frames visible as
    the cause rather than a suspicion.
    """
    for node in graph.nodes.values():
        params = getattr(node, "core_params", None)
        if params is not None and params.tick_start is not None:
            params.tick_start = 0


def gate_activity(graph, timesteps: int) -> None:
    """Close each core's activity window `timesteps` ticks after it opens.

    The compiler already staggers ``tick_start`` by pipeline depth, so every
    core opens exactly when its input arrives. But it leaves ``tick_duration``
    at zero, which means "never stop" — so once the frames run out the cores
    stay live, and a biased layer keeps charging its membrane from the bias
    alone and firing on nothing. Bounding the window to the rate-coding length
    makes each core see exactly the ticks that carry data, which is what the
    synchronous software model sees.
    """
    for node in graph.nodes.values():
        params = getattr(node, "core_params", None)
        if params is not None and params.tick_start is not None:
            params.tick_duration = timesteps


@torch.no_grad()
def chip_accuracy(graph, loader, timesteps: int, batches: int) -> float:
    """Score the mapped graph, counting spikes over the window that carries them.

    ``timesteps`` is the rate-coding window the network was trained with —
    twenty frames — and the answer is the spike count over it, so the run has to
    cover twenty ticks of output rather than one.

    The run is ``timesteps + delay`` ticks long, where ``delay`` is when the
    last core's activity window opens, and only the ``timesteps`` output ticks
    from ``delay`` onward are counted. With the compiler's staggered windows
    that is thirty-three ticks; with ``align_activity`` it is twenty and the
    offset is zero. The offset is read off the windows rather than computed
    separately so the two stay consistent.

    ``PAIIRGraph.run`` simulates one sample at a time, so this walks the batch.

    The last thing the float model does is ``VotingLayer(10)``: the 110 output
    neurons are ten-neuron committees, one per class, and the class score is the
    committee mean. The lowering has a node for it, so the graph usually hands
    back 11 already-pooled channels; when it does not, the committees are summed
    here instead. Either way argmax is over classes, never over raw neurons.
    """
    # The readout offset follows the activity windows the graph actually has,
    # not a depth computed alongside them: whatever the last core's window opens
    # at is when its first output can appear.
    delay = max(
        (
            node.core_params.tick_start
            for node in graph.nodes.values()
            if getattr(node, "core_params", None) is not None
            and node.core_params.tick_start is not None
        ),
        default=0,
    )
    total_ticks = timesteps + delay

    correct = total = 0
    for index, (frames, label) in enumerate(loader):
        if batches and index >= batches:
            break
        for sample in range(frames.shape[0]):
            window = frames[sample : sample + 1].float().transpose(0, 1)  # [T,1,...]
            flush = torch.zeros(
                (total_ticks - timesteps, *window.shape[1:]), dtype=window.dtype
            )
            spikes = torch.as_tensor(graph.run(torch.cat([window, flush]), T=total_ticks))
            carried = spikes[delay : delay + timesteps]
            counts = carried.reshape(carried.shape[0], -1).sum(0)
            if counts.numel() == N_CLASSES:
                scores = counts
            elif counts.numel() == N_CLASSES * VOTES_PER_CLASS:
                scores = counts.reshape(N_CLASSES, VOTES_PER_CLASS).sum(1)
            else:
                raise RuntimeError(
                    f"output is {counts.numel()} wide, expected {N_CLASSES} classes "
                    f"or {N_CLASSES * VOTES_PER_CLASS} committee members"
                )
            correct += int(scores.argmax() == int(label[sample]))
            total += 1
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument(
        "--eval-batches", type=int, default=0, help="batches to score; 0 for all"
    )
    parser.add_argument("--deploy", action="store_true", help="export, compile, place, score")
    parser.add_argument(
        "--source", default="calibrated", choices=("calibrated", "reference-int8"),
        help="whose quantization to deploy: ours, or the reference's global-127 int8",
    )
    parser.add_argument(
        "--no-bias", action="store_true",
        help="zero every bias on both paths, to isolate the bias/pipeline interaction",
    )
    parser.add_argument(
        "--gate-ticks", action="store_true",
        help="close each core's activity window after `timesteps` ticks",
    )
    parser.add_argument(
        "--align-ticks", action="store_true",
        help="open every core at tick 0, matching the simulator's instant dataflow",
    )
    parser.add_argument(
        "--exact-threshold", action="store_true",
        help="skip rounding the integer threshold, to separate it from weight error",
    )
    parser.add_argument(
        "--chip-batches", type=int, default=2,
        help="batches for the chip simulation, which runs one sample at a time",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    if args.source == "reference-int8":
        prepared, spec = reference_int8(args.timesteps)
        _, test_loader = loaders(args.timesteps, args.batch)
        software = accuracy(copy.deepcopy(prepared), test_loader, args.timesteps, args.eval_batches)
        print(f"参考实现 int8(全局 ×127): 软件精度 = {software:.4f}  (其报告值 {REFERENCE_INT8:.4f})")

        graph, groups, over_budget = deploy(prepared, spec, args.timesteps)
        cores = sum(count for _, count, _ in groups)
        budget = HwConfigV2.N_CORE_OFFLINE
        print(
            f"  编译: {cores} 核 / {budget} ({100.0 * cores / budget:.0f}% 预算), "
            f"流水线深度 {pipeline_depth(graph)}"
        )
        if over_budget:
            print(f"  {over_budget}")
        else:
            warn_if_staggered(graph, args.align_ticks)
            if args.align_ticks:
                align_activity(graph)
            if args.gate_ticks:
                gate_activity(graph, args.timesteps)
            chip = chip_accuracy(graph, test_loader, args.timesteps, args.chip_batches)
            print(f"  芯片仿真精度 = {chip:.4f}  (相对软件 {chip - software:+.4f})")
        print(f"\n用时 {time.perf_counter() - started:.0f}s")
        return

    model, blob = build()
    print(
        f"检查点 {CKPT.name}: channels={blob['channels']} T={blob['T']} "
        f"best_test_acc={blob['best_test_acc']:.4f}"
    )
    print(f"参考实现: float {REFERENCE_FLOAT:.4f}, 其 int8 {REFERENCE_INT8:.4f}")

    prepared = tool.paicore.prepare(model)
    print(f"1. prepare: {sum(1 for m in prepared.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))} 个计算层")

    if args.no_bias:
        # The bias reaches the membrane once per tick regardless of whether an
        # input did, so under a pipeline each layer charges for as many ticks as
        # its depth before its signal arrives. Zeroing the bias removes the only
        # term that is not gated by the input; if software and chip then agree,
        # that interaction is the whole gap.
        with torch.no_grad():
            for module in prepared.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
                    module.bias.zero_()
        print("   已清零所有偏置")

    train_loader, test_loader = loaders(args.timesteps, args.batch)
    float_acc = accuracy(copy.deepcopy(prepared), test_loader, args.timesteps, args.eval_batches)
    print(f"   折叠后 float 精度 = {float_acc:.4f}")

    spec, spiking_sites, notes = calibrated_spec(prepared, train_loader, args)
    print(
        f"2. calibrate: w{args.weight_bits}a{args.act_bits}, "
        f"{len(spiking_sites)}/{len(spec.layers)} 个 site 是 spiking"
    )
    for note in notes:
        print(f"   {note}")
    print(f"   {'site':<12}{'v_th':>6}{'s_in':>11}{'s_out':>11}{'s_w':>11}")
    for path, layer in spec.layers.items():
        print(
            f"   {path:<12}{layer.activation.v_threshold:>6.2f}{layer.input_scale:>11.3e}"
            f"{layer.output_scale:>11.3e}{float(layer.weight.scale):>11.3e}"
        )

    integer = materialize(prepared, spec, round_threshold=not args.exact_threshold)
    thresholds = [
        int(dict(integer.named_modules())[layer.activation_path].v_threshold)
        for layer in spec.layers.values()
        if layer.activation_path
    ]
    int_acc = accuracy(integer, test_loader, args.timesteps, args.eval_batches)
    print()
    print("3. 整数模型")
    print(f"   整数阈值 = {thresholds}")
    print(f"   精度 = {int_acc:.4f}  (相对 float {int_acc - float_acc:+.4f})")

    if args.deploy:
        print()
        graph, groups, over_budget = deploy(prepared, spec, args.timesteps)
        cores = sum(count for _, count, _ in groups)
        budget = HwConfigV2.N_CORE_OFFLINE
        lcns = sorted({lcn for _, _, lcn in groups})
        print(
            f"4. export + 编译 + 布放: {len(graph.nodes)} 个 PAIIR 节点, "
            f"{cores} 核 / {budget} ({100.0 * cores / budget:.0f}% 预算), LCN={lcns}, "
            f"流水线深度 {pipeline_depth(graph)}"
        )
        if over_budget:
            print(f"   超出预算。占核最多的 5 个路由组:")
            for name, count, lcn in sorted(groups, key=lambda row: -row[1])[:5]:
                print(f"     {name:<12} {count:>4} 核  LCN={lcn}")
        else:
            warn_if_staggered(graph, args.align_ticks)
            if args.align_ticks:
                align_activity(graph)
            if args.gate_ticks:
                gate_activity(graph, args.timesteps)
            acc = chip_accuracy(graph, test_loader, args.timesteps, args.chip_batches)
            print(f"5. 芯片仿真精度 = {acc:.4f}  (相对整数参考 {acc - int_acc:+.4f})")

    print()
    print(f"用时 {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
