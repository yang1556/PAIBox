"""Where does the chip diverge from the software integer model?

Both the reference's int8 and our calibrated spec lose accuracy when they move
from SpikingJelly to `PAIIRGraph.run`, by a similar amount, which points at the
lowering rather than at either set of scales. State leaking across `run` calls
has already been ruled out — repeated calls are bit-identical.

This walks both executions layer by layer on one sample and reports the total
spike count each layer emits. The two should agree exactly: the integer model
and the chip compute the same accumulate-compare on the same codes. The first
layer where they part is the one to look at.

Counts are compared per layer rather than per neuron because the chip reshapes
and reorders within a layer; the total is order-free and a divergence shows up
in it just as clearly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import warnings

import torch
from spikingjelly.activation_based import functional
from torch import nn

import tool.paicore

from snn_quantize import build, calibrated_spec, deploy, loaders, materialize

TIMESTEPS = 20


def software_counts(model: nn.Module, window: torch.Tensor) -> list[tuple[str, list[float]]]:
    """Per-tick spike counts for each spiking layer of the integer model."""
    counts: list[tuple[str, list[float]]] = []
    handles = []

    def record(path: str):
        def hook(_module, _args, output):
            spikes = torch.as_tensor(output)
            # Multi-step modules hand back [T, N, ...]; reduce everything but T.
            per_tick = spikes.reshape(spikes.shape[0], -1).sum(1)
            counts.append((path, [float(v) for v in per_tick]))

        return hook

    for path, module in model.named_modules():
        if hasattr(module, "v_threshold"):
            handles.append(module.register_forward_hook(record(path)))

    functional.set_step_mode(model, "m")
    functional.reset_net(model)
    with torch.no_grad():
        model(window)
    for handle in handles:
        handle.remove()

    return counts


def chip_counts(
    graph, window: torch.Tensor, total_ticks: int
) -> list[tuple[str, list[float]]]:
    """Per-tick spike counts for each graph node, in call order."""
    totals: dict[str, list[float]] = {}
    originals = {}

    nodes = {
        name: node
        for name, node in graph.nodes.items()
        if hasattr(node, "forward")
    }

    for name, node in nodes.items():
        def wrap(name=name, node=node, forward=node.forward):
            def recording(*args, **kwargs):
                output = forward(*args, **kwargs)
                tensor = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(tensor):
                    totals.setdefault(name, []).append(float(tensor.sum()))
                return output

            return recording

        originals[name] = node.forward
        node.forward = wrap()

    flush = torch.zeros((total_ticks - window.shape[0], *window.shape[1:]), dtype=window.dtype)
    graph.run(torch.cat([window, flush]), T=total_ticks)

    for name, node in nodes.items():
        node.forward = originals[name]

    return [(name, totals[name]) for name in graph.nodes if name in totals]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    knobs = argparse.Namespace(
        weight_bits=8, act_bits=8, calib_batches=4, timesteps=TIMESTEPS, batch=8
    )
    model, _ = build()
    prepared = tool.paicore.prepare(model)
    train_loader, test_loader = loaders(TIMESTEPS, knobs.batch)
    spec, _, _ = calibrated_spec(prepared, train_loader, knobs)
    # `materialize` and `export` are two independent readings of the same spec,
    # and each expects the *float* model: both divide the weights by the scale.
    # Handing the integer model to `export` quantizes twice, which saturates
    # every weight to int8 and widens the accumulator about fourfold.
    reference = materialize(prepared, spec)

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, _, _ = deploy(prepared, spec, TIMESTEPS)

    frames, _ = next(iter(test_loader))
    window = frames[args.sample : args.sample + 1].float().transpose(0, 1)

    software = software_counts(reference, window)
    chip = chip_counts(graph, window, total_ticks=TIMESTEPS + 20)

    def show(title: str, series: list[tuple[str, list[float]]]) -> None:
        print(f"\n{title}")
        for name, per_tick in series:
            head = " ".join(f"{v:6.0f}" for v in per_tick[:12])
            print(f"  {name:24s} 总 {sum(per_tick):9.0f} | 逐拍 {head} ...")

    show("软件整数模型", software)
    show("芯片图", chip)


if __name__ == "__main__":
    main()
