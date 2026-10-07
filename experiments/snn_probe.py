"""Does switching a layer to spiking buy cores, and how many?

The hypothesis worth testing before spending weeks on conversion methods: an
8-bit ANN output occupies eight axon addresses per input, a 1-bit spike occupies
one. Axon demand is ``n_inputs * input_bits``, and LCN follows axon demand, so a
spiking producer should cut its consumer's LCN by up to 8x.

That matters here because ``simplecnn`` maps to 627 cores against a 63-core
budget with **every routing group at LCN 7** — 128x, the hardware maximum. A
saturated LCN is the signature of being neuron-limited rather than
weight-limited, which is exactly the regime a 1-bit producer would relieve. If
the effect is real, mode selection is not a side dimension of the search: it is
the lever that decides whether this workload fits at all.

What this measures and what it does not
---------------------------------------
Cores only. Thresholds are taken from each layer's calibrated output range,
which is a placeholder for threshold balancing, and ``build_spec`` sets a
spiking layer's ``output_scale`` to 1.0 because spikes are unit events — so the
consumer's accumulator scale, and therefore its fitted LUT, is wrong. **The
numerics of these graphs are meaningless.** Placement does not read LUT
thresholds, so the core count is still the number a real conversion would get.

Usage
-----
    uv run python experiments/snn_probe.py --spiking-prefix 0    # all ANN, baseline
    uv run python experiments/snn_probe.py --spiking-prefix 3

    for k in 0 1 2 3 4 5 6 7; do
      uv run python experiments/snn_probe.py --spiking-prefix $k
    done
"""

import argparse
import contextlib
import copy
import io
import sys
import time
from pathlib import Path

import torch
from paicorelib import HwConfigV2
from torch import nn

QUANT_TOOL = Path.home() / "quant_tool"
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))

import tool.mytorch.opt  # noqa: E402
import tool.mytorch.quantization.mode  # noqa: E402
import tool.paicore  # noqa: E402
from deploy import MEAN, STD, DEFAULT_CKPT, load_checkpoint, lower, place  # noqa: E402
from paibox.quant import SnnActivation  # noqa: E402
from quantize import INPUT_SCALE, loaders  # noqa: E402

BUDGET = HwConfigV2.N_CORE_OFFLINE


def spiking_plan(readings, prefix: int) -> dict[str, SnnActivation]:
    """Make the first *prefix* sites spiking, leaving the last one alone.

    The final site has to stay in the value domain: its output feeds an
    ``argmax``, and a spike train would have to be rate-decoded first.

    The threshold is the layer's calibrated output range. A real conversion
    would pick it by threshold balancing; for a core count any value that
    survives ``export``'s checks does, since placement never reads it.
    """
    paths = list(readings)[:-1]
    plan = {}
    for path in paths[:prefix]:
        peak = readings[path].output_max
        if peak is None or peak <= 0:
            raise ValueError(f"'{path}' has no calibrated output range to use as a threshold")
        plan[path] = SnnActivation(v_threshold=float(peak), tau=1.0)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiking-prefix", type=int, default=0, help="leading sites made spiking")
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=16)
    args = parser.parse_args()

    started = time.perf_counter()
    model, _ = load_checkpoint(DEFAULT_CKPT)
    prepared = tool.paicore.prepare(model, mean=MEAN, std=STD)
    train_loader, _ = loaders("cifar10")

    def forward_loop(net: nn.Module) -> None:
        for index, (images, _) in enumerate(train_loader):
            if index >= args.calib_batches:
                break
            net(images)

    config = tool.paicore.paicore_cfg(
        weight_bits=args.weight_bits, act_bits=args.act_bits, per_channel=False
    )
    with contextlib.redirect_stdout(io.StringIO()):
        quantized = tool.mytorch.opt.apply_mode(
            copy.deepcopy(prepared),
            mode=[("quantize", config)],
            registry=tool.mytorch.quantization.mode.QuantizeModeRegistry,
        )
        tool.paicore.calibrate_with(quantized, tool.paicore.MseCalibrator, forward_loop)
        readings = tool.paicore.read_sites(quantized)
        plan = spiking_plan(readings, args.spiking_prefix)
        spec = tool.paicore.build_spec(
            quantized,
            input_scale=INPUT_SCALE,
            timesteps=args.timesteps,
            per_channel_policy=tool.paicore.PerChannelPolicy.PER_TENSOR,
            spiking=plan,
            topology=prepared,
        )

    graph = lower(prepared, spec, args.timesteps)
    groups, _ = place(graph)
    cores = sum(count for _, count, _ in groups)
    lcns = sorted({lcn for _, _, lcn in groups})

    label = f"前 {args.spiking_prefix} 层脉冲" if args.spiking_prefix else "全 ANN"
    print(
        f"{label:<14} 核数={cores:<5} 预算={100.0 * cores / BUDGET:>5.0f}%  "
        f"LCN={lcns}  路由组={len(groups)}  {'装得下' if cores <= BUDGET else ''}"
        f"  {time.perf_counter() - started:.0f}s"
    )
    if plan:
        print(f"{'':>14} 脉冲层: {', '.join(plan)}")


if __name__ == "__main__":
    main()
