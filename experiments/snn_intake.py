"""How far the DVS-Gesture SNN gets through the deployment path, and where it stops.

The point of this script is to fail informatively. The network is a natively
trained SNN — five ``Conv2d -> BatchNorm2d -> IFNode -> AvgPool2d`` blocks, then
two spiking fully connected layers and a voting layer — which is a better first
SNN target than converting an ANN, because there is no ANN/SNN boundary to
rate-decode across, no threshold to balance (training chose it), and nothing to
calibrate between layers since the values there are spikes.

There is also a reference to compare against. ``infer.py`` in the source tree
reports 85.42% for the float model and 81.94% for its own int8 version, which
quantizes by multiplying every weight by a fixed 127 and clipping — the int8
checkpoint is visibly saturated at ±127. Per-layer calibration should recover
part of that 3.5-point drop.

Usage
-----
    uv run python experiments/snn_intake.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

SOURCE = Path(__file__).resolve().parent.parent / "DVSGesture_sw_and_rt"
QUANT_TOOL = Path.home() / "quant_tool"
for extra in (SOURCE, QUANT_TOOL):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import tool.paicore  # noqa: E402
from spikingjelly.activation_based import functional, neuron, surrogate  # noqa: E402

CKPT = SOURCE / "checkpoints" / "dropout_0.1_sum_qat.pt"


def build() -> tuple[nn.Module, dict]:
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
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  state_dict 不完全匹配: missing={len(missing)} unexpected={len(unexpected)}")
        for key in list(missing)[:5]:
            print(f"    missing {key}")
        for key in list(unexpected)[:5]:
            print(f"    unexpected {key}")
    return model.eval(), blob


def describe(model: nn.Module, title: str) -> None:
    print(f"{title}")
    for path, module in model.named_modules():
        if list(module.children()):
            continue
        extra = ""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight
            extra = f"  weight={tuple(weight.shape)} bias={'有' if module.bias is not None else '无'}"
        if isinstance(module, neuron.IFNode):
            extra = f"  v_threshold={module.v_threshold} v_reset={module.v_reset}"
        print(f"  {path:<16} {type(module).__name__:<18}{extra}")


def main() -> None:
    model, blob = build()
    print(
        f"检查点 {CKPT.name}: channels={blob['channels']} T={blob['T']} "
        f"split_by={blob['split_by']} best_test_acc={blob['best_test_acc']:.4f}"
    )
    print()
    describe(model, "原始结构(带 BN)")

    print()
    print("第 1 步 prepare(折叠 BN、剥离推理无操作)")
    try:
        prepared = tool.paicore.prepare(model)
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        print(f"  失败 {type(exc).__name__}: {exc}")
        return
    functional.reset_net(prepared)
    print()
    describe(prepared, "折叠后结构")

    sites = {
        path
        for path, module in prepared.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    print()
    print(f"第 2 步 site_graph({len(sites)} 个量化 site)")
    try:
        graph = tool.paicore.site_graph(prepared, frozenset(sites))
    except Exception as exc:  # noqa: BLE001
        print(f"  失败 {type(exc).__name__}: {exc}")
        return
    print(graph.describe())


if __name__ == "__main__":
    main()
