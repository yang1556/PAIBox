"""Compare the first layer's very first tick, element by element.

At tick zero every membrane potential is zero on both sides, so the spike
pattern is a pure function of weights, bias, threshold and input — no state, no
pipeline, no reset semantics. `snn_param_diff.py` showed those parameters
matching exactly, yet the first layer already fires four times too often on the
chip. Something upstream of the comparison must differ, and this narrows it to
either the accumulator or the input that feeds it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import warnings

import torch

import tool.paicore

from snn_quantize import build, calibrated_spec, deploy, loaders, materialize

TIMESTEPS = 20


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    knobs = argparse.Namespace(
        weight_bits=8, act_bits=8, calib_batches=4, timesteps=TIMESTEPS, batch=8
    )
    model, _ = build()
    prepared = tool.paicore.prepare(model)
    train_loader, test_loader = loaders(TIMESTEPS, knobs.batch)
    spec, _, _ = calibrated_spec(prepared, train_loader, knobs)
    integer = materialize(prepared, spec)
    reference = copy.deepcopy(integer)

    frames, _ = next(iter(test_loader))
    frame0 = frames[0:1].float()[:, 0]  # [1, 2, 128, 128], the first tick only

    print(f"输入第 0 拍: 形状 {tuple(frame0.shape)}, 取值 "
          f"{frame0.min():.0f}..{frame0.max():.0f}, 非零 {int((frame0 != 0).sum())}")

    modules = dict(reference.named_modules())
    conv = modules["conv_fc.0"]
    software_acc = conv(frame0)
    threshold = modules["conv_fc.1"].v_threshold
    print(
        f"软件累加器: 范围 {software_acc.min():.0f}..{software_acc.max():.0f}, "
        f"阈值 {threshold:.0f}, 过阈值 {int((software_acc >= threshold).sum())}"
    )

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, _, _ = deploy(prepared, spec, TIMESTEPS)

    node = graph.nodes["SequentialOp_0"]
    chip_acc = node.comp(frame0)
    print(
        f"芯片累加器: 范围 {chip_acc.min():.0f}..{chip_acc.max():.0f}, "
        f"阈值 {float(node.act.thres_pos):.0f}, "
        f"过阈值 {int((chip_acc >= float(node.act.thres_pos)).sum())}"
    )

    print(f"\n累加器逐元素相同: {torch.equal(software_acc, chip_acc)}")
    if not torch.equal(software_acc, chip_acc):
        delta = (software_acc - chip_acc).abs()
        print(f"  最大偏差 {delta.max():.0f}, 不同元素 {int((delta != 0).sum())}"
              f" / {delta.numel()}")
        print(f"  权重相同: {torch.equal(conv.weight, node.comp.weight)}")
        print(f"  偏置相同: {torch.equal(conv.bias, node.comp.bias)}")
        print(f"  软件 conv: {conv}")
        print(f"  芯片 comp: {node.comp}")

    # Layer two is the first layer fed by a pool, and the first place the two
    # executions part company. Both sides get the *same* layer-one spikes, so
    # anything that differs here belongs to the pool or to layer two itself.
    spikes1 = (software_acc >= threshold).float()
    print(f"\n第 1 层第 0 拍发放 {int(spikes1.sum())}（两边一致，作为第 2 层的共同输入）")

    pool_sw = modules["conv_fc.2"]
    conv2_sw = modules["conv_fc.3"]
    thres2_sw = modules["conv_fc.4"].v_threshold
    acc2_sw = conv2_sw(pool_sw(spikes1))

    pool_chip = graph.nodes["StandaloneCompOp_1"]
    node2 = graph.nodes["SequentialOp_1"]
    acc2_chip = node2.comp(pool_chip.comp(spikes1))
    thres2_chip = float(node2.act.thres_pos)

    print(f"  软件 pool: {pool_sw}  -> 和 {pool_sw(spikes1).sum():.0f}")
    print(f"  芯片 pool: {pool_chip.comp} -> 和 {pool_chip.comp(spikes1).sum():.0f}")
    print(
        f"  软件累加器 {acc2_sw.min():.1f}..{acc2_sw.max():.1f}, 阈值 {thres2_sw:.0f}, "
        f"过阈值 {int((acc2_sw >= thres2_sw).sum())}"
    )
    print(
        f"  芯片累加器 {acc2_chip.min():.1f}..{acc2_chip.max():.1f}, 阈值 {thres2_chip:.0f}, "
        f"过阈值 {int((acc2_chip >= thres2_chip).sum())}"
    )

    # The deferred division makes the chip accumulator four times the software
    # one, with the threshold scaled to match, so this should be all zeros.
    residual = (acc2_chip - 4.0 * acc2_sw).abs()
    print(f"  芯片 - 4x软件: 最大偏差 {residual.max():.1f}, 非零 {int((residual > 0).sum())}"
          f" / {residual.numel()}")


if __name__ == "__main__":
    main()
