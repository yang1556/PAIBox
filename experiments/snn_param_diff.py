"""Compare the neuron parameters the chip got against the ones software uses.

`snn_layer_diff.py` showed the chip over-firing from the very first layer, and
the deeper layers settling into a flat per-tick rate that ignores the input
entirely. Both are symptoms of an accumulate-compare whose threshold or constant
term does not match the software integer model. This prints the two side by side
so the mismatch is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import warnings

import torch
from torch import nn

import tool.paicore

from snn_quantize import build, calibrated_spec, deploy, loaders, materialize

TIMESTEPS = 20


def describe(value) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        flat = value.flatten().float()
        if flat.numel() == 1:
            return f"{float(flat[0]):.4g}"
        return f"[{flat.numel()}] 均值 {flat.mean():.4g} 范围 {flat.min():.4g}..{flat.max():.4g}"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    knobs = argparse.Namespace(
        weight_bits=8, act_bits=8, calib_batches=4, timesteps=TIMESTEPS, batch=8
    )
    model, _ = build()
    prepared = tool.paicore.prepare(model)
    train_loader, _ = loaders(TIMESTEPS, knobs.batch)
    spec, _, _ = calibrated_spec(prepared, train_loader, knobs)
    integer = materialize(prepared, spec)
    reference = copy.deepcopy(integer)

    print("\n软件整数模型")
    modules = dict(reference.named_modules())
    for path, layer in spec.layers.items():
        compute = modules[path]
        activation = modules[layer.activation_path] if layer.activation_path else None
        print(f"  {path:14s} 阈值 {describe(getattr(activation, 'v_threshold', None)):>10s}"
              f"  偏置 {describe(compute.bias)}")

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, _, _ = deploy(prepared, spec, TIMESTEPS)

    print("\n每核活动窗口")
    for name, node in graph.nodes.items():
        params = getattr(node, "core_params", None)
        if params is None:
            continue
        print(
            f"  {name:20s} tick_start {params.tick_start}  "
            f"tick_duration {params.tick_duration}"
        )

    print("\n芯片图节点")
    for name, node in graph.nodes.items():
        act = getattr(node, "act", None)
        comp = getattr(node, "comp", None)
        pool = getattr(node, "avgpool_deploy_metadata", None)
        if act is None and comp is None:
            continue
        bias = getattr(comp, "bias", None) if comp is not None else None
        if act is None:
            print(f"  {name:20s} {type(comp).__name__:14s} 无神经元  偏置 {describe(bias)}")
            continue
        print(
            f"  {name:20s} {type(comp).__name__:14s} thres_pos {describe(act.thres_pos):>10s}"
            f"  leak_v {describe(act.leak_v):>10s}  偏置 {describe(bias)}"
        )
        if pool is not None:
            print(f"  {'':20s}   avgpool 元数据: {pool}")


if __name__ == "__main__":
    main()
