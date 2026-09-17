"""Turning a quantized model into one PAIBox can lower.

``compile_to_paiir`` recognises a specific surface: standard compute modules
whose parameters are already integers, followed by a chip activation. Reaching
that surface is what this module does, given a float model and the decisions a
quantizer made about it.

Two properties of the result matter more than they look:

**No quantizer wrappers survive.** ``fuse_to_offline_cores`` merges a
``comp -> act`` pair into one offline core by matching that exact shape. A
fake-quant module or an observer left between the two blocks the match, which
costs an extra core per layer and may fail deployment validation outright.

**Activations are replaced, not kept.** Lowering maps a bare ``nn.ReLU`` onto
``LutReLU()`` with default bounds, which span the entire int32 range: the second
threshold sits above eight million, so every accumulator a real model produces
lands in the first bin and the layer outputs zeros. A calibrated LUT is not an
optimisation here, it is the difference between working and not.

Replacement happens by module path on a copy of the model, so any topology
survives — branches and custom ``forward`` methods included — because the
structure is never rebuilt.
"""

import copy
import math

import torch
from paicorelib import LeakMultiComparisonOrder, LeakMultiMode
from torch import Tensor, nn

from paibox.paiir.ir.core_neuron import ANNNodeV25, IFNodeV25, LIFNodeV25

from .hw import (
    Constraint,
    UnsupportedConfig,
    check_bias,
    check_shift,
    check_threshold,
)
from .lut import fit_affine_lut, fit_lut
from .spec import AnnActivation, Granularity, LayerQuant, QuantSpec, SnnActivation

__all__ = ["export"]


def export(model: nn.Module, spec: QuantSpec, inplace: bool = False) -> nn.Module:
    """Build a PAIBox-lowerable model from a float model and its quant spec.

    Args:
        model: The model the quantizer inspected. Compute sites keep their
            structure; only parameters and activation modules change.
        spec: Per-site decisions, keyed by dotted module path.
        inplace: Modify *model* instead of a copy.

    Returns:
        A model whose compute modules hold integer parameters and whose
        activations are chip neurons, ready for ``compile_to_paiir``.

    Raises:
        UnsupportedConfig: When a decision cannot be expressed on an offline
            core. The exception names the blocking constraint.
        KeyError: When the spec names a module the model does not contain.
    """
    target = model if inplace else copy.deepcopy(model)
    available = dict(target.named_modules())

    for path, layer in spec.layers.items():
        if path not in available:
            raise KeyError(f"spec names module '{path}', which the model lacks")

        compute = available[path]
        _materialize_parameters(compute, layer, path)

        if layer.activation is None:
            continue

        _place_activation(target, path, layer, _build_activation(layer, path))

    return target.eval()


def _place_activation(
    root: nn.Module, compute_path: str, layer: LayerQuant, activation: nn.Module
) -> None:
    """Put *activation* where the spec says, or after the compute site.

    A site that is the last child of a ``nn.Sequential`` has nowhere to replace,
    which is the normal shape of a final classifier layer: the float model ends
    at the linear layer, while the chip still has to emit codes. Appending is
    unambiguous there. Anywhere else the caller must say which module the
    activation replaces, because guessing would silently drop a real operation.
    """
    if layer.activation_path is not None:
        if layer.activation_path not in dict(root.named_modules()):
            raise KeyError(
                f"activation_path '{layer.activation_path}' names no module. "
                "Make the activation a module rather than a functional call."
            )
        _replace_module(root, layer.activation_path, activation)
        return

    parent_path, _, name = compute_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    names = [child for child, _ in parent.named_children()]
    index = names.index(name)

    if index + 1 < len(names):
        sibling = names[index + 1]
        _replace_module(
            root, f"{parent_path}.{sibling}" if parent_path else sibling, activation
        )
        return

    if isinstance(parent, nn.Sequential):
        parent.append(activation)
        return

    raise KeyError(
        f"'{compute_path}' is the last child of '{parent_path or '<root>'}' and "
        "that parent is not a Sequential, so there is nowhere to put its "
        "activation. Pass activation_path."
    )


def _materialize_parameters(module: nn.Module, layer: LayerQuant, path: str) -> None:
    """Write integer weights and bias into a standard compute module.

    Parameter-free sites such as pooling carry codes through unchanged, so they
    only ever contribute an activation.
    """
    if layer.weight is None:
        return
    if not hasattr(module, "weight") or module.weight is None:
        raise UnsupportedConfig(
            Constraint.BIT_WIDTH,
            f"'{path}' has weight quantization in the spec but no weight tensor.",
        )

    scale = layer.weight.scale
    if layer.weight.granularity is Granularity.PER_GROUP:
        raise UnsupportedConfig(
            Constraint.ACT_RESCALE,
            f"'{path}' asks for per-group weights, which this exporter cannot "
            "place yet. A group scale divides along the reduction axis, so a "
            "single dot product spans several scales and the accumulator has "
            "already summed them before anything can rescale.",
            "fold the scale spread into extra weight bits, or split the "
            "reduction across cores and recombine through direct "
            "membrane-potential accumulation — neither is wired up yet",
        )
    if layer.weight.granularity is Granularity.PER_CHANNEL:
        _check_per_channel_scales(scale, path)

    limit = 2 ** (layer.weight.bits - 1)
    with torch.no_grad():
        codes = _divide_by_scale(module.weight.detach(), scale)
        module.weight.copy_(codes.round().clamp(-limit, limit - 1))

        if layer.bias is not None:
            for value in layer.bias.flatten().tolist():
                check_bias(int(value))
            if getattr(module, "bias", None) is None:
                raise UnsupportedConfig(
                    Constraint.BIAS_RANGE,
                    f"the spec gives '{path}' a bias but the module has none.",
                    "construct the module with bias=True before exporting",
                )
            module.bias.copy_(layer.bias.to(module.bias.dtype))


def _divide_by_scale(weight: Tensor, scale: float | Tensor) -> Tensor:
    """Divide by a scalar, or broadcast a per-output-channel scale over axis 0."""
    if isinstance(scale, Tensor) and scale.numel() > 1:
        shape = [scale.numel()] + [1] * (weight.dim() - 1)
        return weight / scale.reshape(shape).to(weight.dtype)
    return weight / float(scale)


def _check_per_channel_scales(scale: float | Tensor, path: str) -> None:
    """Per-channel scales are only expressible when their ratios are powers of two.

    One LUT serves every neuron on a core, so channels cannot each get their own
    requantization curve. The one per-neuron mechanism left is ``leak_tau``, a
    shift, so the channel-to-channel ratios have to be powers of two — and using
    it at all forces the full neuron tier, eight times the neuron SRAM of a
    folded neuron.
    """
    if not isinstance(scale, Tensor):
        raise ValueError(f"'{path}' declares PER_CHANNEL but its scale is scalar")

    finest = float(scale.min())
    if finest <= 0:
        raise ValueError(f"'{path}' has a non-positive per-channel scale")

    exponents = torch.log2(scale.double() / finest)
    rounded = exponents.round()
    if not torch.allclose(exponents, rounded, atol=1e-6):
        raise UnsupportedConfig(
            Constraint.ACT_RESCALE,
            f"'{path}' has per-channel scales whose ratios are not powers of two, "
            "and the activation LUT is shared by every neuron on the core.",
            "round the per-channel scales to power-of-two ratios, fold them into "
            "the weights, or fall back to a per-tensor scale",
        )
    for exponent in rounded.flatten().tolist():
        check_shift(int(exponent))


def _build_activation(layer: LayerQuant, path: str) -> nn.Module:
    if isinstance(layer.activation, AnnActivation):
        return _build_ann_activation(layer, path)
    if isinstance(layer.activation, SnnActivation):
        return _build_snn_activation(layer, path)
    raise TypeError(f"'{path}' has an unrecognised activation {layer.activation!r}")


def _build_ann_activation(layer: LayerQuant, path: str) -> ANNNodeV25:
    """Fit the LUT that carries requantization, activation and output coding.

    ``levels`` asks for a non-uniform grid, which the LUT supports at no extra
    cost because its thresholds are placed independently.

    With per-channel weight scales the LUT is fitted against the finest of them
    and each channel's own shift makes up the difference, which is the only
    per-neuron rescaling an offline core has.
    """
    act = layer.activation
    assert isinstance(act, AnnActivation)
    acc_scale = _accumulator_scale(layer, path)

    lut = (
        fit_lut(act.levels, acc_scale, signed=act.signed)
        if act.levels is not None
        else fit_affine_lut(acc_scale / layer.output_scale, act.bits, signed=act.signed)
    )
    return ANNNodeV25(lut, **_channel_shift_kwargs(layer))


def _channel_shift_kwargs(layer: LayerQuant) -> dict:
    """Per-channel alignment, expressed as each neuron's ``leak_tau`` shift.

    Channel ``c`` accumulates in units of ``s[c] * s_input``, while the shared
    LUT reads units of ``min(s) * s_input``. Left-shifting by
    ``log2(s[c] / min(s))`` closes that gap exactly, which is why the ratios have
    to be powers of two in the first place.

    Two register choices make the shift land where it is needed. ``DISABLE``
    turns the multiplicative leak into a plain shift rather than a decay toward
    the reset level, and ``BEFORE_COMPARE`` runs it during charging, before the
    lookup, instead of after the reset. Neither is the default.

    Shifting left rather than fitting the LUT at the coarsest scale and shifting
    right keeps the operation lossless: a right shift would discard low bits of
    the accumulator.
    """
    if layer.weight is None:
        return {}

    scale = layer.weight.scale
    if layer.weight.granularity is not Granularity.PER_CHANNEL:
        return {}
    if not isinstance(scale, Tensor) or scale.numel() <= 1:
        return {}

    exponents = torch.log2(scale.double() / float(scale.min())).round().to(torch.int32)
    return {
        "leak_tau_shift": exponents,
        "leak_multi_mode": LeakMultiMode.DISABLE,
        "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
    }


def _build_snn_activation(layer: LayerQuant, path: str) -> nn.Module:
    """Convert real-valued neuron parameters into accumulator units.

    The chip integrates in accumulator steps, so a threshold expressed as a real
    value has to be divided by the worth of one step. Which route produced the
    parameters — conversion from ANN, or training a spiking model directly —
    makes no difference here.
    """
    act = layer.activation
    assert isinstance(act, SnnActivation)
    acc_scale = _accumulator_scale(layer, path)

    threshold = int(round(act.v_threshold / acc_scale))
    if threshold < 1:
        raise UnsupportedConfig(
            Constraint.POW2_SCALE,
            f"'{path}' has threshold {act.v_threshold} which is under one "
            f"accumulator step ({acc_scale:g}); the neuron would fire every step.",
            "raise the threshold, or lower the weight scale so a step is finer",
        )
    check_threshold(threshold)

    reset = None if act.v_reset is None else int(round(act.v_reset / acc_scale))
    leak = int(round(act.leak_v / acc_scale))
    check_bias(leak)

    if act.tau == 1.0:
        return IFNodeV25(v_threshold=float(threshold), v_reset=_as_float(reset))

    check_shift(-int(round(math.log2(act.tau))))
    return LIFNodeV25(
        tau=float(act.tau),
        v_threshold=float(threshold),
        v_reset=_as_float(reset),
    )


def _as_float(value: int | None) -> float | None:
    return None if value is None else float(value)


def _accumulator_scale(layer: LayerQuant, path: str) -> float:
    """Real value of one accumulator step.

    With a per-channel weight scale the step differs per channel; the shared LUT
    is then fitted against the finest scale and the per-channel shift absorbs
    the rest.
    """
    if layer.weight is None:
        return layer.input_scale

    scale = layer.weight.scale
    if isinstance(scale, Tensor) and scale.numel() > 1:
        return float(scale.min()) * layer.input_scale
    return float(scale) * layer.input_scale


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, name, replacement)
