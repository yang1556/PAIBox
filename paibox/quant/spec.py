"""The hand-off between a quantizer and the PAICORE 2.5 deployment stage.

A quantization spec records *decisions* — bit widths, granularity, activation
levels, neuron parameters — and nothing about hardware encoding. Register units,
LUT tables and neuron configuration are derived downstream, so the quantizer
never needs to import PAIBox internals.

Compute attributes such as kernel size or stride are deliberately absent: the
exporter reads those from the module it is walking. A spec describes only what
quantization decided.

Neuron parameters are carried in the **real-valued domain**, not in accumulator
units. That keeps one representation usable by both routes into a spiking layer:
a layer converted from ANN arrives with a threshold chosen by threshold
balancing, while a natively trained spiking layer arrives with the threshold it
was trained with. Downstream both are divided by the same scale chain.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, unique

from torch import Tensor

from .hw import BIT_WIDTHS, Constraint, UnsupportedConfig

__all__ = [
    "AnnActivation",
    "Granularity",
    "LayerQuant",
    "QuantSpec",
    "SnnActivation",
    "WeightQuant",
]


@unique
class Granularity(Enum):
    """Scale sharing pattern for a weight tensor.

    ``PER_CHANNEL`` divides along the output axis, so one scale belongs to one
    neuron and can be absorbed by that neuron's own shift. ``PER_GROUP`` divides
    along the reduction axis, which means a single dot product spans several
    scales — the accumulator has already summed them by the time anything can
    rescale, so it needs either extra weight bits or a split across cores.
    """

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


@dataclass(frozen=True)
class WeightQuant:
    """Integer weights and the scale needed to interpret them.

    Args:
        bits: Code width, one of :data:`~paibox.quant.hw.BIT_WIDTHS`.
        scale: Real value per integer step. Scalar for ``PER_TENSOR``, a tensor
            of per-output-channel or per-group scales otherwise.
        granularity: Scale sharing pattern.
        group_size: Reduction-axis group length, required for ``PER_GROUP``.
    """

    bits: int
    scale: float | Tensor
    granularity: Granularity = Granularity.PER_TENSOR
    group_size: int | None = None

    def __post_init__(self) -> None:
        if self.bits not in BIT_WIDTHS:
            raise UnsupportedConfig(
                Constraint.BIT_WIDTH,
                f"weight width {self.bits} is not selectable; only {BIT_WIDTHS} exist.",
            )
        if self.granularity is Granularity.PER_GROUP and not self.group_size:
            raise ValueError("PER_GROUP weights need a group_size")
        if self.granularity is not Granularity.PER_GROUP and self.group_size:
            raise ValueError(f"group_size is meaningless for {self.granularity}")


@dataclass(frozen=True)
class AnnActivation:
    """Output codes of a value-domain layer, realised by the activation LUT.

    Supplying ``levels`` asks for a non-uniform grid. That costs no more than a
    uniform one here, because the LUT's 256 thresholds are independently
    placeable — a property mainstream quantizers cannot exploit and therefore do
    not offer.

    Args:
        bits: Output code width.
        signed: Emit signed codes. Required wherever the result feeds an
            ``argmax``, since rectifying negative logits to zero destroys it.
        levels: Output levels in the real domain, strictly increasing. ``None``
            asks for a uniform grid over ``[0, max_val]``.
        max_val: Upper end of the uniform grid. Required when ``levels`` is None.
    """

    bits: int
    signed: bool = False
    levels: Sequence[float] | None = None
    max_val: float | None = None

    def __post_init__(self) -> None:
        if self.bits not in BIT_WIDTHS:
            raise UnsupportedConfig(
                Constraint.OUTPUT_WIDTH,
                f"output width {self.bits} is not selectable; only {BIT_WIDTHS} exist.",
            )
        if self.levels is None and self.max_val is None:
            raise ValueError("an ANN activation needs either levels or max_val")
        if self.levels is not None and len(self.levels) < 2:
            raise ValueError("need at least two output levels")


@dataclass(frozen=True)
class SnnActivation:
    """Neuron parameters of a spiking layer, in the real-valued domain.

    The same shape serves a layer converted from ANN and a natively trained
    spiking layer; only the provenance of the numbers differs.

    Args:
        v_threshold: Firing threshold.
        v_reset: Reset potential, or ``None`` for a subtractive (soft) reset.
        tau: Membrane time constant. ``1.0`` means no decay. Larger values must
            be powers of two, because the only multiplicative leak available is
            a shift.
        leak_v: Additive leak per timestep. Folded together with any bias.
        signed: Emit ``{-1, 0, +1}`` rather than ``{0, 1}``, i.e. let the
            negative threshold fire instead of acting as a floor.
    """

    v_threshold: float
    v_reset: float | None = 0.0
    tau: float = 1.0
    leak_v: float = 0.0
    signed: bool = False

    def __post_init__(self) -> None:
        if self.v_threshold <= 0:
            raise ValueError(f"v_threshold must be positive, got {self.v_threshold}")
        if self.tau < 1.0:
            raise ValueError(f"tau must be at least 1, got {self.tau}")
        if self.tau > 1.0 and not _is_power_of_two(self.tau):
            raise UnsupportedConfig(
                Constraint.POW2_SCALE,
                f"tau {self.tau} is not a power of two, but the multiplicative "
                "leak is a shift (leak_tau).",
                "round tau to the nearest power of two, or fold the residue into "
                "the threshold",
            )


def _is_power_of_two(value: float) -> bool:
    exponent = math.log2(value)
    return exponent == math.floor(exponent)


@dataclass(frozen=True)
class LayerQuant:
    """Quantization decisions for one compute site and its activation.

    Args:
        input_scale: Real value per input code. Must equal the producing layer's
            ``output_scale``.
        output_scale: Real value per output code.
        weight: Weight quantization, or ``None`` for a parameter-free site such
            as pooling.
        activation: Activation following this site, or ``None`` when the site
            emits a membrane potential for a downstream accumulation.
        bias: Integer bias, folded into ``leak_v`` downstream. Subject to a
            narrower range than the accumulator.
        activation_path: Dotted path of the module the activation replaces.
            ``None`` means the next sibling of the compute site, which is where
            it sits in a sequential model. The float activation has to be
            replaced rather than left in place: lowering maps a bare
            ``nn.ReLU`` to a LUT spanning the whole int32 range, which sends
            every realistic accumulator value to zero.
    """

    input_scale: float
    output_scale: float
    weight: WeightQuant | None = None
    activation: AnnActivation | SnnActivation | None = None
    bias: Tensor | None = None
    activation_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("input_scale", "output_scale"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        per_channel = (
            self.weight is not None
            and self.weight.granularity is Granularity.PER_CHANNEL
        )
        if per_channel and isinstance(self.activation, AnnActivation):
            # One LUT serves every neuron on a core, so output channels cannot
            # each get their own requantization curve. A per-channel weight
            # scale is only expressible if its channel ratios are powers of two,
            # which the exporter checks once it knows the actual scales.
            if isinstance(self.weight.scale, float):
                raise ValueError(
                    "PER_CHANNEL granularity needs a tensor of per-channel scales"
                )

    @property
    def is_spiking(self) -> bool:
        return isinstance(self.activation, SnnActivation)

    @property
    def accumulator_scale(self) -> float:
        """Real value per accumulator step, for a per-tensor weight scale.

        The accumulator sums ``weight_code * input_code``, so one step is worth
        the product of the two scales. Per-channel and per-group scales make
        this a vector, resolved by the exporter.
        """
        if self.weight is None:
            return self.input_scale
        if not isinstance(self.weight.scale, float):
            raise ValueError(
                "accumulator_scale is scalar only for a per-tensor weight scale"
            )
        return self.weight.scale * self.input_scale


@dataclass(frozen=True)
class QuantSpec:
    """A whole model's quantization decisions, keyed by module path.

    Args:
        input_bits: Width of the codes fed to the first layer.
        input_scale: Real value per input code.
        timesteps: Timesteps one inference runs for. Network-wide: a spiking
            layer needs several to carry rate-coded information, and the value
            must also cover the pipeline depth, since a value takes one timestep
            per cascaded core to reach the output.
        input_signed: Whether input codes are signed.
        layers: Per-site decisions, keyed by the module's dotted name.
    """

    input_bits: int
    input_scale: float
    timesteps: int = 1
    input_signed: bool = False
    layers: dict[str, LayerQuant] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_bits not in BIT_WIDTHS:
            raise UnsupportedConfig(
                Constraint.BIT_WIDTH,
                f"input width {self.input_bits} is not selectable; "
                f"only {BIT_WIDTHS} exist.",
            )
        if self.input_scale <= 0:
            raise ValueError(f"input_scale must be positive, got {self.input_scale}")
        if self.timesteps < 1:
            raise ValueError(f"timesteps must be at least 1, got {self.timesteps}")

    @property
    def has_spiking_layers(self) -> bool:
        return any(layer.is_spiking for layer in self.layers.values())

    def check_scale_chain(self, order: Sequence[str]) -> None:
        """Verify that consecutive layers agree on the scale between them.

        Args:
            order: Module paths in execution order. Sites absent from
                :attr:`layers` are skipped, which lets scale-preserving modules
                such as pooling or flattening sit between two quantized sites.

        Raises:
            ValueError: When a layer's ``input_scale`` does not match the
                previous layer's ``output_scale``. Such a mismatch silently
                rescales activations, so it is worth catching before export.
            KeyError: When ``order`` names a site that has no decisions.
        """
        previous: tuple[str, float] | None = None
        for name in order:
            if name not in self.layers:
                continue
            layer = self.layers[name]
            if previous is not None:
                prior_name, prior_scale = previous
                if layer.input_scale != prior_scale:
                    raise ValueError(
                        f"scale chain breaks between '{prior_name}' and '{name}': "
                        f"output_scale {prior_scale} vs input_scale "
                        f"{layer.input_scale}"
                    )
            previous = (name, layer.output_scale)
