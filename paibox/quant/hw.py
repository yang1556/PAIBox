"""PAICORE 2.5 offline-core capability surface.

This module answers one question: *can this quantization configuration be
expressed on the chip at all?* It deliberately makes no optimality judgement.
A configuration that is legal but wasteful is accepted, and its cost is
reported; a configuration the hardware cannot express is rejected with the
specific constraint that blocks it.

That split matters because a quantization tool must honour explicit user
choices even when they are resource-suboptimal: the baselines a search is
compared against are themselves user-specified configurations, and they have to
go through the same code path as the search's own output.

Register limits are imported from :mod:`paicorelib` rather than transcribed, so
this module stays correct when the hardware definition is updated.

Values not exposed by ``paicorelib`` are defined here with their source noted:
the elastic neuron/weight SRAM geometry, the fan-in base, the timestep field
width, and the weight packing tables all come from the PAICORE 2.5 application
manual, sections 2.2.3 and 2.3.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum, unique

from paicorelib import LCN_EX, DataWidth, HwConfigV2, OfflineNeuRegLimV2, WeightWidth

__all__ = [
    "BIAS_RANGE",
    "BIT_WIDTHS",
    "Constraint",
    "FANIN_BASE",
    "NeuronKind",
    "SRAM_LINES_PER_CORE",
    "TIMESTEP_FIELD_BITS",
    "UnsupportedConfig",
    "WeightFormat",
    "core_count",
    "csc_breakeven_sparsity",
    "data_width_of",
    "lcn_for_axons",
    "neuron_sram_lines",
    "stream_capacity",
    "weight_sram_lines",
    "weight_width_of",
]

#: Offline cores per chip. Total silicon is 81 tiles: 63 offline, 17 online,
#: 1 RISC-V CPU [manual 1.5].
N_CORE_OFFLINE = HwConfigV2.N_CORE_OFFLINE

#: The elastic SRAM shared by neuron parameters and synaptic weights, in
#: 128-bit lines [manual 2.3.3].
SRAM_LINES_PER_CORE = 4096
SRAM_LINE_BITS = 128

#: Inclusive bounds a bias must satisfy, in accumulator steps. A bias is carried
#: by ``leak_v``, which is 20-bit signed and therefore much narrower than the
#: 32-bit accumulator it is added to. Exposed alongside :func:`check_bias` so a
#: caller can report how much of the range a model uses instead of only learning
#: that it overflowed.
BIAS_RANGE = (OfflineNeuRegLimV2.LEAK_V_MIN, OfflineNeuRegLimV2.LEAK_V_MAX)

#: Axon addresses at LCN 1x. The usable count is ``FANIN_BASE * LCN /
#: input_width`` because one address carries one bit of an input code
#: [manual 2.3.1].
FANIN_BASE = 512

#: The work frame splits 17 bits between Time Step and Axon address. The
#: timestep side is 8 bits wide at LCN 1x and loses one bit per LCN doubling
#: [manual 2.2.3]. It bounds *readout*, not computation: see
#: :func:`stream_capacity`.
TIMESTEP_FIELD_BITS = 8

#: Selectable code widths for weights, inputs and outputs. These are per-core
#: 2-bit register fields, so a core cannot mix widths [manual 2.3.1].
BIT_WIDTHS = (1, 2, 4, 8)

_DATA_WIDTH = {
    1: DataWidth.WIDTH_1BIT,
    2: DataWidth.WIDTH_2BIT,
    4: DataWidth.WIDTH_4BIT,
    8: DataWidth.WIDTH_8BIT,
}
_WEIGHT_WIDTH = {
    1: WeightWidth.WEIGHT_WIDTH_1BIT,
    2: WeightWidth.WEIGHT_WIDTH_2BIT,
    4: WeightWidth.WEIGHT_WIDTH_4BIT,
    8: WeightWidth.WEIGHT_WIDTH_8BIT,
}

#: Weights per 128-bit SRAM line, dense [manual 2.3.3].
_DENSE_PER_LINE = {1: 128, 2: 64, 4: 32, 8: 16}

#: Non-zero weights per 128-bit SRAM line under CSC compression. The row
#: indices take the rest of the line, so the count does not scale with width
#: the way the dense count does [manual 2.3.3].
_CSC_PER_LINE = {1: 7, 2: 7, 4: 6, 8: 5}


@unique
class Constraint(Enum):
    """Hardware constraints, labelled to match ``experiments/cost_model.md``."""

    CORE_CAPACITY = "C1"
    CHIP_BUDGET = "C2"
    FANIN_LIMIT = "C3"
    TIMESTEP = "C4"
    BIT_WIDTH = "C5"
    BRANCH_WIDTH = "C6"
    BIAS_RANGE = "C7"
    ACT_RESCALE = "C8"
    OUTPUT_WIDTH = "C9"
    POW2_SCALE = "C10"
    INITIAL_POTENTIAL = "C11"
    MODE_BOUNDARY = "C12"


class UnsupportedConfig(ValueError):
    """A configuration the hardware cannot express.

    Carries the blocking constraint so callers can report *why* a choice was
    refused, and a suggestion for the nearest expressible alternative. This is
    distinct from a merely wasteful configuration, which is accepted.
    """

    def __init__(
        self, constraint: Constraint, message: str, suggestion: str = ""
    ) -> None:
        self.constraint = constraint
        self.suggestion = suggestion
        detail = f"[{constraint.value} {constraint.name}] {message}"
        if suggestion:
            detail = f"{detail} Suggestion: {suggestion}"
        super().__init__(detail)


@unique
class NeuronKind(IntEnum):
    """Neuron storage tiers of the hierarchical neuron memory [manual 2.3.3].

    A folded neuron stores only the membrane potential and shares every
    dynamics parameter, which is why four of them fit one SRAM line. Requesting
    any per-neuron parameter — a per-channel bias or threshold — forces the full
    tier and therefore eight times the neuron SRAM of the folded tier.
    """

    FOLDED = 0
    HALF = 1
    FULL = 2

    @property
    def sram_lines(self) -> float:
        """128-bit SRAM lines occupied per neuron."""
        return {NeuronKind.FOLDED: 0.25, NeuronKind.HALF: 1.0, NeuronKind.FULL: 2.0}[
            self
        ]


@unique
class WeightFormat(Enum):
    """Weight storage layouts of the adaptive weight memory [manual 2.3.3]."""

    DENSE = "dense"
    CSC = "csc"


def data_width_of(bits: int) -> DataWidth:
    """Map a code width to its ``DataWidth`` register value."""
    if bits not in _DATA_WIDTH:
        raise UnsupportedConfig(
            Constraint.BIT_WIDTH,
            f"input/output width {bits} is not selectable; only {BIT_WIDTHS} exist.",
            f"round to one of {BIT_WIDTHS}",
        )
    return _DATA_WIDTH[bits]


def weight_width_of(bits: int) -> WeightWidth:
    """Map a weight width to its ``WeightWidth`` register value."""
    if bits not in _WEIGHT_WIDTH:
        raise UnsupportedConfig(
            Constraint.BIT_WIDTH,
            f"weight width {bits} is not selectable; only {BIT_WIDTHS} exist.",
            f"round to one of {BIT_WIDTHS}",
        )
    return _WEIGHT_WIDTH[bits]


def lcn_for_axons(n_inputs: int, input_bits: int) -> LCN_EX:
    """Smallest LCN whose axon space holds ``n_inputs`` codes of ``input_bits``.

    Mirrors ``RoutingGroup.set_lcn`` in ``paibox/backendv2/routing.py`` so the
    front end predicts the same value the backend will pick.

    Raises:
        UnsupportedConfig: when the fan-in exceeds even ``LCN_128X``. The layer
            then has to be split across cores and recombined through direct
            membrane-potential accumulation.
    """
    addrs = n_inputs * input_bits
    exponent = ((addrs - 1) // FANIN_BASE).bit_length() if addrs > 0 else 0
    if exponent > LCN_EX.LCN_128X.value:
        raise UnsupportedConfig(
            Constraint.FANIN_LIMIT,
            f"{n_inputs} inputs at {input_bits}-bit need {addrs} axon addresses, "
            f"beyond the {FANIN_BASE * (1 << LCN_EX.LCN_128X.value)} reachable at LCN_128X.",
            "split the reduction across cores and accumulate membrane potentials "
            "(AddPotentialMode.DIRECT_ADD), or narrow the input width",
        )
    return LCN_EX(exponent)


def stream_capacity(output_lcn: LCN_EX) -> int:
    """Timesteps that ``STREAM`` readout can tag, given the output group's LCN.

    The 8-bit timestep field is shared with the slot index, so a wide output
    LCN leaves fewer bits to number the steps. Exceeding this does **not** make
    a configuration infeasible: the runtime falls back to ``STEP`` decoding,
    which costs one host round trip per timestep. Compare
    ``_thread_decode_mode`` in ``paibox/backendv2/artifacts/compile_artifacts.py``.
    """
    return 1 << (TIMESTEP_FIELD_BITS - output_lcn.value)


def neuron_sram_lines(n_neurons: int, kind: NeuronKind) -> int:
    """SRAM lines taken by ``n_neurons`` of the given storage tier."""
    return -(-int(n_neurons * kind.sram_lines * 4) // 4)  # ceil on quarter-lines


def csc_breakeven_sparsity(weight_bits: int) -> float:
    """Non-zero fraction below which CSC beats dense storage.

    Dense packing scales with the weight width but CSC does not, because the
    row indices dominate the line. The result is that **low-bit quantization and
    sparse compression compete**: at 1-bit a tensor must be under ~5% non-zero
    for CSC to pay off, against ~31% at 8-bit.
    """
    if weight_bits not in _DENSE_PER_LINE:
        raise UnsupportedConfig(
            Constraint.BIT_WIDTH, f"weight width {weight_bits} is not selectable."
        )
    return _CSC_PER_LINE[weight_bits] / _DENSE_PER_LINE[weight_bits]


def weight_sram_lines(
    n_weights: int,
    weight_bits: int,
    fmt: WeightFormat = WeightFormat.DENSE,
    sparsity: float = 1.0,
) -> int:
    """SRAM lines for one weight set.

    Args:
        n_weights: Weights in the set, before compression.
        weight_bits: Code width, one of :data:`BIT_WIDTHS`.
        fmt: Dense or CSC layout.
        sparsity: Non-zero fraction, used only for CSC.
    """
    if fmt is WeightFormat.DENSE:
        per_line = _DENSE_PER_LINE[weight_bits]
        return -(-n_weights // per_line)

    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    nonzeros = int(round(n_weights * sparsity))
    return -(-nonzeros // _CSC_PER_LINE[weight_bits])


@dataclass(frozen=True, slots=True)
class CoreDemand:
    """SRAM demand of one layer, in 128-bit lines."""

    neuron_lines: int
    weight_lines: int

    @property
    def total(self) -> int:
        return self.neuron_lines + self.weight_lines


def core_count(demand: CoreDemand) -> int:
    """Lower bound on cores needed for a layer's SRAM demand.

    This is the ideal-packing bound. ``RoutingGroup.try_store_neuron`` packs
    first-fit and drops cross-core weight reuse when it rolls over to a new
    core, so the deployed count is greater than or equal to this. Calibrating
    that gap is what the core-count estimator does.
    """
    if demand.total <= 0:
        return 0
    return -(-demand.total // SRAM_LINES_PER_CORE)


def check_bias(value: int) -> None:
    """Bias folds into ``leak_v``, which is narrower than an int32 accumulator."""
    if not OfflineNeuRegLimV2.LEAK_V_MIN <= value <= OfflineNeuRegLimV2.LEAK_V_MAX:
        raise UnsupportedConfig(
            Constraint.BIAS_RANGE,
            f"bias {value} leaves the leak_v range "
            f"[{OfflineNeuRegLimV2.LEAK_V_MIN}, {OfflineNeuRegLimV2.LEAK_V_MAX}].",
            "rescale the layer so the quantized bias fits, or move part of the "
            "offset into the activation LUT",
        )


def check_threshold(value: int) -> None:
    """Positive/negative thresholds are 32-bit signed, like the accumulator."""
    if (
        not OfflineNeuRegLimV2.THRES_POS_MIN
        <= value
        <= OfflineNeuRegLimV2.THRES_POS_MAX
    ):
        raise UnsupportedConfig(
            Constraint.BIAS_RANGE,
            f"threshold {value} leaves the 32-bit signed register range.",
        )


def check_initial_potential(value: int, csc_accelerate: bool = False) -> None:
    """``vjt_initial`` is only 12-bit signed, and CSC acceleration consumes it.

    With ``csc_accelerate`` enabled the field is reused to hold a second copy of
    ``weight_address_start`` [manual 2.3.1], so a non-zero initial potential and
    CSC acceleration are mutually exclusive.
    """
    if csc_accelerate and value != 0:
        raise UnsupportedConfig(
            Constraint.INITIAL_POTENTIAL,
            "vjt_initial is repurposed to store weight_address_start when "
            "csc_accelerate is on, so it cannot also carry an initial potential.",
            "disable csc_accelerate, or keep the initial potential at zero",
        )
    if not (
        OfflineNeuRegLimV2.VJT_INITIAL_MIN
        <= value
        <= OfflineNeuRegLimV2.VJT_INITIAL_MAX
    ):
        raise UnsupportedConfig(
            Constraint.INITIAL_POTENTIAL,
            f"initial potential {value} leaves the vjt_initial range "
            f"[{OfflineNeuRegLimV2.VJT_INITIAL_MIN}, {OfflineNeuRegLimV2.VJT_INITIAL_MAX}].",
        )


def check_shift(exponent: int) -> None:
    """``leak_tau`` is the only multiplicative rescaling after accumulation.

    It is a shift, so any scale the front end wants to realise per neuron must
    be a power of two. Arbitrary ratios have to go into the activation LUT,
    which is shared by every neuron on the core.
    """
    if (
        not OfflineNeuRegLimV2.LEAK_TAU_MIN
        <= exponent
        <= OfflineNeuRegLimV2.LEAK_TAU_MAX
    ):
        raise UnsupportedConfig(
            Constraint.POW2_SCALE,
            f"shift 2**{exponent} leaves the leak_tau range "
            f"[{OfflineNeuRegLimV2.LEAK_TAU_MIN}, {OfflineNeuRegLimV2.LEAK_TAU_MAX}].",
        )
