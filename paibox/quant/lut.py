"""Fitting the offline core's 256-entry activation LUT.

The LUT is the only place on an offline core where an arbitrary mapping can be
applied, so it has to absorb three things at once: requantization of the 32-bit
accumulator, the activation function, and quantization of the output code.
There is no per-channel multiplier after accumulation, and the table is a
*per-core* resource shared by every neuron placed on that core.

Placing a threshold at the accumulator value where the output code steps makes
the hardware's successive-approximation lookup reproduce software
round-to-nearest exactly, which is what the activation-function design targets
at no more than half an LSB of error.

Because the 256 thresholds may be spaced freely, a non-uniform output grid — a
logarithmic or quantile grid, or one that was learned — costs exactly what a
uniform one costs. On a GPU the same choice would need a lookup per element, so
mainstream quantization tools do not offer it; here it is free and unused.

The only ordering requirement is that thresholds increase; the activation
*values* may be non-monotone, which is how non-monotone functions such as GeLU
or Mish are supported [activation-function design doc].
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from paibox.paiir.ir.lut_activation import LUT_TABLE_SIZE, LutCustom

from .hw import Constraint, UnsupportedConfig

__all__ = [
    "code_range",
    "fit_affine_lut",
    "fit_lut",
    "log_levels",
    "quantile_levels",
    "uniform_levels",
    "verify_lut",
]

#: Thresholds below this stand in for "everything more negative", covering the
#: whole negative side of a rectifying activation in one bin.
_FLOOR_THRESHOLD = -(2**30)


def code_range(bits: int, signed: bool) -> tuple[int, int]:
    """Inclusive output code range for a width and signedness."""
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _assemble(thresholds: list[int], values: list[int], signed: bool) -> LutCustom:
    """Pad to the fixed table size and wrap as a ``LutCustom``.

    Padding keeps thresholds strictly increasing past the last real step so the
    unused tail can never be selected, and repeats the final value so a code
    that does land there saturates rather than wrapping.
    """
    if len(thresholds) > LUT_TABLE_SIZE:
        raise UnsupportedConfig(
            Constraint.OUTPUT_WIDTH,
            f"{len(thresholds)} output levels exceed the {LUT_TABLE_SIZE}-entry LUT.",
            "reduce the output width",
        )

    thresholds = list(thresholds)
    values = list(values)
    while len(thresholds) < LUT_TABLE_SIZE:
        thresholds.append(thresholds[-1] + 1)
        values.append(values[-1])

    return LutCustom(
        torch.tensor(thresholds, dtype=torch.int32),
        torch.tensor(values, dtype=torch.int32),
        output_signed=signed,
    )


def fit_affine_lut(ratio: float, bits: int, signed: bool = False) -> LutCustom:
    """LUT for ``clamp(round(acc * ratio), lo, hi)`` over the accumulator.

    This is the uniform case. With per-tensor symmetric weights,
    ``ratio = s_weight * s_input / s_output``: the accumulator is in units of
    ``s_weight * s_input``, and the output code is in units of ``s_output``.

    Rounding is half-*up*, because the core decides a code by comparing
    ``acc >= threshold``. ``torch.round`` rounds halves to even, so the two
    disagree on values exactly between codes. Ties are reachable whenever the
    scale ratio is a power of two, which is the regime this toolchain prefers,
    so a host-side reference implementation should round half up to match.

    Args:
        ratio: Accumulator-to-output-code scale.
        bits: Output code width.
        signed: Signed output codes. Needed for a layer whose result feeds an
            ``argmax``, since rectifying negative logits to zero destroys it.
    """
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError(f"scale ratio must be finite and positive, got {ratio}")

    lo, hi = code_range(bits, signed)
    thresholds = [_FLOOR_THRESHOLD]
    values = [lo]
    for code in range(lo + 1, hi + 1):
        # The code steps where round(acc * ratio) crosses code - 0.5.
        edge = math.ceil((code - 0.5) / ratio)
        if edge <= thresholds[-1]:
            # A ratio this large makes several codes share one accumulator
            # value; the surplus codes are unreachable but must not break the
            # ordering requirement.
            edge = thresholds[-1] + 1
        thresholds.append(edge)
        values.append(code)

    return _assemble(thresholds, values, signed)


def fit_lut(
    levels: Sequence[float], acc_scale: float, signed: bool = False
) -> LutCustom:
    """LUT mapping the accumulator onto an arbitrary set of output levels.

    Each level gets the code equal to its index (offset so signed ranges start
    at their minimum), and the decision boundary between neighbouring levels is
    their midpoint. Dequantization is the caller's table of ``levels``.

    Args:
        levels: Output levels in the real-valued domain, strictly increasing.
        acc_scale: Real value per accumulator unit, i.e. ``s_weight * s_input``.
        signed: Emit signed codes.
    """
    if len(levels) < 2:
        raise ValueError(f"need at least two levels, got {len(levels)}")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly increasing")
    if acc_scale <= 0 or not math.isfinite(acc_scale):
        raise ValueError(f"acc_scale must be finite and positive, got {acc_scale}")

    lo = -(len(levels) // 2) if signed else 0
    thresholds = [_FLOOR_THRESHOLD]
    values = [lo]
    for i in range(1, len(levels)):
        midpoint = 0.5 * (levels[i - 1] + levels[i])
        edge = math.ceil(midpoint / acc_scale)
        if edge <= thresholds[-1]:
            edge = thresholds[-1] + 1
        thresholds.append(edge)
        values.append(lo + i)

    return _assemble(thresholds, values, signed)


def uniform_levels(max_val: float, bits: int, signed: bool = False) -> list[float]:
    """Evenly spaced levels, the grid a conventional quantizer would use."""
    lo, hi = code_range(bits, signed)
    step = max_val / hi if hi > 0 else 0.0
    return [step * code for code in range(lo, hi + 1)]


def log_levels(max_val: float, bits: int, min_ratio: float = 1e-3) -> list[float]:
    """Logarithmic levels over ``[0, max_val]``, denser near zero.

    Useful where activation mass concentrates near zero, as after a rectifier.
    Free here because thresholds are independently placeable; on a GPU this
    would need a per-element lookup.
    """
    n = 2**bits
    smallest = max_val * min_ratio
    return [0.0] + [
        smallest * (max_val / smallest) ** (i / (n - 2)) for i in range(n - 1)
    ]


def quantile_levels(samples: Tensor, bits: int, signed: bool = False) -> list[float]:
    """Levels at equal-mass quantiles of observed activations.

    Spends resolution where the data is rather than where the range is, which
    matters most at 2- and 4-bit widths.
    """
    lo, hi = code_range(bits, signed)
    n = hi - lo + 1
    flat = samples.detach().flatten().float()
    qs = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    levels = torch.quantile(flat.double(), qs).tolist()

    # Quantiles collapse onto each other wherever the distribution has an atom,
    # most often at zero after a rectifier; nudge them apart to stay ordered.
    for i in range(1, len(levels)):
        if levels[i] <= levels[i - 1]:
            levels[i] = math.nextafter(levels[i - 1], math.inf)
    return levels


def verify_lut(lut: LutCustom, expected: Sequence[int], accumulators: Tensor) -> None:
    """Check a fitted table against the mapping it is meant to implement.

    The chip performs a successive-approximation lookup rather than a linear
    scan, and ``LutActivation`` separately proves the two agree at integer
    boundaries; this checks the logical table itself, which is where a fitting
    bug would show up.
    """
    got = lut(accumulators).flatten().tolist()
    want = list(expected)
    if got != want:
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise AssertionError(
            f"LUT disagrees at accumulator {int(accumulators.flatten()[first])}: "
            f"got {got[first]}, expected {want[first]}"
        )
