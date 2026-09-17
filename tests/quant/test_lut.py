import math

import pytest
import torch

from paibox.paiir.ir.lut_activation import LUT_TABLE_SIZE
from paibox.quant.hw import Constraint, UnsupportedConfig
from paibox.quant.lut import (
    code_range,
    fit_affine_lut,
    fit_lut,
    log_levels,
    quantile_levels,
    uniform_levels,
    verify_lut,
)

ACCUMULATORS = torch.tensor(
    [-10_000, -1, 0, 1, 100, 189, 200, 1_000, 5_000, 100_000], dtype=torch.int32
)


def _expected(accumulators, ratio, bits, signed=False):
    """Reference codes under the hardware's rounding rule.

    The core compares ``acc >= threshold``, so a value exactly halfway between
    two codes rounds *up*. ``torch.round`` rounds halves to even instead, which
    disagrees at exact ties — reachable precisely in the power-of-two-scale
    regime this toolchain prefers.
    """
    lo, hi = code_range(bits, signed)
    scaled = accumulators.double() * ratio
    half_up = torch.floor(scaled + 0.5)
    return half_up.clamp(lo, hi).int().tolist()


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [1e-4, 2.65e-3, 0.05, 1.0])
def test_affine_lut_reproduces_software_rounding(bits, ratio):
    """The whole point of placing thresholds at code steps."""
    lut = fit_affine_lut(ratio, bits)

    verify_lut(lut, _expected(ACCUMULATORS, ratio, bits), ACCUMULATORS)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_signed_affine_lut_keeps_negative_codes(bits):
    """A layer feeding ``argmax`` cannot have its negative logits rectified."""
    ratio = 0.01
    lut = fit_affine_lut(ratio, bits, signed=True)
    expected = _expected(ACCUMULATORS, ratio, bits, signed=True)

    verify_lut(lut, expected, ACCUMULATORS)
    assert min(expected) < 0


def test_lut_table_is_always_fully_populated():
    for bits in (1, 2, 4, 8):
        lut = fit_affine_lut(0.01, bits)
        assert lut.thresholds.numel() == LUT_TABLE_SIZE
        assert lut.lut_values.numel() == LUT_TABLE_SIZE


def test_thresholds_are_strictly_increasing():
    """The successive-approximation lookup requires ordered thresholds."""
    for ratio in (1e-5, 1.0, 50.0):
        thresholds = fit_affine_lut(ratio, 8).thresholds
        assert bool(torch.all(thresholds[1:] > thresholds[:-1]))


def test_huge_ratio_stays_ordered_when_codes_become_unreachable():
    """Several codes can share one accumulator value without breaking ordering."""
    lut = fit_affine_lut(100.0, 8)
    assert bool(torch.all(lut.thresholds[1:] > lut.thresholds[:-1]))


@pytest.mark.parametrize("ratio", [0.0, -1.0, math.inf, math.nan])
def test_affine_lut_rejects_degenerate_ratio(ratio):
    with pytest.raises(ValueError):
        fit_affine_lut(ratio, 8)


def test_uniform_levels_round_trip_through_fit_lut():
    """A uniform grid through the general path must match the affine path."""
    max_val, bits, acc_scale = 127.0, 4, 0.01
    levels = uniform_levels(max_val, bits)
    general = fit_lut(levels, acc_scale)

    _, hi = code_range(bits, signed=False)
    affine = fit_affine_lut(acc_scale * hi / max_val, bits)

    assert torch.equal(general.thresholds, affine.thresholds)
    assert torch.equal(general.lut_values, affine.lut_values)


def test_log_levels_are_denser_near_zero():
    """Where a rectifier concentrates activation mass."""
    levels = log_levels(127.0, 4)
    gaps = [b - a for a, b in zip(levels[1:], levels[2:])]

    assert levels[0] == 0.0
    assert all(b > a for a, b in zip(levels, levels[1:]))
    assert gaps[0] < gaps[-1]


def test_quantile_levels_survive_an_atom_at_zero():
    """Post-rectifier data is mostly exact zeros, which collapses quantiles."""
    samples = torch.cat([torch.zeros(8_000), torch.rand(2_000) * 10])
    levels = quantile_levels(samples, 4)

    assert len(levels) == 16
    assert all(b > a for a, b in zip(levels, levels[1:]))


def test_non_uniform_lut_is_usable_and_ordered():
    levels = log_levels(127.0, 4)
    lut = fit_lut(levels, acc_scale=0.005)

    assert bool(torch.all(lut.thresholds[1:] > lut.thresholds[:-1]))
    codes = lut(ACCUMULATORS).flatten()
    assert int(codes.min()) >= 0
    assert int(codes.max()) <= 15


def test_fit_lut_rejects_unordered_levels():
    with pytest.raises(ValueError):
        fit_lut([0.0, 2.0, 1.0], acc_scale=0.01)


def test_too_many_levels_names_the_output_width_constraint():
    with pytest.raises(UnsupportedConfig) as exc:
        fit_lut(list(range(LUT_TABLE_SIZE + 1)), acc_scale=1.0)

    assert exc.value.constraint is Constraint.OUTPUT_WIDTH


def test_verify_lut_reports_the_first_disagreement():
    lut = fit_affine_lut(0.01, 8)
    wrong = [999] * ACCUMULATORS.numel()

    with pytest.raises(AssertionError, match="disagrees at accumulator"):
        verify_lut(lut, wrong, ACCUMULATORS)


def test_ties_round_up_not_to_even():
    """Threshold comparison is ``>=``, so halfway values take the higher code."""
    ratio = 1e-4
    lut = fit_affine_lut(ratio, 8)
    exact_half = torch.tensor([5_000], dtype=torch.int32)  # 5000 * 1e-4 == 0.5

    assert int(lut(exact_half).flatten()[0]) == 1
    assert int(torch.tensor([0.5]).round()[0]) == 0
