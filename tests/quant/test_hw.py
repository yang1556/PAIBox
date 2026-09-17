import pytest
from paicorelib import LCN_EX, OfflineNeuRegLimV2

from paibox.quant.hw import (
    BIT_WIDTHS,
    FANIN_BASE,
    SRAM_LINES_PER_CORE,
    Constraint,
    CoreDemand,
    NeuronKind,
    UnsupportedConfig,
    WeightFormat,
    check_bias,
    check_initial_potential,
    check_shift,
    core_count,
    csc_breakeven_sparsity,
    data_width_of,
    lcn_for_axons,
    neuron_sram_lines,
    stream_capacity,
    weight_sram_lines,
    weight_width_of,
)


@pytest.mark.parametrize(
    ("n_inputs", "input_bits", "expected"),
    [
        (512, 1, LCN_EX.LCN_1X),
        (512, 8, LCN_EX.LCN_8X),
        (513, 8, LCN_EX.LCN_16X),  # 4104 addresses spills past 8x
        (1568, 8, LCN_EX.LCN_32X),
        (8192, 8, LCN_EX.LCN_128X),
        (65536, 1, LCN_EX.LCN_128X),
    ],
)
def test_lcn_matches_backend_derivation(n_inputs, input_bits, expected):
    """Must agree with ``RoutingGroup.set_lcn`` or the front end mispredicts cost."""
    assert lcn_for_axons(n_inputs, input_bits) is expected


def test_lcn_rejects_fanin_beyond_max_with_alternative():
    with pytest.raises(UnsupportedConfig) as exc:
        lcn_for_axons(100_000, 8)

    assert exc.value.constraint is Constraint.FANIN_LIMIT
    assert "AddPotentialMode.DIRECT_ADD" in exc.value.suggestion


def test_stream_capacity_trades_against_fanin():
    """The 8-bit timestep field is shared with the slot index."""
    assert stream_capacity(LCN_EX.LCN_1X) == 256
    assert stream_capacity(LCN_EX.LCN_128X) == 2


@pytest.mark.parametrize(
    ("bits", "breakeven"),
    [(1, 7 / 128), (2, 7 / 64), (4, 6 / 32), (8, 5 / 16)],
)
def test_csc_only_pays_off_when_very_sparse(bits, breakeven):
    """Low-bit quantization and sparse compression compete rather than compose."""
    assert csc_breakeven_sparsity(bits) == pytest.approx(breakeven)


def test_csc_beats_dense_exactly_below_breakeven():
    n = 12_800
    for bits in BIT_WIDTHS:
        s = csc_breakeven_sparsity(bits)
        dense = weight_sram_lines(n, bits, WeightFormat.DENSE)
        assert weight_sram_lines(n, bits, WeightFormat.CSC, s * 0.5) < dense
        assert weight_sram_lines(n, bits, WeightFormat.CSC, min(s * 2, 1.0)) > dense


def test_per_channel_costs_eight_times_folded_neuron_sram():
    """Per-neuron parameters force the full tier, which is the per-channel penalty."""
    n = 4096
    assert neuron_sram_lines(n, NeuronKind.FULL) == 8 * neuron_sram_lines(
        n, NeuronKind.FOLDED
    )
    assert neuron_sram_lines(n, NeuronKind.HALF) == n


def test_core_count_is_ceiling_over_shared_sram():
    assert core_count(CoreDemand(0, 0)) == 0
    assert core_count(CoreDemand(1, 0)) == 1
    assert core_count(CoreDemand(SRAM_LINES_PER_CORE, 0)) == 1
    assert core_count(CoreDemand(SRAM_LINES_PER_CORE, 1)) == 2


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_selectable_widths_round_trip(bits):
    assert data_width_of(bits) is not None
    assert weight_width_of(bits) is not None


@pytest.mark.parametrize("bits", [3, 5, 6, 16])
def test_unselectable_width_names_the_constraint(bits):
    with pytest.raises(UnsupportedConfig) as exc:
        weight_width_of(bits)

    assert exc.value.constraint is Constraint.BIT_WIDTH


def test_weight_lines_track_dense_packing():
    """128-bit lines hold 128/64/32/16 weights at 1/2/4/8 bits."""
    assert weight_sram_lines(128, 1) == 1
    assert weight_sram_lines(16, 8) == 1
    assert weight_sram_lines(17, 8) == 2
    assert weight_sram_lines(3136, 4) == 98


def test_bias_beyond_leak_v_is_rejected():
    check_bias(OfflineNeuRegLimV2.LEAK_V_MAX)

    with pytest.raises(UnsupportedConfig) as exc:
        check_bias(OfflineNeuRegLimV2.LEAK_V_MAX + 1)

    assert exc.value.constraint is Constraint.BIAS_RANGE


def test_initial_potential_conflicts_with_csc_acceleration():
    """``csc_accelerate`` reuses ``vjt_initial`` for ``weight_address_start``."""
    check_initial_potential(0, csc_accelerate=True)
    check_initial_potential(5, csc_accelerate=False)

    with pytest.raises(UnsupportedConfig) as exc:
        check_initial_potential(5, csc_accelerate=True)

    assert exc.value.constraint is Constraint.INITIAL_POTENTIAL


def test_shift_range_follows_leak_tau():
    check_shift(OfflineNeuRegLimV2.LEAK_TAU_MAX)
    check_shift(OfflineNeuRegLimV2.LEAK_TAU_MIN)

    with pytest.raises(UnsupportedConfig) as exc:
        check_shift(OfflineNeuRegLimV2.LEAK_TAU_MAX + 1)

    assert exc.value.constraint is Constraint.POW2_SCALE


def test_fanin_base_matches_backend_constant():
    from paibox.backendv2.routing import FANIN_BASE as backend_fanin

    assert FANIN_BASE == backend_fanin
