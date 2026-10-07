import pytest
import torch

from paibox.quant.hw import Constraint, UnsupportedConfig
from paibox.quant.spec import (
    AnnActivation,
    Granularity,
    LayerQuant,
    QuantSpec,
    SnnActivation,
    WeightQuant,
)


def _layer(**overrides) -> LayerQuant:
    kwargs = {
        "input_scale": 1 / 255,
        "output_scale": 0.005,
        "weight": WeightQuant(bits=4, scale=0.004),
        "activation": AnnActivation(bits=8, max_val=1.5),
    }
    kwargs.update(overrides)
    return LayerQuant(**kwargs)


@pytest.mark.parametrize("bits", [3, 5, 6, 16, 32])
def test_weight_width_must_be_selectable(bits):
    with pytest.raises(UnsupportedConfig) as exc:
        WeightQuant(bits=bits, scale=1.0)

    assert exc.value.constraint is Constraint.BIT_WIDTH


def test_per_group_requires_a_group_size():
    with pytest.raises(ValueError, match="group_size"):
        WeightQuant(bits=4, scale=torch.ones(4), granularity=Granularity.PER_GROUP)

    WeightQuant(
        bits=4, scale=torch.ones(4), granularity=Granularity.PER_GROUP, group_size=128
    )


def test_group_size_is_rejected_for_other_granularities():
    with pytest.raises(ValueError, match="meaningless"):
        WeightQuant(bits=4, scale=1.0, group_size=128)


def test_ann_activation_needs_levels_or_max_val():
    with pytest.raises(ValueError, match="levels or max_val"):
        AnnActivation(bits=4)

    AnnActivation(bits=4, max_val=1.0)
    AnnActivation(bits=4, levels=[0.0, 0.5, 1.0])


@pytest.mark.parametrize("bits", [3, 16])
def test_ann_output_width_names_the_output_constraint(bits):
    with pytest.raises(UnsupportedConfig) as exc:
        AnnActivation(bits=bits, max_val=1.0)

    assert exc.value.constraint is Constraint.OUTPUT_WIDTH


@pytest.mark.parametrize("tau", [1.0, 2.0, 4.0, 8.0, 16.0])
def test_power_of_two_tau_is_accepted(tau):
    assert SnnActivation(v_threshold=1.0, tau=tau).tau == tau


@pytest.mark.parametrize("tau", [1.5, 3.0, 5.0, 7.0])
def test_non_power_of_two_tau_names_the_shift_constraint(tau):
    """The only multiplicative leak is ``leak_tau``, which is a shift."""
    with pytest.raises(UnsupportedConfig) as exc:
        SnnActivation(v_threshold=1.0, tau=tau)

    assert exc.value.constraint is Constraint.POW2_SCALE
    assert "power of two" in str(exc.value)


def test_tau_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        SnnActivation(v_threshold=1.0, tau=0.5)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="v_threshold"):
        SnnActivation(v_threshold=0.0)


def test_soft_reset_is_expressed_as_none():
    assert SnnActivation(v_threshold=1.0, v_reset=None).v_reset is None


@pytest.mark.parametrize("name", ["input_scale", "output_scale"])
def test_scales_must_be_positive(name):
    with pytest.raises(ValueError, match=name):
        _layer(**{name: 0.0})


def test_accumulator_scale_is_the_product_of_both_scales():
    layer = _layer(input_scale=0.5, weight=WeightQuant(bits=8, scale=0.25))
    assert layer.accumulator_scale == pytest.approx(0.125)


def test_accumulator_scale_is_undefined_for_non_scalar_weight_scale():
    layer = _layer(
        weight=WeightQuant(
            bits=4, scale=torch.ones(8), granularity=Granularity.PER_CHANNEL
        )
    )
    with pytest.raises(ValueError, match="per-tensor"):
        layer.accumulator_scale


def test_parameter_free_site_falls_back_to_the_input_scale():
    """Pooling carries codes through without a weight scale."""
    layer = _layer(weight=None, input_scale=0.02)
    assert layer.accumulator_scale == pytest.approx(0.02)


def test_per_channel_needs_a_tensor_of_scales():
    with pytest.raises(ValueError, match="per-channel scales"):
        _layer(
            weight=WeightQuant(bits=4, scale=0.004, granularity=Granularity.PER_CHANNEL)
        )


def test_is_spiking_distinguishes_the_activation_kind():
    assert _layer(activation=SnnActivation(v_threshold=1.0)).is_spiking
    assert not _layer().is_spiking
    assert not _layer(activation=None).is_spiking


def test_spec_reports_whether_any_layer_spikes():
    spec = QuantSpec(input_bits=8, input_scale=1 / 255)
    spec.layers["0"] = _layer()
    assert not spec.has_spiking_layers

    spec.layers["1"] = _layer(activation=SnnActivation(v_threshold=1.0))
    assert spec.has_spiking_layers


def test_timesteps_are_network_wide_and_at_least_one():
    assert QuantSpec(input_bits=8, input_scale=0.1, timesteps=8).timesteps == 8

    with pytest.raises(ValueError, match="timesteps"):
        QuantSpec(input_bits=8, input_scale=0.1, timesteps=0)


@pytest.mark.parametrize("bits", [3, 16])
def test_spec_input_width_must_be_selectable(bits):
    with pytest.raises(UnsupportedConfig) as exc:
        QuantSpec(input_bits=bits, input_scale=0.1)

    assert exc.value.constraint is Constraint.BIT_WIDTH


def test_scale_chain_accepts_matching_neighbours():
    spec = QuantSpec(input_bits=8, input_scale=1 / 255)
    spec.layers["a"] = _layer(input_scale=1 / 255, output_scale=0.01)
    spec.layers["b"] = _layer(input_scale=0.01, output_scale=0.2)

    spec.check_scale_chain(["a", "b"])


def test_scale_chain_skips_scale_preserving_modules():
    """Pooling and flattening sit between quantized sites without a spec entry."""
    spec = QuantSpec(input_bits=8, input_scale=1 / 255)
    spec.layers["conv"] = _layer(input_scale=1 / 255, output_scale=0.01)
    spec.layers["fc"] = _layer(input_scale=0.01, output_scale=0.2)

    spec.check_scale_chain(["conv", "pool", "flatten", "fc"])


def test_scale_chain_break_names_both_sites():
    """A silent mismatch would rescale activations without any error."""
    spec = QuantSpec(input_bits=8, input_scale=1 / 255)
    spec.layers["a"] = _layer(input_scale=1 / 255, output_scale=0.01)
    spec.layers["b"] = _layer(input_scale=0.9, output_scale=0.2)

    with pytest.raises(ValueError, match="scale chain breaks between 'a' and 'b'"):
        spec.check_scale_chain(["a", "b"])
