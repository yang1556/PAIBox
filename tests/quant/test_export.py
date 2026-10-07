import pytest
import torch
from torch import nn

from paibox.paiir.ir.core_neuron import ANNNodeV25, IFNodeV25, LIFNodeV25
from paibox.quant.export import export
from paibox.quant.hw import Constraint, UnsupportedConfig
from paibox.quant.spec import (
    AnnActivation,
    Granularity,
    LayerQuant,
    QuantSpec,
    SnnActivation,
    WeightQuant,
)

INPUT_SCALE = 1 / 255


def _model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 14 * 14, 10, bias=False),
    ).eval()


def _spec(**layer_overrides) -> QuantSpec:
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE, timesteps=8)
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=AnnActivation(bits=8, max_val=2.0),
    )
    spec.layers["4"] = LayerQuant(
        input_scale=0.01,
        output_scale=0.2,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=AnnActivation(bits=8, signed=True, max_val=20.0),
        **layer_overrides,
    )
    return spec


def test_weights_become_integers_in_a_standard_module():
    """A quantizer wrapper here would block comp->act fusion and cost a core."""
    deployed = export(_model(), _spec())

    conv = deployed[0]
    assert isinstance(conv, nn.Conv2d)
    assert torch.equal(conv.weight, conv.weight.round())
    assert conv.weight.abs().max() <= 127


def test_float_activation_is_replaced_by_a_chip_neuron():
    """Lowering maps a bare ReLU onto a LUT spanning all of int32."""
    deployed = export(_model(), _spec())

    assert not isinstance(deployed[1], nn.ReLU)
    assert isinstance(deployed[1], ANNNodeV25)


def test_trailing_compute_site_gets_its_activation_appended():
    """A float classifier ends at the linear layer; the chip still emits codes."""
    original = _model()
    deployed = export(original, _spec())

    assert len(deployed) == len(original) + 1
    assert isinstance(deployed[-1], ANNNodeV25)


def test_scale_preserving_modules_are_left_alone():
    deployed = export(_model(), _spec())

    assert isinstance(deployed[2], nn.MaxPool2d)
    assert isinstance(deployed[3], nn.Flatten)


def test_export_copies_by_default():
    original = _model()
    before = original[0].weight.detach().clone()

    export(original, _spec())

    assert torch.equal(original[0].weight, before)
    assert isinstance(original[1], nn.ReLU)


def test_inplace_export_mutates_the_original():
    original = _model()
    export(original, _spec(), inplace=True)

    assert isinstance(original[1], ANNNodeV25)


def test_explicit_activation_path_is_honoured():
    model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Identity(), nn.ReLU()).eval()
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE)
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=4, scale=0.01),
        activation=AnnActivation(bits=4, max_val=1.0),
        activation_path="2",
    )

    deployed = export(model, spec)

    assert isinstance(deployed[1], nn.Identity)
    assert isinstance(deployed[2], ANNNodeV25)


def test_unknown_module_path_is_reported():
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE)
    spec.layers["nope"] = LayerQuant(input_scale=INPUT_SCALE, output_scale=0.1)

    with pytest.raises(KeyError, match="nope"):
        export(_model(), spec)


def test_missing_activation_path_module_is_reported():
    spec = _spec(activation_path="does.not.exist")

    with pytest.raises(KeyError, match="does.not.exist"):
        export(_model(), spec)


def test_power_of_two_per_channel_scales_are_accepted():
    """The per-channel shift is ``leak_tau``, so ratios must be powers of two."""
    scales = torch.tensor([0.004, 0.008, 0.016, 0.004])
    spec = _spec()
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=8, scale=scales, granularity=Granularity.PER_CHANNEL),
        activation=AnnActivation(bits=8, max_val=2.0),
    )

    deployed = export(_model(), spec)
    assert torch.equal(deployed[0].weight, deployed[0].weight.round())


def test_non_power_of_two_per_channel_scales_name_the_lut_constraint():
    """One LUT serves every neuron on the core, so arbitrary ratios cannot fit."""
    scales = torch.tensor([0.004, 0.005, 0.006, 0.007])
    spec = _spec()
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=8, scale=scales, granularity=Granularity.PER_CHANNEL),
        activation=AnnActivation(bits=8, max_val=2.0),
    )

    with pytest.raises(UnsupportedConfig) as exc:
        export(_model(), spec)

    assert exc.value.constraint is Constraint.ACT_RESCALE
    assert "per-tensor" in exc.value.suggestion


def test_unit_tau_becomes_an_integrate_and_fire_neuron():
    spec = _spec()
    spec.layers["4"] = LayerQuant(
        input_scale=0.01,
        output_scale=1.0,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=SnnActivation(v_threshold=1.0, tau=1.0),
    )

    deployed = export(_model(), spec)
    assert isinstance(deployed[-1], IFNodeV25)


def test_decaying_tau_becomes_a_leaky_neuron():
    spec = _spec()
    spec.layers["4"] = LayerQuant(
        input_scale=0.01,
        output_scale=1.0,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=SnnActivation(v_threshold=1.0, tau=4.0),
    )

    deployed = export(_model(), spec)
    assert isinstance(deployed[-1], LIFNodeV25)


def test_threshold_is_converted_into_accumulator_steps():
    """The chip integrates in steps worth ``weight_scale * input_scale``."""
    weight_scale, input_scale, threshold = 0.004, 0.01, 1.0
    spec = _spec()
    spec.layers["4"] = LayerQuant(
        input_scale=input_scale,
        output_scale=1.0,
        weight=WeightQuant(bits=8, scale=weight_scale),
        activation=SnnActivation(v_threshold=threshold, tau=1.0),
    )

    deployed = export(_model(), spec)
    expected = round(threshold / (weight_scale * input_scale))
    assert float(deployed[-1].thres_pos) == pytest.approx(expected)


def test_threshold_under_one_accumulator_step_is_refused():
    """Such a neuron fires on every step, carrying no information."""
    spec = _spec()
    spec.layers["4"] = LayerQuant(
        input_scale=1.0,
        output_scale=1.0,
        weight=WeightQuant(bits=8, scale=1.0),
        activation=SnnActivation(v_threshold=1e-6, tau=1.0),
    )

    with pytest.raises(UnsupportedConfig) as exc:
        export(_model(), spec)

    assert "accumulator step" in str(exc.value)


def test_soft_reset_survives_as_none():
    spec = _spec()
    spec.layers["4"] = LayerQuant(
        input_scale=0.01,
        output_scale=1.0,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=SnnActivation(v_threshold=1.0, v_reset=None, tau=1.0),
    )

    export(_model(), spec)  # reaching here means the reset mode was accepted


def test_bias_outside_leak_v_range_is_refused():
    model = nn.Sequential(nn.Linear(8, 4, bias=True), nn.ReLU()).eval()
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE)
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=8, scale=0.01),
        activation=AnnActivation(bits=8, max_val=1.0),
        bias=torch.tensor([0.0, 0.0, 0.0, 1e9]),
    )

    with pytest.raises(UnsupportedConfig) as exc:
        export(model, spec)

    assert exc.value.constraint is Constraint.BIAS_RANGE


def test_non_uniform_levels_are_accepted():
    """Free here because LUT thresholds are placed independently."""
    spec = _spec()
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(bits=8, scale=0.004),
        activation=AnnActivation(bits=4, levels=[0.0, 0.1, 0.3, 0.7, 1.5]),
    )

    deployed = export(_model(), spec)
    assert isinstance(deployed[1], ANNNodeV25)


def test_per_group_is_refused_rather_than_misapplied():
    """A group scale divides along the reduction axis, not the output axis.

    Left unchecked, a spec whose group count happens to equal the output channel
    count would broadcast group scales over the wrong axis and quietly produce
    wrong weights.
    """
    model = nn.Sequential(nn.Linear(8, 2, bias=False), nn.ReLU()).eval()
    spec = QuantSpec(input_bits=8, input_scale=INPUT_SCALE)
    spec.layers["0"] = LayerQuant(
        input_scale=INPUT_SCALE,
        output_scale=0.01,
        weight=WeightQuant(
            bits=4,
            scale=torch.tensor([0.01, 0.02]),
            granularity=Granularity.PER_GROUP,
            group_size=4,
        ),
        activation=AnnActivation(bits=4, max_val=1.0),
    )

    with pytest.raises(UnsupportedConfig) as exc:
        export(model, spec)

    assert exc.value.constraint is Constraint.ACT_RESCALE
    assert "reduction axis" in str(exc.value)
