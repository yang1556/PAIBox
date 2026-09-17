"""Hardware-aware quantization for PAICORE 2.5 offline cores.

Fills the stage the toolchain currently leaves to the user: ``compile_to_paiir``
expects a model whose integer semantics are already materialised on modules its
lowering recognises, and getting there by hand is where most deployment effort
goes today.

Two layers, kept separate on purpose:

* :mod:`paibox.quant.hw` is the capability surface. It decides whether a
  configuration is expressible and reports what it costs, without judging
  whether it is a good idea.
* the search layer picks among expressible configurations. A choice the user
  pins is simply a constraint handed to that layer, not a separate code path,
  so an explicitly requested configuration is always honoured even when it
  wastes cores.
"""

from .hw import (
    BIAS_RANGE,
    BIT_WIDTHS,
    Constraint,
    NeuronKind,
    UnsupportedConfig,
    WeightFormat,
    check_bias,
)
from .export import export
from .lut import fit_affine_lut, fit_lut
from .spec import (
    AnnActivation,
    Granularity,
    LayerQuant,
    QuantSpec,
    SnnActivation,
    WeightQuant,
)

__all__ = [
    "BIAS_RANGE",
    "BIT_WIDTHS",
    "AnnActivation",
    "Constraint",
    "check_bias",
    "Granularity",
    "LayerQuant",
    "NeuronKind",
    "QuantSpec",
    "SnnActivation",
    "UnsupportedConfig",
    "WeightFormat",
    "WeightQuant",
    "export",
    "fit_affine_lut",
    "fit_lut",
]
