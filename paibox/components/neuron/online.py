import warnings
from collections.abc import Iterable
from typing import Any, Literal, NoReturn, Optional, Union

import numpy as np

from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    TM,
    CoreMode,
    HwConfig,
    InputWidthFormat,
    MaxPoolingEnable,
    SNNModeEnable,
    SpikeWidthFormat,
    get_core_mode,
)

from paibox.base import DataFlowFormat, NeuDyn
from paibox.types import (
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    LeakVType,
    NeuOutType,
    Shape,
    VoltageType,
)
from paibox.utils import (
    arg_check_non_neg,
    arg_check_non_pos,
    arg_check_pos,
    as_shape,
    shape2num,
)
from paibox.exceptions import ConfigInvalidError, PAIBoxWarning, ShapeError
from .utils import (
    BIT_TRUNCATE_MAX,
    NEG_THRES_MIN,
    _input_width_format,
    _leak_v_check,
    _spike_width_format,
    vjt_overflow,
)

L = Literal
NEU_TARGET_CHIP_NOT_SET = -1


class OnlineBaseNeuron:
    """Meta neuron"""

    # rt_mode_kwds: RTModeKwds
    # mode: CoreMode

    def __init__(
        self,
        shape: Shape,

        leakage_reg: Union[int, LeakVType], #leakage_reg neu
        threshold_reg: int, #neu
        reset_potential_reg: int, # neu
        floor_threshold_reg: int, #neu
        
        
        bit_select: SpikeWidthFormat, # core
        lateral_inhi_value: int, # core
        weight_decay_value: int, # core
        upper_weight: int, # core
        lower_weight: int, # core


        LUT_random_en: list, # core
        decay_random_en: bool, # core
        leakage_order: LCM, # core
        online_mode_en: bool, # core
        
        
        overflow_strict: bool,
        keep_shape: bool = False,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._shape = as_shape(shape)
        self._n_neuron = shape2num(self._shape)

        self.bit_select = bit_select
    

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self.reset_v = reset_potential_reg  # Signed 6-bit
        self.leak_comparison = leakage_order
        #self.leak_v = leakage_reg  # 15-bit
        self.neg_threshold = floor_threshold_reg  # 7 bit
        self.pos_threshold = threshold_reg  # 15-bit

        self.upper_weight = upper_weight  # 8-bit
        self.lower_weight = lower_weight  # 8-bit
        self.lateral_inhi_value = lateral_inhi_value  # 32-bit
        self.weight_decay_value = weight_decay_value  # 8-bit

        self.online_mode_en = online_mode_en  # Enable online mode.
        self.decay_random_en = decay_random_en  # Enable random decay.
        self.LUT_random_en = LUT_random_en  # Enable random LUT.
        # Auxiliary attributes or variables.
        self.overflow_strict = overflow_strict

        if isinstance(leakage_reg, int) or leakage_reg.size == 1:
            # np.array([x]) is treated as a scalar.
            self.leak_v = int(leakage_reg)
        elif np.prod(leakage_reg.shape) == np.prod(self._shape):
            # leak with shape (32,32) == (1,32,32) is allowed.
            self.leak_v = leakage_reg.ravel()
        elif leakage_reg.ndim == 1 and leakage_reg.shape[0] == self._shape[0]:
            self.leak_v = np.repeat(leakage_reg, shape2num(self._shape[1:])).ravel()
        else:
            raise ShapeError(
                f"'leak' is either a scalar or have shape (output channels, ), but got ({self._shape[0]},)."
            )

        _leak_v_check(self.leak_v)

    def init_param(self, param: Any) -> np.ndarray:
        return np.full((self._n_neuron,), param)


    @property
    def _vjt0(self) -> VoltageType:
        return self.init_param(0).astype(VOLTAGE_DTYPE)

    @property
    def _neu_out0(self) -> NeuOutType:
        return self.init_param(0).astype(NEUOUT_U8_DTYPE)

    @property
    def varshape(self) -> tuple[int, ...]:
        return self._shape if self.keep_shape else (self._n_neuron,)
        


class OnlineNeuron(OnlineBaseNeuron, NeuDyn):
    _n_copied = 0
    """Counter of copies."""

    def __init__(
        self,
        shape: Shape,
        reset_v: int = 0, # ram
        leak_comparison: LCM = LCM.LEAK_BEFORE_COMP, #ram leak_post
        neg_threshold: Optional[int] = None, #ram
        pos_threshold: int = 1, #ram
        leak_v: Union[int, LeakVType] = 0, #ram
        lateral_inhi_value: int = 0, #core
        weight_decay_value: int = 0, #core
        upper_weight: int = 127, #ram
        lower_weight: int = 0, #ram
        LUT: list = [], #reg    
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        spike_width: Union[L[1, 8], SpikeWidthFormat] = SpikeWidthFormat.WIDTH_1BIT, #reg
        online_mode: bool = True, #core
        decay_random_en: bool = True, #core
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        target_chip: int = NEU_TARGET_CHIP_NOT_SET,
        name: Optional[str] = None,
    ) -> None:
        if neg_threshold is None:
            neg_threshold = NEG_THRES_MIN

        if neg_threshold > 0:
            # XXX *(-1) if passing a negative threshold > 0
            neg_threshold = (-1) * neg_threshold


        super().__init__(
            shape,
            leak_v,
            arg_check_non_pos(neg_threshold, "negative threshold"),
            reset_v,
            arg_check_non_neg(pos_threshold, "positive threshold"),
            _spike_width_format(spike_width),
            lateral_inhi_value,
            weight_decay_value,
            upper_weight,
            lower_weight,
            LUT,
            decay_random_en,
            leak_comparison,
            online_mode,
            overflow_strict,
            keep_shape,
        )
        super(OnlineBaseNeuron, self).__init__(name)

        """Stateful attributes. Vector."""
        self.set_memory("_vjt", self._vjt0)  # Initial vjt is fixed at 0.
        self.set_memory("_neu_out", self._neu_out0)
        self.set_memory(
            "delay_registers",
            np.zeros(
                (HwConfig.N_TIMESLOT_MAX,) + self._neu_out.shape, dtype=NEUOUT_U8_DTYPE
            ),
        )

        """Non-stateful attributes."""
        self._delay = arg_check_pos(delay, "'delay'")
        self._tws = arg_check_non_neg(tick_wait_start, "'tick_wait_start'")
        self._twe = arg_check_non_neg(tick_wait_end, "'tick_wait_end'")
        self._uf = arg_check_pos(unrolling_factor, "'unrolling_factor'")
        self.target_chip_idx = target_chip
        # Default dataflow is infinite and continuous, starting at tws+0.
        self._oflow_format = DataFlowFormat(0, is_local_time=True)

    def __len__(self) -> int:
        return self._n_neuron

    def __call__(
        self, x: Optional[np.ndarray] = None, *args, **kwargs
    ) -> Optional[NeuOutType]:
        return None
    
    def update(self):
        return None
    

    def attrs(self, all: bool) -> dict[str, Any]:
        attrs = {
            "reset_v": self.reset_v,
            "leak_comparison": self.leak_comparison,
            "neg_threshold": self.neg_threshold,
            "pos_threshold": self.pos_threshold,
            "leak_v": self.leak_v,
            "lateral_inhi_value": self.lateral_inhi_value,
            "weight_decay_value": self.weight_decay_value,
            "upper_weight": self.upper_weight,
            "lower_weight": self.lower_weight,
            "LUT": self.LUT_random_en,
            "decay_random_en": self.decay_random_en,
            "online_mode_en": self.online_mode_en,
            "bit_select": self.bit_select}

        if all:
            attrs |= {
                "shape": self._shape,
                "keep_shape": self.keep_shape,
                "delay": self.delay_relative,
                "tick_wait_start": self.tick_wait_start,
                "tick_wait_end": self.tick_wait_end,
                "unrolling_factor": self.unrolling_factor,
                "overflow_strict": self.overflow_strict,
            }

        return attrs
    
    @property
    def shape_in(self) -> tuple[int, ...]:
        return self._shape

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape

    @property
    def num_in(self) -> int:
        return self._n_neuron

    @property
    def num_out(self) -> int:
        return self._n_neuron

    @property
    def output(self) -> NeuOutType:
        return self._neu_out

    @property
    def spike(self) -> NeuOutType:
        return self._neu_out

    @property
    def feature_map(self) -> NeuOutType:
        return self._neu_out.reshape(self.varshape)

    @property
    def voltage(self) -> VoltageType:
        return self._vjt.reshape(self.varshape)