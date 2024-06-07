import sys
from collections.abc import Sequence
from functools import partial
from typing import Literal, Optional, Union, ClassVar

import numpy as np
from numpy.typing import NDArray
from paicorelib import NTM, RM, TM

from paibox.base import NeuDyn, NodeList
from paibox.exceptions import PAIBoxDeprecationWarning, ShapeError
from paibox.network import DynSysGroup
from paibox.types import SpikeType, VoltageType, DataArrayType
from paibox.utils import arg_check_non_neg, as_shape, shape2num, typical_round

from .modules import (
    BuiltComponentType,
    FunctionalModule,
    FunctionalModule2to1,
    FunctionalModule2to1WithV,
    FunctionalModuleWithV,
    TransposeModule,
)
from .neuron import Neuron
from .neuron.neurons import *
from .neuron.utils import vjt_overflow
from .projection import InputProj
from .synapses import ConnType, FullConnSyn, Conv2dHalfRollSyn
from .synapses.conv_types import _Size2Type
from .synapses.conv_utils import _fm_ndim2_check, _pair
from .synapses.transforms import Conv2dForward, _Pool2dForward

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

__all__ = [
    "BitwiseAND",
    "BitwiseNOT",
    "BitwiseOR",
    "BitwiseXOR",
    "DelayChain",
    "SpikingAdd",
    "SpikingAvgPool2d",
    "SpikingAvgPool2dWithV",
    "SpikingMaxPool2d",
    "SpikingSub",
    "Transpose2d",
    "Transpose3d",
    "Conv_HalfRoll",
    "Filter",
    "Delay_FullConn"
]

_L_SADD = 1  # Literal value for spiking addition.
_L_SSUB = -1  # Literal value for spiking subtraction.


class BitwiseAND(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise AND module. Do a bitwise AND of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.
            - delay: delay between module & another module(or neuron). Default is 1.
            - tick_wait_start: set the moodule to start at timestep `T`. 0 means not working. Default is 1.
            - tick_wait_end: set the module to turn off at time `T`. 0 means always working. Default is 0.
            - unrolling_factor: argument related to the backend. It represents the degree to which modules  \
                are expanded. The larger the value, the more cores required for deployment, but the lower   \
                the latency & the higher the throughput. Default is 1.
            - keep_shape: whether to maintain size information when recording data in the simulation.       \
                Default is `False`.
            - name: name of the module. Optional.

        NOTE: the inherent delay of the module is 0. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+1.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 & x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_and = LIF(
            self.shape_out,
            threshold=1,
            leak_v=-1,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_and,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_and,
            1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_and, syn1, syn2]
        self._rebuild_out_intf(network, n1_and, *generated, **build_options)

        return generated


class BitwiseNOT(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise NOT module. Do a bitwise NOT of the output spike of one neuron & output.

        Args:
            - neuron: the operand.

        NOTE: the inherent delay of the module is 0.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return ~x1

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_not = LIF(
            self.shape_out,
            threshold=1,
            leak_v=1,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_not,
            weights=-1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )

        generated = [n1_not, syn1]
        self._rebuild_out_intf(network, n1_not, *generated, **build_options)

        return generated


class BitwiseOR(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise OR module. Do a bitwise OR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 | x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_or = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_or,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_or,
            1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_or, syn1, syn2]
        self._rebuild_out_intf(network, n1_or, *generated, **build_options)

        return generated


class BitwiseXOR(FunctionalModule2to1):
    inherent_delay = 1

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise XOR module. Do a bitwise XOR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 1. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+2.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 ^ x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        # If neuron_a is of shape (h1, w1) = N, and neuron_b is of shape (h2, w2) = N.
        # The output shape of the module is (N,) or (h1, w1)(if h1 == h2).
        # The shape of n1 is (2N,) or (2, h1, w1).
        n1_aux = SpikingRelu(
            (2,) + self.shape_out,
            delay=1,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=False,
            name=f"n0_{self.name}",
        )

        identity = np.identity(self.num_out, dtype=np.int8)
        # weight of syn1, (-1*(N,), 1*(N,))
        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_aux,
            weights=np.hstack([-1 * identity, identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )
        # weight of syn2, (1*(N,), -1*(N,))
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_aux,
            weights=np.hstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s1_{self.name}",
        )

        # The shape of n2 is (N,) or (h1, w1).
        n2_xor = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=n1_aux.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n1_{self.name}",
        )

        # weight of syn3, identity matrix with shape (2N, N)
        syn3 = FullConnSyn(
            n1_aux,
            n2_xor,
            weights=np.vstack([identity, identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s2_{self.name}",
        )

        generated = [n1_aux, n2_xor, syn1, syn2, syn3]
        self._rebuild_out_intf(network, n2_xor, *generated, **build_options)

        return generated


class DelayChain(FunctionalModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        chain_level: int = 1,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Delay chain. It will add extra neurons (and identity synapses) as buffer.

        Args:
            - neuron: the target neuron to be delayed.
            - chain_level: the level of delay chain.

        NOTE: the inherent delay of the module depends on `chain_level`.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        if chain_level < 1:
            raise ValueError(
                f"the level of delay chain must be positive, but got {chain_level}."
            )

        self.chain_level = chain_level
        self.inherent_delay = chain_level - 1

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return x1

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n_delaychain = NodeList()
        s_delaychain = NodeList()

        # Delay chain of length #D.
        for i in range(self.chain_level - 1):
            n_delay = SpikingRelu(
                self.shape_out,
                tick_wait_start=self.tick_wait_start + i,
                tick_wait_end=self.tick_wait_end,
                delay=1,
                name=f"n{i}_{self.name}",
            )
            n_delaychain.append(n_delay)

        # delay = delay_relative for output neuron
        n_out = SpikingRelu(
            self.shape_out,
            tick_wait_start=self.tick_wait_start + i + 1,
            tick_wait_end=self.tick_wait_end,
            delay=self.delay_relative,
            name=f"n{i+1}_{self.name}",
        )
        n_delaychain.append(n_out)  # Must append to the last.

        syn_in = FullConnSyn(
            self.module_intf.operands[0],
            n_delaychain[0],
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )

        for i in range(self.chain_level - 1):
            s_delay = FullConnSyn(
                n_delaychain[i],
                n_delaychain[i + 1],
                1,
                conn_type=ConnType.One2One,
                name=f"s{i+1}_{self.name}",
            )

            s_delaychain.append(s_delay)

        generated = [*n_delaychain, syn_in, *s_delaychain]
        self._rebuild_out_intf(network, n_out, *generated, **build_options)

        return generated


class SpikingAdd(FunctionalModule2to1WithV):
    inherent_delay = 0

    def __init__(
            self,
            neuron_a: Union[NeuDyn, InputProj],
            neuron_b: Union[NeuDyn, InputProj],
            *,
            keep_shape: bool = True,
            name: Optional[str] = None,
            overflow_strict: bool = False,
            **kwargs,
    ) -> None:
        """Spiking Addition module. The result will be reflected in time dimension.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.
            - overflow_strict: flag of whether to strictly check overflow. If enabled, an exception will be \
                raised if the result overflows during simulation.

        NOTE: the inherent delay of the module is 0.
        """
        self.overflow_strict = overflow_strict
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[SpikeType, VoltageType]:
        """Simplified neuron computing mechanism as the operator function."""
        return _spike_func_sadd_ssub(vjt)

    def synaptic_integr(
            self, x1: SpikeType, x2: SpikeType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, vjt_pre, _L_SADD, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_sadd = Neuron(
            self.shape_out,
            reset_mode=RM.MODE_LINEAR,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_sadd,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_sadd,
            1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_sadd, syn1, syn2]
        self._rebuild_out_intf(network, n1_sadd, *generated, **build_options)

        return generated


class _SpikingPool2dWithV(FunctionalModuleWithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        pos_thres: Optional[int] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling."""
        # C,H,W
        cin, ih, iw = _fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _kernel = np.ones((cin, cin, *_ksize), dtype=np.int8)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        oh = (ih + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        ow = (iw + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (cin, oh, ow)
        else:
            shape_out = (cin * oh * ow,)

        if isinstance(pos_thres, int):
            self.pos_thres = arg_check_non_neg(pos_thres, "positive threshold")
        else:
            self.pos_thres = typical_round(shape2num(_ksize) / 2)

        self.tfm = Conv2dForward((ih, iw), (oh, ow), _kernel, _stride, _padding)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[SpikeType, VoltageType]:
        return _spike_func_avg_pool(vjt, self.pos_thres)

    def synaptic_integr(self, x1: SpikeType, vjt_pre: VoltageType) -> VoltageType:
        return vjt_overflow((vjt_pre + self.tfm(x1).ravel()).astype(np.int32))

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_ap2d = IF(
            self.shape_out,
            threshold=self.pos_thres,
            reset_v=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_ap2d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_ap2d, syn1]
        self._rebuild_out_intf(network, n1_ap2d, *generated, **build_options)

        return generated


class _SpikingPool2d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        # fm_order: _Order3d = "CHW",
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling."""
        if pool_type not in ("avg", "max"):
            raise ValueError("type of pooling must be 'avg' or 'max'.")

        # if fm_order not in ("CHW", "HWC"):
        #     raise ValueError("feature map order must be 'CHW' or 'HWC'.")

        # C,H,W
        cin, ih, iw = _fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        oh = (ih + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        ow = (iw + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (cin, oh, ow)
        else:
            shape_out = (cin * oh * ow,)

        self.tfm = _Pool2dForward(
            cin, (ih, iw), (oh, ow), _ksize, _stride, _padding, pool_type, threshold
        )

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return self.tfm(x1)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        if self.tfm.pool_type == "avg":
            n1_p2d = Neuron(
                self.shape_out,
                leak_v=1 - self.tfm.threshold,
                neg_threshold=0,
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
            )
        else:  # "max"
            n1_p2d = SpikingRelu(
                self.shape_out,
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n0_{self.name}",
            )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_p2d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p2d, syn1]
        self._rebuild_out_intf(network, n1_p2d, *generated, **build_options)

        # for syns in self.module_intf.output:
        #     syns.source = n1_p2d

        # network._add_components(*generated)
        # network._remove_components(self)

        return generated


class SpikingAvgPool2d(_SpikingPool2d):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d average pooling for spike. The input feature map is in 'CHW' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2  \
                integers.
            - threshold: if specified, the pooling result is o = (sum of the pooling window > threshold).   \
                Otherwise the threshold is kernel_size // 2.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "avg",
            stride,
            padding,
            threshold,
            keep_shape,
            name,
            **kwargs,
        )


class SpikingAvgPool2dWithV(_SpikingPool2dWithV):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            neuron, kernel_size, stride, padding, threshold, keep_shape, name, **kwargs
        )


class SpikingMaxPool2d(_SpikingPool2d):
    """
    XXX: By enabling `MaxPoolingEnable` in neurons, the max pooling function can also be implemented.       \
        However, since the second-level cache of the input buffer before the physical core is in 144*8bit   \
        format, it is extremely wasteful when the input data is 1bit (i.e., spike). Therefore, we still     \
        under SNN mode when implementing max pooling of 1-bit input data.
    """

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d max pooling for spike.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2  \
                integers.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "max",
            stride,
            padding,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class SpikingSub(FunctionalModule2to1WithV):
    inherent_delay = 0

    def __init__(
            self,
            neuron_a: Union[NeuDyn, InputProj],
            neuron_b: Union[NeuDyn, InputProj],
            *,
            keep_shape: bool = True,
            name: Optional[str] = None,
            overflow_strict: bool = False,
            **kwargs,
    ) -> None:
        """Spiking subtraction module. The result will be reflected in time dimension.

        Args:
            - neuron_a: the first operand. It is the minuend.
            - neuron_b: the second operand. It is the subtracter.
            - overflow_strict: flag of whether to strictly check overflow. If enabled, an exception will be \
                raised if the result overflows during simulation.

        NOTE: the inherent delay of the module is 0.
        """
        self.overflow_strict = overflow_strict
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[SpikeType, VoltageType]:
        """Simplified neuron computing mechanism to generate output spike."""
        return _spike_func_sadd_ssub(vjt)

    def synaptic_integr(
            self, x1: SpikeType, x2: SpikeType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, vjt_pre, _L_SSUB, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_ssub = Neuron(
            self.shape_out,
            reset_mode=RM.MODE_LINEAR,
            neg_thres_mode=NTM.MODE_SATURATION,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_ssub,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_ssub,
            weights=-1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_ssub, syn1, syn2]
        self._rebuild_out_intf(network, n1_ssub, *generated, **build_options)

        return generated


@deprecated(
    "'Transpose2d' will be removed in a future version. Use 'MatMul2d' instead.",
    category=PAIBoxDeprecationWarning,
)
class Transpose2d(TransposeModule):
    def __init__(
            self,
            neuron: Union[NeuDyn, InputProj],
            *,
            keep_shape: bool = True,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        """2d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim2_check(neuron.shape_out),
            (1, 0),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.T

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t2d = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t2d,
            weights=_transpose2d_mapping(self.shape_in),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_t2d, syn1]
        self._rebuild_out_intf(network, n1_t2d, *generated, **build_options)

        return generated


@deprecated(
    "'Transpose3d' will be removed in a future version. Use 'MatMul2d' instead.",
    category=PAIBoxDeprecationWarning,
)
class Transpose3d(TransposeModule):
    def __init__(
            self,
            neuron: Union[NeuDyn, InputProj],
            axes: Optional[Sequence[int]] = None,
            *,
            keep_shape: bool = True,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        """3d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.
            - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1] \
                where N is the number of axes of output shape of neuron. If not specified, defaults to      \
                `range(ndim)[::-1]`, where `ndim` is the dimension of the output shape, which reverses the  \
                order of the axes.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim3_check(neuron.shape_out),
            axes,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.transpose(self.axes)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t3d = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t3d,
            weights=_transpose3d_mapping(self.shape_in, self.axes),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_t3d, syn1]
        self._rebuild_out_intf(network, n1_t3d, *generated, **build_options)

        return generated

class Delay_FullConn(FunctionalModule):
    def __init__(
            self,
            neuron_s: Union[NeuDyn, InputProj],
            neuron_d: Union[NeuDyn, InputProj],
            delay: int,
            weights: DataArrayType = 1,
            conn_type: ConnType = ConnType.MatConn,
            keep_shape: bool = False,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        self.delay = delay
        self.weights = weights
        self.conn_type = conn_type
        _shape_out = neuron_d.shape_out
        super().__init__(
            neuron_s,
            neuron_d,
            shape_out=_shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return

    def build(self, network: DynSysGroup, **build_options) -> None:
        if len(self.module_intf.operands[0].shape_out)!=2:
            raise ShapeError("The source node must be a successor to the half-convolution")
        delay_shape = self.module_intf.operands[0].shape_out
        delay_neurons = []
        for i in range(self.delay):
            neuron = Neuron(
                shape=delay_shape,
                leak_v=0,
                neg_threshold=0,
                delay=i+1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            delay_neurons.append(neuron)
            # 延时突触
            syn1 = FullConnSyn(
                self.module_intf.operands[0],
                delay_neurons[i],
                weights=_delay_mapping(delay_shape[1], delay_shape[0], 1),
                conn_type=ConnType.MatConn,
                name=f"s{i}_delay",
            )
            #w = np.zeros((neuron.num_out, self.module_intf.operands[1].num_out))
            w = self.weights[i::self.delay, :]
            syn2 = FullConnSyn(  # cin,(kw-1)*ih -> cout * oh
                delay_neurons[i], # 5 -> 3
                self.module_intf.operands[1],
                weights=w,
                conn_type=self.conn_type,
                name=f"s{i}_{self.name}",
            )
            network._add_components(neuron, syn1, syn2)
            network._remove_components(self)
            generated = [*delay_neurons, syn1, syn2]
        return generated


class Conv_HalfRoll(FunctionalModule):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
            self,
            neuron_s: Union[NeuDyn, InputProj],
            neuron_d: Union[NeuDyn, InputProj],
            kernel: np.ndarray,
            stride: Optional[_Size2Type] = None,
            padding: _Size2Type = 0,
            # fm_order: _Order3d = "CHW",
            keep_shape: bool = False,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        """2d conv_halfroll for spike.

        """
        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if len(neuron_s.shape_out) != 2:
            in_ch, in_h, in_w = _fm_ndim2_check(neuron_s.shape_out, "CHW")
            print("变形")
            neuron_s.shape_change((in_ch, in_h))
        in_ch, in_h = neuron_s.shape_out
        cout, cin, kh, kw = kernel.shape
        if len(neuron_d.shape_out) != 2:
            out_ch, out_h, out_w = _fm_ndim2_check(neuron_d.shape_out, "CHW")
            print("变形")
            neuron_d.shape_change((cout, out_h))
        # out_h = (in_h + 2 * padding[0] * (kh - 1) - 1) // stride[
        #     0
        # ] + 1
        if in_ch != cin:
            raise ShapeError(f"input channels mismatch: {in_ch} != {cin}.")
        # if (_output_size := cout * out_h) != neuron_d.num_in:
        #     raise ShapeError(
        #         f"Output size mismatch"
        #     )
        self.kernel = kernel
        _shape_out = neuron_d.shape_out

        super().__init__(
            neuron_s,
            neuron_d,
            shape_out=_shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return

    def build(self, network: DynSysGroup, **build_options) -> None:
        # 延时神经元
        # if len(self.module_intf.operands[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(self.module_intf.operands[0].shape_out, "CHW")
        #     self.module_intf.operands[0].shape_change((in_ch, in_h))
        in_ch, in_h = self.module_intf.operands[0].shape_out
        cout, cin, kh, kw = self.kernel.shape
        # 更改形状
        # out_ch, out_h, out_w = _fm_ndim2_check(self.module_intf.operands[1].shape_out, "CHW")
        # self.module_intf.operands[1].shape_change((cout, out_h))
        n_delays = NodeList()
        s_delays = NodeList()
        for i in range(kw - 1):
            neuron = Neuron(
                (cin, in_h),
                leak_v=0,
                neg_threshold=0,
                delay=i+2,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # 延时突触
            syn1 = FullConnSyn(
                self.module_intf.operands[0],# (2, 5)
                n_delays[i],
                weights=_delay_mapping(in_h, cin, 1),
                conn_type=ConnType.All2All,
                name=f"s{i+1}_delay_{self.name}",
            )
            s_delays.append(syn1)
            syn2 = Conv2dHalfRollSyn(  # cin, ih -> cout * oh
                n_delays[i], self.module_intf.operands[1], kernel=self.kernel, stride=_pair(1), padding=_pair(0),
                order="OIHW", name=f"s{i+1}_{self.name}",
            )
            s_delays.append(syn2)
            # syn2 = FullConnSyn(
            #     delay_neurons[i],
            #     self.module_intf.operands[1],
            #     weights=1,
            #     conn_type=GConnType.All2All,
            #     name=f"s{i}_conv",
            # )
            network._add_components(neuron, syn1, syn2)
        syn3 = Conv2dHalfRollSyn(  # (cin, ih) -> cout * oh
            self.module_intf.operands[0], self.module_intf.operands[1], kernel=self.kernel, stride=(1,1), padding=(0,0),
            name=f"s0_{self.name}",
        )
        generated = [*n_delays, syn3, *s_delays]

        network._add_components(syn3)
        network._remove_components(self)

        return generated

class Filter(FunctionalModule):

    def __init__(
            self,
            neuron: Union[NeuDyn, InputProj],
            time_to_fire: int,
            keep_shape: bool = False,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        """
        """
        shape_out = neuron.shape_out
        self.time_to_fire = time_to_fire
        self.cur_time = 0
        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        if self.cur_time != self.time_to_fire:
            self.cur_time += 1
            return np.zeros_like(x1)
        else:
            self.cur_time = 0
            return x1

    def build(self, network: DynSysGroup, **build_options) -> None:
        inp1 = Always1Neuron((2,))
        n1_filter = Neuron(
            self.shape_out,
            leak_v=0,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name="filter"
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],  # (10,0)
            n1_filter,  # (10,0)
            weights=1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            inp1,  # (2,0)
            n1_filter,  # (10,0)
            weights=-128,
            conn_type=ConnType.All2All,
            name=f"s1_{self.name}",
        )
        network._add_components(n1_filter, syn1, syn2)
        network._remove_components(self)
        generated = [n1_filter, syn1, syn2]
        return generated


def _spike_func_sadd_ssub(vjt: VoltageType) -> tuple[SpikeType, VoltageType]:
    """Function `spike_func()` in spiking addition & subtraction."""
    # Fire
    thres_mode = np.where(
        vjt >= 1,
        TM.EXCEED_POSITIVE,
        np.where(vjt < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
    )
    spike = np.equal(thres_mode, TM.EXCEED_POSITIVE)
    # Reset
    v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, vjt - 1, vjt)

    return spike, v_reset


def _spike_func_avg_pool(
    vjt: VoltageType, pos_thres: int
) -> tuple[SpikeType, VoltageType]:
    """Function `spike_func()` in spiking addition & subtraction."""
    # Fire
    thres_mode = np.where(
        vjt >= pos_thres,
        TM.EXCEED_POSITIVE,
        np.where(vjt < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
    )
    spike = np.equal(thres_mode, TM.EXCEED_POSITIVE)
    # Reset
    v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, 0, vjt)

    return spike, v_reset


def _sum_inputs_sadd_ssub(
        x1: SpikeType,
        x2: SpikeType,
        vjt_pre: VoltageType,
        add_or_sub: Literal[1, -1],
        strict: bool,
) -> VoltageType:
    """Function `sum_input()` for spiking addition & subtraction."""
    incoming_v = (vjt_pre + x1 * 1 + x2 * add_or_sub).astype(np.int32)
    return vjt_overflow(incoming_v, strict)


def _shape_check(shape: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(shape) > ndim:
        raise ShapeError(
            f"expected shape to have dimensions <= {ndim}, but got {len(shape)}."
        )

    return as_shape(shape, min_dim=ndim)


_shape_ndim2_check = partial(_shape_check, ndim=2)
_shape_ndim3_check = partial(_shape_check, ndim=3)


def _transpose2d_mapping(op_shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 2d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y) order.

    Return: transposed index matrix with shape (X*Y, Y*X).
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    for idx in np.ndindex(op_shape):
        mt[idx[0] * op_shape[1] + idx[1], idx[1] * op_shape[0] + idx[0]] = 1

    return mt


def _transpose3d_mapping(
    op_shape: tuple[int, ...], axes: tuple[int, ...]
) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 3d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y,Z) order.
        - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1]     \
            where N is the number of axes of a.

    Return: transposed index matrix with shape (N, N) where N=X*Y*Z.
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    shape_t = tuple(op_shape[i] for i in axes)

    size12 = op_shape[1] * op_shape[2]
    size12_t = shape_t[1] * shape_t[2]

    for idx in np.ndindex(op_shape):
        mt[
            idx[0] * size12 + idx[1] * op_shape[2] + idx[2],
            idx[axes[0]] * size12_t + idx[axes[1]] * shape_t[2] + idx[axes[2]],
        ] = 1

    return mt


def _delay_mapping(h: int, cin: int, n: int) -> NDArray[np.bool_]:
    mt = np.zeros((cin * h, cin * n * h), dtype=np.bool_)
    for i in range(cin):
        for j in range(n * cin):
            for k in range(h):
                mt[i * h + k, j * h + k] = 1
    return mt
