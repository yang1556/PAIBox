from typing import Literal

import numpy as np
import pytest

import paibox as pb


def _out_bypass1(t, data1, *args, **kwargs):
    return data1


def _out_bypass2(t, data2, *args, **kwargs):
    return data2


def _out_bypass3(t, data3, *args, **kwargs):
    return data3


class FModule_ConnWithInput_Net(pb.DynSysGroup):
    """A network where an input node is connected to a module.

    Structure:
        inp1 -> s1 -> n1 ->
                    inp2 -> and1 -> s2 -> n2
    """

    def __init__(self):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        self.and1 = pb.BitwiseAND(self.n1, self.inp2, delay=1, tick_wait_start=2)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=3)
        self.s2 = pb.FullConn(self.and1, self.n2, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.and1, "spike")


class FModule_ConnWithModule_Net(pb.DynSysGroup):
    """A network where one module is connected to another module.

    Structure:
        inp1 -> s1 -> n1 ->
        inp2 -> s2 -> n2 -> and1 ->
               inp3-> s3 ->  n3  -> or1 -> s4 -> n4
    """

    def __init__(self):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.inp3 = pb.InputProj(input=_out_bypass3, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n3 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=2)  # tws = 2!
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.s2 = pb.FullConn(self.inp2, self.n2, conn_type=pb.SynConnType.One2One)
        self.s3 = pb.FullConn(self.inp3, self.n3, conn_type=pb.SynConnType.One2One)

        self.and1 = pb.BitwiseAND(self.n1, self.n2, delay=1, tick_wait_start=2)
        self.or1 = pb.BitwiseOR(self.and1, self.n3, delay=1, tick_wait_start=3)
        self.n4 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=4)
        self.s4 = pb.FullConn(self.or1, self.n4, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.and1, "spike")
        self.probe3 = pb.Probe(self.or1, "spike")


class FModule_ConnWithFModule_Net(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.IF((8, 16, 16), 10, tick_wait_start=1)
        self.n2 = pb.IF((8, 4, 4), 5, tick_wait_start=2)
        self.mp2d = pb.SpikingMaxPool2d(self.n1, (4, 4), tick_wait_start=2)
        self.sub = pb.SpikingSub(self.n2, self.mp2d, tick_wait_start=3)

        self.s1 = pb.FullConn(self.n1, self.n2)


class FunctionalModule_2to1_Net(pb.DynSysGroup):
    def __init__(self, op: Literal["and", "or", "xor", "add", "sub"]):
        super().__init__()
        self.bitwise = 10

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.s2 = pb.FullConn(self.inp2, self.n2, conn_type=pb.SynConnType.One2One)

        if op == "and":
            self.func_node = pb.BitwiseAND(self.n1, self.n2, delay=1, tick_wait_start=2)
        elif op == "or":
            self.func_node = pb.BitwiseOR(self.n1, self.n2, delay=1, tick_wait_start=2)
        elif op == "xor":
            self.func_node = pb.BitwiseXOR(self.n1, self.n2, delay=1, tick_wait_start=2)
        elif op == "add":
            self.func_node = pb.SpikingAdd(self.n1, self.n2, delay=1, tick_wait_start=2)
        elif op == "sub":
            self.func_node = pb.SpikingSub(self.n1, self.n2, delay=1, tick_wait_start=2)

        self.n3 = pb.SpikingRelu(
            (10,),
            delay=1,
            tick_wait_start=self.func_node.tick_wait_start
            + self.func_node.external_delay,
        )
        self.s3 = pb.FullConn(self.func_node, self.n3, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.n2, "spike")
        self.probe3 = pb.Probe(self.func_node, "spike")
        self.probe4 = pb.Probe(self.n3, "spike")

        if hasattr(self.func_node, "voltage"):
            self.probe5 = pb.Probe(self.func_node, "voltage")


class FunctionalModule_1to1_Net(pb.DynSysGroup):
    def __init__(self, op: Literal["not", "delay"]):
        super().__init__()
        self.bitwise = 10

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        if op == "not":
            self.func_node = pb.BitwiseNOT(self.n1, tick_wait_start=2)
        elif op == "delay":
            self.func_node = pb.DelayChain(self.n1, chain_level=5, tick_wait_start=2)

        self.n2 = pb.SpikingRelu(
            (10,),
            delay=1,
            tick_wait_start=self.func_node.tick_wait_start
            + self.func_node.external_delay,
        )
        self.s3 = pb.FullConn(self.func_node, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.func_node, "spike")
        self.probe3 = pb.Probe(self.n2, "spike")

        if hasattr(self.func_node, "voltage"):
            self.probe4 = pb.Probe(self.func_node, "voltage")


class SpikingPool2d_Net(pb.DynSysGroup):
    def __init__(self, fm_shape, ksize, stride, padding, threshold, pool_type):
        super().__init__()
        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=fm_shape)
        self.n1 = pb.SpikingRelu(fm_shape, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        if pool_type == "avg":
            self.pool2d = pb.SpikingAvgPool2d(
                self.n1, ksize, stride, padding, threshold, delay=1, tick_wait_start=2
            )
        elif pool_type == "avgv":
            self.pool2d = pb.SpikingAvgPool2dWithV(
                self.n1, ksize, stride, padding, threshold, delay=1, tick_wait_start=2
            )
        else:  # "max"
            self.pool2d = pb.SpikingMaxPool2d(
                self.n1, ksize, stride, padding, delay=1, tick_wait_start=2
            )

        self.n2 = pb.SpikingRelu(self.pool2d.shape_out, delay=1, tick_wait_start=3)
        self.s3 = pb.FullConn(self.pool2d, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.pool2d, "spike")
        self.probe3 = pb.Probe(self.n2, "spike")


class TransposeModule_T2d_Net(pb.DynSysGroup):
    def __init__(self, shape):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.n1 = pb.IF(shape, 1, 0, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.t2d = pb.Transpose2d(self.n1, tick_wait_start=2)
        self.n2 = pb.SpikingRelu(
            shape, tick_wait_start=self.t2d.tick_wait_start + self.t2d.external_delay
        )
        self.s2 = pb.FullConn(self.t2d, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.t2d, "spike")
        self.probe2 = pb.Probe(self.n2, "spike")


class TransposeModule_T3d_Net(pb.DynSysGroup):
    def __init__(self, shape, axes):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.n1 = pb.IF(shape, 1, 0, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.t3d = pb.Transpose3d(self.n1, axes=axes, tick_wait_start=2)
        self.n2 = pb.SpikingRelu(
            shape, tick_wait_start=self.t3d.tick_wait_start + self.t3d.external_delay
        )
        self.s2 = pb.FullConn(self.t3d, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.t3d, "spike")
        self.probe2 = pb.Probe(self.n2, "spike")

class Conv_HalfRoll_Net1(pb.DynSysGroup):
    def __init__(self, shape, kernel, stride, padding):
        super().__init__()

        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=(1, 11, 11))
        self.conv1 = pb.ConvHalfRoll(self.i1, kernel, stride[0], padding[0], tick_wait_start=1)
        self.conv2 = pb.ConvHalfRoll(self.conv1, kernel, stride[1], padding[1], tick_wait_start=3)
        # self.linear1 = pb.DelayFullConn(
        #     self.conv1,
        #     2,
        #     delay=3,
        #     weights=weight,
        #     conn_type=pb.SynConnType.All2All,
        #     tick_wait_start=3
        # )
class Conv_HalfRoll_Net2(pb.DynSysGroup):
    def __init__(self, shape, kernel, stride, padding, out_feature, weight):
        super().__init__()

        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.conv1 = pb.ConvHalfRoll(self.i1, kernel, stride[0], padding[0], tick_wait_start=1)
        self.conv2 = pb.ConvHalfRoll(self.conv1, kernel, stride[1], padding[1], tick_wait_start=3)
        self.linear1 = pb.DelayFullConn(
            self.conv2,
            out_feature,
            weights=weight,
            conn_type=pb.SynConnType.All2All,
            tick_wait_start=5

        )

class ANNNetwork(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(32, 32))

        n1_bias = np.random.randint(-128, 128, size=(4,), dtype=np.int8)
        self.n1 = pb.LIF(
            (4, 30, 30),
            100,
            bias=n1_bias,
            tick_wait_start=1,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        n2_bias = np.random.randint(-128, 128, size=(4,), dtype=np.int8)
        self.n2 = pb.LIF(
            (4, 28, 28),
            50,
            bias=n2_bias,
            tick_wait_start=2,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        self.n3 = pb.LIF(
            (2, 26, 26),
            20,
            bias=1,
            tick_wait_start=3,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        self.n4 = pb.IF(
            (100,), 10, tick_wait_start=4, input_width=8, spike_width=8, snn_en=False
        )

        kernel_1 = np.random.randint(-128, 128, size=(4, 1, 3, 3), dtype=np.int8)
        self.conv2d_1 = pb.Conv2d(self.inp1, self.n1, kernel_1)

        kernel_2 = np.random.randint(-128, 128, size=(4, 4, 3, 3), dtype=np.int8)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel_2)

        kernel_3 = np.random.randint(-128, 128, size=(2, 4, 3, 3), dtype=np.int8)
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel_3)

        w4 = np.random.randint(-128, 128, size=(2 * 26 * 26, 100), dtype=np.int8)
        self.fc1 = pb.FullConn(self.n3, self.n4, w4)


@pytest.fixture(scope="class")
def build_BitwiseAND_Net():
    return FunctionalModule_2to1_Net("and")


@pytest.fixture(scope="class")
def build_BitwiseNOT_Net():
    return FunctionalModule_1to1_Net("not")


@pytest.fixture(scope="class")
def build_BitwiseOR_Net():
    return FunctionalModule_2to1_Net("or")


@pytest.fixture(scope="class")
def build_BitwiseXOR_Net():
    return FunctionalModule_2to1_Net("xor")


@pytest.fixture(scope="class")
def build_DelayChain_Net():
    return FunctionalModule_1to1_Net("delay")


@pytest.fixture(scope="class")
def build_SpikingAdd_Net():
    return FunctionalModule_2to1_Net("add")


@pytest.fixture(scope="class")
def build_SpikingSub_Net():
    return FunctionalModule_2to1_Net("sub")


@pytest.fixture(scope="class")
def build_FModule_ConnWithInput_Net():
    return FModule_ConnWithInput_Net()


@pytest.fixture(scope="class")
def build_FModule_ConnWithModule_Net():
    return FModule_ConnWithModule_Net()


@pytest.fixture(scope="class")
def build_FModule_ConnWithFModule_Net():
    return FModule_ConnWithFModule_Net()


@pytest.fixture(scope="class")
def build_ANN_Network_1():
    return ANNNetwork()
