import numpy as np

# 读取 .npz 文件
# data = np.load('iodata_016.npz')
#
# # 展示 .npz 文件中的所有数组名称
# print("文件中的数组名称:", data.files)
#
# # # 遍历并展示每个数组的内容
# # for array_name in data.files:
# #     print(f"数组 '{array_name}' 的内容:")
# #     print(data[array_name])
# #     print("-" * 40)
#
# print("input_shape", data["input"].shape)
#
# # 关闭文件
# data.close()
from pathlib import Path

import numpy as np
import pytest

import paibox as pb
from paibox.components.neuron.base import MetaNeuron
from paibox.types import NEUOUT_U8_DTYPE, VOLTAGE_DTYPE, NeuOutType, VoltageType


def _out_bypass1(t, data1, *args, **kwargs):
    return data1

# class Net015(pb.DynSysGroup):
#     def __init__(self, w1, w2, w3):
#         super().__init__()
#         self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
#         self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 2, 1, tick_wait_start=1)
#
#         self.conv2 = pb.Conv2dSemiFolded(
#             self.conv1, w2, 2, 1, tick_wait_start=3
#         )
#
#         self.linear1 = pb.LinearSemiFolded(
#             self.conv2, out_shape[1], weights=w3, bias=2, tick_wait_start=5, rin_buffer_option=True
#         )

class fcnet(pb.Network):
    def __init__(self,):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(10,))
        self.n1 = pb.IF(10, threshold=0, reset_v=0)
        self.s1 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )

        # tick_wait_start = 2 for second layer
        self.n2 = pb.IF(
            10, threshold=0, reset_v=0, tick_wait_start=2, name="batch_dual_port_o1"
        )
        self.n3 = pb.IF(
            10, threshold=0, reset_v=0, tick_wait_start=2, name="batch_dual_port_o2"
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n2,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s4 = pb.FullConn(
            self.n1,
            self.n3,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )


USE_EXISTING_DATA = False




# shape1 = (3, 32, 32)  # C*H*W
# ksize1 = (4, shape1[0], 4, 4)  # O*C*K*K
# ksize2 = (4, ksize1[0], 4, 4)
# out_shape = (4 * 8 * 8, 1300)
shape1 = (10, )
shape2 = (2000, )
shape3 = (10, )

sim_time = 40

USE_EXISTING_DATA = False

FIXED_RNG = np.random.default_rng(seed=42)
CONFIG_CASE_DIR = "config"

# if not USE_EXISTING_DATA:
#     print("Generating new data")
#     # W=8, disable weight bit optimization
#     weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
#     weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
#     weight3 = FIXED_RNG.integers(-3, 5, size=out_shape, dtype=np.int8)
#     inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
#     inpdata1 = np.concatenate(
#         [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
#     )
#     # Shape of reference result is sim_time * refdata
#     refresult1 = np.zeros((sim_time, out_shape[1]), dtype=NEUOUT_U8_DTYPE)

network = fcnet()

#generated = network.build_modules()
#sim = pb.Simulator(network, start_time_zero=False)



# Save weights & input data


mapper = pb.Mapper()
mapper.build(network)
mapper.compile(weight_bit_optimization=False)
mapper.export(
    fp=CONFIG_CASE_DIR, export_core_params=True, format="txt", use_hw_sim=True
)

