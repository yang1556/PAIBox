import argparse
from pathlib import Path

import numpy as np

import paibox as pb
from paibox import DynSysGroup
from paibox.components.synapses.conv_utils import _conv2d_faster
from tests.components.test_functional import _ann_bit_trunc

#
def _out_bypass1(t, data1, *args, **kwargs):
    return data1
#
# kernel1_1 = np.random.randint(0, 10, (64, 3, 3, 3), dtype=np.int8)
# kernel1_2 = np.random.randint(0, 10, (64, 64, 3, 3), dtype=np.int8)
# kernel2_1 = np.random.randint(0, 128, (128, 64, 3, 3), dtype=np.int8)
# kernel2_2 = np.random.randint(0, 128, (128, 128, 3, 3), dtype=np.int8)
# kernel3_1 = np.random.randint(0, 128, (256, 128,3, 3), dtype=np.int8)
# kernel3_2 = np.random.randint(0, 128, (256, 256, 3, 3), dtype=np.int8)
# kernel3_3 = np.random.randint(0, 128, (256, 256, 3, 3), dtype=np.int8)
# kernel4_1 = np.random.randint(0, 128, (512, 256, 3, 3), dtype=np.int8)
# kernel4_2 = np.random.randint(0, 128, (512, 512, 3, 3), dtype=np.int8)
# kernel4_3 = np.random.randint(0, 128, (512, 512, 3, 3), dtype=np.int8)
# kernel5_1 = np.random.randint(0, 128, (512, 512, 3, 3), dtype=np.int8)
# kernel5_2 = np.random.randint(0, 128, (512, 512, 3, 3), dtype=np.int8)
# kernel5_3 = np.random.randint(0, 128, (512, 512, 3, 3), dtype=np.int8)
# weight1 = np.random.randint(0, 128, (512, 4096), dtype=np.int8)
# weight2 = np.random.randint(0, 128, (4096, 4096), dtype=np.int8)
# weight3 = np.random.randint(0, 128, (4096, 10), dtype=np.int8)
#
#
# class VGGNet(pb.DynSysGroup):
#     def __init__(self):
#         super().__init__()
#         pe = pb.simulator.PoissonEncoder()
#         self.i1 = pb.InputProj(input=pe, shape_out=(3, 32))
#
#         # Block 1
#
#         self.conv1_1 = pb.Conv2dSemiFolded(self.i1, kernel1_1, 1, 1, tick_wait_start=1)
#
#
#         self.conv1_2 = pb.Conv2dSemiFolded(self.conv1_1, kernel1_2, 1, 1, tick_wait_start=3)
#
#         self.pool1 = pb.MaxPool2dSemiFolded(self.conv1_2, (2, 2), 2, tick_wait_start=5)
#
#         # Block 2
#
#         self.conv2_1 = pb.Conv2dSemiFolded(self.pool1, kernel2_1, 1, 1, tick_wait_start=7)
#
#
#         self.conv2_2 = pb.Conv2dSemiFolded(self.conv2_1, kernel2_2, 1, 1, tick_wait_start=9)
#
#         self.pool2 = pb.MaxPool2dSemiFolded(self.conv2_2, (2, 2), 2, tick_wait_start=11)
#
#         # Block 3
#
#         self.conv3_1 = pb.Conv2dSemiFolded(self.pool2, kernel3_1, 1, 1, tick_wait_start=13)
#
#         self.conv3_2 = pb.Conv2dSemiFolded(self.conv3_1, kernel3_2, 1, 1, tick_wait_start=15)
#
#         self.conv3_3 = pb.Conv2dSemiFolded(self.conv3_2, kernel3_3, 1, 1, tick_wait_start=17)
#
#         self.pool3 = pb.MaxPool2dSemiFolded(self.conv3_3, (2, 2), 2, tick_wait_start=19)
#
#         # Block 4
#         self.conv4_1 = pb.Conv2dSemiFolded(self.pool3, kernel4_1, 1, 1, tick_wait_start=21)
#
#         self.conv4_2 = pb.Conv2dSemiFolded(self.conv4_1, kernel4_2, 1, 1, tick_wait_start=23)
#
#         self.conv4_3 = pb.Conv2dSemiFolded(self.conv4_2, kernel4_3, 1, 1, tick_wait_start=25)
#
#         self.pool4 = pb.MaxPool2dSemiFolded(self.conv4_3, (2, 2), 2, tick_wait_start=27)
#
#         # Block 5
#         self.conv5_1 = pb.Conv2dSemiFolded(self.pool4, kernel5_1, 1, 1, tick_wait_start=29)
#
#         self.conv5_2 = pb.Conv2dSemiFolded(self.conv5_1, kernel5_2, 1, 1, tick_wait_start=31)
#
#         self.conv5_3 = pb.Conv2dSemiFolded(self.conv5_2, kernel5_3, 1, 1, tick_wait_start=33)
#
#         self.pool5 = pb.MaxPool2dSemiFolded(self.conv5_3, (2, 2), 2, tick_wait_start=35)
#
#         # Fully connected layers would go here
#
#         self.fc1 = pb.LinearSemiFolded(self.pool5, 4096, weight1, bias=0, conn_type=pb.SynConnType.All2All, tick_wait_start=37)
#         self.fc2 = pb.Linear(self.fc1, 4096, weight2, bias=0, conn_type=pb.SynConnType.All2All, tick_wait_start=39)
#         self.fc3 = pb.Linear(self.fc2, 10, weight3, bias=0, conn_type=pb.SynConnType.All2All, tick_wait_start=41)
#
# class snnVGGNet(pb.Network):
#     def __init__(self):
#         super().__init__()
#
#         pe = pb.simulator.PoissonEncoder()
#         self.i1 = pb.InputProj(input=pe, shape_out=(3, 224, 224))
#
#         # Convolutional Block 1
#         self.n1 = pb.IF((64, 224, 224), threshold=1, reset_v=0)
#         self.conv2d_1 = pb.Conv2d(self.i1, self.n1, kernel=kernel1_1, stride=1, padding=1)
#
#         self.n2 = pb.IF((64, 224, 224), threshold=1, reset_v=0)
#         self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=kernel1_2, stride=1, padding=1)
#         self.pool1 = pb.SpikingMaxPool2d(self.n2, kernel_size=2, stride=2)  # Output: (64, 112, 112)
#
#         # Convolutional Block 2
#         self.n3 = pb.IF((128, 112, 112), threshold=1, reset_v=0)
#         self.conv2d_3 = pb.Conv2d(self.pool1, self.n3, kernel=kernel2_1, stride=1, padding=1)
#
#         self.n4 = pb.IF((128, 112, 112), threshold=1, reset_v=0)
#         self.conv2d_4 = pb.Conv2d(self.n3, self.n4, kernel=kernel2_2, stride=1, padding=1)
#         self.pool2 = pb.SpikingMaxPool2d(self.n4, kernel_size=2, stride=2)  # Output: (128, 56, 56)
#
#         # Convolutional Block 3
#         self.n5 = pb.IF((256, 56, 56), threshold=1, reset_v=0)
#         self.conv2d_5 = pb.Conv2d(self.pool2, self.n5, kernel=kernel3_1, stride=1, padding=1)
#
#         self.n6 = pb.IF((256, 56, 56), threshold=1, reset_v=0)
#         self.conv2d_6 = pb.Conv2d(self.n5, self.n6, kernel=kernel3_2, stride=1, padding=1)
#
#         self.n7 = pb.IF((256, 56, 56), threshold=1, reset_v=0)
#         self.conv2d_7 = pb.Conv2d(self.n6, self.n7, kernel=kernel3_3, stride=1, padding=1)
#         self.pool3 = pb.SpikingMaxPool2d(self.n7, kernel_size=2, stride=2)  # Output: (256, 28, 28)
#
#         # Convolutional Block 4
#         self.n8 = pb.IF((512, 28, 28), threshold=1, reset_v=0)
#         self.conv2d_8 = pb.Conv2d(self.pool3, self.n8, kernel=kernel4_1, stride=1, padding=1)
#
#         self.n9 = pb.IF((512, 28, 28), threshold=1, reset_v=0)
#         self.conv2d_9 = pb.Conv2d(self.n8, self.n9, kernel=kernel4_2, stride=1, padding=1)
#
#         self.n10 = pb.IF((512, 28, 28), threshold=1, reset_v=0)
#         self.conv2d_10 = pb.Conv2d(self.n9, self.n10, kernel=kernel4_3, stride=1, padding=1)
#         self.pool4 = pb.SpikingMaxPool2d(self.n10, kernel_size=2, stride=2)  # Output: (512, 14, 14)
#
#         # Convolutional Block 5
#         self.n11 = pb.IF((512, 14, 14), threshold=1, reset_v=0)
#         self.conv2d_11 = pb.Conv2d(self.pool4, self.n11, kernel=kernel5_1, stride=1, padding=1)
#
#         self.n12 = pb.IF((512, 14, 14), threshold=1, reset_v=0)
#         self.conv2d_12 = pb.Conv2d(self.n11, self.n12, kernel=kernel5_2, stride=1, padding=1)
#
#         self.n13 = pb.IF((512, 14, 14), threshold=1, reset_v=0)
#         self.conv2d_13 = pb.Conv2d(self.n12, self.n13, kernel=kernel5_3, stride=1, padding=1)
#         self.pool5 = pb.SpikingMaxPool2d(self.n13, kernel_size=2, stride=2)  # Output: (512, 7, 7)
#
#         # Fully Connected Layers
#         self.n14 = pb.IF(4096, threshold=1, reset_v=0)
#         self.fc1 = pb.FullConn(self.pool5, self.n14, weights=weight1, conn_type=pb.SynConnType.All2All)
#
#         self.n15 = pb.IF(4096, threshold=1, reset_v=0)
#         self.fc2 = pb.FullConn(self.n14, self.n15, weights=weight2, conn_type=pb.SynConnType.All2All)
#
#         self.n16 = pb.IF(10, threshold=1, reset_v=0)
#         self.fc3 = pb.FullConn(self.n15, self.n16, weights=weight3, conn_type=pb.SynConnType.All2All)
#
#
#
#
g_kernel = np.random.randint(0, 10, size=(4, 1, 3, 3), dtype=np.int8)
# print(g_kernel)
class GroupNet(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=input_shape[:2])

        # Block 1

        self.conv1_1 = pb.Conv2dSemiFolded(self.i1, g_kernel, 1, 0, tick_wait_start=1, groups=2)
# # Example of creating a VGG16 instance
input_shape = (2, 5, 5)  # Example input shape (channels, height, width)
group_net = GroupNet()
inpa = np.random.randint(0, 4, size=input_shape, dtype=np.int8)
inp_pad0 = np.concatenate(
    [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
)
#
conv1 = group_net.conv1_1
generated = DynSysGroup.build_fmodule(group_net)
probe1 = pb.Probe(generated[conv1][0], "output")

sim1 = pb.Simulator(group_net, start_time_zero=False)
sim1.add_probe(probe1)
for i in range(inp_pad0.shape[-1]):
    pb.FRONTEND_ENV.save(data1=inp_pad0[:, :, i])
    sim1.run(1)

print(sim1.data[probe1])
x=inpa
x = _ann_bit_trunc(
        _conv2d_faster(
            x,
            (3,3),
            g_kernel,
            (1,1),
            (0,0),
            groups=2,
        )
    )
print(x)
# # sim1 = pb.Simulator(group_net, start_time_zero=False)
# # for i in range(inp_pad0.shape[-1]):
# #     pb.FRONTEND_ENV.save(data1=inp_pad0[:, :, i])
# #     sim1.run(1)
#
# # vgg16_net = VGGNet()
# # pb.BACKEND_CONFIG.target_chip_addr = (0, 0)
# #
# # mapper = pb.Mapper()
# # mapper.build(vgg16_net)
# # graph_info = mapper.compile(
# #         core_estimate_only=True,weight_bit_optimization=True
# #     )
# # print("Core required:", graph_info["n_core_required"])
#
#
# #vgg16_snn = snnVGGNet()
# # mapper1 = pb.Mapper()
# # mapper1.build(vgg16_snn)
# # graph_info1 = mapper1.compile(
# #         core_estimate_only=True,weight_bit_optimization=True
# #     )
# #     # #N of cores required
# # print("Core required:", graph_info1["n_core_required"])

