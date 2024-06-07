import copy

import numpy as np
from numpy._typing import NDArray

import paibox as pb
from paibox.components.functional import Conv_HalfRoll, Filter
from paibox.components.synapses import Conv2dHalfRollSyn
from paibox.components.synapses.conv_utils import _conv2d_halfroll


class fcnet_2layer_dual_port(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(5,))
        self.i2 = pb.InputProj(input=pe,shape_out=(5,))
        self.n1 = pb.IF(10, threshold=Vthr1, reset_v=0, name="delay_1")
        self.s1 = pb.FullConn(
            self.i1,
            self.n1,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n2 = pb.IF(
            5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="delay_2"
        )
        self.s2 = pb.FullConn(
            self.i2,
            self.n2,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n3 = pb.IF(
            20, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="IF_1"
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n3,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        # self.n3 = pb.IF(
        #     20, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="IF_2"
        # )
        # self.s4 = pb.FullConn(
        #     self.n2,
        #     self.n3,
        #     weights=weight1,
        #     conn_type=pb.SynConnType.All2All,
        # )

        # tick_wait_start = 2 for second layer

        # self.n3 = pb.IF(
        #     5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="batch_dual_port_o2"
        # )
        # self.s3 = pb.FullConn(
        #     self.n1,
        #     self.n2,
        #     weights=weight2,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s4 = pb.FullConn(
        #     self.n1,
        #     self.n3,
        #     weights=weight2,
        #     conn_type=pb.SynConnType.All2All,
        # )
        #
        # self.probe1 = pb.Probe(target=self.n2, attr="spike")
        # self.probe2 = pb.Probe(target=self.n3, attr="spike")

class fcnet_3(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(5,))
        self.i2 = pb.InputProj(input=pe,shape_out=(5,))
        self.n1 = pb.IF(5, threshold=Vthr1, reset_v=0, name="n_1")
        self.s1 = pb.FullConn(
            self.i1,
            self.n1,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n2 = pb.IF(
            5, threshold=Vthr2, reset_v=0, delay=2, tick_wait_start=2, name="n_2"
        )
        self.s2 = pb.FullConn(
            self.n1,
            self.n2,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n3 = pb.IF(
            5, threshold=Vthr2, reset_v=0, delay=3, tick_wait_start=2, name="n_3"
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n4 = pb.IF(
            10, threshold=Vthr2, reset_v=0, tick_wait_start=3, name="n_4"
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n4,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        #
        # self.n4 = pb.IF(
        #     20, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="n_4"
        # )
        self.s5 = pb.FullConn(
            self.n3,
            self.n4,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s6 = pb.FullConn(
            self.i2,
            self.n4,
            weights=weight1,
            conn_type=pb.SynConnType.All2All,
        )
# pb_net = fcnet_3(1,0,1,0)
# mapper = pb.Mapper()
# mapper.build(pb_net)
# graph_info = mapper.compile(
#         weight_bit_optimization=True
# )
# print(mapper.core_plm_config)
#     #N of cores required
# print("Core required:", graph_info["n_core_required"])
# for m in graph_info["members"]:
#     print(m)
#     #print(graph_info["members"][m].neuron_configs)
#     for k,v in graph_info["members"][m].neuron_configs.items():
#         print(k.name,v)


# kernel_shape = (1, 1, 3, 3)
# weight = np.random.rand(*kernel_shape)
# class fcnet_4(pb.DynSysGroup):
#     def __init__(self, Vthr1):
#         super().__init__()
#
#         self.n1 = pb.IF((1, 5), threshold=Vthr1, reset_v=0, name="n_1")
#         #self.i2 = pb.InputProj(input=pe,shape_out=(500,))
#         self.n2 = pb.IF((1, 3, 3), threshold=Vthr1, reset_v=0, name="n_2")
#         self.conv = Conv_HalfRoll(self.n1, self.n2, kernel=weight, stride=1)
# pb_net = fcnet_4(1)
# pb_net.module_construct()
# print(pb_net.components)


# x = np.random.randint(0, 2, 5)
# print(x)
weight1 = np.random.randint(0, 10, size=(1, 1, 3, 3), dtype=np.int8)
weight2 = np.random.randint(0, 10, size=(1, 1, 3, 3), dtype=np.int8)
weight3 = np.random.randint(0, 10, size=(1, 1, 3, 3), dtype=np.int8)
class fcnet_4(pb.DynSysGroup):
    def __init__(self):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(2, 5, 5))
        self.n1 = pb.IF((1, 7), threshold=1, reset_v=0, name="n_1")
        self.s0 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.n2 = pb.IF((1, 5, 5), threshold=1, reset_v=0, name="n_2")
        self.conv1 = pb.ConvHalfRoll(self.n1, self.n2, weight1)
        #self.n3 = pb.IF((1, 3, 3), threshold=1, reset_v=0, name="n_3")
        #self.conv2 = pb.ConvHalfRoll(self.n2, self.n3, weight2)
        self.n4 = pb.IF((10,), threshold=1, reset_v=0, name="n_4")
        self.linear = pb.DelayFullConn(
            self.n2,
            self.n4,
            delay=5,
            weights=np.random.randint(0, 10, size=(25, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.filter = pb.Filter(self.n4, 7)
        # self.n4 = pb.IF((1, 5), threshold=1, reset_v=0, name="n_4")
        # self.n5 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_5")
        #self.n4 = pb.IF((3, 5, 5), threshold=1, reset_v=0, name="n_4")
        #self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, weight3)
        # self.s1 = pb.FullConn(
        #     self.n1,
        #     self.n2,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s2 = pb.FullConn(
        #     self.n2,
        #     self.n3,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s3 = pb.FullConn(
        #     self.n1,
        #     self.n3,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s4 = pb.FullConn(
        #     self.n3,
        #     self.n4,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s5 = pb.FullConn(
        #     self.n4,
        #     self.n5,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.s6 = pb.FullConn(
        #     self.n3,
        #     self.n5,
        #     weights=1,
        #     conn_type=pb.SynConnType.All2All,
        # )
        #self.probe1 = pb.Probe(self.n1, "spike")

pb_net = fcnet_4()

mapper = pb.Mapper()
mapper.build(pb_net)
graph_info = mapper.compile()

print("Core required:", graph_info["n_core_required"])
#print(graph_info["members"])
for k, v in graph_info["members"].items():
    for c, coreplm in v.items():
        print(c)
        #print(coreplm.neuron_configs)
        for k, v in coreplm.neuron_configs.items():
            print(k.name,v)





