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
    def __init__(self):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(2, 5, 5))
        self.n1 = pb.IF((1, 7), threshold=1, reset_v=0, name="n_1")
        self.n2 = pb.IF((1, 5, 5), threshold=1, reset_v=0, name="n_2")
        self.n3 = pb.IF((1, 5, 5), threshold=1, reset_v=0, name="n_3")
        self.n4 = pb.IF((1, 5), threshold=1, reset_v=0, name="n_4")
        self.n5 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_5")
        self.n6 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_6")
        self.n7 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_7")
        self.s0 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s1 = pb.FullConn(
            self.n1,
            self.n2,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s2 = pb.FullConn(
            self.n2,
            self.n3,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s4 = pb.FullConn(
            self.n3,
            self.n4,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s5 = pb.FullConn(
            self.n4,
            self.n5,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s6 = pb.FullConn(
            self.n3,
            self.n5,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s7 = pb.FullConn(
            self.n5,
            self.n6,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s8 = pb.FullConn(
            self.n6,
            self.n7,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        self.s9 = pb.FullConn(
            self.n5,
            self.n7,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
weight1 = np.random.randint(0, 10, size=(32, 1, 5, 5), dtype=np.int8)
weight2 = np.random.randint(0, 10, size=(32, 32, 2, 2), dtype=np.int8)
weight3 = np.random.randint(0, 10, size=(64, 32, 5, 5), dtype=np.int8)
weight4 = np.random.randint(0, 10, size=(64, 64, 2, 2), dtype=np.int8)

class Conv2d_Net(pb.Network):
    def __init__(self, Vthr1, Vthr2, Vthr3):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 28, 28))
        self.n1 = pb.IF((32, 24, 24), threshold=Vthr1, reset_v=0)
        self.conv2d_1 = pb.Conv2d(self.i1, self.n1, kernel=weight1, stride=1)

        self.n2 = pb.IF((32, 12, 12), threshold=Vthr2, reset_v=0, tick_wait_start=2)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, stride=2)

        self.n3 = pb.IF((64, 8, 8), threshold=Vthr3, reset_v=0, tick_wait_start=3)
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=weight3, stride=1)
        self.n4 = pb.IF((64, 4, 4), threshold=Vthr3, reset_v=0, tick_wait_start=4)
        self.conv2d_4 = pb.Conv2d(self.n3, self.n4, kernel=weight4, stride=2)
        self.n5 = pb.IF((256,), threshold=Vthr3, reset_v=0, tick_wait_start=5)
        self.fc1 = pb.FullConn(
            self.n4, self.n5, weights=np.random.randint(0, 10, size=(1024, 256), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )
        self.n6 = pb.IF((64,), threshold=Vthr3, reset_v=0, tick_wait_start=6)
        self.fc2 = pb.FullConn(
            self.n5, self.n6, weights=np.random.randint(0, 10, size=(256, 64), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )
        self.n7 = pb.IF((10,), threshold=Vthr3, reset_v=0, tick_wait_start=7)
        self.fc3 = pb.FullConn(
            self.n6, self.n7, weights=np.random.randint(0, 10, size=(64, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.n3, "spike")


def out_bypass1(t, data1, *args, **kwargs):
    return data1

input_data1 = np.array([[1,0,1,0,1],
                       [0,1,0,1,0],
                       [1,1,1,1,1],
                       [1,1,0,1,1],
                       [0,0,1,0,0],
                        [0,0,0,0,0]] , dtype=np.bool_)
input_data2 = np.array([1,0,1,0,1], dtype=np.bool_)
class fcnet_4(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 28, 28))
        #self.i1 = pb.InputProj(input=out_bypass1, shape_out=(1, 5))
        self.n1 = pb.IF((1, 28), threshold=4, reset_v=0, name="n_1")
        self.s0 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        # self.probe1 = pb.Probe(self.n1, "spike")
        self.n2 = pb.IF((32, 24, 24), threshold=0, reset_v=0, name="n_2")
        #self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, np.array([[[[2,1,2],[1,2,1],[1,2,3]]]], dtype=np.int8), 1, tick_wait_start=1)
        self.conv1 = pb.ConvHalfRoll(self.n1, self.n2, weight1, 1)
        self.n3 = pb.IF((32, 12, 12), threshold=1, reset_v=0, name="n_3")
        self.conv2 = pb.ConvHalfRoll(self.n2, self.n3, weight2, 2)
        self.n4 = pb.IF((64, 8, 8), threshold=1, reset_v=0, name="n_4")
        self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, weight3, 1)
        self.n5 = pb.IF((64, 4, 4), threshold=1, reset_v=0, name="n_5")
        self.conv4 = pb.ConvHalfRoll(self.n4, self.n5, weight4, 2)
        self.n6 = pb.IF((256,), threshold=1, reset_v=0, name="n_6")
        self.linear1 = pb.DelayFullConn(
            self.n5,
            self.n6,
            delay=4,
            weights=np.random.randint(0, 10, size=(1024, 256), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.n7 = pb.IF((64,), threshold=1, reset_v=0, name="n_7")
        self.linear2 = pb.FullConn(
            self.n6,
            self.n7,
            weights=np.random.randint(0, 10, size=(256, 64), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.n8 = pb.IF((10,), threshold=1, reset_v=0, name="n_8")
        self.linear2 = pb.FullConn(
            self.n7,
            self.n8,
            weights=np.random.randint(0, 10, size=(64, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.filter = pb.Filter(self.n8, 28)

class fcnet_5(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(3, 224, 224))
        #self.i1 = pb.InputProj(input=out_bypass1, shape_out=(1, 5))
        self.n1 = pb.IF((3, 224), threshold=4, reset_v=0, name="n_1")
        self.s0 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        # self.probe1 = pb.Probe(self.n1, "spike")
        self.n2 = pb.IF((64, 110, 110), threshold=0, reset_v=0, name="n_2")
        #self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, np.array([[[[2,1,2],[1,2,1],[1,2,3]]]], dtype=np.int8), 1, tick_wait_start=1)
        self.conv1 = pb.ConvHalfRoll(self.n1, self.n2, np.random.randint(0, 10, size=(64, 3, 7, 7), dtype=np.int8), 2)
        self.n3 = pb.IF((64, 55, 55), threshold=1, reset_v=0, name="n_3")
        self.conv2 = pb.ConvHalfRoll(self.n2, self.n3, np.random.randint(0, 10, size=(64, 64, 3, 3), dtype=np.int8), 2)

        self.n4 = pb.IF((64, 55, 55), threshold=1, reset_v=0, name="n_4")
        self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, np.random.randint(0, 10, size=(64, 64, 2, 2),  dtype=np.int8),1)
        self.n5 = pb.IF((64, 55, 55), threshold=1, reset_v=0, name="n_5")
        self.conv4 = pb.ConvHalfRoll(self.n4, self.n5, np.random.randint(0, 10, size=(64, 64, 2, 2),  dtype=np.int8), 1)
        self.n6 = pb.IF((128, 27, 27), threshold=1, reset_v=0)
        self.conv4 = pb.ConvHalfRoll(self.n5, self.n6, np.random.randint(0, 10, size=(128, 64, 3, 3), dtype=np.int8), 2)
        # self.n6 = pb.IF((256,), threshold=1, reset_v=0, name="n_6")
        # self.linear1 = pb.DelayFullConn(
        #     self.n5,
        #     self.n6,
        #     delay=4,
        #     weights=np.random.randint(0, 10, size=(1024, 256), dtype=np.int8),
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.n7 = pb.IF((64,), threshold=1, reset_v=0, name="n_7")
        # self.linear2 = pb.FullConn(
        #     self.n6,
        #     self.n7,
        #     weights=np.random.randint(0, 10, size=(256, 64), dtype=np.int8),
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.n8 = pb.IF((10,), threshold=1, reset_v=0, name="n_8")
        # self.linear2 = pb.FullConn(
        #     self.n7,
        #     self.n8,
        #     weights=np.random.randint(0, 10, size=(64, 10), dtype=np.int8),
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.filter = pb.Filter(self.n8, 28)
pb_net = fcnet_5()
#pb_net = Conv2d_Net(1,1,1)
#generated = pb.DynSysGroup.build_fmodule(pb_net)

# neu1 = pb.IF((1, 3, 3), threshold=1, reset_v=0, name="neu1")
# neu1.shape_change((1,3))
# neu1.update(x=np.array([0,0,0]))

# sim1 = pb.Simulator(pb_net, start_time_zero=False)
#
# for i in range(6):
#     pb.FRONTEND_ENV.save(data1=input_data1[i])
#     sim1.run(1)
#     print(pb_net.n1.voltage)
#     # spike_out = sim1.data[pb_net.probe1].astype(np.int8)
#     # print(spike_out[i])

mapper = pb.Mapper()
mapper.build(pb_net)
graph_info = mapper.compile(core_estimate_only=True)

print("Core required:", graph_info["n_core_required"])
# #print(graph_info["members"])
# for k, v in graph_info["members"].items():
#     for c, coreplm in v.items():
#         print(c)
#         for k, v in coreplm.neuron_configs.items():
#             print(k.name,v)
#             for n,s in k.master_nodes.items():
#                 print(s.name)
#                 # print(s.connectivity)


