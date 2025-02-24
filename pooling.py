import numpy as np

import paibox as pb
def _out_bypass1(t, data1, *args, **kwargs):
    return data1
class Net(pb.DynSysGroup):
    def __init__(self, shape):
        super().__init__()

        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape, keep_shape=True)


        self.maxpool = pb.MaxPooling2d(
            self.i1,
            2,
            2,
            tick_wait_start=1
        )

        self.avgpool = pb.AvgPooling2d(
            self.i1,
            2,
            2,
            tick_wait_start=1
        )

shape = (1, 8, 8)
net = Net(shape)

inpa = np.random.default_rng(42).integers(0, 10, size=shape, dtype=np.uint8)
print(inpa)
maxpool = net.maxpool
avgpool = net.avgpool
# generated = net.build_modules()
sim1 = pb.Simulator(net, start_time_zero=False)
# probe_maxpool = pb.Probe(generated[maxpool][0], "output")
# probe_avgpool = pb.Probe(generated[avgpool][0], "output")
probe_maxpool = pb.Probe(maxpool, "spike")
probe_avgpool = pb.Probe(avgpool, "spike")

sim1.add_probe(probe_maxpool)
sim1.add_probe(probe_avgpool)

for i in range(1):
    pb.FRONTEND_ENV.save(data1=inpa)
    sim1.run(1)

print(sim1.data[probe_maxpool])
print(sim1.data[probe_avgpool])


