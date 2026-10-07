import os
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

root = os.path.join(os.path.dirname(__file__), "data", "DVS128Gesture")

ds = DVS128Gesture(root, train=True, data_type="event")
print("len(train):", len(ds))

sample, label = ds[0]
print("label:", label)
print("keys:", sample.keys())
for k, v in sample.items():
    print(k, type(v), getattr(v, "shape", None))