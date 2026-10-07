import os
import numpy as np
import torch
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

root = os.path.join(os.path.dirname(__file__), "data", "DVS128Gesture")
T = 20

ds = DVS128Gesture(root, train=True, data_type="frame", frames_number=T, split_by="number")
frames, label = ds[0]

print("label:", label)
print("frames type:", type(frames))
print("frames shape:", getattr(frames, "shape", None))

# 统一转 torch tensor
if isinstance(frames, np.ndarray):
    frames_t = torch.from_numpy(frames)
else:
    frames_t = frames

print("frames_t type:", type(frames_t))
print("min/max:", frames_t.min().item(), frames_t.max().item())
print("unique values (first frame):", torch.unique(frames_t[0]).tolist()[:10])