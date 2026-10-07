import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

root = os.path.join(os.path.dirname(__file__), "data", "DVS128Gesture")
T = 20

ds = DVS128Gesture(root, train=True, data_type="frame", frames_number=T, split_by="number")
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

frames, label = next(iter(loader))
print("frames type:", type(frames))
print("label type:", type(label))
print("frames shape:", getattr(frames, "shape", None))
print("label shape:", getattr(label, "shape", None))

# 统一 torch
if isinstance(frames, np.ndarray):
    frames = torch.from_numpy(frames)
if isinstance(label, np.ndarray):
    label = torch.from_numpy(label)

print("frames dtype:", frames.dtype)
print("label dtype:", label.dtype)

# 期望通常是 [N, T, 2, 128, 128]
assert frames.dim() == 5, f"Expected 5D frames batch, got {frames.dim()}D"
assert 2 in frames.shape, f"Channel=2 missing? shape={frames.shape}"

# 转成多步输入 [T, N, 2, 128, 128]
if frames.shape[1] == T:
    frames_TN = frames.transpose(0, 1)
elif frames.shape[0] == T:
    frames_TN = frames
else:
    raise RuntimeError(f"Cannot find T dimension. frames.shape={frames.shape}")

print("frames_TN shape:", frames_TN.shape)  # [T,N,2,128,128]