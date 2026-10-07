import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import neuron, surrogate, functional

from model import DVSGestureNet

device = "cuda" if torch.cuda.is_available() else "cpu"

root = os.path.join(os.path.dirname(__file__), "data", "DVS128Gesture")
T = 20

ds = DVS128Gesture(root, train=True, data_type="frame", frames_number=T, split_by="number")
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

frames, label = next(iter(loader))
if isinstance(frames, np.ndarray):
    frames = torch.from_numpy(frames)
if frames.shape[1] == T:
    frames = frames.transpose(0, 1)  # [T,N,2,128,128]
frames = frames.float().to(device)

net = DVSGestureNet(8, neuron.IFNode, surrogate_function=surrogate.ATan()).to(device)
functional.set_step_mode(net, "m")

out = net(frames)   # [T,N,11] 或 [T,N,?]
print("out shape:", out.shape)

assert out.dim() == 3, f"Expected [T,N,C], got {out.shape}"
assert out.shape[0] == T, f"T mismatch: out.shape[0]={out.shape[0]}, T={T}"
assert out.shape[1] == 4, f"Batch mismatch: out.shape[1]={out.shape[1]}"

functional.reset_net(net)