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
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)

frames, label = next(iter(loader))
if isinstance(frames, np.ndarray):
    frames = torch.from_numpy(frames)
if frames.shape[1] == T:
    frames = frames.transpose(0, 1)     # [T,N,2,128,128]
frames = frames.float().to(device)

net = DVSGestureNet(8, neuron.IFNode, surrogate_function=surrogate.ATan()).to(device)
net.eval()

# 多步 m
functional.set_step_mode(net, "m")
functional.reset_net(net)
out_m = net(frames)           # [T,N,11]
logits_m = out_m.mean(0)      # [N,11]

# 单步 s：逐帧喂
functional.set_step_mode(net, "s")
functional.reset_net(net)
acc = 0
for t in range(T):
    y_t = net(frames[t])      # frames[t]: [N,2,128,128] -> [N,11]
    acc = acc + y_t
logits_s = acc / T

diff = (logits_m - logits_s).abs().mean().item()
print("mean abs diff(m vs s):", diff)