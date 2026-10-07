import os
import numpy as np
import torch
import torch.nn.functional as F
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
if isinstance(label, np.ndarray):
    label = torch.from_numpy(label)

if frames.shape[1] == T:
    frames = frames.transpose(0, 1)
frames = frames.float().to(device)
label = label.long().to(device)

net = DVSGestureNet(8, neuron.IFNode, surrogate_function=surrogate.ATan()).to(device)
functional.set_step_mode(net, "m")

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

out = net(frames)            # [T,N,11]
logits = out.mean(0)         # [N,11]
loss = F.cross_entropy(logits, label)

opt.zero_grad(set_to_none=True)
loss.backward()

# 检查梯度是否正常
grad_ok = True
for name, p in net.named_parameters():
    if p.grad is None:
        continue
    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
        print("BAD grad:", name)
        grad_ok = False
        break

print("loss:", float(loss.item()))
print("grad_ok:", grad_ok)

opt.step()
functional.reset_net(net)