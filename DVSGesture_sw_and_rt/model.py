import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer
# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks https://arxiv.org/abs/2007.05785

__all__ = [
    "MNISTNet",
    "FashionMNISTNet",
    "NMNISTNet",
    "CIFAR10Net",
    "CIFAR10DVSNet",
    "DVSGestureNet",
]


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.AvgPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer(10),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


def test_models():
    import torch
    from spikingjelly.activation_based import neuron, surrogate, functional

    x = torch.rand([2, 2, 128, 128])
    net = DVSGestureNet(8, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net)
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x


if __name__ == "__main__":
    test_models()
