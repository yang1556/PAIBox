import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch.utils.data import DataLoader


# ======================
# Original model (with BN)
# ======================
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.AvgPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.1),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.1),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

# ======================
# BN-free model
# ======================
class DVSGestureNet_NoBN(nn.Module):
    def __init__(self, channels=8, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        blocks = []
        for i in range(5):
            in_channels = 2 if i == 0 else channels
            blocks.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
            blocks.append(spiking_neuron(**kwargs))
            blocks.append(layer.AvgPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *blocks,
            layer.Flatten(),
            layer.Dropout(0.1),
            layer.Linear(channels * 4 * 4, 512, bias=True),
            spiking_neuron(**kwargs),

            layer.Dropout(0.1),
            layer.Linear(512, 110, bias=True),
            spiking_neuron(**kwargs),

            layer.VotingLayer(10),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

@torch.no_grad()
def load_fp_checkpoint(net: nn.Module, ckpt_path: str, device):
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    return missing, unexpected

# ======================
# load int checkpoint as float (keep integer values)
# ======================
@torch.no_grad()
def load_int_state_dict_as_float(net: nn.Module, pt_path: str, device):
    int_sd = torch.load(pt_path, map_location=device)
    float_sd = {}
    for k, v in int_sd.items():
        if isinstance(v, torch.Tensor) and v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            float_sd[k] = v.float()
        else:
            float_sd[k] = v
    missing, unexpected = net.load_state_dict(float_sd, strict=False)
    return missing, unexpected


@torch.no_grad()
def evaluate(net, loader, device, T: int):
    net.eval()
    total, correct, total_loss = 0, 0, 0.0

    for frames, label in loader:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        if frames.dim() == 5 and frames.shape[1] == T:
            frames = frames.transpose(0, 1)  # [T,N,2,128,128]

        frames = frames.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        out = net(frames)      # [T, N, 11]
        logits = out.sum(0)    # [N, 11]
        loss = F.cross_entropy(logits, label)

        pred = logits.argmax(1)
        total += label.numel()
        correct += (pred == label).sum().item()
        total_loss += loss.item() * label.size(0)

        functional.reset_net(net)

    return total_loss / max(1, total), correct / max(1, total)


def build_test_loader(args):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    data_root = resolve_data_root(args)

    test_set = DVS128Gesture(
        root=data_root,
        train=False,
        data_type="frame",
        frames_number=args.T,
        split_by=args.split_by,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    return test_loader


def resolve_data_root(args) -> str:
    base_dir = os.path.dirname(__file__)
    return args.data_root or os.path.join(base_dir, "data", "DVS128Gesture")


def test_frame_root(args) -> Path:
    return (
        Path(resolve_data_root(args))
        / f"frames_number_{args.T}_split_by_{args.split_by}"
        / "test"
    )


def class_label_map(classes: list[str]) -> dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def collect_test_frame_samples(args) -> list[tuple[Path, int]]:
    root = test_frame_root(args)
    if not root.is_dir():
        raise FileNotFoundError(
            f"test frame directory not found: {root}. "
            "It should match build_test_loader's DVS128Gesture frame root."
        )

    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class directories found in {root}")
    class_to_idx = class_label_map(classes)

    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        class_dir = root / class_name
        for current_root, _dirs, filenames in sorted(os.walk(class_dir)):
            for filename in sorted(filenames):
                if filename.endswith(".npz"):
                    samples.append(
                        (Path(current_root) / filename, class_to_idx[class_name])
                    )

    if not samples:
        raise FileNotFoundError(f"no .npz frame samples found in {root}")
    return samples


def resolve_sample_path(args) -> tuple[Path, int | None]:
    if args.sample_path is None:
        samples = collect_test_frame_samples(args)
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise IndexError(
                f"--sample_index={args.sample_index} is out of range for "
                f"{len(samples)} test samples"
            )
        return samples[args.sample_index]

    requested = Path(args.sample_path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend(
            [
                Path.cwd() / requested,
                Path(__file__).resolve().parent / requested,
                test_frame_root(args) / requested,
                Path(resolve_data_root(args)) / requested,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            target = target_from_test_frame_path(candidate, args)
            return candidate, target

    raise FileNotFoundError(f"sample path not found: {args.sample_path}")


def target_from_test_frame_path(path: Path, args) -> int | None:
    root = test_frame_root(args).resolve()
    try:
        class_name = path.resolve().relative_to(root).parts[0]
    except (IndexError, ValueError):
        return None

    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    return class_label_map(classes).get(class_name)


def folder_label_from_path(path: Path) -> int | None:
    try:
        return int(path.parent.name)
    except ValueError:
        return None


def load_single_sample(args) -> tuple[np.ndarray, int | None, Path]:
    path, target = resolve_sample_path(args)
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data:
            raise ValueError(f"{path} does not contain a 'frames' array")
        frames = np.asarray(data["frames"], dtype=np.float32)

    expected_shape = (args.T, 2, 128, 128)
    if frames.shape != expected_shape:
        raise ValueError(f"{path} has shape {frames.shape}; expected {expected_shape}")
    return frames, target, path


@torch.no_grad()
def infer_one_sample(net, frames: np.ndarray, target: int | None, device, T: int):
    net.eval()
    frames_tensor = torch.from_numpy(frames)
    if frames_tensor.dim() == 4 and frames_tensor.shape[0] == T:
        frames_tensor = frames_tensor.unsqueeze(1)  # [T, 1, 2, 128, 128]
    elif frames_tensor.dim() == 5 and frames_tensor.shape[1] == T:
        frames_tensor = frames_tensor.transpose(0, 1)
    else:
        raise ValueError(f"unexpected single sample shape: {tuple(frames_tensor.shape)}")

    frames_tensor = frames_tensor.float().to(device, non_blocking=True)

    try:
        out = net(frames_tensor)  # [T, 1, 11]
        logits = out.sum(0)  # [1, 11], same reduction as evaluate()
    finally:
        functional.reset_net(net)

    probabilities = torch.softmax(logits, dim=1)
    pred = int(logits.argmax(1).item())
    confidence = float(probabilities[0, pred].item())
    loss = None
    correct = None
    if target is not None:
        label = torch.tensor([target], dtype=torch.long, device=device)
        loss = float(F.cross_entropy(logits, label).item())
        correct = pred == target

    return {
        "pred": pred,
        "confidence": confidence,
        "loss": loss,
        "correct": correct,
        "logits": logits.squeeze(0).detach().cpu().numpy(),
    }


def print_single_result(mode: str, path: Path, target: int | None, result: dict) -> None:
    try:
        sample = path.resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        sample = path

    folder_label = folder_label_from_path(path)
    print(f"[{mode.upper()} SINGLE] sample={sample}")
    print(
        f"[{mode.upper()} SINGLE] loader_target={target} "
        f"folder_label={folder_label}"
    )
    print(
        f"[{mode.upper()} SINGLE] pred={result['pred']} "
        f"confidence={result['confidence']:.6f} "
        f"loss={result['loss']} correct={result['correct']}"
    )
    print(f"[{mode.upper()} SINGLE] logits={result['logits']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="fused_int",
                        choices=["fused_int", "fp"],
                        help="fused_int: run BN-free net with fused_int8x127; fp: run original net with BN")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--split_by", type=str, default="number", choices=["number", "time"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_all", action="store_true",
                        help="Run the original full test-loader evaluation instead of one sample.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index in the test frame sample list used when --sample_path is not set.")
    parser.add_argument("--sample_path", type=str, default=None,
                        help="A single .npz frame sample under the DVS128Gesture test frame directory.")

    # checkpoints
    parser.add_argument("--ckpt_fp", type=str, default=None,
                        help="Float checkpoint (.pt/.pth) for original model (with BN)")
    parser.add_argument("--ckpt_fused_int", type=str, default=None,
                        help="Fused int checkpoint *_fused_int8x127.pt for BN-free model")

    # threshold control (optional)
    parser.add_argument("--vth", type=float, default=None,
                        help="Override IFNode v_threshold. If not set: fp uses default(1.0), fused_int uses 127.0")

    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_loader = build_test_loader(args) if args.eval_all else None

    if args.mode == "fp":
        if args.ckpt_fp is None:
            raise ValueError("--mode fp requires --ckpt_fp")

        # fp mode: keep default vth (usually 1.0) unless overridden
        kwargs = dict(surrogate_function=surrogate.ATan())
        if args.vth is not None:
            kwargs["v_threshold"] = args.vth

        net = DVSGestureNet(
            channels=args.channels,
            spiking_neuron=neuron.IFNode,
            **kwargs
        ).to(device)

        functional.set_step_mode(net, "m")
        net.eval()

        missing, unexpected = load_fp_checkpoint(net, args.ckpt_fp, device)
        print(f"[FP LOAD] missing={len(missing)} unexpected={len(unexpected)}")

        if args.eval_all:
            loss, acc = evaluate(net, test_loader, device, args.T)
            print(f"[FP EVAL] loss={loss:.4f} acc={acc:.4f}")
        else:
            frames, target, path = load_single_sample(args)
            result = infer_one_sample(net, frames, target, device, args.T)
            print_single_result(args.mode, path, target, result)
        return

    # fused_int mode
    if args.ckpt_fused_int is None:
        raise ValueError("--mode fused_int requires --ckpt_fused_int")

    vth = 127.0 if args.vth is None else args.vth
    net = DVSGestureNet_NoBN(
        channels=args.channels,
        spiking_neuron=neuron.IFNode,
        v_threshold=vth,
        surrogate_function=surrogate.ATan(),
    ).to(device)

    functional.set_step_mode(net, "m")
    net.eval()

    missing, unexpected = load_int_state_dict_as_float(net, args.ckpt_fused_int, device)
    print(f"[FUSED_INT LOAD] missing={len(missing)} unexpected={len(unexpected)}")

    if args.eval_all:
        loss, acc = evaluate(net, test_loader, device, args.T)
        print(f"[FUSED_INT EVAL] loss={loss:.4f} acc={acc:.4f}")
    else:
        frames, target, path = load_single_sample(args)
        result = infer_one_sample(net, frames, target, device, args.T)
        print_single_result(args.mode, path, target, result)


if __name__ == "__main__":
    main()

# 测试命令：
# 跑原版网络
# python infer.py --mode fp --ckpt_fp checkpoints/dropout_0.1_sum_qat.pt --vth 1 --eval_all
# [FP EVAL] loss=0.5760 acc=0.8542 识别正确率为85.42%

# 跑量化后的网络
# python infer.py --mode fused_int --ckpt_fused_int checkpoints/dropout_0.1_sum_qat_int8.pt --eval_all
# [FUSED_INT EVAL] loss=0.6058 acc=0.8194 识别正确率为81.94%

# 临时单样本推理：
# python infer.py --mode fused_int --ckpt_fused_int checkpoints/dropout_0.1_sum_qat_int8.pt --sample_index 0
# python infer.py --mode fused_int --ckpt_fused_int checkpoints/dropout_0.1_sum_qat_int8.pt --sample_path data/DVS128Gesture/frames_number_20_split_by_number/test/0/user24_fluorescent_0.npz

# 考虑到batchnorm不便进行int量化，故训练完毕得到初版权重后，在推理部署时将BN层融入进了conv2d层，如最上方网络结构所示
