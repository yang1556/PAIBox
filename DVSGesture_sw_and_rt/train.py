import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import neuron, surrogate, functional

from model import DVSGestureNet


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(net, loader, device, T: int):
    net.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    for frames, label in loader:
        # frames: np.ndarray or torch.Tensor
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        # [N, T, 2, 128, 128] -> [T, N, 2, 128, 128]
        if frames.dim() == 5 and frames.shape[1] == T:
            frames = frames.transpose(0, 1)

        frames = frames.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        out = net(frames)          # [T, N, 11]
        logits = out.mean(0)       # [N, 11]
        loss = F.cross_entropy(logits, label)

        pred = logits.argmax(1)
        total += label.numel()
        correct += (pred == label).sum().item()
        total_loss += loss.item() * label.size(0)

        functional.reset_net(net)

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU index to use (e.g., 0,1,2...)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data/DVS128Gesture. Default: ./data/DVS128Gesture")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--split_by", type=str, default="number", choices=["number", "time"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="dvsgesture_ch8_ifnode_T20.pt")

    args = parser.parse_args()

    # ===== 指定 GPU =====
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seed_all(args.seed)

    base_dir = os.path.dirname(__file__)
    data_root = args.data_root or os.path.join(base_dir, "data", "DVS128Gesture")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    # ===== Dataset =====
    train_set = DVS128Gesture(
        root=data_root,
        train=True,
        data_type="frame",
        frames_number=args.T,
        split_by=args.split_by,
    )
    test_set = DVS128Gesture(
        root=data_root,
        train=False,
        data_type="frame",
        frames_number=args.T,
        split_by=args.split_by,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
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

    # ===== Model =====
    net = DVSGestureNet(
        channels=args.channels,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
    ).to(device)

    # 多步训练
    functional.set_step_mode(net, "m")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (args.amp_dtype == "fp16"))

    best_test_acc = -1.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for it, (frames, label) in enumerate(train_loader, start=1):
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)

            # [N,T,2,128,128] -> [T,N,2,128,128]
            if frames.dim() == 5 and frames.shape[1] == args.T:
                frames = frames.transpose(0, 1)

            frames = frames.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = net(frames)          # [T,N,11]
                    logits = out.mean(0)       # [N,11]
                    loss = F.cross_entropy(logits, label)

                if args.amp_dtype == "fp16":
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    # bf16 不需要 scaler
                    loss.backward()
                    opt.step()
            else:
                out = net(frames)
                logits = out.mean(0)
                loss = F.cross_entropy(logits, label)
                loss.backward()
                opt.step()

            # 关键：每个 batch reset
            functional.reset_net(net)

            with torch.no_grad():
                pred = logits.argmax(1)
                running_total += label.numel()
                running_correct += (pred == label).sum().item()
                running_loss += loss.item() * label.size(0)

            if it % args.log_interval == 0:
                avg_loss = running_loss / max(1, running_total)
                acc = running_correct / max(1, running_total)
                elapsed = (time.time() - t_start) / 60.0
                print(f"Epoch {epoch:03d} Iter {it:04d} "
                      f"loss={avg_loss:.4f} acc={acc:.4f} "
                      f"elapsed={elapsed:.2f}min")

        # epoch end eval
        test_loss, test_acc = evaluate(net, test_loader, device, args.T)
        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        # save best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt = {
                "epoch": epoch,
                "best_test_acc": best_test_acc,
                "state_dict": net.state_dict(),
                "channels": args.channels,
                "T": args.T,
                "split_by": args.split_by,
                "lr": args.lr,
            }
            torch.save(ckpt, save_path)
            print(f"Saved best checkpoint to: {save_path} (best_test_acc={best_test_acc:.4f})")

    total_min = (time.time() - t_start) / 60.0
    print(f"Training finished. Total time: {total_min:.2f} min. Best test acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()