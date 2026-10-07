import argparse
import csv
import time
from pathlib import Path
from typing import Any, TextIO

import serial

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2] / "debug" / "ref" / "frame_input.txt"
)
DEFAULT_PORT = "/dev/ttyUSB2"
DEFAULT_BAUD = 3_000_000
FRAME_BYTES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send 64-bit frames from debug/ref/frame_input.txt over UART."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--sync-timeout", type=float, default=10.0)
    parser.add_argument("--final-drain", type=float, default=1.0)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV path for every transmitted and received 64-bit frame.",
    )
    parser.add_argument(
        "--no-wait-complete",
        action="store_true",
        help="Do not wait for a completion frame after sync frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = load_frames(args.input_path)
    wait_complete = not args.no_wait_complete

    tx_frames = 0
    rx_frames = 0
    sync_seen = 0
    sync_completed = 0
    started = time.monotonic()

    log_file: TextIO | None = None
    log_writer: csv.writer | None = None
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(("time", "direction", "index", "sync", "frame", "note"))

    try:
        with serial.Serial(
            args.port, args.baud, timeout=0.02, write_timeout=2.0
        ) as ser:
            time.sleep(0.2)
            ser.reset_input_buffer()

            for index, frame in enumerate(frames):
                current_sync = sync_seen if is_sync_frame(frame) else sync_seen - 1
                ser.write(frame.to_bytes(FRAME_BYTES, "big"))
                tx_frames += 1
                log_frame(
                    log_writer,
                    "TX",
                    index,
                    current_sync,
                    frame,
                    "sync" if is_sync_frame(frame) else "",
                )

                if not is_sync_frame(frame):
                    continue

                sync_seen += 1
                if not wait_complete:
                    continue

                complete, count = wait_for_complete(
                    ser,
                    args.sync_timeout,
                    log_writer=log_writer,
                    sync_index=current_sync,
                )
                rx_frames += count
                if not complete:
                    print(
                        f"timeout index={index} sync={current_sync} "
                        f"tx_frames={tx_frames} rx_frames={rx_frames}"
                    )
                    if args.log_path is not None:
                        print(f"log_path={args.log_path}")
                    return 2
                sync_completed += 1

            rx_frames += drain_frames(ser, args.final_drain, log_writer, sync_seen - 1)
    finally:
        if log_file is not None:
            log_file.close()

    print(f"input_path={args.input_path}")
    print(f"port={args.port}")
    print(f"baud={args.baud}")
    print(f"wait_complete={wait_complete}")
    print("stop_reason=completed")
    print(f"input_frames_total={len(frames)}")
    print(f"tx_frames={tx_frames}")
    print(f"rx_frames={rx_frames}")
    print(f"syncs_seen={sync_seen}")
    print(f"syncs_completed={sync_completed}")
    if args.log_path is not None:
        print(f"log_path={args.log_path}")
    print(f"duration_s={time.monotonic() - started:.6f}")
    return 0


def load_frames(path: Path) -> list[int]:
    frames: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            base = 2 if len(text) == 64 and set(text) <= {"0", "1"} else 16
            frame = int(text, base)
            if not 0 <= frame <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"{path}:{line_no}: frame does not fit in uint64")
            frames.append(frame)
    if not frames:
        raise ValueError(f"{path}: no frames found")
    return frames


def is_sync_frame(frame: int) -> bool:
    high = (frame >> 32) & 0xFFFFFFFF
    return ((high >> 30) & 0x3) == 0x3 and ((high >> 28) & 0x3) == 0x0


def is_complete_frame(frame: int) -> bool:
    high = (frame >> 32) & 0xFFFFFFFF
    return ((high >> 28) & 0xF) == 0xE


def wait_for_complete(
    ser: serial.Serial,
    timeout: float,
    *,
    log_writer: Any | None,
    sync_index: int,
) -> tuple[bool, int]:
    deadline = time.monotonic() + timeout
    count = 0
    while time.monotonic() < deadline:
        frame = read_frame(ser, deadline)
        if frame is None:
            break
        count += 1
        note = "complete" if is_complete_frame(frame) else ""
        log_frame(log_writer, "RX", None, sync_index, frame, note)
        if is_complete_frame(frame):
            return True, count
    return False, count


def drain_frames(
    ser: serial.Serial,
    duration: float,
    log_writer: Any | None,
    sync_index: int,
) -> int:
    deadline = time.monotonic() + duration
    count = 0
    while time.monotonic() < deadline:
        frame = read_frame(ser, deadline)
        if frame is None:
            break
        count += 1
        note = "late_complete" if is_complete_frame(frame) else "late"
        log_frame(log_writer, "RX", None, sync_index, frame, note)
    return count


def read_frame(ser: serial.Serial, deadline: float) -> int | None:
    data = bytearray()
    while len(data) < FRAME_BYTES and time.monotonic() < deadline:
        chunk = ser.read(FRAME_BYTES - len(data))
        if chunk:
            data.extend(chunk)
    if len(data) != FRAME_BYTES:
        return None
    return int.from_bytes(data, "big")


def log_frame(
    writer: Any | None,
    direction: str,
    index: int | None,
    sync: int,
    frame: int,
    note: str,
) -> None:
    if writer is None:
        return
    writer.writerow(
        (
            f"{time.time():.6f}",
            direction,
            "" if index is None else index,
            sync,
            f"{frame:016X}",
            note,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
