import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, TextIO

import dv_processing as dv
import numpy as np
import torch
from infer import DVSGestureNet_NoBN, load_int_state_dict_as_float
from spikingjelly.activation_based import functional, neuron, surrogate

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = APP_DIR / "data" / "DVS128Gesture"
DEFAULT_CHECKPOINT = APP_DIR / "checkpoints" / "dropout_0.1_sum_qat_int8.pt"
DEFAULT_FRAME_ROOT = "frames_number_20_split_by_number"
EVENT_ROOT_NAME = "events_np"
TARGET_HW = (128, 128)
CHANNELS = 8
IF_THRESHOLD = 127.0
TIMESTEPS = 20
DATASET_SPLIT = "test"
DATASET_EVENT_SPLIT = "number"
CAMERA_EVENT_SPLIT = "time"
CAMERA_WINDOW_MS = 1000.0
CAMERA_IDLE_SLEEP_MS = 1.0
MIN_CAMERA_EVENTS = 1
DISPLAY_WINDOW_NAME = "DVS Gesture Camera"
DISPLAY_FPS = 30.0
DISPLAY_DECAY_HALFLIFE_MS = 120.0
DISPLAY_MIN_SIDE = 384
DISPLAY_MAX_SIDE = 900

DEFAULT_LABEL_NAMES = {
    0: "hand_clapping",
    1: "right_hand_wave",
    2: "other_gestures",
    3: "left_hand_wave",
    4: "right_arm_clockwise",
    5: "right_arm_counter_clockwise",
    6: "left_arm_clockwise",
    7: "left_arm_counter_clockwise",
    8: "arm_roll",
    9: "air_drums",
    10: "air_guitar",
}


@dataclass(frozen=True)
class EventArrays:
    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray


@dataclass(frozen=True)
class Prediction:
    source: str
    sample: str
    predicted_label: int
    predicted_name: str
    confidence: float
    latency_ms: float
    event_count: int | None
    expected_label: int | None = None
    expected_name: str | None = None
    correct: bool | None = None


class JsonlWriter:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._file: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._file is not None:
            self._file.close()

    def write(self, prediction: Prediction) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")
        self._file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the quantized DVS128 Gesture model from infer.py on either "
            "local DVS128Gesture samples or a live dv-processing camera."
        )
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--jsonl", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser(
        "dataset", help="Use local DVS128Gesture .npz samples as input."
    )
    dataset.add_argument("--source", default="frames", choices=["frames", "events"])
    dataset.add_argument("--sample", type=Path, default=None)
    dataset.add_argument("--limit", type=int, default=8)

    camera = subparsers.add_parser(
        "camera", help="Use a live iniVation-compatible DVS camera."
    )
    camera.add_argument("--camera-name", default=None)
    camera.add_argument("--window-ms", type=float, default=CAMERA_WINDOW_MS)
    camera.add_argument("--max-windows", type=int, default=0)
    camera.add_argument(
        "--display",
        action="store_true",
        help="Show a continuous OpenCV view of the live event stream.",
    )
    camera.add_argument(
        "--resize-mode",
        default="scale",
        choices=["scale", "crop", "strict"],
        help="How to map non-128x128 camera events into the trained input size.",
    )

    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device) -> torch.nn.Module:
    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"checkpoint not found: {DEFAULT_CHECKPOINT}")

    model = DVSGestureNet_NoBN(
        channels=CHANNELS,
        spiking_neuron=neuron.IFNode,
        v_threshold=IF_THRESHOLD,
        surrogate_function=surrogate.ATan(),
    ).to(device)
    functional.set_step_mode(model, "m")
    model.eval()

    missing, unexpected = load_int_state_dict_as_float(
        model, str(DEFAULT_CHECKPOINT), device
    )
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint did not load cleanly: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model


def load_label_names(data_root: Path) -> dict[int, str]:
    names = dict(DEFAULT_LABEL_NAMES)
    mapping = data_root / "download" / "gesture_mapping.csv"
    if not mapping.is_file():
        return names

    for line in mapping.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        action, label = line.rsplit(",", maxsplit=1)
        index = model_label_from_raw_label(int(label.strip()) - 1)
        action = action.strip()
        existing = names.get(index)
        if existing is None:
            names[index] = action
        elif action not in existing.split("/"):
            names[index] = f"{existing}/{action}"
    return names


def iter_dataset_samples(
    source: str,
    sample: Path | None,
    limit: int,
) -> Iterable[tuple[Path, int | None]]:
    if sample is not None:
        label = _label_from_path(sample)
        yield sample, label
        return

    if source == "frames":
        root = DEFAULT_DATA_ROOT / DEFAULT_FRAME_ROOT / DATASET_SPLIT
    else:
        root = DEFAULT_DATA_ROOT / EVENT_ROOT_NAME / DATASET_SPLIT

    if not root.is_dir():
        raise FileNotFoundError(f"dataset input directory not found: {root}")

    yielded = 0
    for path in sorted(root.glob("*/*.npz"), key=_dataset_sort_key):
        yield path, _label_from_path(path)
        yielded += 1
        if limit > 0 and yielded >= limit:
            break


def _dataset_sort_key(path: Path) -> tuple[int, str]:
    label = _label_from_path(path)
    return (999 if label is None else label, path.name)


def _label_from_path(path: Path) -> int | None:
    try:
        return model_label_from_raw_label(int(path.parent.name))
    except ValueError:
        return None


def model_label_from_raw_label(raw_label: int) -> int:
    return {int(name): idx for idx, name in enumerate(sorted(str(i) for i in range(11)))}[
        raw_label
    ]


def load_frames_npz(path: Path, timesteps: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data:
            raise ValueError(f"{path} does not contain a 'frames' array")
        frames = np.asarray(data["frames"], dtype=np.float32)
    return validate_frames(frames, timesteps, path)


def load_event_npz(path: Path) -> EventArrays:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"t", "x", "y", "p"} - keys:
            raise ValueError(f"{path} does not contain t/x/y/p arrays")
        return EventArrays(
            timestamps=np.asarray(data["t"], dtype=np.int64),
            x=np.asarray(data["x"], dtype=np.int64),
            y=np.asarray(data["y"], dtype=np.int64),
            polarity=np.asarray(data["p"], dtype=np.int8),
        )


def validate_frames(
    frames: np.ndarray, timesteps: int, source: Path | str
) -> np.ndarray:
    if frames.shape != (timesteps, 2, *TARGET_HW):
        raise ValueError(
            f"{source} has shape {frames.shape}; expected "
            f"({timesteps}, 2, {TARGET_HW[0]}, {TARGET_HW[1]})"
        )
    return np.ascontiguousarray(frames, dtype=np.float32)


def events_to_frames(
    events: EventArrays,
    timesteps: int,
    split_by: str,
    sensor_hw: tuple[int, int] = TARGET_HW,
    resize_mode: str = "strict",
) -> np.ndarray:
    height, width = sensor_hw
    target_h, target_w = TARGET_HW
    x, y = map_event_coordinates(events.x, events.y, width, height, resize_mode)
    polarity = (events.polarity > 0).astype(np.int64, copy=False)
    timestamps = np.asarray(events.timestamps, dtype=np.int64)

    valid = (
        (x >= 0)
        & (x < target_w)
        & (y >= 0)
        & (y < target_h)
        & (polarity >= 0)
        & (polarity <= 1)
    )
    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    timestamps = timestamps[valid]

    frames = np.zeros((timesteps, 2, target_h, target_w), dtype=np.float32)
    if x.size == 0:
        return frames

    bin_index = event_bin_indices(timestamps, timesteps, split_by)
    np.add.at(frames, (bin_index, polarity, y, x), 1.0)
    return frames


def map_event_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    resize_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    target_h, target_w = TARGET_HW
    if (height, width) == TARGET_HW:
        return x, y

    if resize_mode == "strict":
        raise ValueError(
            f"camera/event resolution {(height, width)} does not match {TARGET_HW}; "
            "use --resize-mode scale or crop for non-DVS128 sensors"
        )

    if resize_mode == "scale":
        scaled_x = np.floor(x.astype(np.float64) * target_w / width).astype(np.int64)
        scaled_y = np.floor(y.astype(np.float64) * target_h / height).astype(np.int64)
        return scaled_x, scaled_y

    if resize_mode == "crop":
        left = max(0, (width - target_w) // 2)
        top = max(0, (height - target_h) // 2)
        return x - left, y - top

    raise ValueError(f"unsupported resize mode: {resize_mode}")


def event_bin_indices(
    timestamps: np.ndarray,
    timesteps: int,
    split_by: str,
) -> np.ndarray:
    if split_by == "number":
        count = timestamps.shape[0]
        return np.minimum(
            (np.arange(count, dtype=np.int64) * timesteps) // count, timesteps - 1
        )

    if split_by != "time":
        raise ValueError(f"unsupported split mode: {split_by}")

    start = int(timestamps.min())
    stop = int(timestamps.max())
    if stop <= start:
        return np.zeros(timestamps.shape[0], dtype=np.int64)
    normalized = (timestamps.astype(np.float64) - start) / (stop - start + 1)
    return np.minimum((normalized * timesteps).astype(np.int64), timesteps - 1)


def predict(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: torch.device,
    names: dict[int, str],
    source: str,
    sample: str,
    event_count: int | None,
    expected_label: int | None = None,
) -> Prediction:
    input_frames = validate_frames(frames, frames.shape[0], sample)
    tensor = (
        torch.from_numpy(input_frames)
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
    )

    start = time.perf_counter()
    with torch.no_grad():
        try:
            out = model(tensor)
            logits = reduce_time_logits(out).detach().cpu()
        finally:
            functional.reset_net(model)
    latency_ms = (time.perf_counter() - start) * 1000.0

    probabilities = torch.softmax(logits, dim=0)
    predicted_label = int(torch.argmax(logits).item())
    expected_name = (
        None
        if expected_label is None
        else names.get(expected_label, str(expected_label))
    )
    correct = None if expected_label is None else predicted_label == expected_label
    return Prediction(
        source=source,
        sample=sample,
        predicted_label=predicted_label,
        predicted_name=names.get(predicted_label, str(predicted_label)),
        confidence=float(probabilities[predicted_label].item()),
        latency_ms=latency_ms,
        event_count=event_count,
        expected_label=expected_label,
        expected_name=expected_name,
        correct=correct,
    )


def reduce_time_logits(out: torch.Tensor) -> torch.Tensor:
    if out.ndim == 3:
        # infer.py evaluates [T, N, C] by summing over T.
        return out.sum(0).squeeze(0)
    if out.ndim == 2:
        return out.sum(0)
    if out.ndim == 1:
        return out
    raise ValueError(f"unexpected model output shape: {tuple(out.shape)}")


def print_prediction(prediction: Prediction) -> None:
    expected = ""
    if prediction.expected_label is not None:
        status = "correct" if prediction.correct else "wrong"
        expected = (
            f" expected={prediction.expected_label}:{prediction.expected_name}"
            f" {status}"
        )
    events = (
        "" if prediction.event_count is None else f" events={prediction.event_count}"
    )
    print(
        f"[{prediction.source}] sample={prediction.sample}"
        f"{expected}"
        f" pred={prediction.predicted_label}:{prediction.predicted_name}"
        f" confidence={prediction.confidence:.4f}"
        f" latency_ms={prediction.latency_ms:.2f}"
        f"{events}",
        flush=True,
    )


class DvsOpenCvDisplay:
    def __init__(self, sensor_hw: tuple[int, int]) -> None:
        self.cv2 = import_cv2()
        height, width = sensor_hw
        self.height = height
        self.width = width
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.window_name = DISPLAY_WINDOW_NAME
        self.last_decay_at = time.perf_counter()
        self.last_show_at = 0.0
        self.show_interval_s = 1.0 / DISPLAY_FPS
        self.display_size = self._display_size(height, width)

        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, *self.display_size)

    def update(
        self,
        events: EventArrays | None,
        prediction: Prediction | None,
    ) -> bool:
        now = time.perf_counter()
        self._decay(now)
        if events is not None:
            self._draw_events(events)

        if now - self.last_show_at < self.show_interval_s:
            return True

        self.last_show_at = now
        view = self.image.copy()
        if self.display_size != (self.width, self.height):
            view = self.cv2.resize(
                view,
                self.display_size,
                interpolation=self.cv2.INTER_NEAREST,
            )
        view = self._with_overlay(view, prediction)
        self.cv2.imshow(self.window_name, view)
        key = self.cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False

        try:
            return self.cv2.getWindowProperty(
                self.window_name, self.cv2.WND_PROP_VISIBLE
            ) >= 1
        except self.cv2.error:
            return False

    def close(self) -> None:
        self.cv2.destroyWindow(self.window_name)

    def _decay(self, now: float) -> None:
        elapsed_ms = max(0.0, (now - self.last_decay_at) * 1000.0)
        self.last_decay_at = now
        if elapsed_ms <= 0.0:
            return
        decay = 0.5 ** (elapsed_ms / DISPLAY_DECAY_HALFLIFE_MS)
        np.multiply(self.image, decay, out=self.image, casting="unsafe")

    def _draw_events(self, events: EventArrays) -> None:
        x = np.asarray(events.x, dtype=np.int64)
        y = np.asarray(events.y, dtype=np.int64)
        polarity = np.asarray(events.polarity)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        if not np.any(valid):
            return

        x = x[valid]
        y = y[valid]
        polarity = polarity[valid]
        positive = polarity > 0
        if np.any(positive):
            self.image[y[positive], x[positive]] = (40, 240, 40)
        if np.any(~positive):
            self.image[y[~positive], x[~positive]] = (220, 80, 220)

    def _with_overlay(
        self,
        view: np.ndarray,
        prediction: Prediction | None,
    ) -> np.ndarray:
        if prediction is None:
            label = "prediction: waiting"
        else:
            label = (
                f"prediction: {prediction.predicted_label} "
                f"{prediction.predicted_name} "
                f"{prediction.confidence:.2f}"
            )
        text_y = 24
        quit_y = max(48, view.shape[0] - 12)
        self.cv2.putText(
            view,
            label,
            (8, text_y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            view,
            label,
            (8, text_y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            view,
            "q/esc: quit",
            (8, quit_y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            view,
            "q/esc: quit",
            (8, quit_y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            self.cv2.LINE_AA,
        )
        return view

    @staticmethod
    def _display_size(height: int, width: int) -> tuple[int, int]:
        largest = max(height, width)
        if largest <= 0:
            return width, height
        if largest < DISPLAY_MIN_SIDE:
            scale = DISPLAY_MIN_SIDE / largest
        else:
            scale = min(1.0, DISPLAY_MAX_SIDE / largest)
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for --display. Install it in the active PAIBox "
            "environment with: python -m pip install opencv-python"
        ) from exc
    return cv2


def run_dataset(
    args: argparse.Namespace, model: torch.nn.Module, device: torch.device
) -> int:
    names = load_label_names(DEFAULT_DATA_ROOT)
    total = 0
    correct = 0

    with JsonlWriter(args.jsonl) as writer:
        for path, expected_label in iter_dataset_samples(
            args.source,
            args.sample,
            args.limit,
        ):
            if args.source == "frames":
                frames = load_frames_npz(path, TIMESTEPS)
                event_count = int(frames.sum())
            else:
                events = load_event_npz(path)
                frames = events_to_frames(events, TIMESTEPS, DATASET_EVENT_SPLIT)
                event_count = int(events.x.shape[0])

            prediction = predict(
                model=model,
                frames=frames,
                device=device,
                names=names,
                source=f"dataset:{args.source}",
                sample=display_path(path),
                event_count=event_count,
                expected_label=expected_label,
            )
            print_prediction(prediction)
            writer.write(prediction)

            total += 1
            if prediction.correct:
                correct += 1

    if total:
        print(f"[dataset] total={total} correct={correct} acc={correct / total:.4f}")
    return 0


def run_camera(
    args: argparse.Namespace, model: torch.nn.Module, device: torch.device
) -> int:
    names = load_label_names(DEFAULT_DATA_ROOT)
    camera = open_camera(args.camera_name)
    sensor_hw = normalize_resolution(camera.getEventResolution())
    display = DvsOpenCvDisplay(sensor_hw) if args.display else None

    slicer = dv.EventStreamSlicer()
    processed = 0
    latest_prediction: Prediction | None = None

    print(
        "[camera] opened"
        f" name={args.camera_name or '<default>'}"
        f" sensor_hw={sensor_hw}"
        f" window_ms={args.window_ms}"
        f" split_by={CAMERA_EVENT_SPLIT}"
        f" resize_mode={args.resize_mode}",
        f" display={bool(display)}",
        flush=True,
    )

    try:
        with JsonlWriter(args.jsonl) as writer:

            def classify_window(event_slice: Any) -> None:
                nonlocal processed, latest_prediction
                arrays = event_store_to_arrays(event_slice)
                event_count = int(arrays.x.shape[0])
                if event_count < MIN_CAMERA_EVENTS:
                    return
                frames = events_to_frames(
                    arrays,
                    timesteps=TIMESTEPS,
                    split_by=CAMERA_EVENT_SPLIT,
                    sensor_hw=sensor_hw,
                    resize_mode=args.resize_mode,
                )
                prediction = predict(
                    model=model,
                    frames=frames,
                    device=device,
                    names=names,
                    source="camera",
                    sample=f"window_{processed:06d}",
                    event_count=event_count,
                )
                latest_prediction = prediction
                print_prediction(prediction)
                writer.write(prediction)
                processed += 1

            slicer.doEveryTimeInterval(
                timedelta(milliseconds=args.window_ms), classify_window
            )

            while camera.isRunning():
                if args.max_windows > 0 and processed >= args.max_windows:
                    break
                events = camera.getNextEventBatch()
                if events is None:
                    if display is not None and not display.update(
                        None, latest_prediction
                    ):
                        break
                    time.sleep(CAMERA_IDLE_SLEEP_MS / 1000.0)
                    continue

                if display is not None:
                    display_events = event_store_to_arrays(events)
                    if not display.update(display_events, latest_prediction):
                        break
                slicer.accept(events)
    finally:
        if display is not None:
            display.close()

    print(f"[camera] processed_windows={processed}")
    return 0


def open_camera(camera_name: str | None) -> Any:
    devices = dv.io.camera.discover()
    if not devices:
        raise RuntimeError(
            "no dv-processing compatible DVS camera was discovered; "
            "check USB connection, permissions, and udev rules"
        )

    if camera_name is not None:
        try:
            return dv.io.camera.open(camera_name)
        except Exception as exc:
            raise RuntimeError(
                f"failed to open camera {camera_name!r}; "
                f"discovered devices: {devices!r}"
            ) from exc

    try:
        return dv.io.camera.open()
    except Exception as exc:
        raise RuntimeError(f"failed to open discovered camera: {devices!r}") from exc


def event_store_to_arrays(event_store: Any) -> EventArrays:
    data = event_store.numpy()
    return EventArrays(
        timestamps=np.asarray(data["timestamp"], dtype=np.int64),
        x=np.asarray(data["x"], dtype=np.int64),
        y=np.asarray(data["y"], dtype=np.int64),
        polarity=np.asarray(data["polarity"], dtype=np.int8),
    )


def normalize_resolution(resolution: Any) -> tuple[int, int]:
    if hasattr(resolution, "height") and hasattr(resolution, "width"):
        return int(resolution.height), int(resolution.width)
    if hasattr(resolution, "rows") and hasattr(resolution, "cols"):
        return int(resolution.rows), int(resolution.cols)
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        first, second = int(resolution[0]), int(resolution[1])
        # dv-processing exposes OpenCV-style sizes in examples; keep square
        # sensors harmless and map non-square values as width, height.
        return (second, first) if first != second else (first, second)
    raise ValueError(f"unsupported camera resolution object: {resolution!r}")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_DIR))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    model = load_model(device)
    print(
        f"[model] checkpoint={display_path(DEFAULT_CHECKPOINT)}"
        f" device={device}"
        f" timesteps={TIMESTEPS}"
        f" threshold={IF_THRESHOLD}",
        flush=True,
    )

    if args.command == "dataset":
        return run_dataset(args, model, device)
    if args.command == "camera":
        return run_camera(args, model, device)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
