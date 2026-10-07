import argparse
import json
import queue
import threading
import time
import numpy as np

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from . import camera_input, dataset
from .runtime import (
    FRAME_SHAPE,
    NUM_CLASSES,
    BoardRuntime,
    RuntimeTiming,
    SampleResult,
    open_serial,
    predict_from_counts,
    run_board,
)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
BOARD_BAUD = 3_000_000
CAMERA_RESIZE_MODE = "scale"
CAMERA_EVENT_FILTER = "background"
CAMERA_WARMUP_WINDOWS = 1
CAMERA_MAX_EVENTS = 0
CAMERA_FRAME_NONZERO_POLICY = "cap"
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


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


@dataclass(frozen=True)
class Prediction:
    source: str
    sample: str
    expected_label: int | None
    expected_name: str | None
    predicted_label: int
    predicted_name: str
    confidence: float
    logits: list[float]
    board_latency_ms: float
    encode_ms: float = 0.0
    tx_work_ms: float = 0.0
    sync_wait_ms: float = 0.0
    work_frame_count: int = 0
    work_write_count: int = 0
    runtime_sync_count: int = 0
    event_count: int | None = None
    raw_event_count: int | None = None
    filtered_event_count: int | None = None
    filter_reduction: float | None = None
    frame_nonzero_count: int | None = None
    source_window_id: int | None = None
    dropped_windows: int = 0
    skipped_windows: int = 0


@dataclass(frozen=True)
class CameraWindow:
    window_id: int
    frames: np.ndarray
    event_count: int
    raw_event_count: int
    filtered_event_count: int
    filter_reduction: float
    frame_nonzero_count: int
    queued_at: float


@dataclass(frozen=True)
class CameraWorkerResult:
    prediction: Prediction | None = None
    error: BaseException | None = None


class JsonlWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.file: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, prediction: Prediction) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")
        self.file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DVSGesture PAICORE 2.5 板卡推理工具。\n"
            "sample/dataset 用于已生成 .npz 帧测试，camera 用于 DVS 实时演示。"
        ),
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample",
        help="单个 .npz 样本上板测试",
        description=(
            "单样本上板测试：从已生成的 DVS128Gesture .npz 帧输入中选择一条送板。\n"
            "调节：板卡等待较久时增大 --timeout；指定样本用 --sample-path。"
        ),
        formatter_class=HelpFormatter,
    )
    add_runtime_options(sample)
    add_logging_options(sample)
    sample_data = sample.add_argument_group("Dataset")
    sample_data.add_argument(
        "--data-root", type=Path, default=dataset.DEFAULT_DATA_ROOT
    )
    sample_data.add_argument("--split", default="test")
    sample_data.add_argument("--sample-index", type=int, default=0)
    sample_data.add_argument("--sample-path", type=Path, default=None)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="测试集连续上板验证",
        description=(
            "测试集流水验证：按数据集顺序连续送板，并实时打印预测、延迟和精度。\n"
            "调节：先用 --limit 做 smoke；串口或板卡等待较慢时增大 --timeout。"
        ),
        formatter_class=HelpFormatter,
    )
    add_runtime_options(dataset_parser)
    add_logging_options(dataset_parser)
    dataset_data = dataset_parser.add_argument_group("Dataset")
    dataset_data.add_argument(
        "--data-root", type=Path, default=dataset.DEFAULT_DATA_ROOT
    )
    dataset_data.add_argument("--split", default="test")
    dataset_data.add_argument("--limit", type=int, default=1)

    camera = subparsers.add_parser(
        "camera",
        help="DVS 相机实时推理演示",
        description=(
            "DVS 实时演示：相机事件 -> 20 帧 -> 板卡流水推理 -> cv2 叠加预测。\n"
            "推荐入口：scripts/run_camera_demo.sh，会自动建立日志目录。\n"
            "调节：难触发则降低 --min-events；延迟高则降低 --max-frame-nonzero；"
            "背景噪声多则增大 --background-activity-ms。"
        ),
        formatter_class=HelpFormatter,
    )
    add_runtime_options(camera)
    camera_opts = camera.add_argument_group("Camera")
    camera_opts.add_argument("--camera-serial", default=None)
    camera_opts.add_argument("--window-ms", type=float, default=1000.0)
    camera_opts.add_argument("--max-windows", type=int, default=0)
    camera_opts.add_argument("--display", action="store_true")
    filter_opts = camera.add_argument_group("Event Filter")
    filter_opts.add_argument("--background-activity-ms", type=float, default=1.0)
    camera_quality = camera.add_argument_group("Window Quality")
    camera_quality.add_argument("--min-events", type=int, default=1000)
    camera_quality.add_argument("--max-frame-nonzero", type=int, default=50000)
    add_logging_options(camera)

    return parser.parse_args()


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    runtime = parser.add_argument_group("Board Runtime")
    runtime.add_argument("--port", default="/dev/ttyUSB0")
    runtime.add_argument("--timeout", type=float, default=3.0)
    runtime.add_argument("--skip-init", action="store_true")


def add_logging_options(parser: argparse.ArgumentParser) -> None:
    logging = parser.add_argument_group("Logging")
    logging.add_argument("--jsonl", type=Path, default=None)
    logging.add_argument("--verbose", action="store_true")
    logging.add_argument("--show-timing", action="store_true")


def main() -> int:
    args = parse_args()
    normalize_run_paths(args)
    if args.command == "camera":
        return run_camera(args)
    return run_samples_or_dataset(args)


def normalize_run_paths(args: argparse.Namespace) -> None:
    paths = [path for path in (getattr(args, "jsonl", None),) if path is not None]
    if not paths:
        return

    parents = {_resolved_parent(path) for path in paths}
    if parents != {LOGS_DIR.resolve()}:
        return

    run_dir = LOGS_DIR / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{args.command}_{Path(args.port).name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.jsonl is not None:
        args.jsonl = run_dir / args.jsonl.name
    print(f"[log] run_dir={run_dir}", flush=True)


def _resolved_parent(path: Path) -> Path:
    parent = path.expanduser().parent
    if path.is_absolute():
        return parent.resolve()
    return (Path.cwd() / parent).resolve()


def run_samples_or_dataset(args: argparse.Namespace) -> int:
    if args.command == "sample":
        samples = [
            dataset.load_sample(
                sample_path=args.sample_path,
                sample_index=args.sample_index,
                data_root=args.data_root,
                split=args.split,
                split_by=dataset.DEFAULT_SPLIT_BY,
            )
        ]
    elif args.command == "dataset":
        samples = dataset.iter_samples(
            data_root=args.data_root,
            split=args.split,
            split_by=dataset.DEFAULT_SPLIT_BY,
            limit=args.limit,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")

    if not samples:
        raise RuntimeError("no samples selected")

    frames = np.stack([sample.frames for sample in samples], axis=0)
    predictions: list[Prediction | None] = [None] * len(samples)
    correct = 0
    labeled = 0

    with JsonlWriter(args.jsonl) as writer:

        def on_sample_done(result: SampleResult) -> None:
            nonlocal correct, labeled
            sample = samples[result.sample_index]
            prediction = build_prediction(
                source=args.command,
                sample=display_path(sample.path),
                expected_label=sample.label,
                counts=result.counts,
                board_latency_ms=result.board_latency_ms,
                timing=result.timing,
                event_count=int(sample.frames.sum()),
            )
            predictions[result.sample_index] = prediction
            if prediction.expected_label is not None:
                labeled += 1
                if prediction.predicted_label == prediction.expected_label:
                    correct += 1
            writer.write(prediction)
            print(
                format_prediction_progress(
                    result.sample_index,
                    len(samples),
                    prediction,
                    correct=correct,
                    labeled=labeled,
                    show_timing=args.show_timing,
                ),
                flush=True,
            )

        counts = run_board(
            frames,
            port=args.port,
            baud=BOARD_BAUD,
            timeout=args.timeout,
            skip_init=args.skip_init,
            dump_path=None,
            on_sample_done=on_sample_done,
        )

    for index, (sample, counts_for_sample) in enumerate(
        zip(samples, counts, strict=True)
    ):
        if predictions[index] is None:
            predictions[index] = build_prediction(
                source=args.command,
                sample=display_path(sample.path),
                expected_label=sample.label,
                counts=counts_for_sample,
                board_latency_ms=0.0,
                timing=RuntimeTiming(),
                event_count=int(sample.frames.sum()),
            )
    return 0


class LatestWindowBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CameraWindow | None = None
        self._closed = False
        self.accepted = 0
        self.dropped = 0

    def put(self, window: CameraWindow) -> None:
        with self._condition:
            if self._latest is not None:
                self.dropped += 1
            self._latest = window
            self.accepted += 1
            self._condition.notify()

    def get(self) -> CameraWindow | None:
        with self._condition:
            while self._latest is None and not self._closed:
                self._condition.wait()
            if self._latest is None:
                return None
            window = self._latest
            self._latest = None
            return window

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def dropped_count(self) -> int:
        with self._condition:
            return self.dropped


class CameraBoardWorker:
    def __init__(
        self,
        *,
        runtime: BoardRuntime,
        buffer: LatestWindowBuffer,
        result_queue: queue.Queue[CameraWorkerResult],
        get_skipped_windows: Callable[[], int],
    ) -> None:
        self.runtime = runtime
        self.buffer = buffer
        self.result_queue = result_queue
        self.get_skipped_windows = get_skipped_windows
        self.outputs = np.zeros((1, FRAME_SHAPE[0], NUM_CLASSES), dtype=np.int32)
        self.sample_windows: dict[int, CameraWindow] = {}
        self.sent_windows = 0
        self.processed_windows = 0

    def run(self) -> None:
        try:
            while True:
                window = self.buffer.get()
                if window is None:
                    break
                sample_index = self.sent_windows
                self._ensure_output_capacity(sample_index + 1)
                self.sample_windows[sample_index] = window
                self.runtime.send_stream_sample(
                    window.frames,
                    outputs=self.outputs,
                    sample_index=sample_index,
                    on_sample_done=self._on_sample_done,
                )
                self.sent_windows += 1
            if self.sent_windows > self.processed_windows:
                self.runtime.flush_stream(
                    self.outputs, on_sample_done=self._on_sample_done
                )
        except BaseException as exc:
            self.result_queue.put(CameraWorkerResult(error=exc))

    def _on_sample_done(self, result: SampleResult) -> None:
        window = self.sample_windows.pop(result.sample_index, None)
        sample = (
            f"sample_{result.sample_index:06d}"
            if window is None
            else f"window_{window.window_id:06d}"
        )
        prediction = build_prediction(
            source="camera",
            sample=sample,
            expected_label=None,
            counts=result.counts,
            board_latency_ms=result.board_latency_ms,
            timing=result.timing,
            event_count=None if window is None else window.event_count,
            raw_event_count=None if window is None else window.raw_event_count,
            filtered_event_count=(
                None if window is None else window.filtered_event_count
            ),
            filter_reduction=None if window is None else window.filter_reduction,
            frame_nonzero_count=None if window is None else window.frame_nonzero_count,
            source_window_id=None if window is None else window.window_id,
            dropped_windows=self.buffer.dropped_count(),
            skipped_windows=self.get_skipped_windows(),
        )
        self.processed_windows = max(self.processed_windows, result.sample_index + 1)
        self.result_queue.put(CameraWorkerResult(prediction=prediction))

    def _ensure_output_capacity(self, required: int) -> None:
        if required <= self.outputs.shape[0]:
            return
        new_capacity = max(required, self.outputs.shape[0] * 2)
        expanded = np.zeros(
            (new_capacity, FRAME_SHAPE[0], NUM_CLASSES), dtype=self.outputs.dtype
        )
        expanded[: self.outputs.shape[0]] = self.outputs
        self.outputs = expanded


def run_camera(args: argparse.Namespace) -> int:
    if args.window_ms <= 0:
        raise ValueError("--window-ms must be positive")
    if args.min_events < 0:
        raise ValueError("--min-events must be non-negative")
    if args.max_frame_nonzero < 0:
        raise ValueError("--max-frame-nonzero must be non-negative")
    if args.background_activity_ms < 0:
        raise ValueError("--background-activity-ms must be non-negative")

    camera = camera_input.open_camera(args.camera_serial)
    sensor_hw = camera_input.normalize_resolution(camera.getEventResolution())
    display = camera_input.DvsOpenCvDisplay(sensor_hw) if args.display else None
    event_filter = camera_input.make_event_filter(
        sensor_hw,
        name=CAMERA_EVENT_FILTER,
        background_activity_ms=args.background_activity_ms,
    )
    slicer = camera_input.dv.EventStreamSlicer()

    max_windows = args.max_windows
    latest_overlay = "prediction: waiting"
    buffer = LatestWindowBuffer()
    result_queue: queue.Queue[CameraWorkerResult] = queue.Queue()
    accepted_windows = 0
    seen_windows = 0
    processed_windows = 0
    skipped_windows = 0
    overactive_windows = 0
    dense_frame_windows = 0
    warmup_dropped_windows = 0
    pending_raw_event_count = 0
    warned_unbounded = False
    start_time = time.perf_counter()

    print(
        "[camera] opened"
        f" serial={args.camera_serial or '<default>'}"
        f" sensor_hw={sensor_hw}"
        f" window_ms={args.window_ms}"
        f" resize_mode={CAMERA_RESIZE_MODE}"
        f" display={bool(display)}",
        f" min_events={args.min_events}" f" warmup_windows={CAMERA_WARMUP_WINDOWS}",
        f" event_filter={CAMERA_EVENT_FILTER}",
        f" max_events={CAMERA_MAX_EVENTS}",
        f" max_frame_nonzero={args.max_frame_nonzero}",
        f" frame_nonzero_policy={CAMERA_FRAME_NONZERO_POLICY}",
        flush=True,
    )

    def classify_window(event_slice: Any) -> None:
        nonlocal accepted_windows, seen_windows
        nonlocal skipped_windows, overactive_windows, dense_frame_windows
        nonlocal warmup_dropped_windows
        nonlocal pending_raw_event_count
        window_id = seen_windows
        seen_windows += 1
        raw_event_count = pending_raw_event_count
        pending_raw_event_count = 0
        arrays = camera_input.event_store_to_arrays(event_slice)
        filtered_event_count = int(arrays.x.shape[0])
        filter_reduction = reduction_ratio(raw_event_count, filtered_event_count)
        if warmup_dropped_windows < CAMERA_WARMUP_WINDOWS:
            warmup_dropped_windows += 1
            if args.verbose:
                print(
                    f"[camera] warmup drop window={window_id} "
                    f"raw_events={raw_event_count} "
                    f"filtered_events={filtered_event_count}",
                    flush=True,
                )
            return
        if filtered_event_count < args.min_events:
            skipped_windows += 1
            update_waiting_overlay(filtered_event_count)
            if args.verbose:
                print(
                    f"[camera] skip window={window_id} "
                    f"raw_events={raw_event_count} "
                    f"filtered_events={filtered_event_count} "
                    f"min_events={args.min_events}",
                    flush=True,
                )
            return
        if exceeds_limit(filtered_event_count, CAMERA_MAX_EVENTS):
            overactive_windows += 1
            update_overactive_overlay(filtered_event_count)
            if args.verbose:
                print(
                    f"[camera] skip overactive window={window_id} "
                    f"raw_events={raw_event_count} "
                    f"filtered_events={filtered_event_count} "
                    f"max_events={CAMERA_MAX_EVENTS}",
                    flush=True,
                )
            return
        if max_windows > 0 and accepted_windows >= max_windows:
            return
        frames = camera_input.events_to_frames(
            arrays, sensor_hw=sensor_hw, resize_mode=CAMERA_RESIZE_MODE
        )
        frame_nonzero_count = int(np.count_nonzero(frames))
        if exceeds_limit(frame_nonzero_count, args.max_frame_nonzero):
            if CAMERA_FRAME_NONZERO_POLICY == "skip":
                dense_frame_windows += 1
                update_dense_frame_overlay(frame_nonzero_count)
                if args.verbose:
                    print(
                        f"[camera] skip dense-frame window={window_id} "
                        f"raw_events={raw_event_count} "
                        f"filtered_events={filtered_event_count} "
                        f"frame_nonzero={frame_nonzero_count} "
                        f"max_frame_nonzero={args.max_frame_nonzero}",
                        flush=True,
                    )
                return
            original_frame_nonzero = frame_nonzero_count
            frames, frame_nonzero_count = camera_input.cap_frame_nonzero(
                frames, args.max_frame_nonzero
            )
            if args.verbose:
                print(
                    f"[camera] cap dense-frame window={window_id} "
                    f"raw_events={raw_event_count} "
                    f"filtered_events={filtered_event_count} "
                    f"frame_nonzero={original_frame_nonzero}->{frame_nonzero_count}",
                    flush=True,
                )
        window = CameraWindow(
            window_id=window_id,
            frames=frames,
            event_count=filtered_event_count,
            raw_event_count=raw_event_count,
            filtered_event_count=filtered_event_count,
            filter_reduction=filter_reduction,
            frame_nonzero_count=frame_nonzero_count,
            queued_at=time.perf_counter(),
        )
        buffer.put(window)
        if args.verbose:
            print(
                f"[camera] queued window={window_id} accepted={accepted_windows} "
                f"raw_events={raw_event_count} "
                f"filtered_events={filtered_event_count} "
                f"filter_reduction={filter_reduction:.6f} "
                f"frame_nonzero={frame_nonzero_count} "
                f"frame_sum={int(frames.sum())} "
                f"dropped={buffer.dropped_count()}",
                flush=True,
            )
        accepted_windows += 1

    def update_waiting_overlay(event_count: int) -> None:
        nonlocal latest_overlay
        latest_overlay = (
            f"prediction: waiting/no gesture events={event_count} "
            f"min={args.min_events} skipped={skipped_windows}"
        )

    def update_overactive_overlay(event_count: int) -> None:
        nonlocal latest_overlay
        latest_overlay = (
            f"prediction: noisy/overactive events={event_count} "
            f"max={CAMERA_MAX_EVENTS} skipped={overactive_windows}"
        )

    def update_dense_frame_overlay(frame_nonzero_count: int) -> None:
        nonlocal latest_overlay
        latest_overlay = (
            f"prediction: dense frame nonzero={frame_nonzero_count} "
            f"max={args.max_frame_nonzero} skipped={dense_frame_windows}"
        )

    def handle_worker_results(writer: JsonlWriter) -> BaseException | None:
        nonlocal latest_overlay, processed_windows
        error: BaseException | None = None
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                return error
            if result.error is not None:
                error = result.error
                continue
            if result.prediction is None:
                continue
            prediction = result.prediction
            writer.write(prediction)
            latest_overlay = format_camera_overlay(prediction)
            print(
                format_camera_prediction(prediction, show_timing=args.show_timing),
                flush=True,
            )
            processed_windows += 1

    slicer.doEveryTimeInterval(timedelta(milliseconds=args.window_ms), classify_window)

    with JsonlWriter(args.jsonl) as writer:
        ser = open_serial(args.port, BOARD_BAUD)
        runtime = BoardRuntime(
            ser,
            timeout=args.timeout,
            skip_init=args.skip_init,
            dump_path=None,
        )
        worker = CameraBoardWorker(
            runtime=runtime,
            buffer=buffer,
            result_queue=result_queue,
            get_skipped_windows=lambda: skipped_windows,
        )
        worker_thread = threading.Thread(
            target=worker.run, name="camera-board-worker", daemon=True
        )
        worker_started = False

        def stop_worker() -> BaseException | None:
            nonlocal worker_started
            buffer.close()
            if worker_started:
                worker_thread.join()
            return handle_worker_results(writer)

        try:
            with runtime:
                worker_thread.start()
                worker_started = True
                while camera.isRunning():
                    if max_windows <= 0 and display is None and args.verbose:
                        if not warned_unbounded:
                            print(
                                "[camera] running without --max-windows; "
                                "stop with Ctrl-C",
                                flush=True,
                            )
                            warned_unbounded = True
                    worker_error = handle_worker_results(writer)
                    if worker_error is not None:
                        raise worker_error
                    if max_windows > 0 and accepted_windows >= max_windows:
                        break
                    raw_events = camera.getNextEventBatch()
                    if raw_events is None:
                        if display is not None and not display.update(
                            None, latest_overlay
                        ):
                            break
                        time.sleep(0.001)
                        continue

                    raw_arrays = camera_input.event_store_to_arrays(raw_events)
                    raw_event_count = int(raw_arrays.x.shape[0])
                    pending_raw_event_count += raw_event_count
                    events = event_filter.accept(raw_events)

                    if display is not None:
                        display_events = camera_input.event_store_to_arrays(events)
                        if not display.update(display_events, latest_overlay):
                            break
                    if max_windows <= 0 or accepted_windows < max_windows:
                        slicer.accept(events)
                worker_error = stop_worker()
                if worker_error is not None:
                    raise worker_error
        finally:
            buffer.close()
            runtime.close()
            if display is not None:
                display.close()

    print(
        "[camera] done"
        f" seen={seen_windows}"
        f" accepted={accepted_windows}"
        f" sent={worker.sent_windows}"
        f" processed={processed_windows}"
        f" skipped={skipped_windows}"
        f" overactive={overactive_windows}"
        f" dense_frame={dense_frame_windows}"
        f" dropped={buffer.dropped_count()}"
        f" warmup_dropped={warmup_dropped_windows}"
        f" elapsed_s={time.perf_counter() - start_time:.3f}",
        flush=True,
    )
    return 0


def build_prediction(
    *,
    source: str,
    sample: str,
    expected_label: int | None,
    counts: np.ndarray,
    board_latency_ms: float,
    timing: RuntimeTiming | None = None,
    event_count: int | None,
    raw_event_count: int | None = None,
    filtered_event_count: int | None = None,
    filter_reduction: float | None = None,
    frame_nonzero_count: int | None = None,
    source_window_id: int | None = None,
    dropped_windows: int = 0,
    skipped_windows: int = 0,
) -> Prediction:
    pred, confidence, logits = predict_from_counts(counts)
    timing = RuntimeTiming() if timing is None else timing
    return Prediction(
        source=source,
        sample=sample,
        expected_label=expected_label,
        expected_name=label_name(expected_label),
        predicted_label=pred,
        predicted_name=label_name(pred),
        confidence=confidence,
        logits=[float(v) for v in logits],
        board_latency_ms=float(board_latency_ms),
        encode_ms=float(timing.encode_ms),
        tx_work_ms=float(timing.tx_work_ms),
        sync_wait_ms=float(timing.sync_wait_ms),
        work_frame_count=int(timing.work_frame_count),
        work_write_count=int(timing.work_write_count),
        runtime_sync_count=int(timing.sync_count),
        event_count=event_count,
        raw_event_count=raw_event_count,
        filtered_event_count=filtered_event_count,
        filter_reduction=filter_reduction,
        frame_nonzero_count=frame_nonzero_count,
        source_window_id=source_window_id,
        dropped_windows=dropped_windows,
        skipped_windows=skipped_windows,
    )


def reduction_ratio(raw_event_count: int, filtered_event_count: int) -> float:
    if raw_event_count <= 0:
        return 0.0
    return max(0.0, 1.0 - (filtered_event_count / raw_event_count))


def exceeds_limit(value: int, limit: int) -> bool:
    return limit > 0 and value > limit


def label_name(label: int | None) -> str | None:
    if label is None:
        return None
    return DEFAULT_LABEL_NAMES.get(label, str(label))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset.APP_DIR))
    except ValueError:
        return str(path)


def format_prediction_progress(
    sample_index: int,
    total: int,
    prediction: Prediction,
    *,
    correct: int,
    labeled: int,
    show_timing: bool = False,
) -> str:
    expected = (
        "expected=NA"
        if prediction.expected_label is None
        else f"expected={prediction.expected_label}:{prediction.expected_name}"
    )
    if labeled:
        accuracy = f"{correct / labeled:.6f}"
        running = f"correct={correct}/{labeled} accuracy={accuracy}"
    else:
        running = "accuracy=NA"
    message = (
        f"[{sample_index + 1}/{total}] sample={prediction.sample} {expected} "
        f"pred={prediction.predicted_label}:{prediction.predicted_name} "
        f"confidence={prediction.confidence:.6f} "
        f"board_latency_ms={prediction.board_latency_ms:.3f}"
    )
    if show_timing:
        message += (
            f" encode_ms={prediction.encode_ms:.3f} "
            f"tx_work_ms={prediction.tx_work_ms:.3f} "
            f"sync_wait_ms={prediction.sync_wait_ms:.3f} "
            f"work_frames={prediction.work_frame_count}"
        )
    return f"{message} {running}"


def format_camera_prediction(
    prediction: Prediction, *, show_timing: bool = False
) -> str:
    message = (
        f"[camera] sample={prediction.sample} events={prediction.event_count} "
        f"raw_events={prediction.raw_event_count} "
        f"filtered_events={prediction.filtered_event_count} "
        f"frame_nonzero={prediction.frame_nonzero_count} "
        f"pred={prediction.predicted_label}:{prediction.predicted_name} "
        f"confidence={prediction.confidence:.6f} "
        f"board_latency_ms={prediction.board_latency_ms:.3f}"
    )
    if show_timing:
        message += (
            f" encode_ms={prediction.encode_ms:.3f} "
            f"tx_work_ms={prediction.tx_work_ms:.3f} "
            f"sync_wait_ms={prediction.sync_wait_ms:.3f} "
            f"work_frames={prediction.work_frame_count}"
        )
    return (
        f"{message} dropped={prediction.dropped_windows} "
        f"skipped={prediction.skipped_windows}"
    )


def format_camera_overlay(prediction: Prediction) -> str:
    return (
        f"pred={prediction.predicted_name} "
        f"conf={prediction.confidence:.2f} "
        f"latency={prediction.board_latency_ms / 1000.0:.1f}s "
        f"events={prediction.filtered_event_count or prediction.event_count}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
