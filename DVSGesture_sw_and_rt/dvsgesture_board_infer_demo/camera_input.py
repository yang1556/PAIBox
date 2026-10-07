import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cv2
import dv_processing as dv
import numpy as np

from .runtime import FRAME_SHAPE, TIMESTEPS

TARGET_HW = (128, 128)
DISPLAY_WINDOW_NAME = "DVS Gesture Board Demo"
DISPLAY_FPS = 30.0
DISPLAY_DECAY_HALFLIFE_MS = 120.0
DISPLAY_MIN_SIDE = 720
DISPLAY_MAX_SIDE = 1280


@dataclass(frozen=True)
class EventArrays:
    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray


@dataclass(frozen=True)
class EventFilterConfig:
    name: str = "none"
    background_activity_ms: float = 1.0
    fast_decay_half_life_ms: float = 10.0
    fast_decay_subdivision: int = 4
    fast_decay_threshold: float = 6.0


class EventFilter:
    def __init__(
        self, *, sensor_hw: tuple[int, int], config: EventFilterConfig
    ) -> None:
        self.config = config
        self._filter = self._build_filter(sensor_hw, config)

    def accept(self, events: Any) -> Any:
        if self._filter is None:
            return events
        self._filter.accept(events)
        return self._filter.generateEvents()

    @staticmethod
    def _build_filter(sensor_hw: tuple[int, int], config: EventFilterConfig) -> Any:
        height, width = sensor_hw
        resolution = (width, height)
        if config.name == "none":
            return None
        if config.name == "background":
            return dv.noise.BackgroundActivityNoiseFilter(
                resolution,
                backgroundActivityDuration=timedelta(
                    milliseconds=config.background_activity_ms
                ),
            )
        if config.name == "fast-decay":
            return dv.noise.FastDecayNoiseFilter(
                resolution,
                halfLife=timedelta(milliseconds=config.fast_decay_half_life_ms),
                subdivisionFactor=config.fast_decay_subdivision,
                noiseThreshold=config.fast_decay_threshold,
            )
        raise ValueError(f"unsupported event filter: {config.name}")


def make_event_filter(
    sensor_hw: tuple[int, int],
    *,
    name: str,
    background_activity_ms: float = 1.0,
    fast_decay_half_life_ms: float = 10.0,
    fast_decay_subdivision: int = 4,
    fast_decay_threshold: float = 6.0,
) -> EventFilter:
    return EventFilter(
        sensor_hw=sensor_hw,
        config=EventFilterConfig(
            name=name,
            background_activity_ms=background_activity_ms,
            fast_decay_half_life_ms=fast_decay_half_life_ms,
            fast_decay_subdivision=fast_decay_subdivision,
            fast_decay_threshold=fast_decay_threshold,
        ),
    )


def open_camera(serial_number: str | None = None) -> Any:
    devices = dv.io.camera.discover()
    if not devices:
        raise RuntimeError(
            "no dv-processing compatible DVS camera was discovered; check USB "
            "connection, permissions, and udev rules"
        )
    if serial_number is not None:
        try:
            return dv.io.camera.open(serial_number)
        except Exception as exc:
            raise RuntimeError(
                _format_open_camera_error(serial_number, devices, exc)
            ) from None
    try:
        return dv.io.camera.open()
    except Exception as exc:
        raise RuntimeError(_format_open_camera_error(None, devices, exc)) from None


def _format_open_camera_error(
    serial_number: str | None, devices: object, exc: Exception
) -> str:
    target = (
        f"camera serial {serial_number!r}"
        if serial_number is not None
        else "discovered camera"
    )
    message = (
        f"failed to open {target}; discovered_serials="
        f"{camera_serial_numbers(devices)}; discovered={devices!r}; error={exc}"
    )
    error_text = str(exc)
    if "LIBUSB_ERROR_ACCESS" in error_text or "requires write access" in error_text:
        message += (
            "\nUSB permission hint: libusb needs write access to the DVS camera "
            "device node. Check /dev/bus/usb permissions, install a udev rule, "
            "or close another process that may already hold the camera."
        )
    return message


def camera_serial_numbers(devices: object) -> list[str]:
    return [
        str(serial)
        for device in devices
        if (serial := getattr(device, "serialNumber", None)) is not None
    ]


def normalize_resolution(resolution: Any) -> tuple[int, int]:
    if hasattr(resolution, "height") and hasattr(resolution, "width"):
        return int(resolution.height), int(resolution.width)
    if hasattr(resolution, "rows") and hasattr(resolution, "cols"):
        return int(resolution.rows), int(resolution.cols)
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        first, second = int(resolution[0]), int(resolution[1])
        return (second, first) if first != second else (first, second)
    raise ValueError(f"unsupported camera resolution object: {resolution!r}")


def event_store_to_arrays(event_store: Any) -> EventArrays:
    data = event_store.numpy()
    return EventArrays(
        timestamps=np.asarray(data["timestamp"], dtype=np.int64),
        x=np.asarray(data["x"], dtype=np.int64),
        y=np.asarray(data["y"], dtype=np.int64),
        polarity=np.asarray(data["polarity"], dtype=np.int8),
    )


def events_to_frames(
    events: EventArrays,
    *,
    sensor_hw: tuple[int, int] = TARGET_HW,
    resize_mode: str = "scale",
) -> np.ndarray:
    height, width = sensor_hw
    target_h, target_w = TARGET_HW
    x, y = map_event_coordinates(events.x, events.y, width, height, resize_mode)
    polarity = (np.asarray(events.polarity) > 0).astype(np.int64, copy=False)
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

    frames = np.zeros(FRAME_SHAPE, dtype=np.uint16)
    if x.size == 0:
        return frames.astype(np.uint8)

    bin_index = event_bin_indices(timestamps)
    np.add.at(frames, (bin_index, polarity, y, x), 1)
    return np.clip(frames, 0, 255).astype(np.uint8)


def cap_frame_nonzero(frames: np.ndarray, max_nonzero: int) -> tuple[np.ndarray, int]:
    frames = np.asarray(frames, dtype=np.uint8)
    nonzero_count = int(np.count_nonzero(frames))
    if max_nonzero <= 0 or nonzero_count <= max_nonzero:
        return frames, nonzero_count

    capped = np.zeros_like(frames)
    positions = np.flatnonzero(frames)
    keep = positions[:max_nonzero]
    capped.flat[keep] = frames.flat[keep]
    return capped, int(keep.size)


def map_event_coordinates(
    x: np.ndarray, y: np.ndarray, width: int, height: int, resize_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    target_h, target_w = TARGET_HW
    if (height, width) == TARGET_HW:
        return x, y

    if resize_mode == "strict":
        raise ValueError(
            f"camera resolution {(height, width)} does not match {TARGET_HW}; "
            "use scale or crop resize mode for non-DVS128 sensors"
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


def event_bin_indices(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.size == 0:
        return np.zeros(0, dtype=np.int64)
    start = int(timestamps.min())
    stop = int(timestamps.max())
    if stop <= start:
        return np.zeros(timestamps.shape[0], dtype=np.int64)
    normalized = (timestamps.astype(np.float64) - start) / (stop - start + 1)
    return np.minimum((normalized * TIMESTEPS).astype(np.int64), TIMESTEPS - 1)


class DvsOpenCvDisplay:
    def __init__(self, sensor_hw: tuple[int, int]) -> None:
        height, width = sensor_hw
        self.height = height
        self.width = width
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_decay_at = time.perf_counter()
        self.last_show_at = 0.0
        self.show_interval_s = 1.0 / DISPLAY_FPS
        self.display_size = self._display_size(height, width)

        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DISPLAY_WINDOW_NAME, *self.display_size)

    def update(self, events: EventArrays | None, overlay: str = "") -> bool:
        now = time.perf_counter()
        self._decay(now)
        if events is not None:
            self._draw_events(events)
        if now - self.last_show_at < self.show_interval_s:
            return True

        self.last_show_at = now
        view = self.image.copy()
        if self.display_size != (self.width, self.height):
            view = cv2.resize(view, self.display_size, interpolation=cv2.INTER_NEAREST)
        self._draw_overlay(view, overlay or "prediction: waiting")
        cv2.imshow(DISPLAY_WINDOW_NAME, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        try:
            return cv2.getWindowProperty(DISPLAY_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        cv2.destroyWindow(DISPLAY_WINDOW_NAME)

    def _decay(self, now: float) -> None:
        elapsed_ms = max(0.0, (now - self.last_decay_at) * 1000.0)
        self.last_decay_at = now
        if elapsed_ms <= 0:
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

    def _draw_overlay(self, view: np.ndarray, text: str) -> None:
        cv2.putText(
            view,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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
