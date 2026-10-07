import csv
import serial
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paicorelib import CoordZXYOffset

TIMESTEPS = 20
FRAME_SHAPE = (TIMESTEPS, 2, 128, 128)
NUM_CLASSES = 11
VOTING_SIZE = 10
OUTPUT_DELAY_SYNCS = 13
FRAME_SIZE_BYTES = 8
INPUT_TARGET_LCN = 7
WORK_WRITE_CHUNK_FRAMES = 256


@dataclass(frozen=True)
class InputRegion:
    channel: int
    y_start: int
    core: CoordZXYOffset


@dataclass(frozen=True)
class SampleResult:
    sample_index: int
    counts: np.ndarray
    board_latency_ms: float
    timing: "RuntimeTiming"


@dataclass(frozen=True)
class RuntimeTiming:
    encode_ms: float = 0.0
    tx_work_ms: float = 0.0
    sync_wait_ms: float = 0.0
    work_frame_count: int = 0
    work_write_count: int = 0
    sync_count: int = 0


@dataclass
class _MutableRuntimeTiming:
    encode_ms: float = 0.0
    tx_work_ms: float = 0.0
    sync_wait_ms: float = 0.0
    work_frame_count: int = 0
    work_write_count: int = 0
    sync_count: int = 0

    def freeze(self) -> RuntimeTiming:
        return RuntimeTiming(
            encode_ms=self.encode_ms,
            tx_work_ms=self.tx_work_ms,
            sync_wait_ms=self.sync_wait_ms,
            work_frame_count=self.work_frame_count,
            work_write_count=self.work_write_count,
            sync_count=self.sync_count,
        )


CONTROL_CORE = CoordZXYOffset(0, 1, 2)
INPUT_REGIONS = (
    InputRegion(channel=0, y_start=0, core=CoordZXYOffset(0, 0, 7)),
    InputRegion(channel=0, y_start=64, core=CoordZXYOffset(0, 0, 6)),
    InputRegion(channel=1, y_start=0, core=CoordZXYOffset(0, 0, 5)),
    InputRegion(channel=1, y_start=64, core=CoordZXYOffset(0, 0, 4)),
)
CLASS_ADDR_TO_INDEX = {class_id * 4: class_id for class_id in range(NUM_CLASSES)}
SampleDoneCallback = Callable[[SampleResult], None]


def open_serial(port: str, baud: int = 3_000_000) -> serial.Serial:
    if serial is None:
        raise RuntimeError("pyserial is required for board inference")
    ser = serial.Serial(port, baud, timeout=0.02)
    time.sleep(0.2)
    ser.reset_input_buffer()
    return ser


def build_work_frame(
    *, timestep: int, core: CoordZXYOffset, axon_addr: int, data: int
) -> int:
    if not 0 <= timestep <= 0xFF:
        raise ValueError(f"timestep must fit in uint8, got {timestep}")
    if not 0 <= axon_addr <= 0xFFFF:
        raise ValueError(f"axon_addr must fit in uint16, got {axon_addr}")
    if not 0 <= data <= 0xFF:
        raise ValueError(f"data must fit in uint8, got {data}")

    high = 0x2 << 30
    high |= ((timestep >> 7) & 0x1) << 28
    high |= (core.z & 0x3F) << 22
    high |= (core.x & 0x3F) << 16
    high |= (core.y & 0x3F) << 10

    # INPUT_TARGET_LCN=7 leaves one physical timestep bit and uses low[23:8]
    # as the direct 16-bit axon address.
    low = ((axon_addr & 0xFFFF) << 8) | data
    return ((high & 0xFFFFFFFF) << 32) | (low & 0xFFFFFFFF)


def build_control_frame(*, ctrl_type: int, core: CoordZXYOffset, load: int) -> int:
    if not 0 <= load <= 0xFFFFFF:
        raise ValueError(f"control load must fit in 24 bits, got {load}")
    high = 0x3 << 30
    high |= (ctrl_type & 0x3) << 28
    high |= (core.z & 0x3F) << 22
    high |= (core.x & 0x3F) << 16
    high |= (core.y & 0x3F) << 10
    return ((high & 0xFFFFFFFF) << 32) | (load & 0xFFFFFF)


def build_init_frame() -> int:
    return build_control_frame(ctrl_type=0x1, core=CONTROL_CORE, load=0)


def build_sync_frame() -> int:
    return build_control_frame(ctrl_type=0x0, core=CONTROL_CORE, load=1)


def frame_to_bytes(frame: int) -> bytes:
    return int(frame).to_bytes(FRAME_SIZE_BYTES, byteorder="big", signed=False)


def frames_to_bytes(frames: np.ndarray) -> bytes:
    frames = np.asarray(frames, dtype=np.uint64)
    big_endian = frames.astype(">u8", copy=False)
    return big_endian.tobytes()


def bytes_to_frame(data: bytes) -> int:
    if len(data) != FRAME_SIZE_BYTES:
        raise ValueError(f"expected 8 bytes, got {len(data)}")
    return int.from_bytes(data, byteorder="big", signed=False)


def frame_hex(frame: int) -> str:
    return f"{frame & 0xFFFFFFFFFFFFFFFF:016X}"


def input_timestep(sample_t: int) -> int:
    if not 0 <= sample_t < TIMESTEPS:
        raise ValueError(f"sample_t must be in [0, {TIMESTEPS}), got {sample_t}")
    return (sample_t & 0x1) << INPUT_TARGET_LCN


def axon_addr(local_y: int, x: int) -> int:
    if not 0 <= local_y < 64:
        raise ValueError(f"local_y must be in [0, 64), got {local_y}")
    if not 0 <= x < 128:
        raise ValueError(f"x must be in [0, 128), got {x}")
    return (local_y * 128 + x) << 3


class _InputFrameEncoder:
    def __init__(self) -> None:
        self.frame_base = self._build_frame_base()

    def encode_array(self, sample_t: int, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.shape != FRAME_SHAPE[1:]:
            raise ValueError(
                f"frame has shape {frames.shape}; expected {FRAME_SHAPE[1:]}"
            )

        parity = sample_t & 0x1
        encoded_regions: list[np.ndarray] = []
        for region_idx, region in enumerate(INPUT_REGIONS):
            half = frames[region.channel, region.y_start : region.y_start + 64, :]
            ys, xs = np.nonzero(half)
            if ys.size == 0:
                continue
            values = half[ys, xs].astype(np.uint64, copy=False)
            encoded_regions.append(self.frame_base[parity, region_idx, ys, xs] | values)

        if not encoded_regions:
            return np.empty(0, dtype=np.uint64)
        return np.concatenate(encoded_regions).astype(np.uint64, copy=False)

    @staticmethod
    def _build_frame_base() -> np.ndarray:
        base = np.empty((2, len(INPUT_REGIONS), 64, 128), dtype=np.uint64)
        local_y = np.arange(64, dtype=np.uint64)[:, np.newaxis]
        local_x = np.arange(128, dtype=np.uint64)[np.newaxis, :]
        low_base = (((local_y * 128) + local_x) << 3) << 8
        for parity, timestep in enumerate((0, 1 << INPUT_TARGET_LCN)):
            for region_idx, region in enumerate(INPUT_REGIONS):
                high = 0x2 << 30
                high |= ((timestep >> 7) & 0x1) << 28
                high |= (region.core.z & 0x3F) << 22
                high |= (region.core.x & 0x3F) << 16
                high |= (region.core.y & 0x3F) << 10
                base[parity, region_idx] = (
                    np.uint64(high & 0xFFFFFFFF) << np.uint64(32)
                ) | low_base
        return base


_INPUT_FRAME_ENCODER = _InputFrameEncoder()


def encode_timestep_frames_array(sample_t: int, frames: np.ndarray) -> np.ndarray:
    return _INPUT_FRAME_ENCODER.encode_array(sample_t, frames)


def encode_timestep_frames(sample_t: int, frames: np.ndarray) -> list[int]:
    return [int(frame) for frame in encode_timestep_frames_array(sample_t, frames)]


def _encode_timestep_frames_reference(sample_t: int, frames: np.ndarray) -> list[int]:
    frames = np.asarray(frames)
    if frames.shape != FRAME_SHAPE[1:]:
        raise ValueError(f"frame has shape {frames.shape}; expected {FRAME_SHAPE[1:]}")

    encoded: list[int] = []
    ts = input_timestep(sample_t)
    for region in INPUT_REGIONS:
        half = frames[region.channel, region.y_start : region.y_start + 64, :]
        ys, xs = np.nonzero(half)
        values = half[ys, xs]
        for local_y, x, value in zip(ys, xs, values, strict=True):
            data = int(value)
            if data == 0:
                continue
            encoded.append(
                build_work_frame(
                    timestep=ts,
                    core=region.core,
                    axon_addr=axon_addr(int(local_y), int(x)),
                    data=data,
                )
            )
    return encoded


def parse_frame_header(frame: int) -> tuple[int, int]:
    high = (frame >> 32) & 0xFFFFFFFF
    return (high >> 30) & 0x3, (high >> 28) & 0x3


def is_complete_frame(frame: int) -> bool:
    high = (frame >> 32) & 0xFFFFFFFF
    return ((high >> 28) & 0xF) == 0xE


def parse_output_work_frame(frame: int) -> tuple[int, int, int]:
    high = (frame >> 32) & 0xFFFFFFFF
    low = frame & 0xFFFFFFFF
    ts_high = (high >> 28) & 0x1
    timestep = (ts_high << 7) | ((low >> 17) & 0x7F)
    addr = (low >> 8) & 0x1FF
    payload = low & 0xFF
    if payload & 0xF0:
        raise ValueError(f"invalid uint4 output payload 0x{payload:02X}")
    if addr not in CLASS_ADDR_TO_INDEX:
        raise ValueError(f"unknown output class axon address {addr}")
    return timestep, CLASS_ADDR_TO_INDEX[addr], payload


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def counts_to_logits(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    if counts.ndim != 2 or counts.shape != (TIMESTEPS, NUM_CLASSES):
        raise ValueError(
            f"counts has shape {counts.shape}; expected {(TIMESTEPS, NUM_CLASSES)}"
        )
    return (counts / VOTING_SIZE).sum(axis=0)


def predict_from_counts(counts: np.ndarray) -> tuple[int, float, np.ndarray]:
    logits = counts_to_logits(counts)
    probabilities = softmax(logits)
    pred = int(np.argmax(logits))
    return pred, float(probabilities[pred]), logits


class BoardRuntime:
    def __init__(
        self,
        ser: serial.Serial,
        *,
        timeout: float = 100.0,
        skip_init: bool = False,
        dump_path: Path | None = None,
    ) -> None:
        self.ser = ser
        self.timeout = timeout
        self.skip_init = skip_init
        self.dump_path = dump_path
        self._dump_file = None
        self.sync_count = 0
        self._sample_started_at: dict[int, float] = {}
        self._sample_timing: dict[int, _MutableRuntimeTiming] = {}

    def __enter__(self) -> "BoardRuntime":
        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            self._dump_file = self.dump_path.open("w", newline="")
            csv.writer(self._dump_file).writerow(("time", "direction", "frame"))
        if not self.skip_init:
            self._send_frame(build_init_frame())
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._dump_file is not None:
            self._dump_file.close()
            self._dump_file = None

    def close(self) -> None:
        self.ser.close()

    def infer_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = self._validate_samples(samples)
        n_samples = samples.shape[0]
        outputs = np.zeros((n_samples, TIMESTEPS, NUM_CLASSES), dtype=np.int32)

        for sample in samples:
            for sample_t in range(TIMESTEPS):
                self._send_sample_timestep(
                    None,
                    sample_t,
                    sample[sample_t],
                    outputs,
                    on_sample_done=None,
                )

        for _ in range(OUTPUT_DELAY_SYNCS):
            self._sync_and_decode(outputs)

        return outputs

    def infer_samples_streaming(
        self,
        samples: np.ndarray,
        *,
        on_sample_done: SampleDoneCallback | None = None,
    ) -> np.ndarray:
        samples = self._validate_samples(samples)
        n_samples = samples.shape[0]
        outputs = np.zeros((n_samples, TIMESTEPS, NUM_CLASSES), dtype=np.int32)

        for sample_index, sample in enumerate(samples):
            self._begin_sample(sample_index)
            for sample_t in range(TIMESTEPS):
                self._send_sample_timestep(
                    sample_index,
                    sample_t,
                    sample[sample_t],
                    outputs,
                    on_sample_done=on_sample_done,
                )

        for _ in range(OUTPUT_DELAY_SYNCS):
            self._sync_and_decode(outputs, on_sample_done=on_sample_done)

        return outputs

    def send_stream_sample(
        self,
        sample: np.ndarray,
        *,
        outputs: np.ndarray,
        sample_index: int,
        on_sample_done: SampleDoneCallback | None = None,
    ) -> None:
        sample = self._validate_single_sample(sample)
        if sample_index < 0 or sample_index >= outputs.shape[0]:
            raise ValueError(f"sample_index {sample_index} is outside outputs")
        self._begin_sample(sample_index)
        for sample_t in range(TIMESTEPS):
            self._send_sample_timestep(
                sample_index,
                sample_t,
                sample[sample_t],
                outputs,
                on_sample_done=on_sample_done,
            )

    def flush_stream(
        self,
        outputs: np.ndarray,
        *,
        on_sample_done: SampleDoneCallback | None = None,
    ) -> None:
        for _ in range(OUTPUT_DELAY_SYNCS):
            self._sync_and_decode(outputs, on_sample_done=on_sample_done)

    def _sync_and_decode(
        self,
        outputs: np.ndarray,
        *,
        on_sample_done: SampleDoneCallback | None = None,
    ) -> float:
        started_at = time.perf_counter()
        self._send_frame(build_sync_frame())
        current_sync = self.sync_count
        self.sync_count += 1
        output_tick = current_sync - OUTPUT_DELAY_SYNCS
        rx_frames = self._recv_until_complete(current_sync)
        sync_wait_ms = (time.perf_counter() - started_at) * 1000.0
        if output_tick < 0:
            return sync_wait_ms

        sample_idx, sample_t = divmod(output_tick, TIMESTEPS)
        if sample_idx < 0 or sample_idx >= outputs.shape[0]:
            return sync_wait_ms

        for frame in rx_frames:
            if is_complete_frame(frame):
                continue
            frame_type, _subtype = parse_frame_header(frame)
            if frame_type != 0x2:
                continue
            try:
                _timestep, class_idx, value = parse_output_work_frame(frame)
            except ValueError:
                continue
            outputs[sample_idx, sample_t, class_idx] += value
        if sample_t == TIMESTEPS - 1:
            started_at = self._sample_started_at.pop(sample_idx, None)
            timing = self._sample_timing.pop(
                sample_idx, _MutableRuntimeTiming()
            ).freeze()
            latency_ms = (
                0.0
                if started_at is None
                else (time.perf_counter() - started_at) * 1000.0
            )
            if on_sample_done is not None:
                on_sample_done(
                    SampleResult(
                        sample_index=sample_idx,
                        counts=outputs[sample_idx].copy(),
                        board_latency_ms=latency_ms,
                        timing=timing,
                    )
                )
        return sync_wait_ms

    def _begin_sample(self, sample_index: int) -> None:
        self._sample_started_at[sample_index] = time.perf_counter()
        self._sample_timing[sample_index] = _MutableRuntimeTiming()

    def _send_sample_timestep(
        self,
        sample_index: int | None,
        sample_t: int,
        frame: np.ndarray,
        outputs: np.ndarray,
        *,
        on_sample_done: SampleDoneCallback | None,
    ) -> None:
        timing = None if sample_index is None else self._sample_timing.get(sample_index)
        started_at = time.perf_counter()
        encoded = encode_timestep_frames_array(sample_t, frame)
        encode_ms = (time.perf_counter() - started_at) * 1000.0

        started_at = time.perf_counter()
        work_write_count = self._send_work_frames(encoded)
        tx_work_ms = (time.perf_counter() - started_at) * 1000.0

        sync_wait_ms = self._sync_and_decode(outputs, on_sample_done=on_sample_done)

        if timing is not None:
            timing.encode_ms += encode_ms
            timing.tx_work_ms += tx_work_ms
            timing.sync_wait_ms += sync_wait_ms
            timing.work_frame_count += int(encoded.size)
            timing.work_write_count += work_write_count
            timing.sync_count += 1

    def _send_frame(self, frame: int) -> None:
        self.ser.write(frame_to_bytes(frame))
        self._dump("TX", frame)

    def _send_work_frames(self, frames: np.ndarray) -> int:
        frames = np.asarray(frames, dtype=np.uint64)
        if frames.size == 0:
            return 0
        write_count = 0
        for start in range(0, frames.size, WORK_WRITE_CHUNK_FRAMES):
            chunk = frames[start : start + WORK_WRITE_CHUNK_FRAMES]
            self.ser.write(frames_to_bytes(chunk))
            write_count += 1
            self._dump_many("TX", chunk)
        return write_count

    def _recv_until_complete(self, sync_index: int) -> list[int]:
        frames: list[int] = []
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            frame = self._read_frame(deadline)
            if frame is None:
                break
            frames.append(frame)
            self._dump("RX", frame)
            if is_complete_frame(frame):
                return frames

        raise TimeoutError(
            "timed out waiting for relay complete frame "
            f"after sync {sync_index} on {self.ser.port} "
            f"(timeout={self.timeout}s, rx_frames={len(frames)}). "
            "Check BOARD_PORT, relay firmware/application state, board "
            "configuration, and whether SKIP_INIT matches the current board "
            "state."
        )

    def _read_frame(self, deadline: float) -> int | None:
        data = bytearray()
        while len(data) < FRAME_SIZE_BYTES and time.monotonic() < deadline:
            chunk = self.ser.read(FRAME_SIZE_BYTES - len(data))
            if chunk:
                data.extend(chunk)
        if len(data) != FRAME_SIZE_BYTES:
            return None
        return bytes_to_frame(bytes(data))

    def _dump(self, direction: str, frame: int) -> None:
        if self._dump_file is None:
            return
        csv.writer(self._dump_file).writerow(
            (f"{time.time():.6f}", direction, frame_hex(frame))
        )
        self._dump_file.flush()

    def _dump_many(self, direction: str, frames: np.ndarray) -> None:
        if self._dump_file is None:
            return
        now = f"{time.time():.6f}"
        writer = csv.writer(self._dump_file)
        writer.writerows((now, direction, frame_hex(int(frame))) for frame in frames)
        self._dump_file.flush()

    @staticmethod
    def _validate_samples(samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.uint8)
        if samples.ndim == 4:
            samples = samples[np.newaxis, ...]
        if samples.ndim != 5 or samples.shape[1:] != FRAME_SHAPE:
            raise ValueError(
                f"samples has shape {samples.shape}; expected "
                f"(N, {FRAME_SHAPE[0]}, {FRAME_SHAPE[1]}, "
                f"{FRAME_SHAPE[2]}, {FRAME_SHAPE[3]})"
            )
        return np.ascontiguousarray(samples)

    @staticmethod
    def _validate_single_sample(sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.uint8)
        if sample.shape != FRAME_SHAPE:
            raise ValueError(f"sample has shape {sample.shape}; expected {FRAME_SHAPE}")
        return np.ascontiguousarray(sample)


def run_board(
    samples: np.ndarray,
    *,
    port: str,
    baud: int = 3_000_000,
    timeout: float = 100.0,
    skip_init: bool = False,
    dump_path: Path | None = None,
    on_sample_done: SampleDoneCallback | None = None,
) -> np.ndarray:
    ser = open_serial(port, baud)
    runtime = BoardRuntime(
        ser, timeout=timeout, skip_init=skip_init, dump_path=dump_path
    )
    try:
        with runtime:
            return runtime.infer_samples_streaming(
                samples, on_sample_done=on_sample_done
            )
    finally:
        runtime.close()
