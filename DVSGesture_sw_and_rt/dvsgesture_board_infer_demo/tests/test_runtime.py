from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys

import numpy as np
import pytest

from dvsgesture_board_infer_demo import dataset
from dvsgesture_board_infer_demo import app as app_module
from dvsgesture_board_infer_demo.app import (
    CameraBoardWorker,
    CameraWorkerResult,
    CameraWindow,
    JsonlWriter,
    LatestWindowBuffer,
    build_prediction,
    exceeds_limit,
    format_camera_prediction,
    format_prediction_progress,
    normalize_run_paths,
    reduction_ratio,
)
from dvsgesture_board_infer_demo.camera_input import (
    EventArrays,
    EventFilterConfig,
    EventFilter,
    camera_serial_numbers,
    cap_frame_nonzero,
    dv,
    event_bin_indices,
    events_to_frames,
    map_event_coordinates,
    open_camera,
)
from dvsgesture_board_infer_demo.runtime import (
    CLASS_ADDR_TO_INDEX,
    CONTROL_CORE,
    INPUT_REGIONS,
    NUM_CLASSES,
    OUTPUT_DELAY_SYNCS,
    WORK_WRITE_CHUNK_FRAMES,
    TIMESTEPS,
    BoardRuntime,
    RuntimeTiming,
    SampleResult,
    _encode_timestep_frames_reference,
    axon_addr,
    build_control_frame,
    build_init_frame,
    build_sync_frame,
    build_work_frame,
    bytes_to_frame,
    CoordZXYOffset,
    counts_to_logits,
    encode_timestep_frames,
    encode_timestep_frames_array,
    frames_to_bytes,
    frame_to_bytes,
    input_timestep,
    is_complete_frame,
    parse_output_work_frame,
)


class FakeSerial:
    def __init__(
        self,
        rx_batches: list[list[int]] | None = None,
        *,
        fill_on_read: bool = False,
    ) -> None:
        self.writes: list[int] = []
        self.write_sizes: list[int] = []
        self.port = "fake://board"
        self._rx_batches = list(rx_batches or [])
        self._buffer = bytearray()
        self._fill_on_read = fill_on_read

    @property
    def in_waiting(self) -> int:
        self._fill_next_batch()
        return len(self._buffer)

    def read(self, size: int = 1) -> bytes:
        if self._fill_on_read:
            self._fill_next_batch()
        if not self._buffer:
            return b""
        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out

    def write(self, data: bytes) -> int:
        if len(data) % 8 != 0:
            raise ValueError(f"write length must be a multiple of 8, got {len(data)}")
        self.write_sizes.append(len(data))
        for start in range(0, len(data), 8):
            self.writes.append(bytes_to_frame(data[start : start + 8]))
        return len(data)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        pass

    def _fill_next_batch(self) -> None:
        if not self._buffer and self._rx_batches:
            for frame in self._rx_batches.pop(0):
                self._buffer.extend(frame_to_bytes(frame))


class FakeWindowBuffer:
    def __init__(self, windows: list[CameraWindow], dropped: int = 0) -> None:
        self._windows = list(windows)
        self._dropped = dropped

    def get(self) -> CameraWindow | None:
        if not self._windows:
            return None
        return self._windows.pop(0)

    def dropped_count(self) -> int:
        return self._dropped


class FakeCameraDescriptor:
    def __init__(self, serial_number: str) -> None:
        self.serialNumber = serial_number


class FakeCameraApi:
    def __init__(self, devices: list[FakeCameraDescriptor]) -> None:
        self.devices = devices
        self.opened: object = None

    def discover(self) -> list[FakeCameraDescriptor]:
        return self.devices

    def open(self, target: object = None) -> str:
        self.opened = target
        return "camera"


def make_camera_window(
    window_id: int,
    *,
    event_count: int,
    queued_at: float = 1.0,
) -> CameraWindow:
    return CameraWindow(
        window_id=window_id,
        frames=np.zeros((TIMESTEPS, 2, 128, 128), dtype=np.uint8),
        event_count=event_count,
        raw_event_count=event_count * 2,
        filtered_event_count=event_count,
        filter_reduction=0.5,
        frame_nonzero_count=event_count // 2,
        queued_at=queued_at,
    )


def test_resolve_first_sample_matches_loader_order() -> None:
    sample = dataset.load_sample(sample_index=0)
    assert sample.label == 0
    assert sample.frames.shape == (TIMESTEPS, 2, 128, 128)
    assert sample.frames.dtype == np.uint8
    assert sample.path.name == "user24_fluorescent_0.npz"


def test_relative_sample_path_resolution() -> None:
    sample = dataset.load_sample(
        sample_path=Path(
            "data/DVS128Gesture/frames_number_20_split_by_number/test/0/"
            "user24_fluorescent_0.npz"
        )
    )
    assert sample.label == 0
    assert sample.frames.shape == (TIMESTEPS, 2, 128, 128)


def test_validate_frames_rejects_non_integer_values() -> None:
    frames = np.zeros((TIMESTEPS, 2, 128, 128), dtype=np.float32)
    frames[0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="non-integer"):
        dataset.validate_frames(frames, "synthetic")


def test_axon_address_layout() -> None:
    assert axon_addr(0, 0) == 0
    assert axon_addr(0, 127) == 1016
    assert axon_addr(1, 0) == 1024
    assert axon_addr(1, 127) == 2040
    assert axon_addr(63, 127) == ((63 * 128 + 127) << 3)


def test_input_timestep_even_odd_ranges() -> None:
    assert input_timestep(0) == 0
    assert input_timestep(2) == 0
    assert input_timestep(1) == 128
    assert input_timestep(19) == 128


def test_quadrant_encoding_targets_four_cores() -> None:
    frame = np.zeros((2, 128, 128), dtype=np.uint8)
    frame[0, 0, 0] = 1
    frame[0, 64, 0] = 2
    frame[1, 0, 0] = 3
    frame[1, 64, 0] = 4

    encoded = encode_timestep_frames(0, frame)
    assert len(encoded) == 4
    assert [decode_core(f) for f in encoded] == [r.core for r in INPUT_REGIONS]
    assert [decode_data(f) for f in encoded] == [1, 2, 3, 4]
    assert [decode_axon(f) for f in encoded] == [0, 0, 0, 0]


def test_vectorized_encoding_matches_reference_for_boundaries() -> None:
    frame = np.zeros((2, 128, 128), dtype=np.uint8)
    frame[0, 0, 0] = 1
    frame[0, 63, 127] = 2
    frame[0, 64, 0] = 3
    frame[0, 127, 127] = 4
    frame[1, 0, 127] = 5
    frame[1, 63, 0] = 6
    frame[1, 64, 127] = 7
    frame[1, 127, 0] = 8

    for sample_t in (0, 1, 18, 19):
        encoded = encode_timestep_frames(sample_t, frame)
        reference = _encode_timestep_frames_reference(sample_t, frame)
        encoded_array = encode_timestep_frames_array(sample_t, frame)
        assert encoded == reference
        assert encoded_array.dtype == np.uint64
        assert encoded_array.tolist() == reference


def test_vectorized_encoding_handles_empty_input() -> None:
    frame = np.zeros((2, 128, 128), dtype=np.uint8)

    encoded = encode_timestep_frames_array(0, frame)

    assert encoded.dtype == np.uint64
    assert encoded.size == 0
    assert encode_timestep_frames(0, frame) == []


def test_frames_to_bytes_is_big_endian_for_multiple_frames() -> None:
    frames = np.array([0x0102030405060708, 0xA0A1A2A3A4A5A6A7], dtype=np.uint64)

    assert frames_to_bytes(frames) == bytes.fromhex("0102030405060708A0A1A2A3A4A5A6A7")


def test_work_frame_encodes_high_axon_bits_directly() -> None:
    frame = build_work_frame(
        timestep=128,
        core=CoordZXYOffset(0, 0, 7),
        axon_addr=2040,
        data=9,
    )
    assert decode_timestep(frame) == 128 + (2040 >> 9)
    assert decode_core(frame) == CoordZXYOffset(0, 0, 7)
    assert decode_axon(frame) == 2040
    assert decode_data(frame) == 9


def test_init_only_once_and_one_sync_per_input_tick() -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    ser = FakeSerial(
        rx_batches=[[complete_frame()] for _ in range(TIMESTEPS + OUTPUT_DELAY_SYNCS)],
        fill_on_read=True,
    )
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=False)

    with runtime:
        runtime.infer_samples(sample)

    assert ser.writes[0] == build_init_frame()
    sync_frame = build_control_frame(ctrl_type=0, core=CONTROL_CORE, load=1)
    assert ser.writes.count(sync_frame) == TIMESTEPS + OUTPUT_DELAY_SYNCS


def test_skip_init_omits_init_frame() -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    ser = FakeSerial(
        rx_batches=[[complete_frame()] for _ in range(TIMESTEPS + OUTPUT_DELAY_SYNCS)],
        fill_on_read=True,
    )
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)

    with runtime:
        runtime.infer_samples(sample)

    assert build_init_frame() not in ser.writes


def test_pipeline_delay_aligns_two_samples_and_flush() -> None:
    n_samples = 2
    total_syncs = n_samples * TIMESTEPS + OUTPUT_DELAY_SYNCS
    batches: list[list[int]] = []
    for sync_idx in range(total_syncs):
        output_tick = sync_idx - OUTPUT_DELAY_SYNCS
        if 0 <= output_tick < n_samples * TIMESTEPS:
            sample_idx, sample_t = divmod(output_tick, TIMESTEPS)
            class_idx = (sample_t + sample_idx) % NUM_CLASSES
            batches.append(
                [output_frame(output_tick + 23, class_idx, 7), complete_frame()]
            )
        else:
            batches.append([complete_frame()])

    samples = np.zeros((n_samples, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    ser = FakeSerial(rx_batches=batches, fill_on_read=True)
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)
    with runtime:
        outputs = runtime.infer_samples(samples)

    assert outputs.shape == (n_samples, TIMESTEPS, NUM_CLASSES)
    for sample_idx in range(n_samples):
        for sample_t in range(TIMESTEPS):
            class_idx = (sample_t + sample_idx) % NUM_CLASSES
            assert outputs[sample_idx, sample_t, class_idx] == 7
    assert outputs.sum() == n_samples * TIMESTEPS * 7


def test_sync_waits_for_complete_frame_before_next_tick() -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    total_syncs = TIMESTEPS + OUTPUT_DELAY_SYNCS
    batches = [[complete_frame()] for _ in range(total_syncs)]
    batches[OUTPUT_DELAY_SYNCS] = [
        output_frame(timestep=33, class_idx=2, value=5),
        complete_frame(),
    ]
    ser = FakeSerial(rx_batches=batches, fill_on_read=True)
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)

    with runtime:
        outputs = runtime.infer_samples(sample)

    assert outputs[0, 0, 2] == 5
    assert sum(1 for frame in ser.writes if frame == build_sync_frame()) == total_syncs
    assert is_complete_frame(complete_frame())


def test_work_frames_are_chunked_before_sync() -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    sample[0, 0, 0, 0, :] = 1
    sample[0, 0, 0, 1, :] = 1
    sample[0, 0, 0, 2, 0] = 1
    total_syncs = TIMESTEPS + OUTPUT_DELAY_SYNCS
    ser = FakeSerial(
        rx_batches=[[complete_frame()] for _ in range(total_syncs)],
        fill_on_read=True,
    )
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)

    with runtime:
        runtime.infer_samples(sample)

    first_sync_index = ser.writes.index(build_sync_frame())
    assert first_sync_index == WORK_WRITE_CHUNK_FRAMES + 1
    assert ser.write_sizes[:3] == [
        WORK_WRITE_CHUNK_FRAMES * 8,
        8,
        8,
    ]
    assert ser.writes[first_sync_index] == build_sync_frame()


def test_dump_path_records_chunked_tx_frames(tmp_path: Path) -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    sample[0, 0, 0, 0, :3] = [1, 2, 3]
    total_syncs = TIMESTEPS + OUTPUT_DELAY_SYNCS
    ser = FakeSerial(
        rx_batches=[[complete_frame()] for _ in range(total_syncs)],
        fill_on_read=True,
    )
    dump_path = tmp_path / "frames.csv"
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True, dump_path=dump_path)

    with runtime:
        runtime.infer_samples(sample)

    rows = dump_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "time,direction,frame"
    tx_rows = [row for row in rows[1:] if ",TX," in row]
    rx_rows = [row for row in rows[1:] if ",RX," in row]
    assert len(tx_rows) == len(ser.writes)
    assert len(rx_rows) == total_syncs


def test_streaming_callback_fires_when_each_sample_is_complete() -> None:
    n_samples = 3
    total_syncs = n_samples * TIMESTEPS + OUTPUT_DELAY_SYNCS
    batches: list[list[int]] = []
    for sync_idx in range(total_syncs):
        output_tick = sync_idx - OUTPUT_DELAY_SYNCS
        if 0 <= output_tick < n_samples * TIMESTEPS:
            sample_idx, sample_t = divmod(output_tick, TIMESTEPS)
            batches.append(
                [
                    output_frame(
                        timestep=output_tick + 23,
                        class_idx=sample_idx,
                        value=1,
                    ),
                    complete_frame(),
                ]
            )
        else:
            batches.append([complete_frame()])

    calls: list[tuple[int, int, float, RuntimeTiming]] = []

    def on_sample_done(result: SampleResult) -> None:
        calls.append(
            (
                result.sample_index,
                int(result.counts.sum()),
                result.board_latency_ms,
                result.timing,
            )
        )

    samples = np.zeros((n_samples, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    ser = FakeSerial(rx_batches=batches, fill_on_read=True)
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)
    with runtime:
        runtime.infer_samples_streaming(samples, on_sample_done=on_sample_done)

    expected_sum = TIMESTEPS
    assert [(idx, total) for idx, total, _latency, _timing in calls] == [
        (0, expected_sum),
        (1, expected_sum),
        (2, expected_sum),
    ]
    assert all(latency >= 0.0 for _idx, _total, latency, _timing in calls)
    assert all(
        timing.encode_ms >= 0.0
        and timing.tx_work_ms >= 0.0
        and timing.sync_wait_ms >= 0.0
        and timing.sync_count == TIMESTEPS
        and timing.work_frame_count == 0
        and timing.work_write_count == 0
        for _idx, _total, _latency, timing in calls
    )


def test_streaming_callback_reports_input_work_timing() -> None:
    total_syncs = TIMESTEPS + OUTPUT_DELAY_SYNCS
    batches = [[complete_frame()] for _ in range(total_syncs)]
    batches[-1] = [output_frame(timestep=19, class_idx=3, value=1), complete_frame()]
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    sample[0, 0, 0, 0, :] = 1
    sample[0, 0, 0, 1, :] = 1
    sample[0, 0, 0, 2, 0] = 1
    sample[0, 0, 0, 2, 1] = 2
    calls: list[SampleResult] = []
    ser = FakeSerial(rx_batches=batches, fill_on_read=True)
    runtime = BoardRuntime(ser, timeout=0.01, skip_init=True)

    with runtime:
        runtime.infer_samples_streaming(sample, on_sample_done=calls.append)

    assert len(calls) == 1
    timing = calls[0].timing
    assert timing.encode_ms >= 0.0
    assert timing.tx_work_ms >= 0.0
    assert timing.sync_wait_ms >= 0.0
    assert timing.sync_count == TIMESTEPS
    assert timing.work_frame_count == WORK_WRITE_CHUNK_FRAMES + 2
    assert timing.work_write_count == 2


def test_events_to_frames_time_bins_polarity_and_clip() -> None:
    events = EventArrays(
        timestamps=np.array([0, 10, 20, 30, 40, 40], dtype=np.int64),
        x=np.array([0, 0, 1, 127, 127, 127], dtype=np.int64),
        y=np.array([0, 0, 1, 127, 127, 127], dtype=np.int64),
        polarity=np.array([0, 1, 1, 0, 1, 1], dtype=np.int8),
    )

    frames = events_to_frames(events)

    assert frames.shape == (TIMESTEPS, 2, 128, 128)
    assert frames.dtype == np.uint8
    assert frames[0, 0, 0, 0] == 1
    assert frames[4, 1, 0, 0] == 1
    assert frames[9, 1, 1, 1] == 1
    assert frames[14, 0, 127, 127] == 1
    assert frames[19, 1, 127, 127] == 2


def test_events_to_frames_clips_uint8_counts() -> None:
    n_events = 300
    events = EventArrays(
        timestamps=np.zeros(n_events, dtype=np.int64),
        x=np.zeros(n_events, dtype=np.int64),
        y=np.zeros(n_events, dtype=np.int64),
        polarity=np.ones(n_events, dtype=np.int8),
    )

    frames = events_to_frames(events)

    assert frames[0, 1, 0, 0] == 255


def test_cap_frame_nonzero_keeps_stable_prefix() -> None:
    frames = np.zeros((TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    frames[0, 0, 0, 0] = 1
    frames[0, 0, 0, 1] = 2
    frames[0, 1, 0, 0] = 3
    frames[1, 0, 0, 0] = 4

    capped, nonzero_count = cap_frame_nonzero(frames, 2)

    assert nonzero_count == 2
    assert np.count_nonzero(capped) == 2
    assert capped[0, 0, 0, 0] == 1
    assert capped[0, 0, 0, 1] == 2
    assert capped[0, 1, 0, 0] == 0
    assert capped[1, 0, 0, 0] == 0


def test_cap_frame_nonzero_is_noop_when_under_limit() -> None:
    frames = np.zeros((TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    frames[0, 0, 0, 0] = 1

    capped, nonzero_count = cap_frame_nonzero(frames, 2)

    assert capped is frames
    assert nonzero_count == 1


def test_event_coordinate_mapping_modes() -> None:
    x = np.array([0, 64, 127], dtype=np.int64)
    y = np.array([0, 64, 127], dtype=np.int64)

    scaled_x, scaled_y = map_event_coordinates(x, y, 256, 256, "scale")
    assert scaled_x.tolist() == [0, 32, 63]
    assert scaled_y.tolist() == [0, 32, 63]

    cropped_x, cropped_y = map_event_coordinates(x + 64, y + 64, 256, 256, "crop")
    assert cropped_x.tolist() == [0, 64, 127]
    assert cropped_y.tolist() == [0, 64, 127]

    with pytest.raises(ValueError, match="does not match"):
        map_event_coordinates(x, y, 256, 256, "strict")


def test_event_bin_indices_handles_empty_and_constant_time() -> None:
    assert event_bin_indices(np.array([], dtype=np.int64)).size == 0
    assert event_bin_indices(np.array([7, 7], dtype=np.int64)).tolist() == [0, 0]


def test_reduction_ratio_handles_empty_and_clamps() -> None:
    assert reduction_ratio(100, 40) == pytest.approx(0.6)
    assert reduction_ratio(0, 40) == pytest.approx(0.0)
    assert reduction_ratio(100, 120) == pytest.approx(0.0)


def test_exceeds_limit_treats_zero_as_disabled() -> None:
    assert not exceeds_limit(100, 0)
    assert not exceeds_limit(100, 100)
    assert exceeds_limit(101, 100)


def test_event_filter_none_is_passthrough() -> None:
    events = dv.EventStore()
    event_filter = EventFilter(
        sensor_hw=(260, 346),
        config=EventFilterConfig(name="none"),
    )

    assert event_filter.accept(events) is events


def test_background_event_filter_accepts_empty_store() -> None:
    event_filter = EventFilter(
        sensor_hw=(260, 346),
        config=EventFilterConfig(name="background", background_activity_ms=1.0),
    )

    filtered = event_filter.accept(dv.EventStore())

    assert filtered.size() == 0


def test_open_camera_uses_requested_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    camera_api = FakeCameraApi(
        [FakeCameraDescriptor("00000595"), FakeCameraDescriptor("AERS0002")]
    )
    monkeypatch.setattr(dv.io, "camera", camera_api)

    camera = open_camera("AERS0002")

    assert camera == "camera"
    assert camera_api.opened == "AERS0002"
    assert camera_serial_numbers(camera_api.devices) == ["00000595", "AERS0002"]


def test_open_camera_error_lists_discovered_serials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCameraApi(FakeCameraApi):
        def open(self, target: object = None) -> str:
            raise RuntimeError("open failed")

    camera_api = FailingCameraApi([FakeCameraDescriptor("00000595")])
    monkeypatch.setattr(dv.io, "camera", camera_api)

    with pytest.raises(RuntimeError, match="00000595"):
        open_camera("missing")


def test_sync_without_complete_frame_times_out() -> None:
    sample = np.zeros((1, TIMESTEPS, 2, 128, 128), dtype=np.uint8)
    ser = FakeSerial(rx_batches=[[]], fill_on_read=True)
    ser.port = "/dev/ttyUSB_TEST"
    runtime = BoardRuntime(ser, timeout=0.001, skip_init=True)

    with pytest.raises(TimeoutError) as exc_info:
        with runtime:
            runtime.infer_samples(sample)
    message = str(exc_info.value)
    assert "relay complete frame" in message
    assert "sync 0" in message
    assert "/dev/ttyUSB_TEST" in message
    assert "BOARD_PORT" in message
    assert "SKIP_INIT" in message


def test_output_decode_uint4_and_invalid_payload() -> None:
    frame = output_frame(timestep=3, class_idx=10, value=15)
    assert parse_output_work_frame(frame) == (3, 10, 15)

    invalid = output_frame(timestep=3, class_idx=10, value=0x10)
    with pytest.raises(ValueError, match="uint4"):
        parse_output_work_frame(invalid)


def test_counts_to_logits_matches_voting_reduction() -> None:
    counts = np.zeros((TIMESTEPS, NUM_CLASSES), dtype=np.int32)
    counts[:, 2] = 10
    logits = counts_to_logits(counts)
    assert logits[2] == pytest.approx(float(TIMESTEPS))
    assert np.count_nonzero(logits) == 1


def test_jsonl_prediction_records_board_latency(tmp_path: Path) -> None:
    counts = np.zeros((TIMESTEPS, NUM_CLASSES), dtype=np.int32)
    counts[:, 3] = 10
    prediction = build_prediction(
        source="camera",
        sample="window_000000",
        expected_label=None,
        counts=counts,
        board_latency_ms=12.5,
        timing=RuntimeTiming(
            encode_ms=1.25,
            tx_work_ms=2.5,
            sync_wait_ms=8.75,
            work_frame_count=321,
            work_write_count=4,
            sync_count=20,
        ),
        event_count=1234,
        raw_event_count=2000,
        filtered_event_count=1234,
        filter_reduction=0.383,
        frame_nonzero_count=456,
        source_window_id=9,
    )
    path = tmp_path / "predictions.jsonl"

    with JsonlWriter(path) as writer:
        writer.write(prediction)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["source"] == "camera"
    assert row["sample"] == "window_000000"
    assert row["predicted_label"] == 3
    assert row["predicted_name"] == "left_hand_wave"
    assert row["board_latency_ms"] == pytest.approx(12.5)
    assert row["encode_ms"] == pytest.approx(1.25)
    assert row["tx_work_ms"] == pytest.approx(2.5)
    assert row["sync_wait_ms"] == pytest.approx(8.75)
    assert row["work_frame_count"] == 321
    assert row["work_write_count"] == 4
    assert row["runtime_sync_count"] == 20
    assert row["event_count"] == 1234
    assert row["raw_event_count"] == 2000
    assert row["filtered_event_count"] == 1234
    assert row["filter_reduction"] == pytest.approx(0.383)
    assert row["frame_nonzero_count"] == 456
    assert row["source_window_id"] == 9
    assert row["dropped_windows"] == 0
    assert row["skipped_windows"] == 0


def test_prediction_progress_hides_timing_by_default() -> None:
    counts = np.zeros((TIMESTEPS, NUM_CLASSES), dtype=np.int32)
    counts[:, 4] = 10
    prediction = build_prediction(
        source="dataset",
        sample="sample.npz",
        expected_label=4,
        counts=counts,
        board_latency_ms=3.25,
        timing=RuntimeTiming(
            encode_ms=0.5,
            tx_work_ms=1.5,
            sync_wait_ms=2.5,
            work_frame_count=64,
        ),
        event_count=200,
    )

    message = format_prediction_progress(
        0, 2, prediction, correct=1, labeled=1
    )

    assert "expected=4:right_arm_clockwise" in message
    assert "pred=4:right_arm_clockwise" in message
    assert "board_latency_ms=3.250" in message
    assert "encode_ms=" not in message
    assert "tx_work_ms=" not in message
    assert "sync_wait_ms=" not in message
    assert "work_frames=" not in message
    assert "accuracy=1.000000" in message


def test_prediction_progress_can_show_timing() -> None:
    counts = np.zeros((TIMESTEPS, NUM_CLASSES), dtype=np.int32)
    counts[:, 4] = 10
    prediction = build_prediction(
        source="dataset",
        sample="sample.npz",
        expected_label=4,
        counts=counts,
        board_latency_ms=3.25,
        timing=RuntimeTiming(
            encode_ms=0.5,
            tx_work_ms=1.5,
            sync_wait_ms=2.5,
            work_frame_count=64,
        ),
        event_count=200,
    )

    message = format_prediction_progress(
        0, 2, prediction, correct=1, labeled=1, show_timing=True
    )

    assert "encode_ms=0.500" in message
    assert "tx_work_ms=1.500" in message
    assert "sync_wait_ms=2.500" in message
    assert "work_frames=64" in message
    assert "accuracy=1.000000" in message


def test_camera_prediction_can_hide_or_show_timing() -> None:
    counts = np.zeros((TIMESTEPS, NUM_CLASSES), dtype=np.int32)
    counts[:, 2] = 10
    prediction = build_prediction(
        source="camera",
        sample="window_000001",
        expected_label=None,
        counts=counts,
        board_latency_ms=22.0,
        timing=RuntimeTiming(
            encode_ms=0.2,
            tx_work_ms=3.4,
            sync_wait_ms=5.6,
            work_frame_count=77,
        ),
        event_count=1000,
    )

    hidden = format_camera_prediction(prediction)
    shown = format_camera_prediction(prediction, show_timing=True)

    assert "board_latency_ms=22.000" in hidden
    assert "encode_ms=" not in hidden
    assert "tx_work_ms=3.400" in shown
    assert "work_frames=77" in shown


def test_log_paths_under_logs_root_are_moved_to_log_subdir(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    args = type(
        "Args",
        (),
        {
            "command": "camera",
            "port": "/dev/ttyUSB2",
            "jsonl": logs_dir / "predictions.jsonl",
        },
    )()

    original_logs_dir = app_module.LOGS_DIR
    app_module.LOGS_DIR = logs_dir
    try:
        normalize_run_paths(args)
    finally:
        app_module.LOGS_DIR = original_logs_dir

    assert args.jsonl.parent.parent == logs_dir
    assert args.jsonl.name == "predictions.jsonl"


def test_latest_window_buffer_keeps_only_newest_window() -> None:
    buffer = LatestWindowBuffer()
    first = make_camera_window(10, event_count=100, queued_at=1.0)
    second = make_camera_window(11, event_count=200, queued_at=2.0)

    buffer.put(first)
    buffer.put(second)

    assert buffer.dropped_count() == 1
    assert buffer.get() == second
    buffer.close()
    assert buffer.get() is None


def test_camera_board_worker_uses_contiguous_sample_indices_for_gapped_windows() -> None:
    n_samples = 2
    total_syncs = n_samples * TIMESTEPS + OUTPUT_DELAY_SYNCS
    batches: list[list[int]] = []
    for sync_idx in range(total_syncs):
        output_tick = sync_idx - OUTPUT_DELAY_SYNCS
        if 0 <= output_tick < n_samples * TIMESTEPS:
            sample_idx, _sample_t = divmod(output_tick, TIMESTEPS)
            batches.append([output_frame(output_tick, sample_idx, 1), complete_frame()])
        else:
            batches.append([complete_frame()])

    buffer = FakeWindowBuffer(
        [
            make_camera_window(7, event_count=700, queued_at=1.0),
            make_camera_window(99, event_count=9900, queued_at=3.0),
        ],
        dropped=1,
    )

    results: "queue.Queue[CameraWorkerResult]" = queue.Queue()
    runtime = BoardRuntime(
        FakeSerial(rx_batches=batches, fill_on_read=True),
        timeout=0.01,
        skip_init=True,
    )
    worker = CameraBoardWorker(
        runtime=runtime,
        buffer=buffer,  # type: ignore[arg-type]
        result_queue=results,
        get_skipped_windows=lambda: 3,
    )
    with runtime:
        worker.run()

    predictions = []
    while not results.empty():
        result = results.get_nowait()
        assert result.error is None
        if result.prediction is not None:
            predictions.append(result.prediction)

    assert [p.source_window_id for p in predictions] == [7, 99]
    assert [p.predicted_label for p in predictions] == [0, 1]
    assert [p.sample for p in predictions] == ["window_000007", "window_000099"]
    assert [p.raw_event_count for p in predictions] == [1400, 19800]
    assert [p.filtered_event_count for p in predictions] == [700, 9900]
    assert [p.frame_nonzero_count for p in predictions] == [350, 4950]
    assert all(p.skipped_windows == 3 for p in predictions)
    assert all(p.dropped_windows == 1 for p in predictions)


def test_run_camera_demo_profiles_support_dry_run(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_camera_demo.sh"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PROFILE": "demo",
            "RUN_DIR": str(tmp_path / "demo_run"),
        }
    )
    demo = subprocess.run(
        ["bash", str(script)],
        cwd=script.parent,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    demo_run_dir = Path(demo.stdout.strip().splitlines()[-1])
    demo_command = (demo_run_dir / "command.txt").read_text(encoding="utf-8")
    demo_env = (demo_run_dir / "env.log").read_text(encoding="utf-8")
    assert "python -m dvsgesture_board_infer_demo.app camera" in demo_command
    assert "--full-frame-log" not in demo_command
    assert "--dump-path" not in demo_command
    assert "--baud" not in demo_command
    assert "--camera-name" not in demo_command
    assert "--resize-mode" not in demo_command
    assert "--display" in demo_command
    assert "--port /dev/ttyUSB2" in demo_command
    assert "--min-events 1000" in demo_command
    assert "--max-events" not in demo_command
    assert "--max-frame-nonzero 50000" in demo_command
    assert "--frame-nonzero-policy" not in demo_command
    assert "--warmup-windows" not in demo_command
    assert "--show-timing" not in demo_command
    assert "--event-filter" not in demo_command
    assert "--background-activity-ms 1" in demo_command
    assert "PROFILE=demo" in demo_env
    assert "MAX_FRAME_NONZERO=50000" in demo_env
    assert "DRY_RUN=1" in demo_env
    assert "BOARD_PORT=/dev/ttyUSB2" in demo_env
    assert "QT_QPA_PLATFORM=xcb" in demo_env

    env.update({"PROFILE": "debug", "RUN_DIR": str(tmp_path / "debug_run")})
    debug = subprocess.run(
        ["bash", str(script)],
        cwd=script.parent,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    debug_run_dir = Path(debug.stdout.strip().splitlines()[-1])
    debug_command = (debug_run_dir / "command.txt").read_text(encoding="utf-8")
    assert "--full-frame-log" not in debug_command
    assert "--dump-path" not in debug_command
    assert "frames.csv" not in debug_command
    assert "--verbose" in debug_command
    assert "--max-windows 3" in debug_command
    assert "--event-filter" not in debug_command
    assert "--max-events" not in debug_command
    assert "--max-frame-nonzero 0" in debug_command
    assert "--frame-nonzero-policy" not in debug_command
    assert "--show-timing" in debug_command

    env.update(
        {
            "PROFILE": "demo",
            "RUN_DIR": str(tmp_path / "serial_run"),
            "CAMERA_SERIAL": "AERS0002",
        }
    )
    serial = subprocess.run(
        ["bash", str(script)],
        cwd=script.parent,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    serial_run_dir = Path(serial.stdout.strip().splitlines()[-1])
    serial_command = (serial_run_dir / "command.txt").read_text(encoding="utf-8")
    serial_env = (serial_run_dir / "env.log").read_text(encoding="utf-8")
    assert "--camera-serial AERS0002" in serial_command
    assert "CAMERA_SERIAL=AERS0002" in serial_env


def output_frame(timestep: int, class_idx: int, value: int) -> int:
    high = 0x2 << 30
    high |= ((timestep >> 7) & 0x1) << 28
    low = ((timestep & 0x7F) << 17) | ((class_idx * 4) << 8) | value
    return (high << 32) | low


def complete_frame() -> int:
    return 0xE000000000000001


def decode_core(frame: int) -> CoordZXYOffset:
    high = (frame >> 32) & 0xFFFFFFFF
    return CoordZXYOffset(
        (high >> 22) & 0x3F,
        (high >> 16) & 0x3F,
        (high >> 10) & 0x3F,
    )


def decode_timestep(frame: int) -> int:
    high = (frame >> 32) & 0xFFFFFFFF
    low = frame & 0xFFFFFFFF
    return (((high >> 28) & 0x1) << 7) | ((low >> 17) & 0x7F)


def decode_axon(frame: int) -> int:
    low = frame & 0xFFFFFFFF
    return (low >> 8) & 0xFFFF


def decode_data(frame: int) -> int:
    return frame & 0xFF


def test_class_address_map_is_fixed() -> None:
    assert CLASS_ADDR_TO_INDEX == {i * 4: i for i in range(NUM_CLASSES)}


def test_app_module_can_show_help() -> None:
    package_parent = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "dvsgesture_board_infer_demo.app", "--help"],
        cwd=package_parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "sample" in result.stdout
    assert "dataset" in result.stdout
    assert "camera" in result.stdout
    assert "replay-txt" not in result.stdout


def test_camera_help_exposes_realtime_options() -> None:
    package_parent = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dvsgesture_board_infer_demo.app",
            "camera",
            "--help",
        ],
        cwd=package_parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "scripts/run_camera_demo.sh" in result.stdout
    assert "--window-ms" in result.stdout
    assert "--camera-serial" in result.stdout
    assert "--display" in result.stdout
    assert "--verbose" in result.stdout
    assert "--min-events" in result.stdout
    assert "--max-frame-nonzero" in result.stdout
    assert "--background-activity-ms" in result.stdout
    assert "--data-root" not in result.stdout
    assert "--split-by" not in result.stdout
    assert "--baud" not in result.stdout
    assert "--dump-path" not in result.stdout
    assert "--camera-name" not in result.stdout
    assert "--resize-mode" not in result.stdout
    assert "--event-filter" not in result.stdout
    assert "--max-events" not in result.stdout
    assert "--frame-nonzero-policy" not in result.stdout
    assert "--warmup-windows" not in result.stdout
    assert "--full-frame-log" not in result.stdout


def test_sample_and_dataset_help_expose_dataset_options() -> None:
    package_parent = Path(__file__).resolve().parents[2]
    for command in ("sample", "dataset"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dvsgesture_board_infer_demo.app",
                command,
                "--help",
            ],
            cwd=package_parent,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0
        assert "--data-root" in result.stdout
        assert "--split" in result.stdout
        assert "--split-by" not in result.stdout
        assert "--baud" not in result.stdout
        assert "--dump-path" not in result.stdout
