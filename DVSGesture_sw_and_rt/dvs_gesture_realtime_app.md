# DVSGesture Realtime Inference App

`dvs_gesture_realtime_app.py` is a standalone inference application for the
quantized DVS128 Gesture model defined in `infer.py`.

It keeps the same inference boundary as `infer.py`:

- model: `DVSGestureNet_NoBN`
- checkpoint: `checkpoints/dropout_0.1_sum_qat_int8.pt`
- threshold: `IFNode(v_threshold=127.0)`
- input tensor: `[T, N, 2, 128, 128]`
- decision: sum logits over time, then `argmax`

## Dataset Smoke Test

Run these commands from the PAIBox repository root. They intentionally use the
existing PAIBox uv environment so the app reuses the already-installed PyTorch
stack.

Use local pre-generated frame samples:

```bash
uv run python tests/app/DVSGesture/dvs_gesture_realtime_app.py \
  dataset \
  --source frames \
  --limit 8
```

Use local raw event samples and run the event-to-frame path:

```bash
uv run python tests/app/DVSGesture/dvs_gesture_realtime_app.py \
  dataset \
  --source events \
  --limit 8
```

Add `--jsonl tests/app/DVSGesture/debug/realtime_predictions.jsonl` if you want
machine-readable prediction records.

## Live Camera

The current `camera` subcommand uses the dv-processing Python API shown by the
iniVation samples:

- `dv.io.camera.open()`
- `camera.getNextEventBatch()`
- `dv.EventStreamSlicer().doEveryTimeInterval(...)`
- `event_store.numpy()`

The Python binding is published on PyPI as `dv-processing` and is imported as
`dv_processing`. In the PAIBox development checkout, install it into the
existing root uv environment:

```bash
uv pip install dv-processing
```

CelePixel/CeleX cameras are a different hardware family and are not discovered
through `dv.io.camera.discover()` in this app. For CelePixel hardware, add a
separate input adapter based on the CeleX/CelePixel SDK, then feed its event
stream into the existing `events_to_frames()` and `predict()` path.

Run with the first available camera:

```bash
uv run python tests/app/DVSGesture/dvs_gesture_realtime_app.py \
  camera \
  --window-ms 1000
```

Run with a named camera:

```bash
uv run python tests/app/DVSGesture/dvs_gesture_realtime_app.py \
  camera \
  --camera-name DVS128_00000000 \
  --window-ms 1000
```

For non-128x128 sensors, choose an explicit mapping:

```bash
uv run python tests/app/DVSGesture/dvs_gesture_realtime_app.py \
  camera \
  --resize-mode scale
```

## Realtime Notes

- `--window-ms` controls the classification cadence. The default is `1000 ms`
  because the training samples are full gesture clips split into 20 timesteps.
- Lower `--window-ms` reduces latency but changes the temporal distribution
  relative to the training data.
- Empty camera windows are ignored.
- `--max-windows` is useful for bounded bench runs.
- For deployment latency measurement, use `--jsonl` and inspect `latency_ms`;
  it measures model inference, not USB capture time.
