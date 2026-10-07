# DVSGesture 2.5 板卡实时推理 Demo

这个目录是一个可迁移的独立 `uv` 工程，用于 PAICORE 2.5 板卡上的
DVSGesture 推理验证：

- `sample`：单个已生成 `.npz` 帧样本上板测试。
- `dataset`：测试集连续上板，终端实时显示预测、延迟和精度。
- `camera`：DVS 相机实时输入，cv2 显示事件画面，并叠加板卡预测结果。

板卡交互协议保持为：每个 timestep 发送 work frames 后发送 sync，并等待
complete；输出按 13 个 sync 延迟对齐。应用侧参数只调整输入、日志和观测行为。

## 环境

建议从包父目录用 `python -m` 启动，这样相对导入稳定：

```bash
cd /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture
uv sync --project dvsgesture_board_infer_demo
```

常用验证：

```bash
cd /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture/dvsgesture_board_infer_demo
uv run python -m py_compile $(find . -name '*.py' -not -path './.venv/*')
uv run --group test ruff check .
uv run --group test pytest -q
```

默认串口是 `/dev/ttyUSB2`，波特率固定为 `3000000`。如果当前 shell 激活了其他
虚拟环境，建议先 `unset VIRTUAL_ENV`，或直接使用下面的 `uv run --project` 命令。

如果机器上有多个 USB 串口，先确认板卡 relay UART 对应哪个设备：

```bash
ls -l /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null
```

确认后用 `BOARD_PORT=/dev/ttyUSBx` 覆盖脚本默认值。注意 DVS 相机的 USB 选择用
`CAMERA_SERIAL`，板卡串口选择用 `BOARD_PORT`，二者是两条独立链路。

## 单样本测试

```bash
APP_DIR=/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture/dvsgesture_board_infer_demo
APP_PARENT="$(dirname "$APP_DIR")"
RUN_DIR="$APP_DIR/logs/$(date +%Y%m%d_%H%M%S)_sample_ttyUSB2"

mkdir -p "$RUN_DIR"
cd "$APP_PARENT"
unset VIRTUAL_ENV

uv run --project "$APP_DIR" python -m dvsgesture_board_infer_demo.app sample \
  --port /dev/ttyUSB2 \
  --timeout 10 \
  --skip-init \
  --split test \
  --sample-index 0 \
  --jsonl "$RUN_DIR/predictions.jsonl" \
  2>&1 | tee "$RUN_DIR/console.log"
```

也可以用 `--sample-path` 指定某个 `.npz` 文件。

## 测试集测试

先用 `--limit 10` 做 smoke，再按需要增大样本数：

```bash
APP_DIR=/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture/dvsgesture_board_infer_demo
APP_PARENT="$(dirname "$APP_DIR")"
RUN_DIR="$APP_DIR/logs/$(date +%Y%m%d_%H%M%S)_dataset_ttyUSB2_limit10"

mkdir -p "$RUN_DIR"
cd "$APP_PARENT"
unset VIRTUAL_ENV

uv run --project "$APP_DIR" python -m dvsgesture_board_infer_demo.app dataset \
  --port /dev/ttyUSB2 \
  --timeout 10 \
  --skip-init \
  --split test \
  --limit 10 \
  --jsonl "$RUN_DIR/predictions.jsonl" \
  2>&1 | tee "$RUN_DIR/console.log"
```

终端会显示每个样本的 `expected`、`pred`、`confidence`、`board_latency_ms`
和累计 `accuracy`。需要拆分耗时细节时加 `--show-timing`。

## DVS 实时演示

推荐使用脚本，它会自动创建时间戳日志目录，记录 `env.log`、`command.txt`、
`console.log` 和 `predictions.jsonl`：

```bash
cd /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture/dvsgesture_board_infer_demo
scripts/run_camera_demo.sh
```

如果 relay 串口不是默认 `/dev/ttyUSB2`：

```bash
BOARD_PORT=/dev/ttyUSB0 scripts/run_camera_demo.sh
```

常用覆盖方式：

```bash
MIN_EVENTS=300 MAX_FRAME_NONZERO=30000 BACKGROUND_ACTIVITY_MS=2 scripts/run_camera_demo.sh
```

如果本机插入了多个 DVS，相机用序列号指定：

```bash
CAMERA_SERIAL=AERS0002 scripts/run_camera_demo.sh
```

可用序列号可用下面命令查看：

```bash
cd /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/app/DVSGesture
uv run --project dvsgesture_board_infer_demo python - <<'PY'
import dv_processing as dv

for index, device in enumerate(dv.io.camera.discover()):
    print(index, "serial=", device.serialNumber, "model=", device.cameraModel)
PY
```

短测试：

```bash
PROFILE=debug scripts/run_camera_demo.sh
```

只查看生成命令，不打开相机和串口：

```bash
DRY_RUN=1 scripts/run_camera_demo.sh
```

## 参数说明

通用参数：

- `--port` / `BOARD_PORT`：板卡串口设备，默认 `/dev/ttyUSB2`。
- `--timeout` / `TIMEOUT`：等待板卡 complete 的秒数。出现串口等待超时时增大。
- `--skip-init` / `SKIP_INIT`：跳过配置初始化。板卡已配置好时保持开启。
- `--jsonl`：预测结果日志路径。每条预测都会 flush，便于中途分析。
- `--verbose` / `VERBOSE`：打印窗口跳过、过滤、排队等调试信息。
- `--show-timing` / `SHOW_TIMING`：终端显示 `encode_ms`、`tx_work_ms`、
  `sync_wait_ms`、`work_frames`。JSONL 中始终保留这些字段。

样本和测试集参数：

- `--data-root`：DVS128Gesture 数据根目录，默认使用本应用所在目录上一级的
  `data/DVS128Gesture`。
- `--split`：数据集 split，默认 `test`。
- `--sample-index`：`sample` 模式按数据集顺序选择第几个样本。
- `--sample-path`：`sample` 模式直接指定 `.npz`。
- `--limit`：`dataset` 模式最多测试多少个样本；`0` 表示不限制。

相机参数：

- `--camera-serial` / `CAMERA_SERIAL`：指定要打开的 DVS 相机序列号。插入多个
  DVS 时设置；不设置时使用 `dv_processing` 发现的默认相机。
- `--window-ms` / `WINDOW_MS`：事件窗口长度。窗口越短，输入动作切片越快，但单窗事件
  可能不足；窗口越长，触发更稳但延迟更高。
- `--max-windows` / `MAX_WINDOWS`：最多接收多少个有效窗口。`0` 表示持续运行。
- `--display` / `DISPLAY_CAMERA`：显示 cv2 DVS 画面并叠加预测。
- `--min-events` / `MIN_EVENTS`：低事件窗口门限。难触发推理时降低；背景噪声导致频繁
  随机预测时升高。
- `--max-frame-nonzero` / `MAX_FRAME_NONZERO`：送板前最多保留多少个非零输入帧位置。
  延迟高时降低；预测不稳定或动作信息被截断时升高。`0` 表示不做 cap。
- `--background-activity-ms` / `BACKGROUND_ACTIVITY_MS`：背景活动滤波时间。背景纹理、
  衣服纹路、光照噪声较多时增大；动作事件也被滤掉、难触发时降低。

## 日志

`scripts/run_camera_demo.sh` 每次默认写入：

- `logs/<timestamp>_camera/env.log`：环境、串口、依赖版本。
- `logs/<timestamp>_camera/command.txt`：实际执行命令。
- `logs/<timestamp>_camera/console.log`：终端输出。
- `logs/<timestamp>_camera/predictions.jsonl`：每次预测的结构化记录。

普通演示不再写全量串口帧 CSV。全量 TX/RX 帧日志很大，适合单独用更小的协议排查脚本。

## 调参建议

- 几乎不出预测：降低 `MIN_EVENTS`，或降低 `BACKGROUND_ACTIVITY_MS`。
- 预测延迟过高：降低 `MAX_FRAME_NONZERO`，也可适当缩短 `WINDOW_MS`。
- 背景或衣服纹理触发太多噪声：增大 `BACKGROUND_ACTIVITY_MS`，提高 `MIN_EVENTS`，
  并让相机背景尽量简洁。
- 预测频率上来但类别不稳：适当增大 `MAX_FRAME_NONZERO`，避免动作信息被 cap 过多。
- `TimeoutError: timed out waiting for relay complete frame after sync 0`：
  DVS 已打开，但板卡 relay 没有在第一个 sync 后返回 complete。优先检查
  `BOARD_PORT` 是否是 relay UART、relay 固件是否正在运行、板卡是否已配置好，以及
  `SKIP_INIT` 是否符合当前板卡状态；最后再增大 `TIMEOUT`。
- OpenCV/Qt 提示找不到 `wayland` 插件或字体：通常只是显示后端警告，不是上板超时
  的原因。脚本默认设置 `QT_QPA_PLATFORM=xcb`；如环境不支持 xcb，可用
  `QT_QPA_PLATFORM=wayland` 或 `DISPLAY_CAMERA=0` 覆盖。
