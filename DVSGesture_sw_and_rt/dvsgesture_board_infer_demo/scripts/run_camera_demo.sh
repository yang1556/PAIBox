#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PARENT="$(cd "$APP_DIR/.." && pwd)"
DEFAULT_RUN_DIR="$APP_DIR/logs/$(date +%Y%m%d_%H%M%S)_camera"
RUN_DIR="${RUN_DIR:-$DEFAULT_RUN_DIR}"

PROFILE="${PROFILE:-demo}"
BOARD_PORT="${BOARD_PORT:-/dev/ttyUSB2}"
TIMEOUT="${TIMEOUT:-10}"
WINDOW_MS="${WINDOW_MS:-1000}"
CAMERA_SERIAL="${CAMERA_SERIAL:-}"
DISPLAY_CAMERA="${DISPLAY_CAMERA:-1}"
SKIP_INIT="${SKIP_INIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

# OpenCV's Qt wheel can print noisy Wayland/font warnings on some desktops.
# For the demo launcher, prefer xcb unless the caller explicitly chooses another
# backend.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

case "$PROFILE" in
  demo)
    MAX_WINDOWS="${MAX_WINDOWS:-0}"
    MIN_EVENTS="${MIN_EVENTS:-1000}"
    MAX_FRAME_NONZERO="${MAX_FRAME_NONZERO:-50000}"
    BACKGROUND_ACTIVITY_MS="${BACKGROUND_ACTIVITY_MS:-1}"
    VERBOSE="${VERBOSE:-0}"
    SHOW_TIMING="${SHOW_TIMING:-0}"
    ;;
  debug)
    MAX_WINDOWS="${MAX_WINDOWS:-3}"
    MIN_EVENTS="${MIN_EVENTS:-1}"
    MAX_FRAME_NONZERO="${MAX_FRAME_NONZERO:-0}"
    BACKGROUND_ACTIVITY_MS="${BACKGROUND_ACTIVITY_MS:-1}"
    VERBOSE="${VERBOSE:-1}"
    SHOW_TIMING="${SHOW_TIMING:-1}"
    ;;
  *)
    echo "unsupported PROFILE=$PROFILE; expected demo or debug" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"
cd "$APP_PARENT"

# Force uv to use this app-local project even if the caller has another venv active.
unset VIRTUAL_ENV

{
  date
  pwd
  id
  groups
  echo "PROFILE=$PROFILE"
  echo "BOARD_PORT=$BOARD_PORT"
  echo "TIMEOUT=$TIMEOUT"
  echo "WINDOW_MS=$WINDOW_MS"
  echo "CAMERA_SERIAL=$CAMERA_SERIAL"
  echo "MAX_WINDOWS=$MAX_WINDOWS"
  echo "MIN_EVENTS=$MIN_EVENTS"
  echo "MAX_FRAME_NONZERO=$MAX_FRAME_NONZERO"
  echo "BACKGROUND_ACTIVITY_MS=$BACKGROUND_ACTIVITY_MS"
  echo "DISPLAY_CAMERA=$DISPLAY_CAMERA"
  echo "SKIP_INIT=$SKIP_INIT"
  echo "VERBOSE=$VERBOSE"
  echo "SHOW_TIMING=$SHOW_TIMING"
  echo "DRY_RUN=$DRY_RUN"
  echo "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
  echo
  echo "[board port]"
  ls -l "$BOARD_PORT" 2>/dev/null || true
  echo
  echo "[serial devices]"
  ls -l /dev/ttyUSB* 2>/dev/null || true
  echo
  echo "[serial by-id]"
  find /dev/serial/by-id -maxdepth 1 -type l -print -exec readlink -f {} \; 2>/dev/null || true
  echo
  echo "[dvs usb]"
  lsusb -d 152a:841a 2>/dev/null || true
  echo
  uv run --project "$APP_DIR" python --version
  uv run --project "$APP_DIR" python - <<'PY'
import serial
import cv2
import dv_processing
import paicorelib
print("pyserial", serial.VERSION)
print("opencv", cv2.__version__)
print("dv_processing", getattr(dv_processing, "__version__", "unknown"))
print("paicorelib", paicorelib.__version__)
PY
} > "$RUN_DIR/env.log" 2>&1

cmd=(
  uv run --project "$APP_DIR" python -m dvsgesture_board_infer_demo.app camera
  --port "$BOARD_PORT"
  --timeout "$TIMEOUT"
  --window-ms "$WINDOW_MS"
  --max-windows "$MAX_WINDOWS"
  --min-events "$MIN_EVENTS"
  --max-frame-nonzero "$MAX_FRAME_NONZERO"
  --background-activity-ms "$BACKGROUND_ACTIVITY_MS"
  --jsonl "$RUN_DIR/predictions.jsonl"
)

if [[ "$SKIP_INIT" != "0" ]]; then
  cmd+=(--skip-init)
fi
if [[ "$DISPLAY_CAMERA" != "0" ]]; then
  cmd+=(--display)
fi
if [[ "$VERBOSE" != "0" ]]; then
  cmd+=(--verbose)
fi
if [[ "$SHOW_TIMING" != "0" ]]; then
  cmd+=(--show-timing)
fi
if [[ -n "$CAMERA_SERIAL" ]]; then
  cmd+=(--camera-serial "$CAMERA_SERIAL")
fi

printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

if [[ "$DRY_RUN" != "0" ]]; then
  echo "[dry-run] $RUN_DIR"
  echo "$RUN_DIR"
  exit 0
fi

echo "[run] $RUN_DIR"
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/console.log"

echo "$RUN_DIR"
