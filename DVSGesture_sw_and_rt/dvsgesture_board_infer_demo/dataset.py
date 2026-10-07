import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIMESTEPS = 20
FRAME_SHAPE = (TIMESTEPS, 2, 128, 128)
APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = APP_DIR / "data" / "DVS128Gesture"
DEFAULT_SPLIT_BY = "number"


@dataclass(frozen=True)
class FrameSample:
    path: Path
    label: int | None
    frames: np.ndarray


def class_label_map(classes: list[str]) -> dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def frame_root(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
) -> Path:
    return data_root / f"frames_number_{TIMESTEPS}_split_by_{split_by}" / split


def collect_samples(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
) -> list[tuple[Path, int | None]]:
    root = frame_root(data_root, split=split, split_by=split_by)
    if not root.is_dir():
        raise FileNotFoundError(f"test frame directory not found: {root}")

    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class directories found in {root}")
    class_to_idx = class_label_map(classes)

    samples: list[tuple[Path, int | None]] = []
    for class_name in classes:
        class_dir = root / class_name
        for current_root, _dirs, filenames in sorted(os.walk(class_dir)):
            for filename in sorted(filenames):
                if filename.endswith(".npz"):
                    samples.append(
                        (Path(current_root) / filename, class_to_idx[class_name])
                    )

    if not samples:
        raise FileNotFoundError(f"no .npz frame samples found in {root}")
    return samples


def resolve_sample_path(
    *,
    sample_path: Path | None,
    sample_index: int,
    data_root: Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
) -> tuple[Path, int | None]:
    if sample_path is None:
        samples = collect_samples(data_root, split=split, split_by=split_by)
        if sample_index < 0 or sample_index >= len(samples):
            raise IndexError(
                f"--sample-index={sample_index} is out of range for "
                f"{len(samples)} samples"
            )
        return samples[sample_index]

    requested = sample_path.expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        root = frame_root(data_root, split=split, split_by=split_by)
        candidates.extend(
            [
                Path.cwd() / requested,
                APP_DIR / requested,
                root / requested,
                data_root / requested,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate, label_from_frame_path(
                candidate, data_root=data_root, split=split, split_by=split_by
            )

    raise FileNotFoundError(f"sample path not found: {sample_path}")


def label_from_frame_path(
    path: Path,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
) -> int | None:
    root = frame_root(data_root, split=split, split_by=split_by).resolve()
    try:
        class_name = path.resolve().relative_to(root).parts[0]
    except (IndexError, ValueError):
        return None

    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    return class_label_map(classes).get(class_name)


def load_frames(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data:
            raise ValueError(f"{path} does not contain a 'frames' array")
        frames = np.asarray(data["frames"])
    return validate_frames(frames, path)


def load_sample(
    *,
    sample_path: Path | None = None,
    sample_index: int = 0,
    data_root: Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
) -> FrameSample:
    path, label = resolve_sample_path(
        sample_path=sample_path,
        sample_index=sample_index,
        data_root=data_root,
        split=split,
        split_by=split_by,
    )
    return FrameSample(path=path, label=label, frames=load_frames(path))


def iter_samples(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    split_by: str = DEFAULT_SPLIT_BY,
    limit: int = 0,
) -> list[FrameSample]:
    paths = collect_samples(data_root, split=split, split_by=split_by)
    if limit > 0:
        paths = paths[:limit]
    return [
        FrameSample(path=path, label=label, frames=load_frames(path))
        for path, label in paths
    ]


def validate_frames(frames: np.ndarray, source: Path | str) -> np.ndarray:
    if frames.shape != FRAME_SHAPE:
        raise ValueError(f"{source} has shape {frames.shape}; expected {FRAME_SHAPE}")
    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError(f"{source} has non-numeric dtype {frames.dtype}")
    if not np.all(np.isfinite(frames)):
        raise ValueError(f"{source} contains NaN or inf values")
    if np.any(frames < 0) or np.any(frames > 255):
        raise ValueError(f"{source} contains values outside uint8 range")
    rounded = np.rint(frames)
    if not np.array_equal(frames, rounded):
        raise ValueError(f"{source} contains non-integer frame values")
    return np.ascontiguousarray(rounded, dtype=np.uint8)
