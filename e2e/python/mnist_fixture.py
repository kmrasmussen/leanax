from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct


IMAGE_SIZE = 28 * 28
NUM_CLASSES = 10
BATCH_SIZE = 2
IDX_IMAGE_MAGIC = 2051
IDX_LABEL_MAGIC = 2049
IDX_ROWS = 28
IDX_COLUMNS = 28
MNIST_FILENAMES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


@dataclass(frozen=True)
class MnistBatch:
    images: list[list[float]]
    labels: list[list[float]]


def fixture_sample(index: int) -> tuple[list[int], int]:
    pixels = [((index * 17) + (pixel * 13)) % 256 for pixel in range(IMAGE_SIZE)]
    label = (index * 3) % NUM_CLASSES
    return pixels, label


def normalize_image(pixels: list[int]) -> list[float]:
    if len(pixels) != IMAGE_SIZE:
        raise ValueError(f"expected {IMAGE_SIZE} pixels, got {len(pixels)}")
    return [pixel / 255.0 for pixel in pixels]


def one_hot(label: int) -> list[float]:
    if label < 0 or label >= NUM_CLASSES:
        raise ValueError(f"label out of range: {label}")
    return [1.0 if index == label else 0.0 for index in range(NUM_CLASSES)]


def fixture_batches(batch_size: int = BATCH_SIZE) -> list[MnistBatch]:
    samples = [fixture_sample(index) for index in range(4)]
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batches.append(
            MnistBatch(
                images=[normalize_image(pixels) for pixels, _label in chunk],
                labels=[one_hot(label) for _pixels, label in chunk],
            )
        )
    return batches


def parse_idx_images(data: bytes) -> list[list[int]]:
    if len(data) < 16:
        raise ValueError("IDX image file is shorter than its 16-byte header")
    magic, count, rows, columns = struct.unpack(">IIII", data[:16])
    if magic != IDX_IMAGE_MAGIC:
        raise ValueError(f"expected IDX image magic {IDX_IMAGE_MAGIC}, got {magic}")
    if rows != IDX_ROWS or columns != IDX_COLUMNS:
        raise ValueError(
            f"expected {IDX_ROWS}x{IDX_COLUMNS} IDX images, got {rows}x{columns}"
        )
    expected_len = 16 + (count * IMAGE_SIZE)
    if len(data) != expected_len:
        raise ValueError(
            f"IDX image payload length mismatch: expected {expected_len} bytes, got {len(data)}"
        )

    images = []
    for index in range(count):
        start = 16 + (index * IMAGE_SIZE)
        images.append(list(data[start : start + IMAGE_SIZE]))
    return images


def parse_idx_labels(data: bytes) -> list[int]:
    if len(data) < 8:
        raise ValueError("IDX label file is shorter than its 8-byte header")
    magic, count = struct.unpack(">II", data[:8])
    if magic != IDX_LABEL_MAGIC:
        raise ValueError(f"expected IDX label magic {IDX_LABEL_MAGIC}, got {magic}")
    expected_len = 8 + count
    if len(data) != expected_len:
        raise ValueError(
            f"IDX label payload length mismatch: expected {expected_len} bytes, got {len(data)}"
        )

    labels = list(data[8:])
    for label in labels:
        if label >= NUM_CLASSES:
            raise ValueError(f"label out of range: {label}")
    return labels


def batches_from_idx(
    images_bytes: bytes, labels_bytes: bytes, batch_size: int = BATCH_SIZE
) -> list[MnistBatch]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    images = parse_idx_images(images_bytes)
    labels = parse_idx_labels(labels_bytes)
    if len(images) != len(labels):
        raise ValueError(f"image/label count mismatch: {len(images)} images, {len(labels)} labels")
    if len(images) % batch_size != 0:
        raise ValueError(
            f"IDX sample count {len(images)} is not divisible by batch size {batch_size}"
        )

    batches = []
    for start in range(0, len(images), batch_size):
        image_chunk = images[start : start + batch_size]
        label_chunk = labels[start : start + batch_size]
        batches.append(
            MnistBatch(
                images=[normalize_image(pixels) for pixels in image_chunk],
                labels=[one_hot(label) for label in label_chunk],
            )
        )
    return batches


def load_idx_files(
    images_path: str | Path, labels_path: str | Path, batch_size: int = BATCH_SIZE
) -> list[MnistBatch]:
    return batches_from_idx(
        Path(images_path).read_bytes(),
        Path(labels_path).read_bytes(),
        batch_size=batch_size,
    )


def default_mnist_cache_dir(env: dict[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    if xdg_cache := values.get("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "leanax" / "mnist"
    if home := values.get("HOME"):
        return Path(home) / ".cache" / "leanax" / "mnist"
    raise ValueError("cannot resolve MNIST cache: neither XDG_CACHE_HOME nor HOME is set")


def split_idx_paths(split: str, cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    if split not in MNIST_FILENAMES:
        raise ValueError(f"unknown MNIST split {split!r}; expected one of {sorted(MNIST_FILENAMES)}")
    root = Path(cache_dir) if cache_dir is not None else default_mnist_cache_dir()
    image_name, label_name = MNIST_FILENAMES[split]
    return root / image_name, root / label_name


def resolve_idx_paths(
    split: str = "train",
    cache_dir: str | Path | None = None,
    images_path: str | Path | None = None,
    labels_path: str | Path | None = None,
) -> tuple[Path, Path]:
    if (images_path is None) != (labels_path is None):
        raise ValueError("provide both images_path and labels_path, or neither")
    if images_path is not None and labels_path is not None:
        resolved = (Path(images_path), Path(labels_path))
    else:
        resolved = split_idx_paths(split, cache_dir)

    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing MNIST IDX cache file(s): " + ", ".join(missing))
    return resolved


def load_mnist_split(
    split: str = "train",
    cache_dir: str | Path | None = None,
    images_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[MnistBatch]:
    resolved_images, resolved_labels = resolve_idx_paths(split, cache_dir, images_path, labels_path)
    return load_idx_files(resolved_images, resolved_labels, batch_size=batch_size)


def verify() -> None:
    first = fixture_batches()[0]
    assert len(first.images) == BATCH_SIZE
    assert len(first.labels) == BATCH_SIZE
    assert all(len(image) == IMAGE_SIZE for image in first.images)
    assert all(len(label) == NUM_CLASSES for label in first.labels)
    assert all(0.0 <= pixel <= 1.0 for image in first.images for pixel in image)
    assert all(sum(label) == 1.0 for label in first.labels)
    assert fixture_batches()[0] == fixture_batches()[0]
    print(
        "mnist-fixture "
        f"batch={len(first.images)}x{len(first.images[0])} "
        f"labels={len(first.labels)}x{len(first.labels[0])}"
    )


if __name__ == "__main__":
    verify()
