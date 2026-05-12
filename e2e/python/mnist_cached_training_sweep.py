from __future__ import annotations

import struct
from pathlib import Path
from tempfile import TemporaryDirectory

from mnist_fixture import (
    BATCH_SIZE,
    IDX_COLUMNS,
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    IDX_ROWS,
    IMAGE_SIZE,
    split_idx_paths,
)
from mnist_train_command import (
    OptionalDatasetMissing,
    cached_metrics,
    render_metrics,
    require_checked_artifacts,
)


SAMPLE_COUNT = 16
EPOCHS = 4


def sweep_images() -> bytes:
    header = struct.pack(">IIII", IDX_IMAGE_MAGIC, SAMPLE_COUNT, IDX_ROWS, IDX_COLUMNS)
    pixel_values = []
    for sample in range(SAMPLE_COUNT):
        label = (sample * 7) % 10
        image = [0] * IMAGE_SIZE
        for pixel in range(label * 8, (label + 1) * 8):
            image[pixel] = 255
        pixel_values.extend(image)
    pixels = bytes(pixel_values)
    return header + pixels


def sweep_labels() -> bytes:
    labels = bytes((sample * 7) % 10 for sample in range(SAMPLE_COUNT))
    return struct.pack(">II", IDX_LABEL_MAGIC, SAMPLE_COUNT) + labels


def write_cache(root: Path) -> None:
    images_path, labels_path = split_idx_paths("train", root)
    images_path.write_bytes(sweep_images())
    labels_path.write_bytes(sweep_labels())


def verify_missing_cache(root: Path) -> None:
    try:
        cached_metrics(
            split="train",
            epochs=1,
            cache_dir=str(root),
            images_idx=None,
            labels_idx=None,
            max_samples=SAMPLE_COUNT,
        )
    except OptionalDatasetMissing as err:
        if "missing MNIST IDX cache file" not in str(err):
            raise AssertionError(f"missing-cache error was not specific: {err}") from err
    else:
        raise AssertionError("cached metrics unexpectedly succeeded without IDX files")


def verify() -> None:
    artifacts = require_checked_artifacts()
    with TemporaryDirectory(prefix="leanax-mnist-cached-sweep-") as tmp:
        root = Path(tmp)
        verify_missing_cache(root)
        write_cache(root)
        metrics = cached_metrics(
            split="train",
            epochs=EPOCHS,
            cache_dir=str(root),
            images_idx=None,
            labels_idx=None,
            max_samples=SAMPLE_COUNT,
        )

    if metrics.mode != "cached-train":
        raise AssertionError(f"unexpected mode {metrics.mode}")
    if metrics.samples != SAMPLE_COUNT or metrics.batches != SAMPLE_COUNT // BATCH_SIZE:
        raise AssertionError(f"unexpected cached sweep shape: {metrics}")
    if metrics.epochs != EPOCHS:
        raise AssertionError(f"unexpected epoch count: {metrics.epochs}")
    if not metrics.final_loss < metrics.first_loss:
        raise AssertionError(
            f"cached sweep loss did not decrease: {metrics.first_loss} -> {metrics.final_loss}"
        )
    if metrics.final_accuracy < metrics.first_accuracy:
        raise AssertionError(
            f"cached sweep accuracy regressed: {metrics.first_accuracy} -> {metrics.final_accuracy}"
        )

    print("mnist-cached-training-sweep " + render_metrics(metrics, artifacts).removeprefix("mnist-train "))


if __name__ == "__main__":
    verify()
