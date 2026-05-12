from __future__ import annotations

import struct

from mnist_fixture import (
    BATCH_SIZE,
    IDX_COLUMNS,
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    IDX_ROWS,
    IMAGE_SIZE,
    NUM_CLASSES,
    batches_from_idx,
)


SAMPLE_COUNT = 4


def sample_images() -> bytes:
    header = struct.pack(">IIII", IDX_IMAGE_MAGIC, SAMPLE_COUNT, IDX_ROWS, IDX_COLUMNS)
    pixels = bytes(
        ((sample * 29) + (pixel * 7)) % 256
        for sample in range(SAMPLE_COUNT)
        for pixel in range(IMAGE_SIZE)
    )
    return header + pixels


def sample_labels() -> bytes:
    return struct.pack(">II", IDX_LABEL_MAGIC, SAMPLE_COUNT) + bytes([0, 3, 6, 9])


def verify() -> None:
    batches = batches_from_idx(sample_images(), sample_labels())
    if len(batches) != SAMPLE_COUNT // BATCH_SIZE:
        raise AssertionError(f"expected two batches, got {len(batches)}")
    first = batches[0]
    if len(first.images) != BATCH_SIZE or len(first.labels) != BATCH_SIZE:
        raise AssertionError("IDX batches do not match the fixture batch size")
    if any(len(image) != IMAGE_SIZE for batch in batches for image in batch.images):
        raise AssertionError("IDX image vectors are not flattened 28x28 images")
    if any(len(label) != NUM_CLASSES for batch in batches for label in batch.labels):
        raise AssertionError("IDX labels are not one-hot vectors")
    if any(
        not 0.0 <= pixel <= 1.0
        for batch in batches
        for image in batch.images
        for pixel in image
    ):
        raise AssertionError("IDX pixels are not normalized into [0, 1]")
    expected_labels = [0, 3, 6, 9]
    actual_labels = [
        label.index(1.0) for batch in batches for label in batch.labels
    ]
    if actual_labels != expected_labels:
        raise AssertionError(f"unexpected labels: {actual_labels}")
    if batches != batches_from_idx(sample_images(), sample_labels()):
        raise AssertionError("IDX parser is not deterministic")

    print(
        "mnist-idx-sample "
        f"samples={SAMPLE_COUNT} "
        f"batch={len(first.images)}x{len(first.images[0])} "
        f"labels={len(first.labels)}x{len(first.labels[0])}"
    )


if __name__ == "__main__":
    verify()
