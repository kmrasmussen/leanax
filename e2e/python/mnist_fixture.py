from __future__ import annotations

from dataclasses import dataclass


IMAGE_SIZE = 28 * 28
NUM_CLASSES = 10
BATCH_SIZE = 2


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
