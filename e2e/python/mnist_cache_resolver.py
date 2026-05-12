from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from mnist_fixture import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    load_mnist_split,
    resolve_idx_paths,
    split_idx_paths,
)
from mnist_idx_sample import sample_images, sample_labels


def write_split(root: Path, split: str) -> tuple[Path, Path]:
    images_path, labels_path = split_idx_paths(split, root)
    images_path.write_bytes(sample_images())
    labels_path.write_bytes(sample_labels())
    return images_path, labels_path


def assert_batches(split: str, root: Path) -> None:
    batches = load_mnist_split(split, cache_dir=root)
    if len(batches) != 2:
        raise AssertionError(f"expected two {split} batches, got {len(batches)}")
    first = batches[0]
    if len(first.images) != BATCH_SIZE or len(first.labels) != BATCH_SIZE:
        raise AssertionError(f"{split} batch size mismatch")
    if any(len(image) != IMAGE_SIZE for batch in batches for image in batch.images):
        raise AssertionError(f"{split} images are not flattened 28x28")
    if any(len(label) != NUM_CLASSES for batch in batches for label in batch.labels):
        raise AssertionError(f"{split} labels are not one-hot class vectors")


def verify() -> None:
    with TemporaryDirectory(prefix="leanax-mnist-cache-") as tmp:
        root = Path(tmp)
        train_images, train_labels = write_split(root, "train")
        test_images, test_labels = write_split(root, "test")

        assert resolve_idx_paths("train", cache_dir=root) == (train_images, train_labels)
        assert resolve_idx_paths("test", cache_dir=root) == (test_images, test_labels)
        assert resolve_idx_paths(images_path=train_images, labels_path=train_labels) == (train_images, train_labels)
        assert_batches("train", root)
        assert_batches("test", root)

        train_labels.unlink()
        try:
            resolve_idx_paths("train", cache_dir=root)
        except FileNotFoundError as err:
            if "train-labels-idx1-ubyte" not in str(err):
                raise AssertionError(f"missing-cache diagnostic did not name the missing label file: {err}") from err
        else:
            raise AssertionError("partial cache unexpectedly resolved")

    print("mnist-cache-resolver splits=train,test batch=2x784 labels=2x10")


if __name__ == "__main__":
    verify()
