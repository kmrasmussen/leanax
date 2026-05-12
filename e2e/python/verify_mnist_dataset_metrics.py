from __future__ import annotations

import json
from pathlib import Path


METRICS_ARTIFACT = Path("generated/mnist-real-dataset-metrics.json")
REQUIRED_SCHEMA = "leanax.mnist_dataset_metrics.v1"
DERIVED_MASK_ARTIFACT = "generated/mnist-train-step-derived-mask.mlir"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_int(data: dict, key: str) -> int:
    value = data.get(key)
    require(isinstance(value, int), f"{key} must be an integer")
    return value


def require_number(data: dict, key: str) -> float:
    value = data.get(key)
    require(isinstance(value, int | float), f"{key} must be numeric")
    return float(value)


def verify() -> None:
    require(METRICS_ARTIFACT.is_file(), f"missing metrics artifact {METRICS_ARTIFACT}")
    data = json.loads(METRICS_ARTIFACT.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "metrics artifact root must be an object")
    require(data.get("schema") == REQUIRED_SCHEMA, "unexpected metrics schema")
    require(data.get("mode") == "cached-train", "metrics mode must be cached-train")
    require(data.get("split") == "train", "metrics split must be train")

    epochs = require_int(data, "epochs")
    samples = require_int(data, "samples")
    batches = require_int(data, "batches")
    require(epochs == 4, f"unexpected epoch count {epochs}")
    require(samples == 16, f"unexpected sample count {samples}")
    require(batches == 8, f"unexpected batch count {batches}")

    first_loss = require_number(data, "first_loss")
    final_loss = require_number(data, "final_loss")
    first_accuracy = require_number(data, "first_accuracy")
    final_accuracy = require_number(data, "final_accuracy")
    require(first_loss > 0.0, "first_loss must be positive")
    require(0.0 < final_loss < first_loss, "final_loss must be positive and lower than first_loss")
    require(0.0 <= first_accuracy <= 1.0, "first_accuracy must be in [0, 1]")
    require(0.0 <= final_accuracy <= 1.0, "final_accuracy must be in [0, 1]")
    require(final_accuracy >= first_accuracy, "final_accuracy must not regress")

    artifacts = data.get("artifacts")
    require(isinstance(artifacts, list), "artifacts must be a list")
    require(all(isinstance(path, str) for path in artifacts), "artifacts must contain paths")
    require(DERIVED_MASK_ARTIFACT in artifacts, "metrics must mention the derived-mask train-step artifact")
    for artifact in artifacts:
        require(Path(artifact).is_file(), f"missing referenced artifact {artifact}")

    print(
        "mnist-dataset-metrics "
        f"path={METRICS_ARTIFACT} "
        f"samples={samples} "
        f"batches={batches} "
        f"first_loss={first_loss:.6f} "
        f"final_loss={final_loss:.6f} "
        f"artifacts={len(artifacts)}"
    )


if __name__ == "__main__":
    verify()
