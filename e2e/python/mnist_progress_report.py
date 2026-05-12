from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "e2e/manifest.txt"
DATASET_METRICS = REPO / "generated/mnist-real-dataset-metrics.json"


EXPECTED = {
    "affine_external_runtime": True,
    "artifact_composed_train_step": True,
    "cache_resolver": True,
    "compare_select_artifact": True,
    "compare_select_validation": True,
    "dense_runtime": True,
    "derived_relu_mask_artifact": True,
    "direct_mnist_external_runtime": False,
    "derived_mask_train_command_wiring": True,
    "fixture_only_default": True,
    "full_dataset_training": True,
    "idx_full_dataset_loader": True,
    "cached_dataset_training_sweep": True,
    "runtime_codegen_skeleton": True,
    "runtime_operation_inventory": True,
    "runtime_generated_dense_fixture": True,
    "runtime_generated_mnist_forward": True,
    "runtime_generated_train_step": True,
    "runtime_reduce_fixtures": True,
    "runtime_scalar_math_fixture": True,
    "runtime_shape_ops_fixtures": True,
    "runtime_tiny_train_step_fixture": True,
    "runtime_readiness_v6": True,
    "mnist_forward_runtime": True,
    "mnist_train_command": True,
    "monolithic_mnist_train_step": True,
    "mnist_cross_entropy_artifact": True,
    "mnist_forward_artifact": True,
    "mnist_parameter_tree_artifact": True,
    "mnist_train_step_derived_mask": True,
    "optional_full_dataset_smoke": True,
    "relu_dense_gradient_artifact": True,
    "runtime_capability_matrix": True,
    "softmax_dense_gradient_artifact": True,
    "structured_dataset_training_metrics": True,
    "ten_class_fixture_training": True,
}


def manifest_entries() -> set[tuple[str, str]]:
    entries = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            entries.add((fields[0], fields[1]))
    return entries


def artifact_contains(path: str, required_text: list[str]) -> bool:
    artifact = REPO / path
    if not artifact.is_file():
        return False
    text = artifact.read_text(encoding="utf-8")
    return all(required in text for required in required_text)


def dataset_metrics_ready() -> bool:
    if not DATASET_METRICS.is_file():
        return False
    data = json.loads(DATASET_METRICS.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts")
    return (
        data.get("schema") == "leanax.mnist_dataset_metrics.v1"
        and data.get("mode") == "cached-train"
        and data.get("split") == "train"
        and data.get("samples") == 16
        and data.get("batches") == 8
        and data.get("epochs") == 4
        and isinstance(data.get("first_loss"), int | float)
        and isinstance(data.get("final_loss"), int | float)
        and data["final_loss"] < data["first_loss"]
        and isinstance(data.get("first_accuracy"), int | float)
        and isinstance(data.get("final_accuracy"), int | float)
        and data["final_accuracy"] >= data["first_accuracy"]
        and isinstance(artifacts, list)
        and "generated/mnist-train-step-derived-mask.mlir" in artifacts
    )


def report() -> dict[str, bool]:
    entries = manifest_entries()
    actual = {
        "affine_external_runtime": ("runtime", "affine-runtime") in entries,
        "artifact_composed_train_step": (
            ("training-loop", "mnist-train-step-artifact") in entries
            and artifact_contains("generated/mnist-forward.mlir", ["tensor<2x784xf32>"])
            and artifact_contains("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential"])
            and artifact_contains("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
            and artifact_contains("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1"])
            and artifact_contains("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1"])
            and artifact_contains("generated/mnist-train-step-derived-mask.mlir", ["%next_w1", "%loss"])
        ),
        "cache_resolver": ("data-loader", "mnist-cache-resolver") in entries,
        "compare_select_artifact": (
            ("numeric", "compare-select") in entries
            and artifact_contains(
                "generated/compare-select.mlir",
                ["stablehlo.compare", "stablehlo.select", "tensor<2x3xi1>"],
            )
        ),
        "compare_select_validation": (
            ("validation-fail", "bad-compare-shape") in entries
            and ("validation-fail", "bad-select-predicate-shape") in entries
        ),
        "dense_runtime": (
            ("runtime", "dense-runtime") in entries
            and artifact_contains("e2e/golden/dense-runtime.mlir", ["llvm.func @main", "llvm.return"])
        ),
        "derived_relu_mask_artifact": (
            ("numeric", "relu-derived-mask") in entries
            and artifact_contains(
                "generated/relu-derived-mask.mlir",
                ["stablehlo.compare", "stablehlo.select", "%relu_mask", "tensor<2x8xi1>"],
            )
        ),
        "direct_mnist_external_runtime": False,
        "derived_mask_train_command_wiring": (
            ("training-loop", "mnist-train-command") in entries
            and artifact_contains(
                "generated/mnist-train-step-derived-mask.mlir",
                ["stablehlo.compare", "stablehlo.select", "%loss"],
            )
        ),
        "fixture_only_default": ("data-loader", "mnist-fixture") in entries,
        "full_dataset_training": (
            ("training-loop", "mnist-cached-training-sweep") in entries
            and ("data-loader", "mnist-dataset-metrics") in entries
            and dataset_metrics_ready()
        ),
        "idx_full_dataset_loader": ("data-loader", "mnist-idx-sample") in entries,
        "cached_dataset_training_sweep": (
            ("training-loop", "mnist-cached-training-sweep") in entries
            and dataset_metrics_ready()
        ),
        "runtime_operation_inventory": (
            ("data-loader", "runtime-operation-inventory") in entries
        ),
        "runtime_codegen_skeleton": (
            ("runtime", "generated-arithmetic-runtime") in entries
            and artifact_contains(
                "e2e/golden/generated-arithmetic-runtime.mlir",
                ["%checksum", "llvm.return"],
            )
        ),
        "runtime_generated_dense_fixture": (
            ("runtime", "generated-dense-runtime") in entries
            and artifact_contains(
                "e2e/golden/generated-dense-runtime.mlir",
                ["%y00", "%y01", "%y10", "%y11", "%generated_dense_acc0"],
            )
        ),
        "runtime_generated_mnist_forward": (
            ("runtime", "generated-mnist-forward-runtime") in entries
            and artifact_contains(
                "e2e/golden/generated-mnist-forward-runtime.mlir",
                ["%hidden_pre0", "llvm.fcmp", "llvm.select", "%logit0", "%generated_forward_acc0"],
            )
        ),
        "runtime_generated_train_step": (
            ("runtime", "generated-derived-mask-train-step-runtime") in entries
            and artifact_contains(
                "e2e/golden/generated-derived-mask-train-step-runtime.mlir",
                ["%mask_bool", "llvm.intr.exp", "llvm.intr.log", "%next_w1", "%checksum"],
            )
        ),
        "runtime_reduce_fixtures": (
            ("runtime", "reduce-row-runtime") in entries
            and ("runtime", "reduce-all-runtime") in entries
            and ("runtime", "reduce-keepdim-runtime") in entries
            and artifact_contains("e2e/golden/reduce-row-runtime.mlir", ["%row0", "%row1", "%reduce_row_acc0"])
            and artifact_contains("e2e/golden/reduce-all-runtime.mlir", ["%sum04", "%checksum"])
            and artifact_contains("e2e/golden/reduce-keepdim-runtime.mlir", ["%reduce_keepdim_p5", "%reduce_keepdim_acc0"])
        ),
        "runtime_scalar_math_fixture": (
            ("runtime", "softmax-loss-runtime") in entries
            and artifact_contains(
                "e2e/golden/softmax-loss-runtime.mlir",
                ["llvm.intr.exp", "llvm.intr.log", "llvm.fdiv"],
            )
        ),
        "runtime_shape_ops_fixtures": (
            ("runtime", "broadcast-shape-runtime") in entries
            and ("runtime", "reshape-shape-runtime") in entries
            and ("runtime", "transpose-shape-runtime") in entries
            and artifact_contains("e2e/golden/broadcast-shape-runtime.mlir", ["%broadcast_target_v0", "%broadcast_target_acc0"])
            and artifact_contains("e2e/golden/reshape-shape-runtime.mlir", ["%reshape_target_v5", "%reshape_target_acc0"])
            and artifact_contains("e2e/golden/transpose-shape-runtime.mlir", ["%transpose_target_v1", "%transpose_target_acc0"])
        ),
        "runtime_tiny_train_step_fixture": (
            ("runtime", "tiny-train-step-runtime") in entries
            and artifact_contains(
                "e2e/golden/tiny-train-step-runtime.mlir",
                ["llvm.select", "llvm.intr.exp", "llvm.intr.log", "%checksum"],
            )
        ),
        "mnist_forward_runtime": (
            ("runtime", "mnist-forward-runtime") in entries
            and artifact_contains("e2e/golden/mnist-forward-runtime.mlir", ["llvm.fcmp", "llvm.select"])
        ),
        "mnist_train_command": ("training-loop", "mnist-train-command") in entries,
        "monolithic_mnist_train_step": (
            ("numeric", "mnist-train-step") in entries
            and artifact_contains(
                "generated/mnist-train-step.mlir",
                ["%next_w1", "%next_b1", "%next_w2", "%next_b2", "%loss"],
            )
        ),
        "mnist_cross_entropy_artifact": (
            ("numeric", "mnist-cross-entropy") in entries
            and artifact_contains(
                "generated/mnist-cross-entropy.mlir",
                ["stablehlo.exponential", "tensor<2x10xf32>"],
            )
        ),
        "mnist_forward_artifact": (
            ("numeric", "mnist-forward") in entries
            and artifact_contains(
                "generated/mnist-forward.mlir",
                ["tensor<2x784xf32>", "stablehlo.maximum", "tensor<2x10xf32>"],
            )
        ),
        "mnist_parameter_tree_artifact": (
            ("numeric", "mnist-parameter-tree") in entries
            and artifact_contains(
                "generated/mnist-parameter-tree.mlir",
                ["%next_w1", "%next_b1", "%next_w2", "%next_b2"],
            )
        ),
        "mnist_train_step_derived_mask": (
            ("numeric", "mnist-train-step-derived-mask") in entries
            and artifact_contains(
                "generated/mnist-train-step-derived-mask.mlir",
                ["stablehlo.compare", "stablehlo.select", "%relu_mask", "%next_w1", "%loss"],
            )
        ),
        "optional_full_dataset_smoke": ("training-loop", "mnist-full-dataset-smoke") in entries,
        "relu_dense_gradient_artifact": (
            ("numeric", "grad-relu-dense") in entries
            and artifact_contains("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1", "%relu_mask"])
        ),
        "runtime_capability_matrix": ("data-loader", "runtime-capability-matrix") in entries,
        "softmax_dense_gradient_artifact": (
            ("numeric", "grad-softmax-dense") in entries
            and artifact_contains("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
        ),
        "structured_dataset_training_metrics": (
            ("data-loader", "mnist-dataset-metrics") in entries
            and dataset_metrics_ready()
        ),
        "ten_class_fixture_training": ("training-loop", "mnist-classifier-smoke") in entries,
    }
    actual["runtime_readiness_v6"] = (
        actual["runtime_codegen_skeleton"]
        and actual["runtime_shape_ops_fixtures"]
        and actual["runtime_reduce_fixtures"]
        and actual["runtime_generated_dense_fixture"]
        and actual["runtime_generated_mnist_forward"]
        and actual["runtime_generated_train_step"]
        and not actual["direct_mnist_external_runtime"]
    )
    return actual


def main() -> int:
    actual = report()
    if actual != EXPECTED:
        raise AssertionError(
            "MNIST progress report drifted:\n"
            f"expected={json.dumps(EXPECTED, sort_keys=True)}\n"
            f"actual={json.dumps(actual, sort_keys=True)}"
        )

    print("mnist-progress " + json.dumps(actual, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
