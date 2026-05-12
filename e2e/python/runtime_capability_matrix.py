from __future__ import annotations

import json
import shutil
import subprocess


REQUIRED = ["mlir-opt", "mlir-runner"]
OPTIONAL = ["stablehlo-opt", "iree-compile", "iree-run-module"]


def tool_info(name: str) -> dict[str, str | bool]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": "", "version": ""}
    version = ""
    try:
        output = subprocess.run(
            [name, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        version = output.stdout.splitlines()[0] if output.stdout else ""
    except Exception as err:
        version = f"version probe failed: {err}"
    return {"available": True, "path": path, "version": version}


def matrix() -> dict[str, object]:
    tools = {name: tool_info(name) for name in REQUIRED + OPTIONAL}
    return {
        "required": REQUIRED,
        "optional": OPTIONAL,
        "tools": tools,
        "direct_stablehlo_runtime": bool(
            tools["stablehlo-opt"]["available"]
            and tools["iree-compile"]["available"]
            and tools["iree-run-module"]["available"]
        ),
    }


def verify() -> None:
    report = matrix()
    tools = report["tools"]
    missing_required = [name for name in REQUIRED if not tools[name]["available"]]
    if missing_required:
        raise AssertionError(f"missing required runtime tool(s): {', '.join(missing_required)}")
    print("runtime-capabilities " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    verify()
