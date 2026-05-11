from __future__ import annotations

import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"stablehlo-text verification failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def verify(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require(text.endswith("\n"), "file must end with a newline")
    require(text.count("{") == text.count("}"), "unbalanced braces")
    require(re.search(r"^module @leanax_[A-Za-z0-9_]+ \{$", text, re.M), "missing module header")
    require("func.func @main(" in text, "missing main function")
    require(re.search(r"return %[A-Za-z0-9_]+ : tensor<", text), "missing typed return")

    ops = re.findall(r"stablehlo\.([A-Za-z0-9_]+)", text)
    require(ops, "module has no stablehlo operations")
    allowed = {"add", "multiply", "dot_general"}
    unknown = sorted(set(ops) - allowed)
    require(not unknown, f"unknown operations: {', '.join(unknown)}")

    defined = set(re.findall(r"^\s+%([A-Za-z0-9_]+) = stablehlo\.", text, re.M))
    params = set()
    signature = re.search(r"func\.func @main\((.*?)\) ->", text)
    require(signature is not None, "missing function signature")
    for name in re.findall(r"%([A-Za-z0-9_]+): tensor<", signature.group(1)):
        params.add(name)

    for ref in re.findall(r"%([A-Za-z0-9_]+)", text):
        require(ref in defined or ref in params, f"undefined SSA reference %{ref}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_stablehlo_text.py PATH", file=sys.stderr)
        return 2
    verify(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
