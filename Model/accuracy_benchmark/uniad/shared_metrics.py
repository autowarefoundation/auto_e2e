# SPDX-License-Identifier: Apache-2.0
"""Execute only hash-verified original AutoE2E ADE/FDE reduction statements."""

import ast
import hashlib
import numpy as np

SOURCE_SHA256 = "02568b4cd213171c14dfb59825811fded7db1449a60b02f14cfbd0a082d6f722"
SOURCE_COMMIT = "40a75cf8a26b60f6b094125e3924fe0bdb427b02"


def load_metric_code(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned shared evaluator source checksum differs")
    tree = ast.parse(source)
    (function,) = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_reactive_multitask"
    ]
    statements = []
    for name in ("displacement", "errors"):
        (statement,) = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == name
        ]
        statements.append(statement)
    (loop,) = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Call)
        and isinstance(n.iter.func, ast.Attribute)
        and isinstance(n.iter.func.value, ast.Name)
        and n.iter.func.value.id == "horizon_steps"
        and n.iter.func.attr == "items"
    ]
    statements.append(loop)
    # Only the three inspected numerical statements are compiled, not imports,
    # data loading, model execution, or other code from the source module.
    return compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec")


def evaluate(code, predicted, target, valid=None):
    import torch

    predicted, target = (
        np.asarray(predicted, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
    )
    if (
        predicted.shape != target.shape
        or predicted.ndim != 3
        or predicted.shape[1:] != (30, 2)
    ):
        raise ValueError("Expected matching [N,30,2] trajectories, never extrapolated")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite trajectories")
    if valid is None:
        valid = np.ones(predicted.shape[:2], dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != predicted.shape[:2]:
        raise ValueError("Mask shape mismatch")
    horizons = {"1s": 10, "2s": 20, "3s": 30}
    scope = dict(
        torch=torch,
        predicted_xy=torch.from_numpy(predicted),
        target_xy=torch.from_numpy(target),
        valid=torch.from_numpy(valid),
        horizon_steps=horizons,
    )
    for metric in ("ade", "fde"):
        scope[f"trajectory_{metric}_sum"] = {h: 0.0 for h in horizons}
        scope[f"trajectory_{metric}_count"] = {h: 0 for h in horizons}
    exec(code, scope)
    values, counts = {}, {}
    for h in horizons:
        values[h[:-1]], counts[h[:-1]] = {}, {}
        for metric in ("ade", "fde"):
            count = scope[f"trajectory_{metric}_count"][h]
            values[h[:-1]][metric + "_m"] = (
                scope[f"trajectory_{metric}_sum"][h] / count if count else None
            )
            counts[h[:-1]][metric] = count
    return values, counts


def self_test(code):
    predicted = np.ones((2, 30, 2), dtype=np.float32)
    predicted[..., 0] = 3.0
    predicted[..., 1] = 4.0
    target = np.zeros_like(predicted)
    valid = np.ones((2, 30), dtype=bool)
    valid[1, 20:] = False
    values, counts = evaluate(code, predicted, target, valid)
    assert all(v == {"ade_m": 5.0, "fde_m": 5.0} for v in values.values())
    assert counts["3"] == {"ade": 2, "fde": 1}
    assert "5" not in values
