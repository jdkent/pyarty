from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyarty.reader import InferredBundle, infer_bundle_from_directory


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _collect_files(path: Path) -> dict[str, object]:
    collected: dict[str, object] = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(file.relative_to(path))
        suffix = file.suffix.lower()
        if suffix == ".json":
            collected[rel] = json.loads(file.read_text(encoding="utf-8"))
        elif suffix == ".jsonl":
            collected[rel] = [
                json.loads(line)
                for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            collected[rel] = file.read_text(encoding="utf-8")
    return collected


def test_infer_bundle_round_trip(tmp_path):
    root = tmp_path / "input-tree"
    (root / "experiments" / "run-001").mkdir(parents=True)
    (root / "experiments" / "run-002").mkdir(parents=True)

    (root / "experiments" / "run-001" / "payload.txt").write_text(
        "alpha", encoding="utf-8"
    )
    _write_json(
        root / "experiments" / "run-001" / "metrics.json",
        {"accuracy": 0.9, "passed": True},
    )

    (root / "experiments" / "run-002" / "payload.txt").write_text(
        "beta", encoding="utf-8"
    )
    _write_jsonl(
        root / "experiments" / "run-002" / "metrics.jsonl",
        [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.3}],
    )

    _write_json(root / "leaderboard.json", {"best": "run-001", "scores": [1, 2]})

    inferred = infer_bundle_from_directory(root, root_class_name="ExperimentBundle")

    assert isinstance(inferred, InferredBundle)
    assert inferred.root_class.__name__ == "ExperimentBundle"

    bundle_instance = inferred.instance
    experiments = bundle_instance.experiments
    run_001 = experiments.run_001
    run_002 = experiments.run_002

    assert run_001.payload == "alpha"
    assert run_001.metrics["accuracy"] == 0.9
    assert run_002.payload == "beta"
    assert run_002.metrics[0]["step"] == 1
    assert bundle_instance.leaderboard["best"] == "run-001"

    out_dir = tmp_path / "round-trip"
    bundle_instance.write(out_dir)

    assert _collect_files(root) == _collect_files(out_dir)

    schema = inferred.schema
    assert schema["$ref"] == "#/$defs/ExperimentBundle"
    defs = schema["$defs"]
    experiments_def = defs[experiments.__class__.__name__]
    assert experiments_def["properties"]["run_001"]["$ref"] == "#/$defs/Run001"
    leaderboard_schema = defs["ExperimentBundle"]["properties"]["leaderboard"]
    assert leaderboard_schema["type"] == "object"
    assert leaderboard_schema["x-pyarty"]["extension"] == "json"


def test_infer_bundle_rejects_unknown_extension(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "notes.md").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        infer_bundle_from_directory(root)
