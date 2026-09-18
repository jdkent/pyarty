"""Scaffolding emits source code, and that source must actually work.

The strongest assertion available is to exec the generated module and use the
resulting class to read the very tree it was generated from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyarty import scaffold_from_directory
from pyarty.errors import ReadError


def _corpus(root: Path) -> Path:
    for name in ("alpha", "beta"):
        directory = root / "reports" / name
        directory.mkdir(parents=True)
        (directory / "body.txt").write_text(f"text for {name}")
        (directory / "metrics.json").write_text(json.dumps({"score": 1}))
    (root / "summary.json").write_text(json.dumps({"count": 2}))
    return root


def _load(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, "<scaffold>", "exec"), namespace)
    return namespace


def test_generated_source_is_valid_python(tmp_path):
    source = scaffold_from_directory(_corpus(tmp_path / "corpus"))
    compile(source, "<scaffold>", "exec")


def test_generated_classes_read_the_original_tree(tmp_path):
    root = _corpus(tmp_path / "corpus")
    source = scaffold_from_directory(root, root_class_name="Corpus")
    namespace = _load(source)

    corpus = namespace["Corpus"].read(root)
    assert len(corpus.reports) == 2
    assert {r.name for r in corpus.reports} == {"alpha", "beta"}
    assert corpus.summary == {"count": 2}


def test_generated_classes_round_trip_the_tree(tmp_path):
    root = _corpus(tmp_path / "corpus")
    namespace = _load(scaffold_from_directory(root, root_class_name="Corpus"))

    corpus = namespace["Corpus"].read(root)
    clone = tmp_path / "clone"
    corpus.write(clone)

    def snapshot(base: Path) -> dict:
        """Compare by *value*, since JSON formatting is pyarty's to choose.

        The hand-written fixture uses compact JSON while pyarty writes it
        indented, so byte-identity only holds between two pyarty-written
        trees (asserted in test_roundtrip).
        """
        out = {}
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            key = path.relative_to(base).as_posix()
            if path.suffix == ".json":
                out[key] = json.loads(path.read_text())
            else:
                out[key] = path.read_bytes()
        return out

    assert snapshot(root) == snapshot(clone)

    # A second pyarty-written copy *is* byte-identical.
    again = tmp_path / "again"
    namespace["Corpus"].read(clone).write(again)
    assert {
        p.relative_to(clone).as_posix(): p.read_bytes()
        for p in sorted(clone.rglob("*"))
        if p.is_file()
    } == {
        p.relative_to(again).as_posix(): p.read_bytes()
        for p in sorted(again.rglob("*"))
        if p.is_file()
    }


def test_scaffold_recovers_a_bundle_pyarty_itself_wrote(tmp_path):
    """The strongest property: scaffold inverts write.

    A tree written from a ``{name}.txt`` pattern must scaffold back to that
    same pattern, not to one class per directory.
    """
    from pyarty import Dir, File, at, bundle

    @bundle
    class Report:
        name: str
        body: File[str] = at("{name}.txt")
        metrics: File[dict] = at("metrics.json")

    @bundle
    class Corpus:
        reports: Dir[list[Report]] = at("reports/{name}")
        summary: File[dict] = at("summary.json")

    root = tmp_path / "written"
    Corpus(
        reports=[
            Report(name="alpha", body="A", metrics={"score": 1}),
            Report(name="beta", body="B", metrics={"score": 2}),
        ],
        summary={"count": 2},
    ).write(root)

    source = scaffold_from_directory(root, root_class_name="Corpus")
    assert 'at("{name}.txt")' in source
    assert 'at("reports/{name}")' in source
    assert source.count("@bundle") == 2

    regenerated = _load(source)["Corpus"].read(root)
    assert {r.name for r in regenerated.reports} == {"alpha", "beta"}
    assert {r.body for r in regenerated.reports} == {"A", "B"}


def test_files_with_colliding_field_names_are_disambiguated(tmp_path):
    root = tmp_path / "collide"
    root.mkdir()
    (root / "my-data.json").write_text("{}")
    (root / "my_data.txt").write_text("t")

    source = scaffold_from_directory(root)
    compile(source, "<scaffold>", "exec")
    instance = _load(source)["Root"].read(root)
    # Both files must be present under distinct field names.
    assert len([v for v in vars(instance).values()]) == 2


def test_repeated_siblings_collapse_into_one_class(tmp_path):
    """Two same-shaped directories must not produce two classes."""
    source = scaffold_from_directory(_corpus(tmp_path / "corpus"))
    assert source.count("@bundle") == 2  # root + the repeated report
    assert 'Dir[list[' in source
    assert 'at("reports/{name}")' in source


def test_repeated_child_gains_a_name_field(tmp_path):
    source = scaffold_from_directory(_corpus(tmp_path / "corpus"))
    assert "name: str" in source


def test_annotations_are_inferred_from_extensions(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "notes.txt").write_text("text")
    (root / "config.json").write_text('{"a": 1}')
    (root / "rows.jsonl").write_text('{"r": 1}\n{"r": 2}\n')
    (root / "listing.json").write_text('[{"x": 1}]')
    (root / "blob.bin").write_bytes(b"\x00")

    source = scaffold_from_directory(root)
    assert "File[str]" in source
    assert "File[dict]" in source
    assert "File[list[dict]]" in source
    assert "File[bytes]" in source


def test_distinct_shapes_get_distinct_classes(tmp_path):
    root = tmp_path / "varied"
    (root / "one").mkdir(parents=True)
    (root / "one" / "a.txt").write_text("a")
    (root / "two").mkdir(parents=True)
    (root / "two" / "b.json").write_text("{}")

    source = scaffold_from_directory(root)
    namespace = _load(source)
    # Root plus one class per distinct shape.
    assert source.count("@bundle") == 3
    assert namespace["Root"].read(root) is not None


def test_hidden_entries_are_skipped(tmp_path):
    root = tmp_path / "hidden"
    root.mkdir()
    (root / "keep.txt").write_text("keep")
    (root / ".secret").write_text("skip")
    (root / ".git").mkdir()

    source = scaffold_from_directory(root)
    assert "keep" in source
    assert "secret" not in source


def test_empty_directory_produces_a_valid_class(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    source = scaffold_from_directory(root)
    compile(source, "<scaffold>", "exec")
    assert "pass" in source


def test_names_are_sanitized_into_identifiers(tmp_path):
    root = tmp_path / "odd"
    root.mkdir()
    (root / "my-file.name.txt").write_text("x")
    (root / "123numeric.json").write_text("{}")

    source = scaffold_from_directory(root)
    compile(source, "<scaffold>", "exec")


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(ReadError):
        scaffold_from_directory(tmp_path / "absent")


def test_root_class_name_is_honoured(tmp_path):
    root = tmp_path / "corpus"
    _corpus(root)
    source = scaffold_from_directory(root, root_class_name="MyBundle")
    assert "class MyBundle:" in source
    assert "MyBundle.read" in source
