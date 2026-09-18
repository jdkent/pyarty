"""The central contract: ``read`` is the inverse of ``write``.

Every layout worth supporting gets exercised here as a full cycle, because a
round trip is the only test that can catch a lossy name or a one-way codec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from pyarty import Dir, File, at, bundle


# ----------------------------------------------------------------------
# Fixtures: bundles covering each field kind
# ----------------------------------------------------------------------
@bundle
class Report:
    name: str
    body: File[str] = at("{name}.txt")
    metrics: File[dict] = at("metrics.json")


@bundle
class Corpus:
    reports: Dir[list[Report]] = at("reports/{name}")
    summary: File[dict] = at("summary.json")


@bundle
class Payloads:
    text: File[str]
    blob: File[bytes]
    count: File[int]
    ratio: File[float]
    flag: File[bool]
    mapping: File[dict]
    rows: File[list[dict]]
    lines: File[list[str]]


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def assert_round_trips(instance, cls, tmp_path: Path) -> None:
    """Assert value identity and byte identity across a full cycle."""
    first = tmp_path / "first"
    instance.write(first)

    recovered = cls.read(first)
    assert recovered == instance, "read(write(x)) must equal x"

    second = tmp_path / "second"
    recovered.write(second)
    assert _files(first) == _files(second), "write(read(x)) must be byte-identical"


# ----------------------------------------------------------------------
# Round trips
# ----------------------------------------------------------------------
def test_nested_collection_round_trips(tmp_path):
    corpus = Corpus(
        reports=[
            Report(name="alpha", body="Hello alpha", metrics={"score": 10}),
            Report(name="beta", body="Hello beta", metrics={"score": 20}),
        ],
        summary={"count": 2},
    )
    assert_round_trips(corpus, Corpus, tmp_path)


def test_name_baked_into_filename_is_recovered(tmp_path):
    """The defect that made the old design unusable: a name in a path was lost."""
    corpus = Corpus(
        reports=[Report(name="alpha", body="x", metrics={})],
        summary={},
    )
    corpus.write(tmp_path / "out")

    recovered = Corpus.read(tmp_path / "out")
    assert recovered.reports[0].name == "alpha"
    assert recovered.reports[0].body == "x"


def test_expected_paths_on_disk(tmp_path):
    Corpus(
        reports=[Report(name="alpha", body="A", metrics={"n": 1})],
        summary={"count": 1},
    ).write(tmp_path / "out")

    assert sorted(_files(tmp_path / "out")) == [
        "reports/alpha/alpha.txt",
        "reports/alpha/metrics.json",
        "summary.json",
    ]


def test_every_codec_round_trips(tmp_path):
    payloads = Payloads(
        text="hello\nworld",
        blob=b"\x00\x01\xff binary",
        count=42,
        ratio=0.5,
        flag=True,
        mapping={"nested": {"a": [1, 2, 3]}, "unicode": "héllo"},
        rows=[{"step": 1}, {"step": 2}],
        lines=["first", "second", "third"],
    )
    assert_round_trips(payloads, Payloads, tmp_path)


def test_default_patterns_come_from_field_name_and_type(tmp_path):
    Payloads(
        text="t",
        blob=b"b",
        count=1,
        ratio=1.0,
        flag=False,
        mapping={},
        rows=[],
        lines=[],
    ).write(tmp_path / "out")

    assert sorted(_files(tmp_path / "out")) == [
        "blob.bin",
        "count.txt",
        "flag.json",
        "lines.txt",
        "mapping.json",
        "ratio.txt",
        "rows.jsonl",
        "text.txt",
    ]


def test_empty_collection_round_trips(tmp_path):
    corpus = Corpus(reports=[], summary={"count": 0})
    assert_round_trips(corpus, Corpus, tmp_path)


def test_unicode_and_multiline_payloads_survive(tmp_path):
    corpus = Corpus(
        reports=[Report(name="ünïcode", body="line1\nline2\n", metrics={"é": "ü"})],
        summary={},
    )
    assert_round_trips(corpus, Corpus, tmp_path)


def test_deeply_nested_bundles_round_trip(tmp_path):
    @bundle
    class Leaf:
        leaf_id: str
        data: File[dict] = at("data.json")

    @bundle
    class Middle:
        mid: str
        leaves: Dir[list[Leaf]] = at("leaves/{leaf_id}")

    @bundle
    class Top:
        middles: Dir[list[Middle]] = at("{mid}")

    top = Top(
        middles=[
            Middle(mid="m1", leaves=[Leaf(leaf_id="l1", data={"v": 1})]),
            Middle(
                mid="m2",
                leaves=[
                    Leaf(leaf_id="l2", data={"v": 2}),
                    Leaf(leaf_id="l3", data={"v": 3}),
                ],
            ),
        ]
    )
    assert_round_trips(top, Top, tmp_path)


def test_single_non_collection_dir_round_trips(tmp_path):
    @bundle
    class Config:
        settings: File[dict] = at("settings.json")

    @bundle
    class Project:
        config: Dir[Config] = at("conf")
        readme: File[str] = at("README.md")

    project = Project(config=Config(settings={"debug": True}), readme="# hi")
    assert_round_trips(project, Project, tmp_path)


def test_multi_variable_pattern_round_trips(tmp_path):
    @bundle
    class Scan:
        subject: str
        session: str
        image: File[bytes] = at("sub-{subject}_ses-{session}.nii")

    @bundle
    class Study:
        scans: Dir[list[Scan]] = at("{subject}_{session}")

    study = Study(
        scans=[
            Scan(subject="01", session="a", image=b"\x01"),
            Scan(subject="02", session="b", image=b"\x02"),
        ]
    )
    assert_round_trips(study, Study, tmp_path)


def test_typed_value_fields_are_coerced_back(tmp_path):
    """A captured path fragment is text; the declared type must be restored."""

    @bundle
    class Run:
        index: int
        label: str
        log: File[str] = at("run-{index}-{label}.txt")

    @bundle
    class Sweep:
        runs: Dir[list[Run]] = at("runs/{index}")

    sweep = Sweep(runs=[Run(index=1, label="a", log="x"), Run(index=2, label="b", log="y")])
    assert_round_trips(sweep, Sweep, tmp_path)

    recovered = Sweep.read((tmp_path / "first"))
    assert [r.index for r in recovered.runs] == [1, 2]
    assert all(isinstance(r.index, int) for r in recovered.runs)


def test_optional_file_absent_round_trips(tmp_path):
    @bundle
    class WithOptional:
        required: File[str] = at("required.txt")
        extra: Optional[File[dict]] = at("extra.json", default=None)

    present = WithOptional(required="r", extra={"a": 1})
    assert_round_trips(present, WithOptional, tmp_path)

    absent = WithOptional(required="r", extra=None)
    assert_round_trips(absent, WithOptional, tmp_path / "absent")


def test_copy_field_round_trips(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")

    @bundle
    class Release:
        model: File[Path] = at("model.bin")

    out = tmp_path / "out"
    Release(model=source).write(out)
    assert (out / "model.bin").read_bytes() == b"weights"

    recovered = Release.read(out)
    # A copy field reads back as the path it found, so it can be re-copied.
    assert isinstance(recovered.model, Path)
    assert recovered.model.read_bytes() == b"weights"

    again = tmp_path / "again"
    recovered.write(again)
    assert (again / "model.bin").read_bytes() == b"weights"
