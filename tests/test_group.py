"""``Group[list[T]]``: one record assembled across parallel trees.

``Dir`` repeats a bundle over subdirectories, each child rooted in its own
directory. Some layouts instead spread one logical record across sibling trees
that agree on a filename — a YOLO image and its label, a BIDS image and its
sidecar. No single directory holds a record, so nesting cannot express it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from pyarty import Dir, File, Files, Group, at, bundle, bundle_schema
from pyarty.errors import PayloadTypeError, ReadError, SpecError
from pyarty.schema import FieldKind


@bundle
class Sample:
    stem: str
    image: File[bytes] = at("images/train/{stem}.jpg")
    label: File[str] = at("labels/train/{stem}.txt")


@bundle
class Dataset:
    names: File[dict] = at("data.json")
    samples: Group[list[Sample]] = at(key="stem")


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _sample_dataset() -> Dataset:
    return Dataset(
        names={"names": ["cat", "dog"]},
        samples=[
            Sample(stem="img001", image=b"JPG1", label="0 .5 .5 .1 .1\n"),
            Sample(stem="img002", image=b"JPG2", label="1 .2 .2 .3 .3\n"),
        ],
    )


# ----------------------------------------------------------------------
# The motivating layouts
# ----------------------------------------------------------------------
def test_yolo_paired_record_round_trips(tmp_path):
    original = _sample_dataset()
    first = tmp_path / "first"
    original.write(first)

    assert sorted(_files(first)) == [
        "data.json",
        "images/train/img001.jpg",
        "images/train/img002.jpg",
        "labels/train/img001.txt",
        "labels/train/img002.txt",
    ]

    recovered = Dataset.read(first)
    assert recovered == original

    second = tmp_path / "second"
    recovered.write(second)
    assert _files(first) == _files(second)


def test_each_record_pairs_its_own_files(tmp_path):
    """The point of the feature: one object holding both halves."""
    _sample_dataset().write(tmp_path / "out")
    recovered = Dataset.read(tmp_path / "out")

    by_stem = {s.stem: s for s in recovered.samples}
    assert by_stem["img001"].image == b"JPG1"
    assert by_stem["img001"].label == "0 .5 .5 .1 .1\n"
    assert by_stem["img002"].image == b"JPG2"


def test_bids_image_and_sidecar_as_one_record(tmp_path):
    @bundle
    class Bold:
        sub: str
        image: File[bytes] = at(
            "sub-{sub:alnum}/func/sub-{sub:alnum}_task-rest_bold.nii.gz"
        )
        sidecar: File[dict] = at(
            "sub-{sub:alnum}/func/sub-{sub:alnum}_task-rest_bold.json"
        )

    @bundle
    class BidsSet:
        description: File[dict] = at("dataset_description.json")
        scans: Group[list[Bold]] = at(key="sub")

    original = BidsSet(
        description={"Name": "ds", "BIDSVersion": "1.9.0"},
        scans=[
            Bold(sub="01", image=b"n1", sidecar={"RepetitionTime": 2.0}),
            Bold(sub="02", image=b"n2", sidecar={"RepetitionTime": 2.0}),
        ],
    )
    out = tmp_path / "out"
    original.write(out)
    assert (out / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz").exists()
    assert BidsSet.read(out) == original


# ----------------------------------------------------------------------
# Discovery semantics
# ----------------------------------------------------------------------
def test_records_are_ordered_by_key(tmp_path):
    out = tmp_path / "out"
    Dataset(
        names={},
        samples=[
            Sample(stem="c", image=b"3", label="c\n"),
            Sample(stem="a", image=b"1", label="a\n"),
            Sample(stem="b", image=b"2", label="b\n"),
        ],
    ).write(out)
    assert [s.stem for s in Dataset.read(out).samples] == ["a", "b", "c"]


def test_an_incomplete_record_is_surfaced_not_dropped(tmp_path):
    """Keys are the union across patterns, so a half-record is visible."""
    out = tmp_path / "out"
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)
    (out / "data.json").write_text("{}")
    (out / "images" / "train" / "a.jpg").write_bytes(b"A")
    (out / "labels" / "train" / "a.txt").write_text("ok\n")
    (out / "images" / "train" / "orphan.jpg").write_bytes(b"O")

    with pytest.raises(ReadError, match="labels/train"):
        Dataset.read(out)


def test_optional_member_tolerates_an_incomplete_record(tmp_path):
    @bundle
    class LooseSample:
        stem: str
        image: File[bytes] = at("images/train/{stem}.jpg")
        label: Optional[File[str]] = at("labels/train/{stem}.txt", default=None)

    @bundle
    class LooseDataset:
        samples: Group[list[LooseSample]] = at(key="stem")

    out = tmp_path / "out"
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "train" / "a.jpg").write_bytes(b"A")
    (out / "images" / "train" / "b.jpg").write_bytes(b"B")
    (out / "labels" / "train").mkdir(parents=True)
    (out / "labels" / "train" / "a.txt").write_text("ok\n")

    found = {s.stem: s.label for s in LooseDataset.read(out).samples}
    assert found == {"a": "ok\n", "b": None}


def test_a_record_does_not_absorb_its_siblings_files(tmp_path):
    """The key must constrain each record's patterns, or all of them match."""
    _sample_dataset().write(tmp_path / "out")
    recovered = Dataset.read(tmp_path / "out")
    assert len(recovered.samples) == 2
    assert recovered.samples[0].image != recovered.samples[1].image


def test_empty_group_round_trips(tmp_path):
    original = Dataset(names={}, samples=[])
    out = tmp_path / "out"
    original.write(out)
    assert Dataset.read(out) == original


def test_group_reads_empty_when_trees_are_absent(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "data.json").write_text("{}")
    assert Dataset.read(out).samples == []


def test_unrelated_files_are_ignored(tmp_path):
    out = tmp_path / "out"
    _sample_dataset().write(out)
    (out / "images" / "train" / "notes.md").write_text("skip")

    assert len(Dataset.read(out).samples) == 2


def test_strict_mode_counts_grouped_files_as_claimed(tmp_path):
    out = tmp_path / "out"
    _sample_dataset().write(out)
    assert len(Dataset.read(out, strict=True).samples) == 2

    (out / "stray.txt").write_text("z")
    with pytest.raises(ReadError, match="no field"):
        Dataset.read(out, strict=True)


# ----------------------------------------------------------------------
# Prefix and composition
# ----------------------------------------------------------------------
def test_static_prefix_roots_the_records(tmp_path):
    @bundle
    class Pair:
        k: str
        a: File[str] = at("A/{k}.txt")
        b: File[str] = at("B/{k}.txt")

    @bundle
    class Prefixed:
        pairs: Group[list[Pair]] = at("train", key="k")

    original = Prefixed(pairs=[Pair(k="x", a="1", b="2")])
    out = tmp_path / "out"
    original.write(out)
    assert sorted(_files(out)) == ["train/A/x.txt", "train/B/x.txt"]
    assert Prefixed.read(out) == original


def test_group_nested_inside_a_dir(tmp_path):
    """The composition a dynamic prefix would have meant."""

    @bundle
    class Pair:
        k: str
        a: File[str] = at("A/{k}.txt")
        b: File[str] = at("B/{k}.txt")

    @bundle
    class Split:
        split: str
        pairs: Group[list[Pair]] = at(key="k")

    @bundle
    class Dataset2:
        splits: Dir[list[Split]] = at("{split}")

    original = Dataset2(
        splits=[
            Split(split="train", pairs=[Pair(k="x", a="1", b="2")]),
            Split(split="val", pairs=[Pair(k="y", a="3", b="4")]),
        ]
    )
    out = tmp_path / "out"
    original.write(out)
    assert sorted(_files(out)) == [
        "train/A/x.txt",
        "train/B/x.txt",
        "val/A/y.txt",
        "val/B/y.txt",
    ]
    assert Dataset2.read(out) == original


def test_group_records_may_hold_files_fields(tmp_path):
    @bundle
    class Record:
        rid: str
        head: File[str] = at("heads/{rid}.txt")
        parts: Files[dict[str, str]] = at("parts/{rid}-{n}.txt", key="n")

    @bundle
    class Holder:
        records: Group[list[Record]] = at(key="rid")

    original = Holder(
        records=[
            Record(rid="r1", head="H1", parts={"a": "1", "b": "2"}),
            Record(rid="r2", head="H2", parts={"a": "3"}),
        ]
    )
    out = tmp_path / "out"
    original.write(out)
    assert sorted(_files(out)) == [
        "heads/r1.txt",
        "heads/r2.txt",
        "parts/r1-a.txt",
        "parts/r1-b.txt",
        "parts/r2-a.txt",
    ]
    assert Holder.read(out) == original


# ----------------------------------------------------------------------
# Write-side validation
# ----------------------------------------------------------------------
def test_duplicate_key_is_refused(tmp_path):
    with pytest.raises(PayloadTypeError, match="two records"):
        Dataset(
            names={},
            samples=[
                Sample(stem="same", image=b"1", label="a\n"),
                Sample(stem="same", image=b"2", label="b\n"),
            ],
        ).write(tmp_path / "out")


def test_none_key_is_refused(tmp_path):
    with pytest.raises(PayloadTypeError, match="cannot be blank"):
        Dataset(
            names={}, samples=[Sample(stem=None, image=b"1", label="a\n")]
        ).write(tmp_path / "out")


def test_wrong_element_type_is_refused(tmp_path):
    with pytest.raises(PayloadTypeError, match="element 0"):
        Dataset(names={}, samples=["not a bundle"]).write(tmp_path / "out")


def test_mapping_value_is_refused(tmp_path):
    with pytest.raises(PayloadTypeError, match="sequence of"):
        Dataset(names={}, samples={"a": 1}).write(tmp_path / "out")


# ----------------------------------------------------------------------
# Declaration errors
# ----------------------------------------------------------------------
def test_key_is_required():
    with pytest.raises(SpecError, match="needs the variable"):

        @bundle
        class Bad:
            samples: Group[list[Sample]] = at()


def test_key_must_be_a_field_on_the_record():
    with pytest.raises(SpecError, match="not a field on"):

        @bundle
        class Bad:
            samples: Group[list[Sample]] = at(key="nope")


def test_key_must_appear_in_a_record_pattern():
    with pytest.raises(SpecError, match="cannot be discovered"):

        @bundle
        class Unkeyed:
            tag: str
            data: File[dict] = at("data.json")

        @bundle
        class Bad:
            items: Group[list[Unkeyed]] = at(key="tag")


def test_group_must_wrap_a_list_of_bundles():
    with pytest.raises(SpecError, match="list of a @bundle"):

        @bundle
        class Bad:
            samples: Group[Sample] = at(key="stem")


def test_dynamic_prefix_is_rejected_with_guidance():
    with pytest.raises(SpecError) as info:

        @bundle
        class Bad:
            split: str
            samples: Group[list[Sample]] = at("{split}", key="stem")

    message = str(info.value)
    assert "must be a fixed path" in message
    assert "Dir[list[Child]]" in message


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------
def test_group_field_is_classified():
    item = bundle_schema(Dataset).field("samples")
    assert item.kind is FieldKind.GROUP
    assert item.key == "stem"
    assert item.child is Sample
    assert item.pattern is None


def test_layout_describes_the_grouping():
    layout = Dataset.layout()
    assert "grouped by {stem}" in layout
    assert "images/train/{stem}.jpg" in layout
