"""Writing is strict and all-or-nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyarty import Dir, File, at, bundle, plan_bundle
from pyarty.errors import PayloadTypeError, WriteError


@bundle
class Doc:
    title: str
    body: File[str] = at("{title}.txt")
    meta: File[dict] = at("meta.json")


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


# ----------------------------------------------------------------------
# Strict payload validation
# ----------------------------------------------------------------------
def test_dict_field_rejects_a_string(tmp_path):
    """Previously this wrote a .json file containing raw, invalid JSON."""
    with pytest.raises(PayloadTypeError, match="expects a mapping"):
        Doc(title="t", body="b", meta="not a dict").write(tmp_path / "out")


def test_text_field_rejects_a_dict(tmp_path):
    with pytest.raises(PayloadTypeError, match="expects str"):
        Doc(title="t", body={"a": 1}, meta={}).write(tmp_path / "out")


def test_error_names_the_field_and_the_declared_type(tmp_path):
    with pytest.raises(PayloadTypeError) as info:
        Doc(title="t", body="b", meta=["wrong"]).write(tmp_path / "out")
    message = str(info.value)
    assert "Doc.meta" in message
    assert "File[dict]" in message


def test_jsonl_field_rejects_non_mapping_rows(tmp_path):
    @bundle
    class Rows:
        rows: File[list[dict]] = at("rows.jsonl")

    with pytest.raises(PayloadTypeError):
        Rows(rows=[{"ok": 1}, "not a mapping"]).write(tmp_path / "out")


def test_int_field_rejects_a_bool(tmp_path):
    """bool is an int subclass; conflating them breaks the round trip."""

    @bundle
    class Counter:
        count: File[int] = at("count.txt")

    with pytest.raises(PayloadTypeError):
        Counter(count=True).write(tmp_path / "out")


def test_nothing_is_written_when_validation_fails(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(PayloadTypeError):
        Doc(title="t", body="fine", meta="broken").write(out)
    # The valid sibling must not have been written either.
    assert not out.exists() or _files(out) == []


def test_required_none_field_is_rejected(tmp_path):
    with pytest.raises(PayloadTypeError, match="not optional"):
        Doc(title="t", body=None, meta={}).write(tmp_path / "out")


# ----------------------------------------------------------------------
# Overwrite behaviour
# ----------------------------------------------------------------------
def test_existing_files_are_refused_without_overwrite(tmp_path):
    out = tmp_path / "out"
    Doc(title="t", body="first", meta={}).write(out)

    with pytest.raises(WriteError, match="overwrite=True"):
        Doc(title="t", body="second", meta={}).write(out)


def test_overwrite_replaces_content(tmp_path):
    out = tmp_path / "out"
    Doc(title="t", body="first", meta={}).write(out)
    Doc(title="t", body="second", meta={}).write(out, overwrite=True)
    assert (out / "t.txt").read_text() == "second"


def test_writing_beside_unrelated_files_is_allowed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "unrelated.log").write_text("keep me")

    Doc(title="t", body="b", meta={}).write(out)
    assert (out / "unrelated.log").read_text() == "keep me"


def test_overwrite_merges_rather_than_syncing(tmp_path):
    """Documented behaviour: overwrite replaces declared files, removes nothing.

    A file left by a previous write of an optional field therefore survives,
    and a later read would pick it up.
    """
    from typing import Optional

    @bundle
    class Two:
        first: File[str] = at("first.txt")
        second: Optional[File[str]] = at("second.txt", default=None)

    out = tmp_path / "out"
    Two(first="1", second="2").write(out)
    Two(first="1", second=None).write(out, overwrite=True)

    assert (out / "second.txt").exists()
    assert Two.read(out).second == "2"


def test_literal_glob_metacharacters_round_trip(tmp_path):
    @bundle
    class Odd:
        payload: File[str] = at("a[1].txt")

    out = tmp_path / "out"
    Odd(payload="here").write(out)
    assert (out / "a[1].txt").read_text() == "here"
    assert Odd.read(out).payload == "here"


def test_destination_that_is_a_file_is_refused(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    with pytest.raises(WriteError, match="not a directory"):
        Doc(title="t", body="b", meta={}).write(target)


def test_write_returns_the_directory(tmp_path):
    out = Doc(title="t", body="b", meta={}).write(tmp_path / "out")
    assert out == tmp_path / "out"


# ----------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------
def test_each_element_gets_its_own_directory(tmp_path):
    """Two elements must never collapse into one directory."""

    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    out = tmp_path / "out"
    Shelf(
        docs=[
            Doc(title="a", body="AAA", meta={"n": 1}),
            Doc(title="b", body="BBB", meta={"n": 2}),
        ]
    ).write(out)

    assert _files(out) == [
        "docs/a/a.txt",
        "docs/a/meta.json",
        "docs/b/b.txt",
        "docs/b/meta.json",
    ]
    assert (out / "docs" / "a" / "a.txt").read_text() == "AAA"
    assert (out / "docs" / "b" / "b.txt").read_text() == "BBB"


def test_duplicate_variable_values_are_refused(tmp_path):
    """Two elements sharing a name would silently clobber each other."""

    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    with pytest.raises(WriteError, match="would both write"):
        Shelf(
            docs=[
                Doc(title="same", body="A", meta={}),
                Doc(title="same", body="B", meta={}),
            ]
        ).write(tmp_path / "out")


def test_wrong_element_type_is_refused(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    with pytest.raises(PayloadTypeError, match="element 0"):
        Shelf(docs=["not a bundle"]).write(tmp_path / "out")


def test_empty_collection_still_creates_no_stray_files(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    out = tmp_path / "out"
    Shelf(docs=[]).write(out)
    assert _files(out) == []


# ----------------------------------------------------------------------
# Pattern safety
# ----------------------------------------------------------------------
def test_path_separator_in_a_variable_is_refused(tmp_path):
    from pyarty.errors import PatternError

    with pytest.raises(PatternError, match="path separator"):
        Doc(title="a/../../etc/passwd", body="b", meta={}).write(tmp_path / "out")


def test_empty_variable_value_is_refused(tmp_path):
    from pyarty.errors import PatternError

    with pytest.raises(PatternError, match="empty value"):
        Doc(title="", body="b", meta={}).write(tmp_path / "out")


# ----------------------------------------------------------------------
# Copy fields
# ----------------------------------------------------------------------
def test_copy_field_copies_bytes(tmp_path):
    source = tmp_path / "src.bin"
    source.write_bytes(b"\x00\x01payload")

    @bundle
    class Release:
        blob: File[Path] = at("blob.bin")

    out = tmp_path / "out"
    Release(blob=source).write(out)
    assert (out / "blob.bin").read_bytes() == b"\x00\x01payload"


def test_copy_field_with_missing_source_is_refused(tmp_path):
    """Previously this warned and wrote the path string as the file body."""

    @bundle
    class Release:
        blob: File[Path] = at("blob.bin")

    with pytest.raises(WriteError, match="does not exist"):
        Release(blob=tmp_path / "nope.bin").write(tmp_path / "out")


def test_copy_field_pointing_at_a_directory_is_refused(tmp_path):
    @bundle
    class Release:
        blob: File[Path] = at("blob.bin")

    directory = tmp_path / "adir"
    directory.mkdir()
    with pytest.raises(WriteError, match="is a directory"):
        Release(blob=directory).write(tmp_path / "out")


def test_copy_field_keeps_the_source_extension_when_pattern_omits_one(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"m")

    @bundle
    class Release:
        blob: File[Path]

    out = tmp_path / "out"
    Release(blob=source).write(out)
    assert (out / "blob.onnx").exists()


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------
def test_plan_bundle_previews_without_writing(tmp_path):
    planned = plan_bundle(Doc(title="t", body="b", meta={}))
    assert sorted(p.relative_path for p in planned) == ["meta.json", "t.txt"]
    assert list(tmp_path.iterdir()) == []


def test_plan_bundle_rejects_non_bundles():
    with pytest.raises(WriteError, match="not a @bundle"):
        plan_bundle(object())
