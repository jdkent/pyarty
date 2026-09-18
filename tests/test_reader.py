"""Reading is lenient by default, strict on request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from pyarty import Dir, File, at, bundle
from pyarty.errors import MissingFileError, ReadError


@bundle
class Doc:
    title: str
    body: File[str] = at("{title}.txt")
    meta: File[dict] = at("meta.json")


def _make_doc(root: Path, title: str = "t") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{title}.txt").write_text("body text")
    (root / "meta.json").write_text(json.dumps({"n": 1}))
    return root


# ----------------------------------------------------------------------
# Basic reading
# ----------------------------------------------------------------------
def test_reads_a_hand_made_tree(tmp_path):
    """Trees pyarty did not write must still read, or the contract is useless."""
    doc = Doc.read(_make_doc(tmp_path / "doc"))
    assert doc.title == "t"
    assert doc.body == "body text"
    assert doc.meta == {"n": 1}


def test_value_field_is_recovered_from_the_filename(tmp_path):
    doc = Doc.read(_make_doc(tmp_path / "doc", title="my-report"))
    assert doc.title == "my-report"


def test_codec_follows_the_annotation_not_the_extension(tmp_path):
    """A .cfg file declared File[dict] is still parsed as JSON."""

    @bundle
    class Odd:
        payload: File[dict] = at("payload.cfg")

    root = tmp_path / "odd"
    root.mkdir()
    (root / "payload.cfg").write_text('{"parsed": true}')

    assert Odd.read(root).payload == {"parsed": True}


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(ReadError, match="does not exist"):
        Doc.read(tmp_path / "absent")


def test_path_that_is_a_file_is_reported(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    with pytest.raises(ReadError, match="not a directory"):
        Doc.read(target)


# ----------------------------------------------------------------------
# Leniency
# ----------------------------------------------------------------------
def test_unclaimed_files_are_ignored_by_default(tmp_path):
    root = _make_doc(tmp_path / "doc")
    (root / "notes.md").write_text("ignore me")
    (root / "stray.bin").write_bytes(b"\x00")

    doc = Doc.read(root)
    assert doc.body == "body text"


def test_strict_mode_rejects_unclaimed_files(tmp_path):
    root = _make_doc(tmp_path / "doc")
    (root / "notes.md").write_text("ignore me")

    with pytest.raises(ReadError, match="no field"):
        Doc.read(root, strict=True)


def test_strict_mode_accepts_an_exact_tree(tmp_path):
    root = _make_doc(tmp_path / "doc")
    assert Doc.read(root, strict=True).body == "body text"


def test_optional_field_becomes_none_when_absent(tmp_path):
    @bundle
    class WithOptional:
        required: File[str] = at("required.txt")
        extra: Optional[File[dict]] = at("extra.json", default=None)

    root = tmp_path / "opt"
    root.mkdir()
    (root / "required.txt").write_text("here")

    result = WithOptional.read(root)
    assert result.required == "here"
    assert result.extra is None


def test_missing_required_file_is_reported(tmp_path):
    root = tmp_path / "doc"
    root.mkdir()
    (root / "meta.json").write_text("{}")

    with pytest.raises(MissingFileError, match="body"):
        Doc.read(root)


def test_undecodable_payload_names_the_field_and_file(tmp_path):
    root = tmp_path / "doc"
    root.mkdir()
    (root / "t.txt").write_text("fine")
    (root / "meta.json").write_text("this is not json")

    with pytest.raises(ReadError) as info:
        Doc.read(root)
    message = str(info.value)
    assert "Doc.meta" in message
    assert "meta.json" in message


def test_ambiguous_file_match_is_reported(tmp_path):
    root = tmp_path / "doc"
    root.mkdir()
    (root / "one.txt").write_text("a")
    (root / "two.txt").write_text("b")
    (root / "meta.json").write_text("{}")

    with pytest.raises(ReadError, match="matched 2 files"):
        Doc.read(root)


# ----------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------
def test_collection_reads_every_matching_directory(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    root = tmp_path / "shelf"
    _make_doc(root / "docs" / "a", title="a")
    _make_doc(root / "docs" / "b", title="b")

    shelf = Shelf.read(root)
    assert [d.title for d in shelf.docs] == ["a", "b"]


def test_collection_is_empty_when_no_directories_match(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    root = tmp_path / "shelf"
    root.mkdir()
    assert Shelf.read(root).docs == []


def test_collection_ignores_directories_that_do_not_match(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/run-{title}")

    root = tmp_path / "shelf"
    _make_doc(root / "docs" / "run-a", title="a")
    (root / "docs" / "scratch").mkdir()

    shelf = Shelf.read(root)
    assert [d.title for d in shelf.docs] == ["a"]


def test_collection_results_are_ordered_deterministically(tmp_path):
    @bundle
    class Shelf:
        docs: Dir[list[Doc]] = at("docs/{title}")

    root = tmp_path / "shelf"
    for title in ["c", "a", "b"]:
        _make_doc(root / "docs" / title, title=title)

    assert [d.title for d in Shelf.read(root).docs] == ["a", "b", "c"]


def test_single_dir_matching_many_is_reported(tmp_path):
    @bundle
    class Wrapper:
        only: Dir[Doc] = at("{title}")

    root = tmp_path / "w"
    _make_doc(root / "a", title="a")
    _make_doc(root / "b", title="b")

    with pytest.raises(ReadError, match="Dir\\[list"):
        Wrapper.read(root)


def test_unrecoverable_value_field_explains_how_to_fix_it(tmp_path):
    """A value no pattern captures cannot be reconstructed; say so clearly."""

    @bundle
    class Orphan:
        label: str
        payload: File[dict] = at("payload.json")

    root = tmp_path / "orphan"
    root.mkdir()
    (root / "payload.json").write_text("{}")

    with pytest.raises(ReadError) as info:
        Orphan.read(root)
    message = str(info.value)
    assert "label" in message
    assert 'at("{label}.txt")' in message


def test_value_field_with_a_default_is_left_alone(tmp_path):
    @bundle
    class Defaulted:
        label: str = "fallback"
        payload: File[dict] = at("payload.json", default_factory=dict)

    root = tmp_path / "d"
    root.mkdir()
    (root / "payload.json").write_text("{}")

    assert Defaulted.read(root).label == "fallback"


def test_read_rejects_non_bundles(tmp_path):
    from pyarty import read_bundle

    with pytest.raises(ReadError, match="not a @bundle"):
        read_bundle(int, tmp_path)
