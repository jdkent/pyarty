"""Declaration-time validation.

A layout that cannot round-trip should fail on import, not after it has already
written a tree. Each test here asserts an error arrives at ``@bundle`` time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from pyarty import Dir, File, at, bundle, bundle_schema, is_bundle
from pyarty.errors import LayoutError, PatternError, SpecError
from pyarty.schema import FieldKind


@bundle
class Child:
    child_id: str
    data: File[dict] = at("data.json")


def test_collection_without_variable_is_rejected():
    """The old silent-data-loss bug, now a declaration-time error."""
    with pytest.raises(LayoutError, match="no {variable}"):

        @bundle
        class Root:
            kids: Dir[list[Child]] = at("kids")


def test_collection_error_suggests_a_fix():
    with pytest.raises(LayoutError) as info:

        @bundle
        class Root:
            kids: Dir[list[Child]] = at("kids")

    message = str(info.value)
    assert 'at("kids/{name}")' in message
    assert "Child" in message


def test_default_pattern_for_collection_is_also_rejected():
    # No at() at all: the default pattern is the field name, still static.
    with pytest.raises(LayoutError):

        @bundle
        class Root:
            kids: Dir[list[Child]]


def test_two_fields_at_the_same_path_are_rejected():
    with pytest.raises(LayoutError, match="both map to"):

        @bundle
        class Clash:
            first: File[str] = at("same.txt")
            second: File[str] = at("same.txt")


def test_unknown_pattern_variable_is_rejected_with_field_list():
    with pytest.raises(SpecError) as info:

        @bundle
        class Bad:
            body: File[str] = at("{nope}.txt")

    assert "{nope}" in str(info.value)
    assert "'Bad' has" in str(info.value)


def test_dir_pattern_may_reference_child_fields():
    @bundle
    class Parent:
        kids: Dir[list[Child]] = at("kids/{child_id}")

    assert bundle_schema(Parent).field("kids").pattern.variables == ("child_id",)


def test_dir_pattern_may_also_reference_owner_fields():
    @bundle
    class Parent:
        group: str
        kids: Dir[list[Child]] = at("{group}/{child_id}")

    assert is_bundle(Parent)


def test_malformed_pattern_is_rejected_immediately():
    with pytest.raises(PatternError):

        @bundle
        class Bad:
            body: File[str] = at("/absolute.txt")


def test_dir_must_reference_a_bundle():
    with pytest.raises(SpecError, match="@bundle"):

        @bundle
        class Bad:
            kids: Dir[list[int]] = at("kids/{x}")


def test_file_payload_cannot_nest_structure():
    with pytest.raises(SpecError, match="nest"):

        @bundle
        class Bad:
            weird: File[Dir[Child]] = at("weird")


def test_at_on_a_plain_value_field_is_rejected():
    with pytest.raises(SpecError, match="Only File"):

        @bundle
        class Bad:
            label: str = at("label")


def test_field_kinds_are_classified():
    @bundle
    class Mixed:
        label: str
        body: File[str] = at("{label}.txt")
        kids: Dir[list[Child]] = at("kids/{child_id}")

    schema = bundle_schema(Mixed)
    assert schema.field("label").kind is FieldKind.VALUE
    assert schema.field("body").kind is FieldKind.FILE
    assert schema.field("kids").kind is FieldKind.DIR
    assert schema.field("kids").is_collection is True
    assert schema.field("kids").child is Child


def test_default_pattern_is_field_name_plus_codec_extension():
    @bundle
    class Defaults:
        notes: File[str]
        config: File[dict]
        rows: File[list[dict]]

    schema = bundle_schema(Defaults)
    assert schema.field("notes").pattern.raw == "notes.txt"
    assert schema.field("config").pattern.raw == "config.json"
    assert schema.field("rows").pattern.raw == "rows.jsonl"


def test_explicit_extension_in_pattern_wins_over_codec_default():
    @bundle
    class Custom:
        payload: File[dict] = at("payload.cfg")

    assert bundle_schema(Custom).field("payload").pattern.raw == "payload.cfg"


def test_optional_field_is_detected():
    @bundle
    class WithOptional:
        maybe: Optional[File[dict]] = at("maybe.json", default=None)

    field = bundle_schema(WithOptional).field("maybe")
    assert field.optional is True
    assert field.has_default is True


def test_path_annotation_selects_the_copy_codec():
    @bundle
    class Copies:
        blob: File[Path] = at("blob.bin")

    assert bundle_schema(Copies).field("blob").is_copy is True


def test_bundle_accepts_an_existing_dataclass():
    from dataclasses import dataclass

    @bundle
    @dataclass
    class Already:
        body: File[str] = at("body.txt")

    assert is_bundle(Already)


def test_bundle_schema_rejects_non_bundles():
    with pytest.raises(SpecError):
        bundle_schema(int)


def test_layout_renders_a_readable_tree():
    @bundle
    class Parent:
        kids: Dir[list[Child]] = at("kids/{child_id}")
        summary: File[dict] = at("summary.json")

    layout = Parent.layout()
    assert "kids/{child_id}/" in layout
    assert "data.json" in layout
    assert "summary.json" in layout
