"""Self-referential bundles, for arbitrarily deep trees.

``Dir[list["Self"]]`` used to fail because ``@bundle`` attached the schema only
*after* resolving fields, so a class was not yet recognised as a bundle during
its own resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyarty import Dir, File, Files, at, bundle, is_bundle
from pyarty.errors import ReadError


@bundle
class Catalog:
    """A STAC-shaped recursive catalog."""

    catalog: File[dict] = at("catalog.json")
    children: Dir[list["Catalog"]] = at("{child_id}", default_factory=list)
    # The root has no parent to name it, so it needs a fallback.
    child_id: str = "root"


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _sample() -> Catalog:
    return Catalog(
        catalog={"type": "Catalog", "id": "root"},
        children=[
            Catalog(
                catalog={"type": "Catalog", "id": "a"},
                child_id="a",
                children=[
                    Catalog(catalog={"type": "Catalog", "id": "a1"}, child_id="a1")
                ],
            ),
            Catalog(catalog={"type": "Catalog", "id": "b"}, child_id="b"),
        ],
    )


def test_self_reference_is_accepted():
    assert is_bundle(Catalog)
    assert Catalog.__pyarty_schema__.field("children").child is Catalog


def test_recursive_tree_round_trips(tmp_path):
    original = _sample()
    first = tmp_path / "first"
    original.write(first)

    assert sorted(_files(first)) == [
        "a/a1/catalog.json",
        "a/catalog.json",
        "b/catalog.json",
        "catalog.json",
    ]

    recovered = Catalog.read(first)
    assert recovered == original

    second = tmp_path / "second"
    recovered.write(second)
    assert _files(first) == _files(second)


def test_nested_levels_are_reachable(tmp_path):
    _sample().write(tmp_path / "out")
    recovered = Catalog.read(tmp_path / "out")

    assert recovered.catalog["id"] == "root"
    assert {c.child_id for c in recovered.children} == {"a", "b"}
    deep = next(c for c in recovered.children if c.child_id == "a")
    assert deep.children[0].catalog["id"] == "a1"
    assert deep.children[0].children == []


def test_leaf_has_no_children(tmp_path):
    leaf = Catalog(catalog={"type": "Catalog", "id": "only"})
    out = tmp_path / "leaf"
    leaf.write(out)
    assert sorted(_files(out)) == ["catalog.json"]
    assert Catalog.read(out).children == []


def test_deep_recursion_is_read_back(tmp_path):
    """A chain far deeper than a typical tree still reads."""
    depth = 12
    node = Catalog(catalog={"depth": depth}, child_id=f"n{depth}")
    for level in range(depth - 1, -1, -1):
        node = Catalog(
            catalog={"depth": level},
            child_id=f"n{level}",
            children=[node],
        )

    out = tmp_path / "deep"
    node.write(out)

    recovered = Catalog.read(out)
    walker = recovered
    for level in range(depth):
        assert walker.catalog["depth"] == level
        walker = walker.children[0]
    assert walker.catalog["depth"] == depth


def test_recursion_limit_is_reported(tmp_path):
    """A pathological tree reports a limit instead of hitting RecursionError."""
    from pyarty.reader import MAX_DEPTH

    out = tmp_path / "toodeep"
    current = out
    for level in range(MAX_DEPTH + 3):
        current.mkdir(parents=True, exist_ok=True)
        (current / "catalog.json").write_text("{}")
        current = current / f"n{level}"

    with pytest.raises(ReadError, match="Recursion limit"):
        Catalog.read(out)


def test_mutual_recursion_between_two_bundles(tmp_path):
    """Two bundles referring to each other, resolved by the caller's scope."""

    @bundle
    class Folder:
        name: str = "root"
        meta: File[dict] = at("folder.json", default_factory=dict)
        subfolders: Dir[list["Folder"]] = at("{name}", default_factory=list)
        blobs: Files[dict[str, bytes]] = at("blob-{key}.bin", default_factory=dict)

    tree = Folder(
        meta={"kind": "root"},
        blobs={"a": b"A"},
        subfolders=[Folder(name="child", meta={"kind": "leaf"}, blobs={"b": b"B"})],
    )
    out = tmp_path / "out"
    tree.write(out)
    assert sorted(_files(out)) == [
        "blob-a.bin",
        "child/blob-b.bin",
        "child/folder.json",
        "folder.json",
    ]
    assert Folder.read(out) == tree


def test_layout_marks_the_recursion():
    layout = Catalog.layout()
    assert "recursive" in layout
    assert "{child_id}/*" in layout
