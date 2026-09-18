"""``Files[dict[K, T]]``: many files sharing one pattern.

``File[...]`` matches exactly one file, but "N files, one pattern" is the
commonest shape in real layouts. Each test here is drawn from a directory
standard that the single-file form could not express.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyarty import Dir, File, Files, at, bundle, bundle_schema
from pyarty.errors import PayloadTypeError, ReadError, SpecError
from pyarty.schema import FieldKind


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def assert_round_trips(instance, cls, tmp_path: Path) -> None:
    first = tmp_path / "first"
    instance.write(first)
    recovered = cls.read(first)
    assert recovered == instance

    second = tmp_path / "second"
    recovered.write(second)
    assert _files(first) == _files(second)


# ----------------------------------------------------------------------
# Real layouts
# ----------------------------------------------------------------------
def test_bagit_multiple_manifests(tmp_path):
    """RFC 8493 allows manifest-md5.txt and manifest-sha512.txt together."""

    @bundle
    class Bag:
        declaration: File[str] = at("bagit.txt")
        manifests: Files[dict[str, str]] = at("manifest-{algorithm:alnum}.txt")

    bag = Bag(
        declaration="BagIt-Version: 1.0\n",
        manifests={"md5": "abc data/x\n", "sha512": "def data/x\n"},
    )
    assert_round_trips(bag, Bag, tmp_path)
    assert sorted(_files(tmp_path / "first")) == [
        "bagit.txt",
        "manifest-md5.txt",
        "manifest-sha512.txt",
    ]


def test_huggingface_sharded_checkpoint(tmp_path):
    """model-00001-of-00002.safetensors: key variable plus a shared one."""

    @bundle
    class Repo:
        total: str
        config: File[dict] = at("config.json")
        shards: Files[dict[str, bytes]] = at(
            "model-{shard:digits}-of-{total:digits}.safetensors", key="shard"
        )

    repo = Repo(
        total="00002",
        config={"model_type": "llama"},
        shards={"00001": b"w1", "00002": b"w2"},
    )
    assert_round_trips(repo, Repo, tmp_path)

    # The non-key variable lands back in its own field.
    assert Repo.read(tmp_path / "first").total == "00002"


def test_bids_multiple_runs(tmp_path):
    """Several runs live side by side in one datatype directory."""

    @bundle
    class Func:
        sub: str
        task: str
        bold: Files[dict[int, bytes]] = at(
            "sub-{sub:alnum}_task-{task:alnum}_run-{run:digits}_bold.nii.gz",
            key="run",
        )

    func = Func(sub="01", task="rest", bold={1: b"r1", 2: b"r2", 3: b"r3"})
    assert_round_trips(func, Func, tmp_path)

    # dict[int, ...] means the key comes back as an int, not "1".
    assert list(Func.read(tmp_path / "first").bold) == [1, 2, 3]


def test_yolo_parallel_trees_pair_by_key(tmp_path):
    """images/train/{stem}.jpg and labels/train/{stem}.txt pair on the stem."""

    @bundle
    class Split:
        images: Files[dict[str, bytes]] = at("images/train/{stem}.jpg")
        labels: Files[dict[str, str]] = at("labels/train/{stem}.txt")

    split = Split(
        images={"a": b"jpgA", "b": b"jpgB"},
        labels={"a": "0 .5 .5 .1 .1\n", "b": "1 .2 .2 .3 .3\n"},
    )
    assert_round_trips(split, Split, tmp_path)

    recovered = Split.read(tmp_path / "first")
    assert set(recovered.images) == set(recovered.labels) == {"a", "b"}


# ----------------------------------------------------------------------
# Behaviour
# ----------------------------------------------------------------------
def test_files_field_is_classified_and_keyed():
    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    item = bundle_schema(Bag).field("manifests")
    assert item.kind is FieldKind.FILES
    assert item.key == "algorithm"
    assert item.is_collection is True


def test_default_pattern_and_extension_apply(tmp_path):
    @bundle
    class Keyed:
        parts: Files[dict[str, dict]] = at("part-{n}")

    assert bundle_schema(Keyed).field("parts").pattern.raw == "part-{n}.json"
    Keyed(parts={"a": {"v": 1}}).write(tmp_path / "out")
    assert (tmp_path / "out" / "part-a.json").exists()


def test_empty_mapping_round_trips(tmp_path):
    @bundle
    class Bag:
        declaration: File[str] = at("bagit.txt")
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    bag = Bag(declaration="v1", manifests={})
    assert_round_trips(bag, Bag, tmp_path)


def test_missing_files_read_as_empty_not_an_error(tmp_path):
    """A directory still being filled is a normal state, not a failure."""

    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    out = tmp_path / "out"
    out.mkdir()
    assert Bag.read(out).manifests == {}


def test_unrelated_files_are_not_claimed(tmp_path):
    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm:alnum}.txt")

    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest-md5.txt").write_text("a\n")
    (out / "tagmanifest-md5.txt").write_text("b\n")
    (out / "notes.md").write_text("c\n")

    assert Bag.read(out).manifests == {"md5": "a\n"}


def test_payload_is_validated_per_entry(tmp_path):
    @bundle
    class Bag:
        manifests: Files[dict[str, dict]] = at("manifest-{algorithm}.json")

    with pytest.raises(PayloadTypeError, match="expects a mapping"):
        Bag(manifests={"md5": "not a mapping"}).write(tmp_path / "out")


def test_non_mapping_value_is_rejected(tmp_path):
    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    with pytest.raises(PayloadTypeError, match="mapping of key"):
        Bag(manifests=["md5"]).write(tmp_path / "out")


def test_empty_key_is_rejected(tmp_path):
    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    with pytest.raises(PayloadTypeError, match="empty key"):
        Bag(manifests={"": "x"}).write(tmp_path / "out")


def test_conflicting_shared_variable_is_reported(tmp_path):
    """A non-key variable describes the field, so it must agree everywhere."""

    @bundle
    class Repo:
        total: str
        shards: Files[dict[str, bytes]] = at(
            "model-{shard:digits}-of-{total:digits}.bin", key="shard"
        )

    out = tmp_path / "out"
    out.mkdir()
    (out / "model-00001-of-00002.bin").write_bytes(b"a")
    (out / "model-00002-of-00009.bin").write_bytes(b"b")

    with pytest.raises(ReadError, match="conflicting values"):
        Repo.read(out)


def test_copy_payload_works_per_entry(tmp_path):
    source_a = tmp_path / "a.bin"
    source_a.write_bytes(b"AAA")
    source_b = tmp_path / "b.bin"
    source_b.write_bytes(b"BBB")

    @bundle
    class Blobs:
        blobs: Files[dict[str, Path]] = at("blob-{name}.bin")

    out = tmp_path / "out"
    Blobs(blobs={"a": source_a, "b": source_b}).write(out)
    assert (out / "blob-a.bin").read_bytes() == b"AAA"

    recovered = Blobs.read(out)
    assert set(recovered.blobs) == {"a", "b"}
    assert recovered.blobs["b"].read_bytes() == b"BBB"


def test_files_inside_a_nested_dir(tmp_path):
    @bundle
    class Run:
        run_id: str
        outputs: Files[dict[str, dict]] = at("out-{name}.json")

    @bundle
    class Experiment:
        runs: Dir[list[Run]] = at("runs/{run_id}")

    experiment = Experiment(
        runs=[
            Run(run_id="a", outputs={"x": {"v": 1}, "y": {"v": 2}}),
            Run(run_id="b", outputs={"x": {"v": 3}}),
        ]
    )
    assert_round_trips(experiment, Experiment, tmp_path)
    assert sorted(_files(tmp_path / "first")) == [
        "runs/a/out-x.json",
        "runs/a/out-y.json",
        "runs/b/out-x.json",
    ]


# ----------------------------------------------------------------------
# Declaration errors
# ----------------------------------------------------------------------
def test_list_form_is_rejected_with_guidance():
    """A bare list cannot round trip: write would not know the filenames."""
    with pytest.raises(SpecError, match="cannot round trip"):

        @bundle
        class Bad:
            shards: Files[list[bytes]] = at("model-{shard}.bin")


def test_static_pattern_is_rejected():
    with pytest.raises(SpecError, match="no {variable}"):

        @bundle
        class Bad:
            manifests: Files[dict[str, str]] = at("manifest.txt")


def test_ambiguous_key_must_be_named():
    with pytest.raises(SpecError, match="ambiguous"):

        @bundle
        class Bad:
            shards: Files[dict[str, bytes]] = at("model-{shard}-of-{total}.bin")


def test_named_key_must_exist_in_the_pattern():
    with pytest.raises(SpecError, match="not a variable"):

        @bundle
        class Bad:
            shards: Files[dict[str, bytes]] = at("model-{shard}.bin", key="nope")


def test_non_mapping_annotation_is_rejected():
    with pytest.raises(SpecError, match="dict\\[key, payload\\]"):

        @bundle
        class Bad:
            shards: Files[bytes] = at("model-{shard}.bin")


def test_files_payload_cannot_nest_structure():
    with pytest.raises(SpecError, match="nest"):

        @bundle
        class Child:
            data: File[dict] = at("data.json")

        @bundle
        class Bad:
            weird: Files[dict[str, Dir[Child]]] = at("{k}")


def test_layout_shows_the_key():
    @bundle
    class Bag:
        manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

    assert "many keyed by {algorithm}" in Bag.layout()


# ----------------------------------------------------------------------
# Interaction with the other pattern features
# ----------------------------------------------------------------------
def test_files_with_an_optional_group(tmp_path):
    """A Files pattern may also carry optional entities."""
    from typing import Optional

    @bundle
    class Func:
        sub: str
        bold: Files[dict[int, bytes]] = at(
            "sub-{sub:alnum}[_ses-{ses:alnum}]_run-{run:digits}_bold.nii",
            key="run",
            default_factory=dict,
        )
        ses: Optional[str] = None

    for ses, expected in [
        (None, ["sub-01_run-1_bold.nii", "sub-01_run-2_bold.nii"]),
        ("pre", ["sub-01_ses-pre_run-1_bold.nii", "sub-01_ses-pre_run-2_bold.nii"]),
    ]:
        out = tmp_path / f"ses-{ses}"
        original = Func(sub="01", ses=ses, bold={1: b"x", 2: b"y"})
        original.write(out)
        assert sorted(p.name for p in out.iterdir()) == expected
        assert Func.read(out) == original


def test_files_key_must_satisfy_the_charset(tmp_path):
    @bundle
    class Parts:
        parts: Files[dict[str, str]] = at("p-{k:digits}.txt")

    with pytest.raises(Exception, match="digits"):
        Parts(parts={"abc": "x"}).write(tmp_path / "out")


def test_keys_colliding_after_coercion_are_reported(tmp_path):
    """run-1 and run-01 both coerce to int 1; that must not silently drop one."""

    @bundle
    class Runs:
        runs: Files[dict[int, bytes]] = at("run-{n:digits}.bin")

    out = tmp_path / "out"
    out.mkdir()
    (out / "run-1.bin").write_bytes(b"a")
    (out / "run-01.bin").write_bytes(b"b")

    with pytest.raises(ReadError, match="two files with key"):
        Runs.read(out)


def test_strict_mode_counts_files_as_claimed(tmp_path):
    @bundle
    class Parts:
        parts: Files[dict[str, str]] = at("p-{k}.txt")

    out = tmp_path / "out"
    Parts(parts={"a": "1", "b": "2"}).write(out)
    # Everything on disk is claimed by the Files field.
    assert Parts.read(out, strict=True).parts == {"a": "1", "b": "2"}

    (out / "stray.txt").write_text("z")
    with pytest.raises(ReadError, match="no field"):
        Parts.read(out, strict=True)


def test_files_inside_a_recursive_bundle(tmp_path):
    @bundle
    class Node:
        blobs: Files[dict[str, bytes]] = at("b-{k}.bin", default_factory=dict)
        kids: Dir[list["Node"]] = at("{nid}", default_factory=list)
        nid: str = "root"

    tree = Node(blobs={"x": b"X"}, kids=[Node(blobs={"y": b"Y"}, nid="k1")])
    out = tmp_path / "out"
    tree.write(out)
    assert sorted(_files(out)) == ["b-x.bin", "k1/b-y.bin"]
    assert Node.read(out) == tree
