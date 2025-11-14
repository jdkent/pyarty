from __future__ import annotations

import json
from typing import List

from pyarty import Dir, File, bundle, twig


@bundle
class WriterReport:
    name: str
    body: File[str] = twig(name="{name}")
    details: File[dict]


@bundle
class WriterReportSet:
    label: str
    reports: Dir[List[WriterReport]] = twig(name=("{name}", "field"))
    summary: File[dict]


def test_render_simple_bundle(tmp_path):
    report_a = WriterReport(name="alpha", body="hello", details={"score": 10})
    report_b = WriterReport(name="beta", body="world", details={"score": 20})

    bundle_instance = WriterReportSet(
        label="demo",
        reports=[report_a, report_b],
        summary={"reports": 2},
    )

    out_dir = tmp_path / "output"
    bundle_instance.write(out_dir)

    body_a = (out_dir / "alpha" / "alpha.txt").read_text()
    meta_a = json.loads((out_dir / "alpha" / "details.json").read_text())
    summary = json.loads((out_dir / "summary.json").read_text())

    assert body_a == "hello"
    assert meta_a["score"] == 10
    assert summary["reports"] == 2


def _custom_node_name(node, index=None):
    suffix = f"-{index}" if index is not None else ""
    return f"{node.slug}{suffix}"


@bundle
class WriterNode:
    slug: str
    payload: File[str]


@bundle
class WriterTree:
    label: str
    nodes: Dir[List[WriterNode]] = twig(name=(_custom_node_name, "field"))


def test_callable_directory_names(tmp_path):
    bundle_instance = WriterTree(
        label="tree",
        nodes=[WriterNode(slug="n1", payload="x"), WriterNode(slug="n2", payload="y")],
    )

    out_dir = tmp_path / "tree"
    bundle_instance.write(out_dir)
    assert (out_dir / "n1-0" / "payload.txt").read_text() == "x"


@bundle
class ListDirChild:
    payload: File[str]


@bundle
class ListDirRoot:
    processed: Dir[List[ListDirChild]]


def test_dir_list_defaults_to_field_name(tmp_path):
    bundle_instance = ListDirRoot(processed=[ListDirChild(payload="hello")])
    out_dir = tmp_path / "article"
    bundle_instance.write(out_dir)
    assert (out_dir / "processed" / "payload.txt").read_text() == "hello"


@bundle
class PrefixedDirChild:
    slug: str
    payload: File[str]


@bundle
class PrefixedDirRoot:
    nodes: Dir[List[PrefixedDirChild]] = twig(
        prefix="processed", name=("{slug}", "field")
    )


def test_dir_prefix_applied(tmp_path):
    bundle_instance = PrefixedDirRoot(
        nodes=[PrefixedDirChild(slug="n1", payload="x")]
    )
    out_dir = tmp_path / "prefixed"
    bundle_instance.write(out_dir)
    assert (out_dir / "processed" / "n1" / "payload.txt").read_text() == "x"


@bundle
class PrefixedFileRoot:
    report: File[str] = twig(prefix="reports", name="summary")


def test_file_prefix_applied(tmp_path):
    bundle_instance = PrefixedFileRoot(report="hello world")
    out_dir = tmp_path / "file-prefix"
    bundle_instance.write(out_dir)
    assert (out_dir / "reports" / "summary.txt").read_text() == "hello world"
