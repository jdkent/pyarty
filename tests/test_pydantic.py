"""Pydantic models as payloads.

A model is validated in both directions: a write takes only an instance of the
declared model, and a read parses the file into that model, so a file that
breaks the contract fails at the read instead of travelling on as a dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

pydantic = pytest.importorskip("pydantic")

from pydantic import BaseModel, Field  # noqa: E402

from pyarty import Dir, File, Files, at, bundle, bundle_schema  # noqa: E402
from pyarty.codecs import codec_for_annotation, is_pydantic_model  # noqa: E402
from pyarty.errors import PayloadTypeError, ReadError  # noqa: E402


class Point(BaseModel):
    x: float
    y: float
    z: float
    label: Optional[str] = None


class Analysis(BaseModel):
    key: str
    points: list[Point] = []


class Aliased(BaseModel):
    table_id: str = Field(alias="tableId")


@bundle
class Paper:
    pmid: str
    parse: File[Analysis] = at("{pmid}/parse.json")
    points: File[list[Point]] = at("{pmid}/points.jsonl")


@bundle
class Corpus:
    papers: Dir[list[Paper]] = at("papers/{pmid}")
    by_table: Files[dict[str, Analysis]] = at("tables/{table}.json")


@bundle
class Defaults:
    analysis: File[Analysis]
    rows: File[list[Point]]


@bundle
class WithAlias:
    record: File[Aliased] = at("record.json")


@bundle
class OptionalModel:
    extra: Optional[File[Analysis]] = at("extra.json", default=None)


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _paper(pmid: str) -> Paper:
    pts = [Point(x=1, y=2, z=3, label="amygdala"), Point(x=-4, y=5.5, z=6)]
    return Paper(pmid=pmid, parse=Analysis(key="tbl1#ab12", points=pts), points=pts)


def test_round_trip_returns_models(tmp_path: Path) -> None:
    corpus = Corpus(
        papers=[_paper("111"), _paper("222")],
        by_table={"tbl1": Analysis(key="tbl1#ab12"), "tbl2": Analysis(key="tbl2#cd34")},
    )
    corpus.write(tmp_path / "a")
    back = Corpus.read(tmp_path / "a")
    assert back == corpus
    assert isinstance(back.papers[0].parse, Analysis)
    assert isinstance(back.papers[0].points[0], Point)
    assert isinstance(back.by_table["tbl2"], Analysis)

    back.write(tmp_path / "b")
    assert _files(tmp_path / "a") == _files(tmp_path / "b")


def test_formats_on_disk(tmp_path: Path) -> None:
    _paper("111").write(tmp_path)
    parse = (tmp_path / "111/parse.json").read_text()
    assert parse.startswith("{\n") and parse.endswith("}\n")
    lines = (tmp_path / "111/points.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert Point.model_validate_json(lines[1]) == Point(x=-4, y=5.5, z=6)


def test_default_extensions(tmp_path: Path) -> None:
    Defaults(analysis=Analysis(key="k"), rows=[Point(x=0, y=0, z=0)]).write(tmp_path)
    assert sorted(_files(tmp_path)) == ["analysis.json", "rows.jsonl"]


def test_write_refuses_a_dict(tmp_path: Path) -> None:
    with pytest.raises(PayloadTypeError, match="expects a Analysis"):
        Paper(pmid="1", parse={"key": "k"}, points=[]).write(tmp_path)
    assert not any(tmp_path.iterdir())


def test_write_refuses_a_mixed_list(tmp_path: Path) -> None:
    with pytest.raises(PayloadTypeError, match="a list of Point"):
        Paper(
            pmid="1", parse=Analysis(key="k"), points=[Point(x=0, y=0, z=0), {"x": 1}]
        ).write(tmp_path)


def test_read_rejects_a_file_that_breaks_the_model(tmp_path: Path) -> None:
    _paper("111").write(tmp_path)
    (tmp_path / "111/parse.json").write_text('{"key": "k", "points": [{"x": 1}]}')
    with pytest.raises(ReadError, match="pydantic:Analysis"):
        Paper.read(tmp_path)


def test_read_rejects_a_bad_line(tmp_path: Path) -> None:
    _paper("111").write(tmp_path)
    with (tmp_path / "111/points.jsonl").open("a") as handle:
        handle.write('{"x": "north"}\n')
    with pytest.raises(ReadError, match="pydantic-jsonl:Point"):
        Paper.read(tmp_path)


def test_aliases_read_back(tmp_path: Path) -> None:
    original = WithAlias(record=Aliased(tableId="tbl3"))
    original.write(tmp_path)
    assert '"tableId"' in (tmp_path / "record.json").read_text()
    assert WithAlias.read(tmp_path) == original


def test_optional_model(tmp_path: Path) -> None:
    OptionalModel().write(tmp_path / "none")
    assert OptionalModel.read(tmp_path / "none").extra is None
    OptionalModel(extra=Analysis(key="k")).write(tmp_path / "some")
    assert OptionalModel.read(tmp_path / "some").extra == Analysis(key="k")


def test_codec_selection() -> None:
    assert is_pydantic_model(Analysis)
    assert not is_pydantic_model(dict)
    assert not is_pydantic_model(Analysis(key="k"))
    assert codec_for_annotation(Analysis).extension == "json"
    assert codec_for_annotation(list[Point]).extension == "jsonl"
    assert codec_for_annotation(Optional[Analysis]) is codec_for_annotation(Analysis)


def test_schema_describes_the_model() -> None:
    names = {
        item.name: item.codec.name
        for item in bundle_schema(Paper).fields
        if item.codec is not None
    }
    assert names["parse"] == "pydantic:Analysis"
    assert names["points"] == "pydantic-jsonl:Point"
