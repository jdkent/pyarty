from typing import Dict, List, Union

from pyarty import Dir, FieldKind, File, HintKind, bundle, twig


def test_bundle():

    def _generate_file_name(owner, index=None) -> str:
        suffix = index if index is not None else "00"
        return f"{owner.slug}_{suffix}"

    @bundle
    class FileInDir:
        text: File[str]

    @bundle
    class EmbeddedFile:
        number: File[float]

    @bundle
    class EmbeddedDir:
        embedded_dir: Dir[EmbeddedFile]

    @bundle
    class MyBundle:
        slug: str
        my_dir: Dir[FileInDir]
        my_file: File[int] = twig(extension=".bin", name=_generate_file_name)
        another_dir: Dir[EmbeddedDir]
        slug_file: File[str] = twig(name="{slug}")
        static_file: File[str] = twig(name="static")

    bundle_instance = MyBundle(
        slug="bundle-001",
        my_dir=FileInDir("file_contents"),
        my_file=42,
        another_dir=EmbeddedDir(embedded_dir=EmbeddedFile(number=3.14)),
        slug_file="slug file",
        static_file="static payload",
        __bundle_metadata__={"my_file": {"indexer": "fortytwo"}},
    )

    assert (
        bundle_instance.__bundle_instance_metadata__["my_file"]["indexer"] == "fortytwo"
    )
    bundle_definition = MyBundle.__bundle_definition__
    assert [field.name for field in bundle_definition.fields] == [
        "slug",
        "my_dir",
        "my_file",
        "another_dir",
        "slug_file",
        "static_file",
    ]
    my_file_field = next(
        field for field in bundle_definition.fields if field.name == "my_file"
    )
    my_file_metadata = my_file_field.metadata[0]
    assert my_file_metadata.layer is File
    callable_hint = my_file_metadata.data["name"]
    assert callable_hint.kind is HintKind.CALLABLE
    assert callable_hint.value is _generate_file_name

    slug_file_field = next(
        field for field in bundle_definition.fields if field.name == "slug_file"
    )
    slug_file_metadata = slug_file_field.metadata[0]
    name_hint = slug_file_metadata.data["name"]
    assert name_hint.kind is HintKind.TEMPLATE
    assert name_hint.value == "{slug}"
    assert name_hint.source == "self"

    static_file_field = next(
        field for field in bundle_definition.fields if field.name == "static_file"
    )
    static_file_metadata = static_file_field.metadata[0]
    static_name_hint = static_file_metadata.data["name"]
    assert static_name_hint.kind is HintKind.LITERAL
    assert static_name_hint.value == "static"

    ## expected directory structure:
    # my_bundle/
    # ├── my_dir/
    # │   └── text.txt
    # ├── my_file_fortytwo.bin
    # └── another_dir/
    #     └── embedded_dir/
    #        └── number.bin


def test_bundle_allows_metadata_field_and_runtime_metadata_kwarg():

    @bundle
    class Report:
        name: str
        body: File[str]
        metadata: File[dict]

    report = Report(
        name="alpha",
        body="payload",
        metadata={"score": 10},
        __bundle_metadata__={"runtime": True},
    )

    assert report.metadata == {"score": 10}
    assert report.__bundle_instance_metadata__["runtime"] is True


def test_article_corpus_structure():

    @bundle
    class SingleTable:
        name: str
        xml: File[str] = twig(extension=".xml", name="{name}")

    @bundle
    class SingleAnalysis:
        name: str
        jsonl: File[str] = twig(extension=".jsonl", name="{name}")

    @bundle
    class ProcessedPayload:
        analyses: Dir[List[SingleAnalysis]]
        article_data: File[dict]
        article_metadata: File[dict]
        tables: Dir[List[SingleTable]]
        tables_manifest: File[dict]

    @bundle
    class ProcessedSection:
        vendor: str
        vendor_dir: Dir[ProcessedPayload] = twig(name="{vendor}")

    @bundle
    class SourcePayload:
        article: File[str]

    @bundle
    class SourceSection:
        vendor: str
        vendor_dir: Dir[SourcePayload] = twig(name="{vendor}")

    @bundle
    class ArticleBundle:
        slug: str
        identifiers: File[dict]
        processed: Dir[ProcessedSection]
        source: Dir[SourceSection]

    @bundle
    class ArticleCorpus:
        label: str
        articles: Dir[List[ArticleBundle]] = twig(
            name=(lambda corpus, index: f"{corpus.label}_{index}", "self")
        )

    article_specs = [
        (
            "15820229|10.1016_j.biopsych.2004.12.015|PMC7976178",
            "elsevier",
            ["tbl2"],
            ["tbl1", "tbl2"],
        ),
        (
            "16753205|10.1016_j.bandl.2006.04.013|PMC7976178",
            "elsevier",
            ["tbl1", "tbl2", "tbl3"],
            ["tbl1", "tbl2", "tbl3", "tbl4"],
        ),
        (
            "17412610|10.1016_j.neuroimage.2007.02.027|",
            "elsevier",
            ["tbl2"],
            ["tbl1", "tbl2", "tbl3"],
        ),
        (
            "21256231|10.1016_j.neuroimage.2011.01.028|",
            "elsevier",
            ["t0005"],
            ["t0005", "t0010"],
        ),
        (
            "22102882|10.1371_journal.pone.0027240|PMC3213127",
            "pubget",
            [
                "pone-0027240-t003",
                "pone-0027240-t004",
                "pone-0027240-t005",
            ],
            [
                "pone-0027240-t001",
                "pone-0027240-t002",
                "pone-0027240-t003",
                "pone-0027240-t004",
                "pone-0027240-t005",
            ],
        ),
        (
            "27498135|10.1016_j.neuroimage.2016.08.002|",
            "elsevier",
            ["t0005"],
            ["t0005", "t0010"],
        ),
        (
            "27769891|10.1016_j.neulet.2016.10.029|",
            "elsevier",
            ["tbl0005"],
            ["tbl0005"],
        ),
        (
            "33882403|10.1016_j.bandc.2021.105728|",
            "elsevier",
            ["t0010", "t0015", "t0020"],
            ["t0005", "t0010", "t0015", "t0020"],
        ),
    ]

    def build_article(slug, vendor, analyses, tables):
        processed_payload = ProcessedPayload(
            analyses=[
                SingleAnalysis(name=name, jsonl=f"analysis:{name}") for name in analyses
            ],
            article_data={"slug": slug, "vendor": vendor},
            article_metadata={"doi": slug.split("|")[1]},
            tables=[
                SingleTable(name=name, xml=f"<table id='{name}'/>") for name in tables
            ],
            tables_manifest={"tables": tables},
        )
        processed_section = ProcessedSection(
            vendor=vendor, vendor_dir=processed_payload
        )
        source_section = SourceSection(
            vendor=vendor,
            vendor_dir=SourcePayload(article=f"article-source:{slug}"),
        )
        return ArticleBundle(
            slug=slug,
            identifiers={"slug": slug},
            processed=processed_section,
            source=source_section,
        )

    corpus = ArticleCorpus(
        label="neuro",
        articles=[build_article(*spec) for spec in article_specs],
    )

    assert len(corpus.articles) == len(article_specs)
    assert corpus.articles[0].slug == article_specs[0][0]
    assert corpus.articles[4].processed.vendor == "pubget"

    table_def = SingleTable.__bundle_definition__
    assert table_def.fields[0].kind == FieldKind.VALUE
    table_name_meta = table_def.fields[1].metadata[0].data["name"]
    assert table_name_meta.kind is HintKind.TEMPLATE
    assert table_name_meta.value == "{name}"
    assert table_name_meta.source == "self"

    corpus_def = ArticleCorpus.__bundle_definition__
    articles_field = next(
        field for field in corpus_def.fields if field.name == "articles"
    )
    assert articles_field.kind == FieldKind.DIR
    articles_name_hint = articles_field.metadata[0].data["name"]
    assert articles_name_hint.kind is HintKind.CALLABLE


def test_field_based_template_naming():

    @bundle
    class Report:
        name: str
        contents: File[str]

    @bundle
    class ReportSet:
        reports: Dir[List[Report]] = twig(name=("{name}", "field"))

    report_def = ReportSet.__bundle_definition__
    reports_field = next(
        field for field in report_def.fields if field.name == "reports"
    )
    name_hint = reports_field.metadata[0].data["name"]
    assert name_hint.kind is HintKind.TEMPLATE
    assert name_hint.source == "field"
    assert name_hint.value == "{name}"


def test_extension_inference_variants():

    @bundle
    class Extensions:
        text: File[str]
        mixed: File[Union[str, int]]
        settings: File[dict]
        records: File[List[Dict[str, int]]]
        logs: File[List[str]]

    definition = Extensions.__bundle_definition__
    extensions = {field.name: field for field in definition.fields}

    def _extension_for(name: str) -> str:
        entry = extensions[name]
        return entry.metadata[0].data["extension"]

    assert _extension_for("text") == "txt"
    assert _extension_for("mixed") == "txt"
    assert _extension_for("settings") == "json"
    assert _extension_for("records") == "jsonl"
    assert _extension_for("logs") == "txt"


def test_callable_name_with_varargs():

    def variant_name(owner, index=None, *extras):
        suffix = index if index is not None else "x"
        return f"{owner.prefix}-{suffix}"

    @bundle
    class Variant:
        prefix: str
        payload: File[int] = twig(name=variant_name)

    definition = Variant.__bundle_definition__
    payload_field = next(
        field for field in definition.fields if field.name == "payload"
    )
    hint = payload_field.metadata[0].data["name"]
    assert hint.kind is HintKind.CALLABLE
    assert hint.value is variant_name
