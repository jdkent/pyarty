"""pyarty — a two-way contract between directory trees and dataclasses.

Declare where each field lives with a relative path pattern, then move between
a dataclass and a directory in either direction:

    from pyarty import Dir, File, at, bundle

    @bundle
    class Report:
        name: str                                  # filled FROM the path
        body: File[str] = at("{name}.txt")
        metrics: File[dict] = at("metrics.json")

    @bundle
    class Corpus:
        reports: Dir[list[Report]] = at("reports/{name}")
        summary: File[dict] = at("summary.json")

    Corpus(...).write("./out")
    Corpus.read("./out")        # == the original

The guarantee is that ``read`` is the inverse of ``write``: patterns are parsed
as well as rendered, so a value baked into a filename comes back as a field.
"""

from __future__ import annotations

from .codecs import Codec, codec_for_annotation
from .errors import (
    LayoutError,
    MissingFileError,
    PatternError,
    PayloadTypeError,
    PyartyError,
    ReadError,
    SpecError,
    WriteError,
)
from .pattern import PathPattern
from .reader import read_bundle
from .scaffold import scaffold_from_directory
from .schema import (
    BundleField,
    BundleSchema,
    Dir,
    FieldKind,
    File,
    PathSpec,
    at,
    bundle,
    bundle_schema,
    is_bundle,
)
from .writer import plan_bundle, write_bundle

__version__ = "0.1.0"

__all__ = [
    # Declaration
    "bundle",
    "at",
    "File",
    "Dir",
    # Round trip
    "write_bundle",
    "read_bundle",
    "plan_bundle",
    # Introspection
    "bundle_schema",
    "is_bundle",
    "BundleSchema",
    "BundleField",
    "FieldKind",
    "PathSpec",
    "PathPattern",
    "Codec",
    "codec_for_annotation",
    "scaffold_from_directory",
    # Errors
    "PyartyError",
    "PatternError",
    "SpecError",
    "LayoutError",
    "PayloadTypeError",
    "WriteError",
    "ReadError",
    "MissingFileError",
    "__version__",
]
