"""Annotation-driven serialization.

A codec is chosen from a field's *declared type*, never from the runtime value.
That inversion is deliberate: it means the on-disk format is a property of the
schema, so ``File[dict]`` can never emit a bare string and call it ``.json``.

The mapping is intentionally small and predictable:

===========================  =========  ==============================
Annotation                   Extension  On disk
===========================  =========  ==============================
``str``                      ``.txt``   raw UTF-8 text
``bytes``                    ``.bin``   raw bytes
``int`` / ``float``          ``.txt``   the number as text
``bool``                     ``.json``  ``true`` / ``false``
``dict`` / ``Mapping``       ``.json``  indented JSON object
``list[str]``                ``.txt``   one item per line
``list[dict]``               ``.jsonl`` one JSON object per line
``list`` / ``tuple`` (other) ``.json``  indented JSON array
``Path``                     (source)   copied from the given path
===========================  =========  ==============================

An explicit extension in the pattern always wins over the default above; only
the encode/decode behaviour comes from the annotation.
"""

from __future__ import annotations

import collections.abc as abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

__all__ = ["Codec", "codec_for_annotation", "unwrap_optional", "describe_annotation"]


_MAPPING_ORIGINS = (dict, abc.Mapping, abc.MutableMapping)
_SEQUENCE_ORIGINS = (list, tuple, set, frozenset, abc.Sequence, abc.MutableSequence)


@dataclass(frozen=True)
class Codec:
    """Encode/decode pair plus the default extension for an annotation."""

    name: str
    extension: str
    encode: Callable[[Any], bytes]
    decode: Callable[[bytes], Any]
    #: Runtime type check used for strict-write validation.
    accepts: Callable[[Any], bool]
    #: Human-readable description of what ``accepts`` allows.
    expects: str
    #: True when the payload is a path to copy rather than data to serialize.
    is_copy: bool = False


# ----------------------------------------------------------------------
# Primitive codecs
# ----------------------------------------------------------------------
def _json_encode(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _json_decode(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def _jsonl_encode(rows: Any) -> bytes:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    return ("".join(f"{line}\n" for line in lines)).encode("utf-8")


def _jsonl_decode(raw: bytes) -> list[Any]:
    return [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]


def _lines_encode(items: Any) -> bytes:
    return ("".join(f"{item}\n" for item in items)).encode("utf-8")


def _lines_decode(raw: bytes) -> list[str]:
    text = raw.decode("utf-8")
    if not text:
        return []
    # A single trailing newline is a terminator, not an empty final item.
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")


TEXT = Codec(
    name="text",
    extension="txt",
    encode=lambda value: str(value).encode("utf-8"),
    decode=lambda raw: raw.decode("utf-8"),
    accepts=lambda value: isinstance(value, str),
    expects="str",
)

BYTES = Codec(
    name="bytes",
    extension="bin",
    encode=lambda value: bytes(value),
    decode=lambda raw: raw,
    accepts=lambda value: isinstance(value, (bytes, bytearray)),
    expects="bytes",
)

INTEGER = Codec(
    name="int",
    extension="txt",
    encode=lambda value: str(value).encode("utf-8"),
    decode=lambda raw: int(raw.decode("utf-8").strip()),
    accepts=lambda value: isinstance(value, int) and not isinstance(value, bool),
    expects="int",
)

FLOAT = Codec(
    name="float",
    extension="txt",
    encode=lambda value: repr(float(value)).encode("utf-8"),
    decode=lambda raw: float(raw.decode("utf-8").strip()),
    accepts=lambda value: isinstance(value, float) and not isinstance(value, bool),
    expects="float",
)

BOOLEAN = Codec(
    name="bool",
    extension="json",
    encode=_json_encode,
    decode=_json_decode,
    accepts=lambda value: isinstance(value, bool),
    expects="bool",
)

JSON_OBJECT = Codec(
    name="json",
    extension="json",
    encode=_json_encode,
    decode=_json_decode,
    accepts=lambda value: isinstance(value, abc.Mapping),
    expects="a mapping",
)

JSON_ARRAY = Codec(
    name="json-array",
    extension="json",
    encode=_json_encode,
    decode=_json_decode,
    accepts=lambda value: isinstance(value, (list, tuple))
    and not isinstance(value, (str, bytes)),
    expects="a list or tuple",
)

JSON_LINES = Codec(
    name="jsonl",
    extension="jsonl",
    encode=_jsonl_encode,
    decode=_jsonl_decode,
    accepts=lambda value: isinstance(value, (list, tuple))
    and all(isinstance(row, abc.Mapping) for row in value),
    expects="a list of mappings",
)

TEXT_LINES = Codec(
    name="lines",
    extension="txt",
    encode=_lines_encode,
    decode=_lines_decode,
    accepts=lambda value: isinstance(value, (list, tuple))
    and all(isinstance(item, str) for item in value),
    expects="a list of str",
)

JSON_ANY = Codec(
    name="json-any",
    extension="json",
    encode=_json_encode,
    decode=_json_decode,
    accepts=lambda value: True,
    expects="any JSON-serializable value",
)


class PurePathLike:
    """Marker accepted in ``File[...]`` to mean "copy this file".

    ``File[Path]`` means the same thing and reads better; this alias exists for
    callers who want the intent spelled out.
    """


def _copy_accepts(value: Any) -> bool:
    return isinstance(value, (str, Path))


COPY = Codec(
    name="copy",
    extension="",
    # Never invoked: the writer short-circuits to a filesystem copy.
    encode=lambda value: b"",
    decode=lambda raw: raw,
    accepts=_copy_accepts,
    expects="a path to an existing file",
    is_copy=True,
)


# ----------------------------------------------------------------------
# Annotation resolution
# ----------------------------------------------------------------------
def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Split ``T | None`` into ``(T, True)``; anything else into ``(T, False)``."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) != len(args):
            if len(non_none) == 1:
                return non_none[0], True
            return Union[tuple(non_none)], True  # type: ignore[return-value]
    return annotation, False


def codec_for_annotation(annotation: Any) -> Codec:
    """Select the codec for a declared payload annotation."""
    inner, _ = unwrap_optional(annotation)

    if inner is Any:
        return JSON_ANY
    if inner is Path or inner is PurePathLike:
        return COPY
    if inner is str:
        return TEXT
    if inner in (bytes, bytearray):
        return BYTES
    if inner is bool:
        return BOOLEAN
    if inner is int:
        return INTEGER
    if inner is float:
        return FLOAT
    if inner is dict:
        return JSON_OBJECT
    if inner in (list, tuple):
        return JSON_ARRAY

    origin = get_origin(inner)
    if origin in _MAPPING_ORIGINS:
        return JSON_OBJECT
    if origin in _SEQUENCE_ORIGINS:
        args = [arg for arg in get_args(inner) if arg is not Ellipsis]
        if args:
            element, _ = unwrap_optional(args[0])
            if element is str:
                return TEXT_LINES
            if element is dict or get_origin(element) in _MAPPING_ORIGINS:
                return JSON_LINES
        return JSON_ARRAY

    # Unknown but JSON-shaped types (TypedDict, Any-ish aliases) fall back to
    # JSON rather than failing: reading is lenient by design.
    return JSON_ANY


def describe_annotation(annotation: Any) -> str:
    """Render an annotation for error messages."""
    if annotation is Any:
        return "Any"
    name = getattr(annotation, "__name__", None)
    if name and not get_args(annotation):
        return name
    text = str(annotation)
    return text.replace("typing.", "")
