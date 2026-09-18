"""The declaration layer: ``File``, ``Dir``, ``at()`` and ``@bundle``.

A bundle is an ordinary dataclass whose fields fall into exactly three kinds:

``File[T]``
    A single file. ``T`` selects the codec (see :mod:`pyarty.codecs`).
``Dir[T]`` / ``Dir[list[T]]``
    A subdirectory (or one per element) described by another bundle class.
``T`` (anything else)
    A plain value. It is *never* written as its own file; it is carried by the
    pattern variables of its siblings, which is how a name baked into a path
    survives the round trip.

Every ``File``/``Dir`` field owns a :class:`~pyarty.pattern.PathPattern`. When
omitted, the pattern defaults to the field name plus the codec's extension, so
the common case needs no annotation at all.

Validation runs at decoration time, so a layout that could not round-trip
raises on import rather than after it has already clobbered files.
"""

from __future__ import annotations

import sys
from dataclasses import MISSING, dataclass, field as dataclass_field, fields, is_dataclass
from dataclasses import Field as DataclassField
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from .codecs import Codec, codec_for_annotation, describe_annotation, unwrap_optional
from .errors import LayoutError, SpecError
from .pattern import PathPattern

__all__ = [
    "Dir",
    "File",
    "at",
    "bundle",
    "is_bundle",
    "bundle_schema",
    "BundleSchema",
    "BundleField",
    "FieldKind",
    "PathSpec",
]

_T = TypeVar("_T")

#: Key under which a :class:`PathSpec` is stashed in ``dataclasses.field`` metadata.
SPEC_KEY = "pyarty"

SCHEMA_ATTR = "__pyarty_schema__"


class File(Generic[_T]):
    """Marks a field as one file whose payload type is ``_T``."""


class Dir(Generic[_T]):
    """Marks a field as a directory described by bundle class ``_T``.

    ``Dir[list[Child]]`` means one directory per element, in which case the
    pattern's last segment must contain a variable to keep them distinct.
    """


class FieldKind(str, Enum):
    FILE = "file"
    DIR = "dir"
    VALUE = "value"


@dataclass(frozen=True)
class PathSpec:
    """User-supplied placement options for one field, as built by :func:`at`."""

    pattern: str | None = None
    copy: bool | None = None

    def __post_init__(self) -> None:
        if self.pattern is not None:
            # Fail fast on a malformed pattern, at declaration time.
            PathPattern.parse(self.pattern)


def at(
    pattern: str | None = None,
    /,
    *,
    copy: bool | None = None,
    **field_kwargs: Any,
) -> DataclassField[Any]:
    """Declare where a field lives, as a relative path pattern.

    ``pattern`` may contain ``{variable}`` placeholders that map onto sibling
    (or child) fields by name; it is rendered on write and parsed on read.

        body:    File[str]          = at("{name}.txt")
        metrics: File[dict]         = at("metrics.json")
        reports: Dir[list[Report]]  = at("reports/{name}")

    Set ``copy=True`` on a ``File[Path]`` field to copy an existing file rather
    than serialize a payload. Remaining keyword arguments are forwarded to
    :func:`dataclasses.field`, so ``default``/``default_factory`` still work.
    """
    spec = PathSpec(pattern=pattern, copy=copy)
    metadata = dict(field_kwargs.pop("metadata", {}) or {})
    metadata[SPEC_KEY] = spec
    return dataclass_field(metadata=metadata, **field_kwargs)


@dataclass(frozen=True)
class BundleField:
    """A fully resolved field: kind, pattern, codec and child class."""

    name: str
    kind: FieldKind
    annotation: Any
    #: Payload annotation inside ``File[...]``, or the child class for ``Dir[...]``.
    payload: Any
    pattern: PathPattern | None
    codec: Codec | None
    #: Child bundle class for ``Dir`` fields.
    child: type[Any] | None
    is_collection: bool
    optional: bool
    has_default: bool
    is_copy: bool

    @property
    def variables(self) -> tuple[str, ...]:
        return self.pattern.variables if self.pattern else ()


@dataclass(frozen=True)
class BundleSchema:
    """The resolved layout of one bundle class."""

    cls: type[Any]
    fields: tuple[BundleField, ...]

    def by_kind(self, kind: FieldKind) -> tuple[BundleField, ...]:
        return tuple(f for f in self.fields if f.kind is kind)

    def field(self, name: str) -> BundleField | None:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        return None

    @property
    def value_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.by_kind(FieldKind.VALUE))

    def describe(self) -> str:
        """Render the layout as an indented tree, for docs and debugging."""
        return "\n".join(self._describe_lines(set()))

    def _describe_lines(self, seen: set[type[Any]]) -> list[str]:
        lines = [f"{self.cls.__name__}/"]
        if self.cls in seen:
            return [f"{self.cls.__name__}/ (recursive)"]
        seen = seen | {self.cls}
        for item in self.fields:
            if item.kind is FieldKind.VALUE:
                lines.append(
                    f"  ({item.name}: {describe_annotation(item.annotation)}"
                    " — from path)"
                )
            elif item.kind is FieldKind.FILE:
                detail = "copied" if item.is_copy else item.codec.name  # type: ignore[union-attr]
                lines.append(f"  {item.pattern}  [{detail}]")
            else:
                marker = "*" if item.is_collection else ""
                lines.append(f"  {item.pattern}/{marker}")
                child_schema = bundle_schema(item.child)  # type: ignore[arg-type]
                for line in child_schema._describe_lines(seen)[1:]:
                    lines.append(f"  {line}")
        return lines


def is_bundle(candidate: Any) -> bool:
    """Whether ``candidate`` is a class decorated with :func:`bundle`."""
    return isinstance(candidate, type) and hasattr(candidate, SCHEMA_ATTR)


def bundle_schema(cls: type[Any]) -> BundleSchema:
    """Return the :class:`BundleSchema` for a bundle class."""
    schema = getattr(cls, SCHEMA_ATTR, None)
    if schema is None:
        raise SpecError(
            f"{getattr(cls, '__name__', cls)!r} is not a @bundle class."
        )
    return schema


# ----------------------------------------------------------------------
# Decoration
# ----------------------------------------------------------------------
def bundle(cls: type[Any] | None = None, /, **dataclass_kwargs: Any):
    """Turn a class into a bundle: a dataclass with ``.write()`` and ``.read()``.

    Classes that are not already dataclasses are converted. The resolved layout
    is attached as ``__pyarty_schema__`` and validated immediately.
    """
    # Capture the defining scope so bundles declared inside a function (tests,
    # factories, notebooks) can still reference sibling bundle classes, which
    # `typing.get_type_hints` would otherwise fail to resolve because it only
    # consults module globals.
    try:
        caller_locals = dict(sys._getframe(1).f_locals)
    except (AttributeError, ValueError):  # pragma: no cover - exotic runtimes
        caller_locals = {}

    def decorate(target: type[Any]) -> type[Any]:
        actual = target if is_dataclass(target) else dataclass(**dataclass_kwargs)(target)
        # The class being decorated is not yet bound in the defining scope, so
        # add it explicitly to support self-referential layouts.
        localns = {**caller_locals, target.__name__: actual}
        schema = _resolve_schema(actual, localns)
        setattr(actual, SCHEMA_ATTR, schema)
        _attach_io(actual)
        return actual

    return decorate(cls) if cls is not None else decorate


def _attach_io(cls: type[Any]) -> None:
    from .reader import read_bundle
    from .writer import write_bundle

    if "write" not in cls.__dict__:

        def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
            """Materialize this bundle under ``path``; returns that path."""
            return write_bundle(self, path, overwrite=overwrite)

        cls.write = write  # type: ignore[attr-defined]

    if "read" not in cls.__dict__:

        def read(target_cls, path: str | Path, *, strict: bool = False):
            """Read ``path`` into an instance of this bundle class."""
            return read_bundle(target_cls, path, strict=strict)

        cls.read = classmethod(read)  # type: ignore[attr-defined]

    if "layout" not in cls.__dict__:

        def layout(target_cls) -> str:
            """Return the declared on-disk layout as a tree."""
            return bundle_schema(target_cls).describe()

        cls.layout = classmethod(layout)  # type: ignore[attr-defined]


def _resolve_schema(
    cls: type[Any], localns: dict[str, Any] | None = None
) -> BundleSchema:
    hints = _type_hints(cls, localns)
    resolved: list[BundleField] = []

    for dc_field in fields(cls):
        annotation = hints.get(dc_field.name, dc_field.type)
        spec = dc_field.metadata.get(SPEC_KEY) if dc_field.metadata else None
        if spec is not None and not isinstance(spec, PathSpec):
            raise SpecError(
                f"Field '{dc_field.name}' on '{cls.__name__}' has unexpected "
                f"{SPEC_KEY!r} metadata; use at(...) to declare placement."
            )
        resolved.append(_resolve_field(cls, dc_field, annotation, spec))

    schema = BundleSchema(cls=cls, fields=tuple(resolved))
    _validate_schema(schema)
    return schema


def _type_hints(
    cls: type[Any], localns: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        return get_type_hints(cls, localns=localns, include_extras=True)
    except NameError as exc:
        raise SpecError(
            f"Could not resolve type hints for '{cls.__name__}': {exc}. "
            "Forward references must be importable when @bundle runs."
        ) from exc


def _resolve_field(
    cls: type[Any],
    dc_field: DataclassField[Any],
    annotation: Any,
    spec: PathSpec | None,
) -> BundleField:
    base, optional = unwrap_optional(annotation)
    origin = get_origin(base)
    has_default = (
        dc_field.default is not MISSING or dc_field.default_factory is not MISSING
    )

    if origin is File:
        return _resolve_file(cls, dc_field, annotation, base, spec, optional, has_default)
    if origin is Dir:
        return _resolve_dir(cls, dc_field, annotation, base, spec, optional, has_default)

    if spec is not None:
        raise SpecError(
            f"Field '{dc_field.name}' on '{cls.__name__}' uses at(...) but is "
            f"annotated '{describe_annotation(annotation)}'. Only File[...] and "
            "Dir[...] fields have a path; plain values come from pattern "
            "variables on sibling fields."
        )
    return BundleField(
        name=dc_field.name,
        kind=FieldKind.VALUE,
        annotation=annotation,
        payload=annotation,
        pattern=None,
        codec=None,
        child=None,
        is_collection=False,
        optional=optional,
        has_default=has_default,
        is_copy=False,
    )


def _resolve_file(
    cls: type[Any],
    dc_field: DataclassField[Any],
    annotation: Any,
    base: Any,
    spec: PathSpec | None,
    optional: bool,
    has_default: bool,
) -> BundleField:
    args = get_args(base)
    if len(args) != 1:
        raise SpecError(
            f"File[...] on field '{dc_field.name}' of '{cls.__name__}' needs "
            "exactly one payload type, e.g. File[dict]."
        )
    payload = args[0]
    if _mentions(payload, Dir) or _mentions(payload, File):
        raise SpecError(
            f"File payload for '{dc_field.name}' on '{cls.__name__}' cannot "
            "nest File[...] or Dir[...]; a file holds data, not structure."
        )

    codec = codec_for_annotation(payload)
    is_copy = spec.copy if spec and spec.copy is not None else codec.is_copy
    if is_copy and not codec.is_copy:
        # copy=True on a non-Path annotation: honour the flag, copy semantics win.
        from .codecs import COPY

        codec = COPY

    pattern = _pattern_for(spec, dc_field.name)
    if not pattern.suffix and codec.extension and not is_copy:
        pattern = pattern.with_suffix(codec.extension)

    return BundleField(
        name=dc_field.name,
        kind=FieldKind.FILE,
        annotation=annotation,
        payload=payload,
        pattern=pattern,
        codec=codec,
        child=None,
        is_collection=False,
        optional=optional,
        has_default=has_default,
        is_copy=bool(is_copy),
    )


def _resolve_dir(
    cls: type[Any],
    dc_field: DataclassField[Any],
    annotation: Any,
    base: Any,
    spec: PathSpec | None,
    optional: bool,
    has_default: bool,
) -> BundleField:
    args = get_args(base)
    if len(args) != 1:
        raise SpecError(
            f"Dir[...] on field '{dc_field.name}' of '{cls.__name__}' needs "
            "exactly one argument, e.g. Dir[Child] or Dir[list[Child]]."
        )

    inner = args[0]
    child, is_collection = _child_of(inner)
    if child is None:
        raise SpecError(
            f"Dir[...] on field '{dc_field.name}' of '{cls.__name__}' must "
            f"reference a @bundle class (or a list of one); got "
            f"'{describe_annotation(inner)}'."
        )

    pattern = _pattern_for(spec, dc_field.name)
    return BundleField(
        name=dc_field.name,
        kind=FieldKind.DIR,
        annotation=annotation,
        payload=inner,
        pattern=pattern,
        codec=None,
        child=child,
        is_collection=is_collection,
        optional=optional,
        has_default=has_default,
        is_copy=False,
    )


def _pattern_for(spec: PathSpec | None, field_name: str) -> PathPattern:
    if spec is not None and spec.pattern is not None:
        return PathPattern.parse(spec.pattern)
    return PathPattern.parse(field_name)


_COLLECTION_ORIGINS = (list, tuple, set, frozenset)


def _child_of(inner: Any) -> tuple[type[Any] | None, bool]:
    """Resolve ``Dir``'s argument into ``(child_class, is_collection)``."""
    if is_bundle(inner):
        return inner, False
    origin = get_origin(inner)
    if origin in _COLLECTION_ORIGINS:
        args = [arg for arg in get_args(inner) if arg is not Ellipsis]
        if args and is_bundle(args[0]):
            return args[0], True
    return None, False


def _mentions(annotation: Any, marker: type[Any]) -> bool:
    if get_origin(annotation) is marker:
        return True
    return any(_mentions(arg, marker) for arg in get_args(annotation))


# ----------------------------------------------------------------------
# Whole-schema validation
# ----------------------------------------------------------------------
def _validate_schema(schema: BundleSchema) -> None:
    cls_name = schema.cls.__name__
    placements: dict[str, str] = {}

    for item in schema.fields:
        if item.pattern is None:
            continue

        # A collection of directories must vary per element or they collide.
        if item.kind is FieldKind.DIR and item.is_collection:
            if not item.pattern.names_distinct_children:
                raise LayoutError(
                    f"Dir[list[...]] field '{item.name}' on '{cls_name}' uses "
                    f"pattern '{item.pattern}', whose last segment has no "
                    f"{{variable}}. Every element would render to the same "
                    f"directory and all but one would be lost.\n"
                    f"  Try: at(\"{item.pattern}/{{name}}\") — where 'name' is "
                    f"a field on {item.child.__name__}."  # type: ignore[union-attr]
                )

        # Two fields writing the same literal path would clobber each other.
        if not item.pattern.is_dynamic:
            previous = placements.get(item.pattern.raw)
            if previous is not None:
                raise LayoutError(
                    f"Fields '{previous}' and '{item.name}' on '{cls_name}' "
                    f"both map to '{item.pattern}'. Give at least one its own "
                    "pattern."
                )
            placements[item.pattern.raw] = item.name

    _validate_variables(schema)


def _validate_variables(schema: BundleSchema) -> None:
    """Check that every pattern variable names a real field.

    A ``File`` pattern resolves against its owner; a ``Dir`` pattern resolves
    against the child first, then falls back to the owner.

    Whether a plain value field is *recoverable* is deliberately not checked
    here: a child's value is very often supplied by its parent's ``Dir``
    pattern, which is unknowable at the child's decoration time. The reader
    reports that case against the actual tree instead.
    """
    cls_name = schema.cls.__name__
    own_names = {f.name for f in schema.fields}

    for item in schema.fields:
        if item.pattern is None:
            continue

        if item.kind is FieldKind.DIR:
            child_names = {f.name for f in bundle_schema(item.child).fields}  # type: ignore[arg-type]
            resolvable = child_names | own_names
            hint = (
                f"'{item.child.__name__}' has: "  # type: ignore[union-attr]
                f"{', '.join(sorted(child_names)) or '(no fields)'}"
            )
        else:
            resolvable = own_names
            hint = f"'{cls_name}' has: {', '.join(sorted(own_names))}"

        for variable in item.pattern.variables:
            if variable not in resolvable:
                raise SpecError(
                    f"Pattern '{item.pattern}' on field '{item.name}' of "
                    f"'{cls_name}' refers to '{{{variable}}}', which is not a "
                    f"field. {hint}"
                )
