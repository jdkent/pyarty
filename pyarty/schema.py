"""The declaration layer: ``File``, ``Dir``, ``at()`` and ``@bundle``.

A bundle is an ordinary dataclass whose fields fall into exactly four kinds:

``File[T]``
    A single file. ``T`` selects the codec (see :mod:`pyarty.codecs`).
``Files[dict[K, T]]``
    Many files sharing one pattern, keyed by one of its variables.
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

import collections.abc as abc
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
    "Files",
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

#: Set while a class's own layout is being resolved, so a bundle that refers to
#: itself (``Dir[list["Catalog"]]``) counts as a bundle during that window.
RESOLVING_ATTR = "__pyarty_resolving__"


class File(Generic[_T]):
    """Marks a field as one file whose payload type is ``_T``."""


class Files(Generic[_T]):
    """Marks a field as *many* files sharing one pattern.

    Declared as ``Files[dict[str, T]]``: a mapping from the value of the
    pattern's key variable to each file's payload. One pattern matching many
    files is extremely common — ``manifest-{algorithm}.txt``,
    ``model-{shard}-of-{total}.safetensors``, ``..._run-{run}_bold.nii.gz``,
    ``images/train/{stem}.jpg`` — and ``File[...]`` deliberately matches
    exactly one.

    Only the mapping form exists, because it is the only one that can round
    trip: a bare ``list`` of payloads cannot tell ``write`` what to name each
    file. The key supplies one variable; any others resolve against the owning
    instance, exactly as they do for ``Dir``.
    """


class Dir(Generic[_T]):
    """Marks a field as a directory described by bundle class ``_T``.

    ``Dir[list[Child]]`` means one directory per element, in which case the
    pattern's last segment must contain a variable to keep them distinct.
    ``Dir[list["Self"]]`` is allowed, for recursive trees.
    """


class FieldKind(str, Enum):
    FILE = "file"
    FILES = "files"
    DIR = "dir"
    VALUE = "value"


@dataclass(frozen=True)
class PathSpec:
    """User-supplied placement options for one field, as built by :func:`at`."""

    pattern: str | None = None
    copy: bool | None = None
    charset: str | None = None
    key: str | None = None

    def __post_init__(self) -> None:
        if self.pattern is not None:
            # Fail fast on a malformed pattern, at declaration time.
            PathPattern.parse(self.pattern, charset=self.charset)


def at(
    pattern: str | None = None,
    /,
    *,
    copy: bool | None = None,
    charset: str | None = None,
    key: str | None = None,
    **field_kwargs: Any,
) -> DataclassField[Any]:
    """Declare where a field lives, as a relative path pattern.

    ``pattern`` may contain ``{variable}`` placeholders that map onto sibling
    (or child) fields by name; it is rendered on write and parsed on read.

        body:     File[str]              = at("{name}.txt")
        metrics:  File[dict]             = at("metrics.json")
        reports:  Dir[list[Report]]      = at("reports/{name}")
        manifest: Files[dict[str, str]]  = at("manifest-{algorithm}.txt")

    ``charset`` sets the default character class for unqualified variables in
    this pattern; a single variable can override it inline as
    ``{name:alnum}``.

    ``key`` names which variable keys a :class:`Files` mapping. It defaults to
    the pattern's only variable and is required when there is more than one.

    Set ``copy=True`` on a ``File[Path]`` field to copy an existing file rather
    than serialize a payload. Remaining keyword arguments are forwarded to
    :func:`dataclasses.field`, so ``default``/``default_factory`` still work.
    """
    spec = PathSpec(pattern=pattern, copy=copy, charset=charset, key=key)
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
    #: For ``Files`` fields: the variable whose value keys the mapping.
    key: str | None = None
    #: For ``Files`` fields: the mapping's declared key annotation.
    key_type: Any = None

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
            elif item.kind is FieldKind.FILES:
                detail = "copied" if item.is_copy else item.codec.name  # type: ignore[union-attr]
                lines.append(
                    f"  {item.pattern}  [{detail}, many keyed by {{{item.key}}}]"
                )
            else:
                marker = "*" if item.is_collection else ""
                child = item.child
                if child in seen:
                    # A self-referential layout nests forever; say so instead.
                    lines.append(
                        f"  {item.pattern}/{marker}  "
                        f"[{child.__name__}, recursive]"  # type: ignore[union-attr]
                    )
                    continue
                lines.append(f"  {item.pattern}/{marker}")
                child_schema = bundle_schema(child)  # type: ignore[arg-type]
                for line in child_schema._describe_lines(seen)[1:]:
                    lines.append(f"  {line}")
        return lines


def is_bundle(candidate: Any) -> bool:
    """Whether ``candidate`` is a class decorated with :func:`bundle`.

    True while a class is still resolving its own layout, so that a bundle may
    refer to itself.
    """
    if not isinstance(candidate, type):
        return False
    return hasattr(candidate, SCHEMA_ATTR) or bool(
        candidate.__dict__.get(RESOLVING_ATTR)
    )


def _field_names_of(cls: type[Any]) -> set[str]:
    """Field names of a bundle class, usable mid-resolution.

    A self-referential class has no schema yet when its own ``Dir`` patterns
    are validated, so fall back to the dataclass fields.
    """
    schema = getattr(cls, SCHEMA_ATTR, None)
    if schema is not None:
        return {item.name for item in schema.fields}
    if is_dataclass(cls):
        return {item.name for item in fields(cls)}
    return set(getattr(cls, "__annotations__", {}))


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
        # Mark the class as a bundle *before* resolving its own fields, so a
        # self-reference resolves. Without this, Dir[list["Catalog"]] inside
        # Catalog fails: the schema attribute would not exist yet.
        setattr(actual, RESOLVING_ATTR, True)
        try:
            schema = _resolve_schema(actual, localns)
        finally:
            try:
                delattr(actual, RESOLVING_ATTR)
            except AttributeError:  # pragma: no cover - inherited marker
                pass
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
    if origin is Files:
        return _resolve_files(cls, dc_field, annotation, base, spec, optional, has_default)
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
    if _mentions(payload, Dir) or _mentions(payload, File) or _mentions(payload, Files):
        raise SpecError(
            f"File payload for '{dc_field.name}' on '{cls.__name__}' cannot "
            "nest File/Files/Dir; a file holds data, not structure."
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


def _resolve_files(
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
            f"Files[...] on field '{dc_field.name}' of '{cls.__name__}' needs "
            "exactly one argument, e.g. Files[dict[str, bytes]]."
        )

    container, _ = unwrap_optional(args[0])
    container_origin = get_origin(container) or container
    inner = get_args(container)

    if container_origin in (list, tuple, set, frozenset):
        raise SpecError(
            f"Files[...] on field '{dc_field.name}' of '{cls.__name__}' must be "
            f"a mapping, not {describe_annotation(container)}. A bare list of "
            "payloads cannot round trip, because write would not know what to "
            "name each file.\n"
            f"  Try: Files[dict[str, {describe_annotation(inner[0]) if inner else '...'}]]"
        )
    if container_origin not in _MAPPING_ORIGINS or len(inner) != 2:
        raise SpecError(
            f"Files[...] on field '{dc_field.name}' of '{cls.__name__}' must be "
            f"dict[key, payload]; got {describe_annotation(container)}."
        )

    key_type, payload = inner
    if _mentions(payload, Dir) or _mentions(payload, File) or _mentions(payload, Files):
        raise SpecError(
            f"Files payload for '{dc_field.name}' on '{cls.__name__}' cannot "
            "nest File/Files/Dir; a file holds data, not structure."
        )

    codec = codec_for_annotation(payload)
    is_copy = spec.copy if spec and spec.copy is not None else codec.is_copy
    if is_copy and not codec.is_copy:
        from .codecs import COPY

        codec = COPY

    pattern = _pattern_for(spec, dc_field.name)
    if not pattern.suffix and codec.extension and not is_copy:
        pattern = pattern.with_suffix(codec.extension)

    key = _resolve_files_key(cls, dc_field.name, pattern, spec)
    return BundleField(
        name=dc_field.name,
        kind=FieldKind.FILES,
        annotation=annotation,
        payload=payload,
        pattern=pattern,
        codec=codec,
        child=None,
        is_collection=True,
        optional=optional,
        has_default=has_default,
        is_copy=bool(is_copy),
        key=key,
        key_type=key_type,
    )


def _resolve_files_key(
    cls: type[Any], field_name: str, pattern: PathPattern, spec: PathSpec | None
) -> str:
    """Pick which pattern variable keys a ``Files`` mapping."""
    declared = spec.key if spec else None
    variables = pattern.variables

    if declared is not None:
        if declared not in variables:
            raise SpecError(
                f"Files field '{field_name}' on '{cls.__name__}' declares "
                f"key={declared!r}, which is not a variable in "
                f"'{pattern}'. Available: {', '.join(variables) or '(none)'}."
            )
        return declared

    if not variables:
        raise SpecError(
            f"Files field '{field_name}' on '{cls.__name__}' uses pattern "
            f"'{pattern}', which has no {{variable}}. Every entry would map to "
            "the same file.\n"
            f"  Add one — e.g. at(\"{pattern}-{{key}}\") — or use File[...] for "
            "a single file."
        )
    if len(variables) > 1:
        raise SpecError(
            f"Files field '{field_name}' on '{cls.__name__}' uses pattern "
            f"'{pattern}', which has {len(variables)} variables "
            f"({', '.join(variables)}), so the mapping key is ambiguous.\n"
            f"  Name it: at(\"{pattern}\", key=\"{variables[0]}\"). The rest "
            "resolve against the owning instance."
        )
    return variables[0]


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
    charset = spec.charset if spec is not None else None
    raw = spec.pattern if spec is not None and spec.pattern is not None else field_name
    return PathPattern.parse(raw, charset=charset)


_COLLECTION_ORIGINS = (list, tuple, set, frozenset)
_MAPPING_ORIGINS = (dict, abc.Mapping, abc.MutableMapping)


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

        # Files keys off a variable, checked in _resolve_files_key; a mapping
        # whose pattern never varies is rejected there.
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
            child_names = _field_names_of(item.child)  # type: ignore[arg-type]
            resolvable = child_names | own_names
            hint = (
                f"'{item.child.__name__}' has: "  # type: ignore[union-attr]
                f"{', '.join(sorted(child_names)) or '(no fields)'}"
            )
        elif item.kind is FieldKind.FILES:
            # The key variable comes from the mapping key, not from a field;
            # every other variable still has to resolve against the owner.
            resolvable = own_names | {item.key} if item.key else own_names
            hint = (
                f"'{cls_name}' has: {', '.join(sorted(own_names))} "
                f"(plus '{item.key}', supplied by the mapping key)"
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
