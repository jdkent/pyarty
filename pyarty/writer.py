"""Writing bundles to disk.

Writing is *strict*: the whole tree is planned and validated in memory before
a single byte is written. A bundle therefore either produces a complete, valid
directory or raises without leaving a half-written one behind.
"""

from __future__ import annotations

import collections.abc as abc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .codecs import describe_annotation
from .errors import PayloadTypeError, WriteError
from .schema import BundleField, FieldKind, bundle_schema, is_bundle

__all__ = ["write_bundle", "plan_bundle"]


@dataclass(frozen=True)
class _PlannedFile:
    """One resolved file write, ready to execute."""

    relative_path: str
    payload: bytes | None
    copy_from: Path | None
    #: Dotted field path, for error messages.
    origin: str


def write_bundle(
    instance: Any, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write ``instance`` into directory ``path``.

    Returns the directory written. Raises before touching disk if any payload
    contradicts its declared annotation.
    """
    base = Path(path)
    planned = plan_bundle(instance)

    if base.exists():
        if not base.is_dir():
            raise WriteError(f"Destination '{base}' exists and is not a directory.")
        clashes = [p for p in planned if (base / p.relative_path).exists()]
        if clashes and not overwrite:
            listed = ", ".join(sorted(p.relative_path for p in clashes)[:5])
            raise WriteError(
                f"Destination '{base}' already contains {len(clashes)} of the "
                f"files to be written ({listed}"
                f"{', ...' if len(clashes) > 5 else ''}). "
                "Pass overwrite=True to replace them."
            )

    for item in planned:
        target = base / item.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.copy_from is not None:
            shutil.copy2(item.copy_from, target)
        else:
            assert item.payload is not None  # guaranteed by _plan_file
            target.write_bytes(item.payload)

    # Directories that contain no files still belong to the declared layout.
    base.mkdir(parents=True, exist_ok=True)
    for directory in _planned_directories(instance):
        (base / directory).mkdir(parents=True, exist_ok=True)

    return base


def plan_bundle(instance: Any) -> tuple[_PlannedFile, ...]:
    """Resolve ``instance`` into the exact files it would write.

    Exposed so callers can preview or test a layout without writing it.
    """
    if not is_bundle(type(instance)):
        raise WriteError(
            f"{type(instance).__name__} is not a @bundle class instance."
        )
    planned: list[_PlannedFile] = []
    _plan(instance, prefix="", origin=type(instance).__name__, into=planned)
    _reject_duplicates(planned)
    return tuple(planned)


def _reject_duplicates(planned: Iterable[_PlannedFile]) -> None:
    seen: dict[str, str] = {}
    for item in planned:
        previous = seen.get(item.relative_path)
        if previous is not None:
            raise WriteError(
                f"Two fields would both write '{item.relative_path}': "
                f"{previous} and {item.origin}. This usually means a pattern "
                "variable holds the same value for two elements."
            )
        seen[item.relative_path] = item.origin


def _plan(
    instance: Any, *, prefix: str, origin: str, into: list[_PlannedFile]
) -> None:
    schema = bundle_schema(type(instance))
    for item in schema.fields:
        value = getattr(instance, item.name, None)
        where = f"{origin}.{item.name}"

        if item.kind is FieldKind.VALUE:
            continue
        if value is None:
            if item.optional or item.has_default:
                continue
            raise PayloadTypeError(
                f"Field '{where}' is None but is not optional. Declare it as "
                f"'{describe_annotation(item.annotation)} | None' to allow "
                "omitting it."
            )

        if item.kind is FieldKind.FILE:
            into.append(_plan_file(item, value, instance, prefix, where))
        elif item.kind is FieldKind.FILES:
            _plan_files(item, value, instance, prefix, where, into)
        elif item.kind is FieldKind.GROUP:
            _plan_group(item, value, instance, prefix, where, into)
        else:
            _plan_dir(item, value, instance, prefix, where, into)


def _plan_file(
    item: BundleField,
    value: Any,
    owner: Any,
    prefix: str,
    where: str,
    *,
    scope_override: Mapping[str, Any] | None = None,
) -> _PlannedFile:
    """Plan one file. ``scope_override`` is used by ``Files`` to inject its key."""
    assert item.pattern is not None and item.codec is not None
    scope = scope_override if scope_override is not None else _variables(owner)

    if item.is_copy:
        source = _resolve_source(value, where)
        pattern = item.pattern
        if not pattern.suffix and source.suffix:
            # Preserve the source extension so read() can find it again.
            pattern = pattern.with_suffix(source.suffix)
        relative = _join(prefix, pattern.format(scope))
        return _PlannedFile(
            relative_path=relative, payload=None, copy_from=source, origin=where
        )

    if not item.codec.accepts(value):
        raise PayloadTypeError(
            f"Field '{where}' is declared "
            f"File[{describe_annotation(item.payload)}], which expects "
            f"{item.codec.expects}, but got {type(value).__name__}: "
            f"{_preview(value)}"
        )

    try:
        payload = item.codec.encode(value)
    except (TypeError, ValueError) as exc:
        raise PayloadTypeError(
            f"Field '{where}' could not be encoded as {item.codec.name}: {exc}"
        ) from exc

    relative = _join(prefix, item.pattern.format(scope))
    return _PlannedFile(
        relative_path=relative, payload=payload, copy_from=None, origin=where
    )


def _plan_files(
    item: BundleField,
    value: Any,
    owner: Any,
    prefix: str,
    where: str,
    into: list[_PlannedFile],
) -> None:
    """Plan one file per entry of a ``Files`` mapping.

    The mapping key supplies the field's key variable; every other variable in
    the pattern resolves against the owning instance.
    """
    if not isinstance(value, abc.Mapping):
        raise PayloadTypeError(
            f"Field '{where}' is declared Files[dict[...]] and expects a "
            f"mapping of key -> payload; got {type(value).__name__}."
        )

    owner_scope = _variables(owner)
    for key, payload in value.items():
        if key is None or str(key) == "":
            raise PayloadTypeError(
                f"Field '{where}' has an empty key; it names a file, so it "
                "cannot be blank."
            )
        entry = f"{where}[{key!r}]"
        scope = {**owner_scope, item.key: key}
        # A single-entry view of the field, reusing the File planner so copy
        # handling and payload validation stay in one place.
        into.append(
            _plan_file(item, payload, owner, prefix, entry, scope_override=scope)
        )


def _plan_group(
    item: BundleField,
    value: Any,
    owner: Any,
    prefix: str,
    where: str,
    into: list[_PlannedFile],
) -> None:
    """Plan a ``Group``: records rooted in the *owner's* directory.

    Unlike ``Dir``, a group creates no directory of its own — each record's
    fields carry full relative paths that spread across parallel trees. So the
    records are planned at the same prefix as their owner, and any pattern on
    the group field acts purely as a prefix directory.
    """
    if isinstance(value, abc.Mapping) or not isinstance(value, abc.Iterable):
        raise PayloadTypeError(
            f"Field '{where}' is declared Group[list[...]] and expects a "
            f"sequence of '{item.child.__name__}'; got "  # type: ignore[union-attr]
            f"{type(value).__name__}."
        )

    root = prefix
    if item.pattern is not None:
        root = _join(prefix, item.pattern.format(_variables(owner)))

    seen_keys: dict[Any, int] = {}
    for index, record in enumerate(value):
        if not is_bundle(type(record)):
            raise PayloadTypeError(
                f"Field '{where}' is declared Group[list[...]] of "
                f"'{item.child.__name__}' but element {index} is a "  # type: ignore[union-attr]
                f"{type(record).__name__}."
            )
        key = getattr(record, item.key, None)  # type: ignore[arg-type]
        if key is None:
            raise PayloadTypeError(
                f"Field '{where}' groups by '{item.key}', but element {index} "
                f"has {item.key}=None. It names the record's files, so it "
                "cannot be blank."
            )
        if key in seen_keys:
            raise PayloadTypeError(
                f"Field '{where}' has two records with {item.key}={key!r} "
                f"(elements {seen_keys[key]} and {index}). The grouping key "
                "must be unique, or their files would collide."
            )
        seen_keys[key] = index
        _plan(record, prefix=root, origin=f"{where}[{key!r}]", into=into)


def _plan_dir(
    item: BundleField,
    value: Any,
    owner: Any,
    prefix: str,
    where: str,
    into: list[_PlannedFile],
) -> None:
    assert item.pattern is not None
    children = list(value) if item.is_collection else [value]

    for index, child in enumerate(children):
        if not is_bundle(type(child)):
            raise PayloadTypeError(
                f"Field '{where}' is declared Dir[...] of "
                f"'{item.child.__name__}' but element {index} is a "  # type: ignore[union-attr]
                f"{type(child).__name__}."
            )
        # Child values win; the owner supplies anything the child lacks.
        scope = {**_variables(owner), **_variables(child)}
        relative = _join(prefix, item.pattern.format(scope))
        child_origin = f"{where}[{index}]" if item.is_collection else where
        _plan(child, prefix=relative, origin=child_origin, into=into)


def _planned_directories(instance: Any) -> list[str]:
    """Relative directories implied by ``Dir`` fields, including empty ones."""
    found: list[str] = []

    def walk(current: Any, prefix: str) -> None:
        schema = bundle_schema(type(current))
        # Group records live in their owner's directory, so any Dir fields they
        # declare still contribute directories at that same prefix.
        for item in schema.by_kind(FieldKind.GROUP):
            records = getattr(current, item.name, None) or []
            root = prefix
            if item.pattern is not None:
                root = _join(prefix, item.pattern.format(_variables(current)))
            if item.pattern is not None:
                found.append(root)
            for record in records:
                if is_bundle(type(record)):
                    walk(record, root)

        for item in schema.by_kind(FieldKind.DIR):
            value = getattr(current, item.name, None)
            if value is None:
                continue
            assert item.pattern is not None
            children = list(value) if item.is_collection else [value]
            for child in children:
                if not is_bundle(type(child)):
                    continue
                scope = {**_variables(current), **_variables(child)}
                relative = _join(prefix, item.pattern.format(scope))
                found.append(relative)
                walk(child, relative)

    walk(instance, "")
    return found


def _variables(instance: Any) -> dict[str, Any]:
    """Values on ``instance`` usable as pattern variables.

    Only plain values qualify; a ``File``/``Dir`` field holds a payload, not a
    name, so including it would let a dict leak into a path.
    """
    schema = bundle_schema(type(instance))
    return {
        item.name: getattr(instance, item.name, None)
        for item in schema.by_kind(FieldKind.VALUE)
    }


def _resolve_source(value: Any, where: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PayloadTypeError(
            f"Field '{where}' is a copy field, so it expects a path; got "
            f"{type(value).__name__}."
        )
    source = Path(value).expanduser()
    if not source.exists():
        raise WriteError(
            f"Field '{where}' copies from '{source}', which does not exist."
        )
    if source.is_dir():
        raise WriteError(
            f"Field '{where}' copies from '{source}', which is a directory; "
            "use Dir[...] for directory structure."
        )
    return source


def _join(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}" if prefix else relative


def _preview(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."
