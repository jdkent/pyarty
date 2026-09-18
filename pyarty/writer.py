"""Writing bundles to disk.

Writing is *strict*: the whole tree is planned and validated in memory before
a single byte is written. A bundle therefore either produces a complete, valid
directory or raises without leaving a half-written one behind.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
        else:
            _plan_dir(item, value, instance, prefix, where, into)


def _plan_file(
    item: BundleField,
    value: Any,
    owner: Any,
    prefix: str,
    where: str,
) -> _PlannedFile:
    assert item.pattern is not None and item.codec is not None

    if item.is_copy:
        source = _resolve_source(value, where)
        pattern = item.pattern
        if not pattern.suffix and source.suffix:
            # Preserve the source extension so read() can find it again.
            pattern = pattern.with_suffix(source.suffix)
        relative = _join(prefix, pattern.format(_variables(owner)))
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

    relative = _join(prefix, item.pattern.format(_variables(owner)))
    return _PlannedFile(
        relative_path=relative, payload=payload, copy_from=None, origin=where
    )


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
