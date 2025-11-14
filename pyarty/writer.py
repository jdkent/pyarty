from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Type

from .dsl import (
    BundleDefinition,
    BundleField,
    BundleMetadata,
    FieldKind,
    File,
    Dir,
    Hint,
    HintKind,
)


class RenderError(RuntimeError):
    """Raised when a bundle cannot be rendered to disk."""


def write_bundle(
    bundle: Any, output_path: str | Path, *, overwrite: bool = False
) -> None:
    """Render a bundle instance to ``output_path``."""

    path = Path(output_path)
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise RenderError(f"Destination '{path}' already exists and is not empty.")
    else:
        path.mkdir(parents=True, exist_ok=True)

    definition = getattr(bundle.__class__, "__bundle_definition__", None)
    if definition is None or not isinstance(definition, BundleDefinition):
        raise RenderError("Object is not a bundle-decorated dataclass instance.")

    _render_fields(definition, bundle, path)


def _render_fields(
    definition: BundleDefinition, instance: Any, base_path: Path
) -> None:
    for field in definition.fields:
        value = getattr(instance, field.name)
        if value is None:
            continue
        if field.kind is FieldKind.DIR:
            _render_dir_field(field, value, instance, base_path)
        elif field.kind is FieldKind.FILE:
            _render_file_field(field, value, instance, base_path)
        # VALUE fields are metadata-only and skipped.


def _render_dir_field(
    field: BundleField, value: Any, owner: Any, base_path: Path
) -> None:
    entries = _iter_dir_entries(value)
    if not entries:
        return

    for index, child in entries:
        name = _compute_name(field, owner, child, index)
        dir_path = base_path / name
        dir_path.mkdir(parents=True, exist_ok=True)
        child_def = getattr(child.__class__, "__bundle_definition__", None)
        if child_def is None:
            raise RenderError(
                f"Directory field '{field.name}' expected bundle data; got {type(child).__name__}."
            )
        _render_fields(child_def, child, dir_path)


def _render_file_field(
    field: BundleField, value: Any, owner: Any, base_path: Path
) -> None:
    metadata = _metadata_for_layer(field.metadata, layer_type=File)
    extension = metadata.get("extension")
    name = _compute_name(field, owner, value, None)
    if not name:
        raise RenderError(f"File field '{field.name}' produced an empty name.")
    filename = name
    if extension:
        filename = f"{filename}.{extension}"
    target = base_path / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(target, value, extension)


def _iter_dir_entries(value: Any) -> list[tuple[int | None, Any]]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        entries: list[tuple[int | None, Any]] = []
        for idx, item in enumerate(value):
            entries.append((idx, item))
        return entries
    return [(None, value)]


def _metadata_for_layer(
    metadata: tuple[BundleMetadata, ...], layer_type: Type[Any]
) -> Mapping[str, Any]:
    for entry in metadata:
        if entry.layer is layer_type:
            return entry.data
    return {}


def _compute_name(
    field: BundleField, owner: Any, subject: Any, index: int | None
) -> str:
    layer = Dir if field.kind is FieldKind.DIR else File
    metadata = _metadata_for_layer(field.metadata, layer)
    name_hint = metadata.get("name")
    name = _resolve_hint(name_hint, owner, subject, index, field.name)
    if name is None:
        if field.kind is FieldKind.DIR and field.is_collection and index is not None:
            name = field.name
        else:
            name = _default_name(field.name, index)
    prefix_hint = metadata.get("prefix")
    if prefix_hint:
        prefix_value = _resolve_hint(prefix_hint, owner, subject, index, field.name)
        if prefix_value:
            return _apply_prefix(prefix_value, name)
    return name


def _resolve_hint(
    hint: Hint | None,
    owner: Any,
    subject: Any,
    index: int | None,
    field_name: str,
) -> str | None:
    if hint is None:
        return None
    context = owner if hint.source == "self" else subject
    if hint.kind is HintKind.LITERAL:
        return str(hint.value)
    if hint.kind is HintKind.TEMPLATE:
        context_mapping = dict(_context_from(context))
        if index is not None:
            context_mapping.setdefault("index", index)
        try:
            return hint.value.format(**context_mapping)
        except KeyError as exc:
            raise RenderError(
                f"Missing template variable {exc.args[0]!r} for field '{field_name}'."
            ) from exc
    if hint.kind is HintKind.CALLABLE:
        return _invoke_hint_callable(hint.value, context, index)
    raise RenderError(f"Unsupported hint kind for field '{field_name}'.")


def _context_from(source: Any) -> Mapping[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return source
    if hasattr(source, "__dict__"):
        return vars(source)
    return {}


def _default_name(field_name: str, index: int | None) -> str:
    if index is None:
        return field_name
    return f"{field_name}_{index}"


def _apply_prefix(prefix: str, name: str) -> str:
    return str(Path(prefix) / name)


def _invoke_hint_callable(
    func: Callable[..., Any], context: Any, index: int | None
) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return str(func(context) if index is None else func(context, index))

    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    args: list[Any] = []
    if not params and not has_varargs:
        return str(func())
    if params:
        args.append(context)
    if index is not None and (has_varargs or len(args) < len(params)):
        args.append(index)
    return str(func(*args))


def _write_payload(target: Path, payload: Any, extension: str | None) -> None:
    if isinstance(payload, bytes):
        target.write_bytes(payload)
        return
    if isinstance(payload, str):
        target.write_text(payload)
        return
    if extension == "json":
        with target.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        return
    if extension == "jsonl":
        if not isinstance(payload, Iterable) or isinstance(
            payload, (str, bytes, bytearray)
        ):
            raise RenderError("jsonl payload must be an iterable of records.")
        with target.open("w", encoding="utf-8") as fp:
            for row in payload:
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
        return
    if hasattr(payload, "read"):
        target.write_bytes(payload.read())
        return
    raise RenderError(
        f"Unsupported payload type for field output: {type(payload).__name__}."
    )
