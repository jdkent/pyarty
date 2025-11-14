"""DSL primitives for describing filesystem bundle schemas."""

from __future__ import annotations

import collections.abc as abc
from dataclasses import Field as DataclassField
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


try:  # Python < 3.9 compatibility for typing.Annotated
    from typing import Annotated  # type: ignore
except ImportError:  # pragma: no cover - best effort fallback
    Annotated = None  # type: ignore

__all__ = [
    "Dir",
    "File",
    "bundle",
    "twig",
    "BundleError",
    "BundleMetadataError",
    "BundleMetadata",
    "BundleField",
    "BundleDefinition",
    "FieldKind",
    "Hint",
    "HintKind",
]


_T = TypeVar("_T")


class Dir(Generic[_T]):
    """Marker generic describing a directory node that contains nested schema nodes."""


class File(Generic[_T]):
    """Marker generic describing a file node that holds runtime payload data."""


class BundleError(TypeError):
    """Base error type for DSL misconfiguration."""


class InvalidBundleAnnotation(BundleError):
    """Raised when a dataclass field uses an unsupported annotation."""


class BundleMetadataError(BundleError):
    """Raised when metadata cannot be assigned to a DSL node unambiguously."""


class FieldKind(str, Enum):
    """Classification for bundle fields."""

    DIR = "dir"
    FILE = "file"
    VALUE = "value"


class HintKind(str, Enum):
    """Classification for naming/prefix hints."""

    LITERAL = "literal"
    TEMPLATE = "template"
    CALLABLE = "callable"


@dataclass(frozen=True)
class BundleMetadata:
    """Normalized metadata targeting a specific DSL layer."""

    layer: Type[Any]
    index: int
    data: Mapping[str, Any]


@dataclass(frozen=True)
class BundleField:
    """Representation of a dataclass field participating in the bundle DSL."""

    name: str
    kind: FieldKind
    annotation: Any
    dataclass_field: DataclassField[Any]
    metadata: Tuple[BundleMetadata, ...] = ()
    raw_metadata: Mapping[str, Any] = MappingProxyType({})
    is_collection: bool = False


@dataclass(frozen=True)
class Hint:
    """Normalized runtime naming hint."""

    kind: HintKind
    value: Any
    source: str


@dataclass(frozen=True)
class BundleDefinition:
    """Container describing the structure of a bundle-decorated dataclass."""

    cls: Type[Any]
    fields: Tuple[BundleField, ...]


DEFAULT_DATACLASS_KWARGS: Mapping[str, Any] = MappingProxyType(
    {
        "eq": True,
        "frozen": False,
    }
)


BundleClass = TypeVar("BundleClass", bound=type)

INSTANCE_METADATA_ATTR = "__bundle_instance_metadata__"
INIT_WRAPPED_ATTR = "__bundle_init_wrapped__"
RUNTIME_METADATA_KWARG = "__bundle_metadata__"
_HINT_SOURCES = ("self", "field")
_DEFAULT_HINT_SOURCE = "self"


def bundle(
    cls: BundleClass | None = None, /, **dataclass_kwargs: Any
) -> BundleClass | Callable[[BundleClass], BundleClass]:
    """Decorator registering a dataclass as part of the bundle DSL.

    If the class is not already a dataclass it will be converted using the provided
    ``dataclass`` keyword arguments (falling back to :data:`DEFAULT_DATACLASS_KWARGS`).
    A ``__bundle_definition__`` attribute is injected so downstream code can
    introspect the declared structure without re-parsing annotations.
    """

    def _decorate(target_cls: BundleClass) -> BundleClass:
        actual_cls = target_cls
        if not is_dataclass(actual_cls):
            merged_kwargs = dict(DEFAULT_DATACLASS_KWARGS)
            merged_kwargs.update(dataclass_kwargs)
            actual_cls = dataclass(**merged_kwargs)(actual_cls)
        definition = _build_bundle_definition(actual_cls)
        setattr(actual_cls, "__bundle_definition__", definition)
        if not hasattr(actual_cls, "write"):
            setattr(actual_cls, "write", _bundle_write)
        _ensure_instance_metadata_storage(actual_cls)
        return actual_cls

    if cls is not None:
        return _decorate(cls)
    return _decorate


def twig(
    *,
    name: Any | None = None,
    extension: str | None = None,
    prefix: Any | None = None,
    **field_kwargs: Any,
) -> DataclassField[Any]:
    """Helper for declaring bundle-aware dataclass fields.

    Wraps :func:`dataclasses.field`, merging naming and extension hints with any provided metadata.
    Works for both :class:`Dir` and :class:`File` annotations.
    """

    metadata: Dict[str, Any] = {}
    if name is not None:
        metadata["name"] = name
    if extension is not None:
        metadata["extension"] = _normalize_extension_value(extension)
    if prefix is not None:
        metadata["prefix"] = prefix
    return dataclass_field(metadata=metadata if metadata else None, **field_kwargs)


def _bundle_write(self, path: str | Path, overwrite: bool = False) -> None:
    from .writer import write_bundle

    write_bundle(self, path, overwrite=overwrite)


def _get_type_hints_with_extras(target: Type[Any]) -> Dict[str, Any]:
    try:
        return get_type_hints(target, include_extras=True)
    except TypeError:
        return get_type_hints(target)


def _freeze_metadata_map(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not metadata:
        return MappingProxyType({})
    if isinstance(metadata, MappingProxyType):
        return metadata
    return MappingProxyType(dict(metadata))


def _build_bundle_definition(cls: Type[Any]) -> BundleDefinition:
    annotations = _get_type_hints_with_extras(cls)
    collected: list[BundleField] = []
    for dc_field in fields(cls):
        annotation = annotations.get(dc_field.name, dc_field.type)
        kind = _classify_field(annotation, cls, dc_field.name)
        raw_metadata = _freeze_metadata_map(dc_field.metadata)
        normalized_metadata = (
            _normalize_metadata(
                annotation,
                raw_metadata,
                infer_extension=(kind is FieldKind.FILE),
            )
            if kind in (FieldKind.DIR, FieldKind.FILE)
            else ()
        )
        is_collection = (
            kind is FieldKind.DIR and _dir_annotation_is_collection(annotation)
        )
        collected.append(
            BundleField(
                name=dc_field.name,
                kind=kind,
                annotation=annotation,
                dataclass_field=dc_field,
                metadata=normalized_metadata,
                raw_metadata=raw_metadata,
                is_collection=is_collection,
            )
        )
    definition = BundleDefinition(cls=cls, fields=tuple(collected))
    return definition


def _ensure_instance_metadata_storage(cls: Type[Any]) -> None:
    original_init = cls.__init__
    if getattr(original_init, INIT_WRAPPED_ATTR, False):
        return

    @wraps(original_init)
    def __bundle_init__(self, *args, **kwargs):  # type: ignore[override]
        runtime_metadata = _extract_runtime_metadata(kwargs)
        _set_instance_metadata(self, runtime_metadata)
        original_init(self, *args, **kwargs)

    setattr(__bundle_init__, INIT_WRAPPED_ATTR, True)
    cls.__init__ = __bundle_init__  # type: ignore[assignment]


def _set_instance_metadata(instance: Any, metadata: Any) -> None:
    payload = metadata if metadata is not None else {}
    try:
        setattr(instance, INSTANCE_METADATA_ATTR, payload)
    except AttributeError:
        object.__setattr__(instance, INSTANCE_METADATA_ATTR, payload)


def _extract_runtime_metadata(kwargs: MutableMapping[str, Any]) -> Any:
    if RUNTIME_METADATA_KWARG in kwargs:
        return kwargs.pop(RUNTIME_METADATA_KWARG)
    return None


def _strip_annotated(annotation: Any) -> Any:
    current = annotation
    while Annotated is not None and get_origin(current) is Annotated:
        args = get_args(current)
        if not args:
            break
        current = args[0]
    return current


def _classify_field(annotation: Any, cls: Type[Any], field_name: str) -> FieldKind:
    base = _strip_annotated(annotation)
    origin = get_origin(base)
    if origin is Dir:
        _validate_dir_annotation(base, cls, field_name)
        return FieldKind.DIR
    if origin is File:
        _validate_file_annotation(base, cls, field_name)
        return FieldKind.FILE
    return FieldKind.VALUE


def _validate_dir_annotation(annotation: Any, cls: Type[Any], field_name: str) -> None:
    args = get_args(annotation)
    if len(args) != 1:
        raise InvalidBundleAnnotation(
            f"Dir annotation for field '{field_name}' on bundle '{cls.__name__}' requires exactly one argument."
        )
    payload = _extract_dir_payload(args[0])
    if payload is None or not _is_bundle_class(payload):
        raise InvalidBundleAnnotation(
            f"Dir annotation for field '{field_name}' on bundle '{cls.__name__}' must reference another "
            "@bundle-decorated class."
        )


def _validate_file_annotation(annotation: Any, cls: Type[Any], field_name: str) -> None:
    args = get_args(annotation)
    if len(args) != 1:
        raise InvalidBundleAnnotation(
            f"File annotation for field '{field_name}' on bundle '{cls.__name__}' requires exactly one argument."
        )
    payload = args[0]
    if _contains_dir(payload):
        raise InvalidBundleAnnotation(
            f"File payload for field '{field_name}' on bundle '{cls.__name__}' cannot contain Dir[...] annotations."
        )


def _contains_dir(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return False
    if origin is Dir:
        return True
    return any(_contains_dir(arg) for arg in get_args(annotation))


def _is_bundle_class(candidate: Any) -> bool:
    return isinstance(candidate, type) and hasattr(candidate, "__bundle_definition__")


_DIR_COLLECTION_ORIGINS: Tuple[Any, ...] = (
    list,
    tuple,
    set,
    frozenset,
    abc.Sequence,
    abc.MutableSequence,
    abc.Set,
    abc.Iterable,
)


def _extract_dir_payload(argument: Any) -> Type[Any] | None:
    candidate = _strip_annotated(argument)
    if _is_bundle_class(candidate):
        return candidate
    origin = get_origin(candidate)
    if origin in _DIR_COLLECTION_ORIGINS:
        args = get_args(candidate)
        if not args:
            return None
        inner = args[0]
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
        return _extract_dir_payload(inner)
    return None


def _dir_annotation_is_collection(annotation: Any) -> bool:
    args = get_args(_strip_annotated(annotation))
    if len(args) != 1:
        return False
    candidate = _strip_annotated(args[0])
    origin = get_origin(candidate)
    if origin in _DIR_COLLECTION_ORIGINS:
        return True
    return False


def _normalize_metadata(
    annotation: Any,
    metadata: Mapping[str, Any],
    *,
    infer_extension: bool = False,
) -> Tuple[BundleMetadata, ...]:
    root_layer = _top_layer(annotation)
    regular_keys: Dict[Any, Any] = {}
    layer_keys: Dict[Type[Any], Any] = {}
    for key, value in metadata.items():
        if key in (Dir, Dir.__name__):
            _register_layer(layer_keys, Dir, value)
        elif key in (File, File.__name__):
            _register_layer(layer_keys, File, value)
        else:
            regular_keys[key] = value
    normalized: list[BundleMetadata] = []
    if regular_keys:
        normalized.append(
            BundleMetadata(
                layer=root_layer,
                index=0,
                data=_normalize_metadata_layer(root_layer, regular_keys),
            )
        )
    for layer, value in layer_keys.items():
        normalized.extend(_expand_layer_metadata(layer, value))
    return _maybe_attach_extension(tuple(normalized), annotation, infer_extension)


def _register_layer(
    target: MutableMapping[Type[Any], Any], layer: Type[Any], value: Any
) -> None:
    if layer in target:
        raise BundleMetadataError(
            f"Duplicate metadata assignment for layer {layer.__name__}. Use a sequence of mappings instead."
        )
    target[layer] = value


def _expand_layer_metadata(layer: Type[Any], value: Any) -> Iterable[BundleMetadata]:
    if isinstance(value, Mapping):
        yield BundleMetadata(
            layer=layer, index=0, data=_normalize_metadata_layer(layer, value)
        )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise BundleMetadataError(
                    f"Layer metadata for {layer.__name__} must be mappings; got {type(entry).__name__}."
                )
            yield BundleMetadata(
                layer=layer, index=index, data=_normalize_metadata_layer(layer, entry)
            )
        return
    raise BundleMetadataError(
        f"Layer metadata for {layer.__name__} must be a mapping or sequence of mappings; got {type(value).__name__}."
    )


def _top_layer(annotation: Any) -> Type[Any]:
    base = _strip_annotated(annotation)
    origin = get_origin(base)
    if origin in (Dir, File):
        return origin
    raise InvalidBundleAnnotation("Top-level annotation must be Dir or File")


def _normalize_metadata_layer(
    layer: Type[Any], data: Mapping[str, Any]
) -> Mapping[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "name":
            hint = _normalize_hint_entry("name", value)
            if hint is not None:
                normalized["name"] = hint
            continue
        if key == "extension":
            normalized["extension"] = _normalize_extension_value(value)
            continue
        if key == "prefix":
            hint = _normalize_hint_entry("prefix", value)
            if hint is not None:
                normalized["prefix"] = hint
            continue
        normalized[key] = value
    return MappingProxyType(normalized)


def _normalize_hint_entry(entry_name: str, value: Any) -> Hint | None:
    if value is None:
        return None
    hint_value = value
    hint_source = _DEFAULT_HINT_SOURCE
    if isinstance(value, tuple):
        if len(value) != 2:
            raise BundleMetadataError(
                f"Metadata entry '{entry_name}' must be a value/source tuple."
            )
        hint_value, hint_source = value
    if isinstance(hint_source, str):
        normalized_source = hint_source.strip().lower()
    else:
        normalized_source = ""
    if normalized_source not in _HINT_SOURCES:
        raise BundleMetadataError(
            f"Metadata entry '{entry_name}' source must be one of {_HINT_SOURCES}; got {hint_source!r}."
        )
    hint_source = normalized_source
    if isinstance(hint_value, Hint):
        if hint_value.source not in _HINT_SOURCES:
            raise BundleMetadataError(
                f"Hint source must be one of {_HINT_SOURCES}; got {hint_value.source!r}."
            )
        return hint_value
    if callable(hint_value):
        return Hint(HintKind.CALLABLE, hint_value, hint_source)
    if isinstance(hint_value, str):
        if not hint_value:
            raise BundleMetadataError(
                f"Metadata entry '{entry_name}' cannot be an empty string."
            )
        kind = (
            HintKind.TEMPLATE
            if ("{" in hint_value and "}" in hint_value)
            else HintKind.LITERAL
        )
        return Hint(kind, hint_value, hint_source)
    raise BundleMetadataError(
        f"Unsupported value for '{entry_name}' metadata entry."
    )


def _maybe_attach_extension(
    metadata: Tuple[BundleMetadata, ...], annotation: Any, infer_extension: bool
) -> Tuple[BundleMetadata, ...]:
    if not infer_extension:
        return metadata
    if _metadata_has_extension(metadata):
        return metadata
    inferred = _infer_extension_from_file_annotation(annotation)
    if not inferred:
        return metadata
    normalized_ext = _normalize_extension_value(inferred)
    if metadata:
        updated = list(metadata)
        first = updated[0]
        if first.layer is File:
            merged = dict(first.data)
            merged.setdefault("extension", normalized_ext)
            updated[0] = BundleMetadata(
                layer=first.layer,
                index=first.index,
                data=MappingProxyType(merged),
            )
        else:
            updated.insert(
                0,
                BundleMetadata(
                    layer=File,
                    index=0,
                    data=MappingProxyType({"extension": normalized_ext}),
                ),
            )
        return tuple(updated)
    return (
        BundleMetadata(
            layer=File,
            index=0,
            data=MappingProxyType({"extension": normalized_ext}),
        ),
    )


def _metadata_has_extension(metadata: Tuple[BundleMetadata, ...]) -> bool:
    for meta in metadata:
        if meta.layer is File and "extension" in meta.data:
            return True
    return False


_TEXT_SCALAR_TYPES = (str, int, float)
_LIST_ORIGINS: Tuple[Any, ...] = (list, abc.Sequence, abc.MutableSequence)
_MAPPING_ORIGINS: Tuple[Any, ...] = (dict, abc.Mapping, abc.MutableMapping)


def _infer_extension_from_file_annotation(annotation: Any) -> str | None:
    base = _strip_annotated(annotation)
    origin = get_origin(base)
    if origin is not File:
        return None
    args = get_args(base)
    if not args:
        return None
    payload = args[0]
    hinted = _infer_extension_from_payload(payload)
    return _normalize_extension_value(hinted) if hinted else None


def _infer_extension_from_payload(payload: Any) -> str | None:
    payload = _strip_annotated(payload)
    if _is_text_scalar(payload):
        return "txt"
    if _is_mapping_type(payload):
        return "json"
    origin = get_origin(payload)
    if origin in _LIST_ORIGINS:
        args = get_args(payload)
        if not args:
            return None
        inner = args[0]
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
        if _is_mapping_type(inner):
            return "jsonl"
        if _is_text_scalar(inner):
            return "txt"
    if origin in _MAPPING_ORIGINS:
        return "json"
    if origin is Union:
        if all(_is_text_scalar(arg) for arg in get_args(payload)):
            return "txt"
    return None


def _is_text_scalar(annotation: Any) -> bool:
    ann = _strip_annotated(annotation)
    origin = get_origin(ann)
    if origin is None:
        return ann in _TEXT_SCALAR_TYPES
    if origin is Union:
        args = get_args(ann)
        return bool(args) and all(_is_text_scalar(arg) for arg in args)
    return False


def _is_mapping_type(annotation: Any) -> bool:
    ann = _strip_annotated(annotation)
    if ann in _MAPPING_ORIGINS:
        return True
    origin = get_origin(ann)
    return origin in _MAPPING_ORIGINS if origin is not None else False


def _normalize_extension_value(extension: str) -> str:
    ext = extension.strip()
    if not ext:
        raise BundleMetadataError("Extension values cannot be empty.")
    if ext.startswith("."):
        ext = ext[1:]
    return ext
