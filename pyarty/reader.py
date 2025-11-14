"""Directory-to-bundle inference utilities."""

from __future__ import annotations

import json
import keyword
import re
from collections import defaultdict
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .dsl import Dir, File, bundle, twig


__all__ = ["InferredBundle", "infer_bundle_from_directory"]


SUPPORTED_EXTENSIONS = {".txt", ".json", ".jsonl"}


@dataclass(frozen=True)
class InferredBundle:
    """Container for dynamically inferred bundle information."""

    root_class: type[Any]
    instance: Any
    schema: Mapping[str, Any]


def infer_bundle_from_directory(
    directory: str | Path, *, root_class_name: str | None = None
) -> InferredBundle:
    """Infer a bundle dataclass tree and JSON Schema from ``directory``.

    The resulting dataclass instances can be rendered back to disk via ``.write``.
    """

    root_path = Path(directory).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Directory '{root_path}' does not exist or is not a directory.")

    builder = _BundleBuilder(root_path, root_class_name)
    root_cls, instance = builder.build()
    schema = builder.schema()
    return InferredBundle(root_cls, instance, schema)


class _BundleBuilder:
    def __init__(self, root_path: Path, root_override: str | None) -> None:
        self.root_path = root_path
        self.root_override = root_override
        self._class_cache: dict[Path, type[Any]] = {}
        self._instance_cache: dict[Path, Any] = {}
        self._schema_defs: dict[str, Mapping[str, Any]] = {}
        self._name_counts: MutableMapping[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> tuple[type[Any], Any]:
        return self._build_dir(self.root_path, preferred_name=self.root_override)

    def schema(self) -> Mapping[str, Any]:
        root_class = self._class_cache[self.root_path]
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"pyarty://{self.root_path.name}",
            "$ref": f"#/$defs/{root_class.__name__}",
            "$defs": dict(self._schema_defs),
        }

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_dir(
        self, path: Path, *, preferred_name: str | None = None
    ) -> tuple[type[Any], Any]:
        if path in self._class_cache:
            return self._class_cache[path], self._instance_cache[path]

        class_name = self._unique_class_name(
            preferred_name or (path.name or "root"),
            trust_input=preferred_name is not None,
        )

        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        fields: list[tuple[str, Any, Any]] = []
        init_kwargs: dict[str, Any] = {}
        schema_properties: dict[str, Any] = {}
        required_fields: list[str] = []

        used_field_names: set[str] = set()

        for entry in entries:
            if entry.is_dir():
                child_cls, child_instance = self._build_dir(entry)
                field_name = _unique_field_name(
                    used_field_names, _snake_case(entry.name)
                )
                fields.append(
                    (
                        field_name,
                        Dir[child_cls],
                        twig(name=entry.name),
                    )
                )
                init_kwargs[field_name] = child_instance
                schema_properties[field_name] = {
                    "$ref": f"#/$defs/{child_cls.__name__}",
                    "description": f"Directory '{entry.name}'",
                    "x-pyarty": {
                        "kind": "dir",
                        "path": self._relative(entry),
                        "name": entry.name,
                    },
                }
                required_fields.append(field_name)
                continue

            if entry.is_file():
                suffix = entry.suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported file extension '{entry.suffix}' in '{entry}'."
                    )
                annotation, value, schema = self._build_file(entry)
                field_name = _unique_field_name(
                    used_field_names, _snake_case(entry.stem)
                )
                fields.append(
                    (
                        field_name,
                        annotation,
                        twig(name=entry.stem, extension=suffix.lstrip(".")),
                    )
                )
                init_kwargs[field_name] = value
                schema["x-pyarty"] = {
                    "kind": "file",
                    "path": self._relative(entry),
                    "name": entry.name,
                    "extension": suffix.lstrip("."),
                }
                schema_properties[field_name] = schema
                required_fields.append(field_name)

        namespace = {"__module__": __name__}
        dataclass_type = make_dataclass(class_name, fields, namespace=namespace)
        bundle_class = bundle(dataclass_type)
        instance = bundle_class(**init_kwargs)

        self._class_cache[path] = bundle_class
        self._instance_cache[path] = instance
        self._schema_defs[bundle_class.__name__] = self._directory_schema(
            bundle_class.__name__, schema_properties, required_fields, path
        )
        return bundle_class, instance

    def _build_file(self, path: Path) -> tuple[Any, Any, Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            contents = path.read_text(encoding="utf-8")
            return File[str], contents, {"type": "string"}
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            annotation = File[self._annotation_for_json_value(payload)]
            schema = _infer_json_schema(payload)
            return annotation, payload, schema
        if suffix == ".jsonl":
            payload_list: list[Any] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.rstrip("\n\r")
                    if not stripped:
                        continue
                    payload_list.append(json.loads(stripped))
            annotation = File[List[self._annotation_for_jsonl(payload_list)]]
            schema = {
                "type": "array",
                "items": _merge_schemas(
                    _infer_json_schema(entry) for entry in payload_list
                ),
            }
            return annotation, payload_list, schema
        raise ValueError(f"Unsupported file extension '{path.suffix}'.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _unique_class_name(self, base: str, *, trust_input: bool = False) -> str:
        sanitized = (
            _sanitize_class_name(base) if trust_input else _camelcase(base)
        )
        count = self._name_counts[sanitized]
        self._name_counts[sanitized] += 1
        if count == 0:
            return sanitized
        return f"{sanitized}{count+1}"

    def _relative(self, target: Path) -> str:
        rel = target.relative_to(self.root_path)
        rel_str = rel.as_posix()
        return rel_str if rel_str else "."

    def _directory_schema(
        self,
        class_name: str,
        properties: Mapping[str, Any],
        required: Sequence[str],
        path: Path,
    ) -> Mapping[str, Any]:
        schema: Dict[str, Any] = {
            "title": class_name,
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "x-pyarty": {
                "kind": "dir",
                "path": self._relative(path),
                "name": path.name or "root",
            },
        }
        if required:
            schema["required"] = list(required)
        return schema

    def _annotation_for_json_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return Dict[str, Any]
        if isinstance(value, list):
            return List[Any]
        if isinstance(value, bool):
            return bool
        if isinstance(value, int):
            return int
        if isinstance(value, float):
            return float
        if value is None:
            return Any
        return str

    def _annotation_for_jsonl(self, items: list[Any]) -> Any:
        if not items:
            return Dict[str, Any]
        first = items[0]
        if isinstance(first, dict):
            return Dict[str, Any]
        if isinstance(first, list):
            return List[Any]
        if isinstance(first, bool):
            return bool
        if isinstance(first, int):
            return int
        if isinstance(first, float):
            return float
        if first is None:
            return Any
        return str


def _merge_schemas(schemas: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    unique: list[Mapping[str, Any]] = []
    signatures: set[str] = set()
    for schema in schemas:
        key = json.dumps(schema, sort_keys=True)
        if key in signatures:
            continue
        signatures.add(key)
        unique.append(schema)
    if not unique:
        return {}
    if len(unique) == 1:
        return unique[0]
    return {"anyOf": unique}


def _infer_json_schema(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        schema: Dict[str, Any] = {"type": "array"}
        if value:
            schema["items"] = _merge_schemas(
                _infer_json_schema(entry) for entry in value
            )
        return schema
    if isinstance(value, dict):
        properties = {
            key: _infer_json_schema(inner) for key, inner in sorted(value.items())
        }
        schema = {"type": "object"}
        if properties:
            schema["properties"] = properties
            schema["required"] = list(properties.keys())
        return schema
    return {"type": "string"}


def _camelcase(value: str) -> str:
    tokens = re.split(r"[^0-9a-zA-Z]+", value)
    filtered = [token for token in tokens if token]
    if not filtered:
        return "Node"
    combined = "".join(token.capitalize() for token in filtered)
    if combined[0].isdigit():
        combined = f"N{combined}"
    return combined


def _sanitize_class_name(value: str) -> str:
    if not value:
        return "Node"
    filtered = re.sub(r"[^0-9a-zA-Z]+", "", value)
    if not filtered:
        filtered = "Node"
    if filtered[0].isdigit():
        filtered = f"N{filtered}"
    return filtered


def _snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    lowered = value.lower() or "node"
    if lowered[0].isdigit():
        lowered = f"n_{lowered}"
    if keyword.iskeyword(lowered):
        lowered = f"{lowered}_" 
    return lowered


def _unique_field_name(existing: set[str], candidate: str) -> str:
    base = candidate
    counter = 1
    while candidate in existing:
        counter += 1
        candidate = f"{base}_{counter}"
    existing.add(candidate)
    return candidate
