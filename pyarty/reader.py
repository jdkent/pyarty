"""Reading a directory back into a bundle instance.

Reading is *lenient* by default, which is the asymmetry that makes the contract
usable against real trees:

* files that no field claims are ignored;
* an ``Optional`` field with no match becomes ``None``;
* a required field with no match raises :class:`MissingFileError`.

Pass ``strict=True`` to also reject unclaimed files, which is how you assert
that a tree contains *exactly* what the schema describes.

Reading is driven entirely by the schema: each field globs for candidates, then
parses them with its own pattern. Nothing is inferred from file extensions, so
``File[dict]`` decodes as JSON because it was *declared* that way.
"""

from __future__ import annotations

from dataclasses import MISSING, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping

from .codecs import describe_annotation
from .errors import MissingFileError, ReadError
from .pattern import PathPattern
from .schema import BundleField, FieldKind, bundle_schema, is_bundle

__all__ = ["read_bundle"]


def read_bundle(
    cls: type[Any], path: str | Path, *, strict: bool = False
) -> Any:
    """Read directory ``path`` into an instance of bundle class ``cls``."""
    if not is_bundle(cls):
        raise ReadError(f"{getattr(cls, '__name__', cls)!r} is not a @bundle class.")

    base = Path(path).expanduser()
    if not base.exists():
        raise ReadError(f"Directory '{base}' does not exist.")
    if not base.is_dir():
        raise ReadError(f"'{base}' is not a directory.")

    claimed: set[Path] = set()
    instance = _read_into(
        cls, base, inherited={}, claimed=claimed, strict=strict, depth=0
    )

    if strict:
        _reject_unclaimed(base, claimed, cls)
    return instance


#: A recursive layout (Dir[list["Self"]]) could otherwise follow a symlink
#: loop forever. Real trees are nowhere near this deep.
MAX_DEPTH = 64


def _read_into(
    cls: type[Any],
    base: Path,
    *,
    inherited: Mapping[str, Any],
    claimed: set[Path],
    strict: bool,
    depth: int = 0,
) -> Any:
    """Build one instance of ``cls`` from directory ``base``.

    ``inherited`` carries pattern variables captured by an ancestor's ``Dir``
    pattern — this is how a name encoded in a directory name lands back in the
    child's own field.
    """
    if depth > MAX_DEPTH:
        raise ReadError(
            f"Recursion limit ({MAX_DEPTH}) exceeded reading '{base}' as "
            f"'{cls.__name__}'. A recursive layout may be following a "
            "symlink loop."
        )
    schema = bundle_schema(cls)
    kwargs: dict[str, Any] = {}
    captured: dict[str, Any] = dict(inherited)

    # Files first: their patterns may capture variables that value fields need.
    for item in schema.by_kind(FieldKind.FILE):
        value, found_vars = _read_file(item, base, cls, claimed)
        if value is not _ABSENT:
            kwargs[item.name] = value
            captured.update(found_vars)

    for item in schema.by_kind(FieldKind.FILES):
        value, found_vars = _read_files(item, base, cls, claimed)
        if value is not _ABSENT:
            kwargs[item.name] = value
            captured.update(found_vars)

    for item in schema.by_kind(FieldKind.DIR):
        value = _read_dir(item, base, cls, captured, claimed, strict, depth)
        if value is not _ABSENT:
            kwargs[item.name] = value

    for item in schema.by_kind(FieldKind.VALUE):
        value = _resolve_value(item, captured, cls, base)
        if value is not _ABSENT:
            kwargs[item.name] = value

    try:
        return cls(**kwargs)
    except TypeError as exc:
        missing = _missing_required(cls, kwargs)
        detail = (
            f" Missing: {', '.join(missing)}." if missing else ""
        )
        raise ReadError(
            f"Could not construct '{cls.__name__}' from '{base}': {exc}.{detail}"
        ) from exc


class _Absent:
    """Marker distinguishing "no value" from a legitimate ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def _read_file(
    item: BundleField,
    base: Path,
    cls: type[Any],
    claimed: set[Path],
) -> tuple[Any, dict[str, Any]]:
    assert item.pattern is not None and item.codec is not None
    pattern = item.pattern
    where = f"{cls.__name__}.{item.name}"

    # A copy field whose pattern has no extension kept the source's on write.
    glob = pattern.glob()
    match_pattern: PathPattern | None = pattern
    if item.is_copy and not pattern.suffix:
        glob = f"{glob}.*"
        match_pattern = None

    matches: list[tuple[Path, dict[str, str]]] = []
    for candidate in sorted(base.glob(glob)):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(base).as_posix()
        if match_pattern is None:
            matches.append((candidate, {}))
            continue
        found = match_pattern.match(relative)
        if found is not None:
            matches.append((candidate, found))

    if not matches:
        if item.optional or item.has_default:
            return (None if item.optional else _ABSENT), {}
        raise MissingFileError(
            f"No file matching '{pattern}' for required field '{where}' "
            f"under '{base}'."
        )

    if len(matches) > 1:
        listed = ", ".join(p.name for p, _ in matches[:5])
        raise ReadError(
            f"Field '{where}' pattern '{pattern}' matched "
            f"{len(matches)} files under '{base}' ({listed}"
            f"{', ...' if len(matches) > 5 else ''}). A File[...] field must "
            "match exactly one; use Dir[list[...]] for repeated structure."
        )

    found_path, found_vars = matches[0]
    claimed.add(found_path)

    if item.is_copy:
        return found_path, found_vars

    raw = found_path.read_bytes()
    try:
        return item.codec.decode(raw), found_vars
    except Exception as exc:
        raise ReadError(
            f"Field '{where}' declared File[{describe_annotation(item.payload)}] "
            f"could not decode '{found_path}' as {item.codec.name}: {exc}"
        ) from exc


def _read_files(
    item: BundleField,
    base: Path,
    cls: type[Any],
    claimed: set[Path],
) -> tuple[Any, dict[str, Any]]:
    """Read every file matching a ``Files`` pattern into a mapping.

    Unlike ``File``, matching many files is the expected case; matching none
    yields an empty mapping rather than an error, since "no shards yet" is a
    normal state for a directory that is still being filled.

    Returns the mapping plus any *non-key* variables the pattern captured.
    Those describe the field as a whole rather than one entry (the ``total`` in
    ``model-{shard}-of-{total}.safetensors``), so they feed sibling value
    fields exactly as a ``File`` pattern's captures do, and must agree across
    every match.
    """
    assert item.pattern is not None and item.codec is not None and item.key
    where = f"{cls.__name__}.{item.name}"
    found: dict[Any, Any] = {}
    shared: dict[str, Any] = {}

    for candidate in sorted(base.glob(item.pattern.glob())):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(base).as_posix()
        variables = item.pattern.match(relative)
        if variables is None or item.key not in variables:
            continue

        for name, value in variables.items():
            if name == item.key:
                continue
            if name in shared and shared[name] != value:
                raise ReadError(
                    f"Field '{where}' pattern '{item.pattern}' captured "
                    f"conflicting values for '{{{name}}}': "
                    f"{shared[name]!r} and {value!r}. A variable other than "
                    f"the key '{{{item.key}}}' describes the whole field, so "
                    "it must be the same in every matching filename."
                )
            shared[name] = value

        key = _coerce(variables[item.key], item.key_type)
        if key in found:
            raise ReadError(
                f"Field '{where}' found two files with key {key!r} under "
                f"'{base}'. The key variable '{{{item.key}}}' must be unique "
                "across matches."
            )
        claimed.add(candidate)

        if item.is_copy:
            found[key] = candidate
            continue
        try:
            found[key] = item.codec.decode(candidate.read_bytes())
        except Exception as exc:
            raise ReadError(
                f"Field '{where}' declared Files[..., "
                f"{describe_annotation(item.payload)}] could not decode "
                f"'{candidate}' as {item.codec.name}: {exc}"
            ) from exc

    if not found and item.has_default:
        return _ABSENT, shared
    return found, shared


# ----------------------------------------------------------------------
# Directories
# ----------------------------------------------------------------------
def _read_dir(
    item: BundleField,
    base: Path,
    cls: type[Any],
    captured: Mapping[str, Any],
    claimed: set[Path],
    strict: bool,
    depth: int = 0,
) -> Any:
    assert item.pattern is not None and item.child is not None
    pattern = item.pattern
    where = f"{cls.__name__}.{item.name}"

    matches: list[tuple[Path, dict[str, str]]] = []
    for candidate in sorted(base.glob(pattern.glob())):
        if not candidate.is_dir():
            continue
        relative = candidate.relative_to(base).as_posix()
        found = pattern.match(relative)
        if found is not None:
            matches.append((candidate, found))

    if item.is_collection:
        children = [
            _read_into(
                item.child,
                directory,
                inherited={**captured, **found},
                claimed=claimed,
                strict=strict,
                depth=depth + 1,
            )
            for directory, found in matches
        ]
        if not children and (item.optional or item.has_default):
            return _ABSENT if item.has_default else []
        return children

    if not matches:
        if item.optional or item.has_default:
            return None if item.optional else _ABSENT
        raise MissingFileError(
            f"No directory matching '{pattern}' for required field '{where}' "
            f"under '{base}'."
        )
    if len(matches) > 1:
        raise ReadError(
            f"Field '{where}' pattern '{pattern}' matched {len(matches)} "
            f"directories under '{base}'. Declare it as Dir[list[...]] to "
            "accept more than one."
        )

    directory, found = matches[0]
    return _read_into(
        item.child,
        directory,
        inherited={**captured, **found},
        claimed=claimed,
        strict=strict,
        depth=depth + 1,
    )


# ----------------------------------------------------------------------
# Plain values
# ----------------------------------------------------------------------
def _resolve_value(
    item: BundleField,
    captured: Mapping[str, Any],
    cls: type[Any],
    base: Path,
) -> Any:
    if item.name in captured:
        return _coerce(captured[item.name], item.annotation)
    if item.has_default:
        return _ABSENT
    if item.optional:
        return None
    raise ReadError(
        f"Field '{item.name}' on '{cls.__name__}' is a plain value, but no "
        f"pattern captured '{{{item.name}}}' while reading '{base}', so its "
        f"value cannot be recovered.\n"
        f"  Reference it in a sibling pattern — e.g. "
        f"at(\"{{{item.name}}}.txt\") — or have the parent name the directory "
        f"with at(\".../{{{item.name}}}\"), or give the field a default."
    )


def _coerce(raw: Any, annotation: Any) -> Any:
    """Convert a captured path fragment to the field's declared type.

    Pattern captures are always strings; a field declared ``int`` should get an
    ``int`` back so the round trip compares equal.
    """
    from .codecs import unwrap_optional

    target, _ = unwrap_optional(annotation)
    if not isinstance(raw, str) or target is str:
        return raw
    if target is int:
        try:
            return int(raw)
        except ValueError:
            return raw
    if target is float:
        try:
            return float(raw)
        except ValueError:
            return raw
    if target is bool:
        lowered = raw.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    return raw


# ----------------------------------------------------------------------
# Strict mode
# ----------------------------------------------------------------------
def _reject_unclaimed(base: Path, claimed: set[Path], cls: type[Any]) -> None:
    unclaimed = sorted(
        p for p in base.rglob("*") if p.is_file() and p not in claimed
    )
    if not unclaimed:
        return
    listed = ", ".join(p.relative_to(base).as_posix() for p in unclaimed[:5])
    raise ReadError(
        f"strict=True: '{base}' contains {len(unclaimed)} file(s) that no "
        f"field of '{cls.__name__}' claims ({listed}"
        f"{', ...' if len(unclaimed) > 5 else ''}). "
        f"Declare them, or read with strict=False to ignore them."
    )


def _missing_required(cls: type[Any], supplied: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for item in dataclass_fields(cls):
        if item.name in supplied:
            continue
        if item.default is MISSING and item.default_factory is MISSING:
            missing.append(item.name)
    return missing
