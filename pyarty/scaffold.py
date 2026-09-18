"""Generate bundle source code from an existing directory tree.

The old ``infer_bundle_from_directory`` built anonymous dataclasses at runtime,
which could not be the class you already had and could not be edited. This
emits **Python source** instead: paste it into your project, rename what you
like, and it is a normal bundle.

The interesting part is collapsing repetition. Given::

    corpus/
      reports/alpha/{body.txt,metrics.json}
      reports/beta/{body.txt,metrics.json}
      summary.json

sibling directories with identical shapes become one class plus a
``Dir[list[...]]`` field, rather than one class per directory::

    @bundle
    class Report:
        name: str
        body: File[str] = at("{name}.txt")
        metrics: File[dict] = at("metrics.json")
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ReadError

__all__ = ["scaffold_from_directory"]


#: Extension -> (annotation, decoder for shape sniffing)
_KNOWN_TEXT = {".txt", ".md", ".rst", ".log"}


@dataclass
class _Node:
    """A directory in the scanned tree."""

    path: Path
    files: list[Path]
    subdirs: list["_Node"]

    def shape(self) -> str:
        """A structural fingerprint used to detect sibling repetition.

        A file whose stem matches its own directory name is normalized to
        ``{name}``, so ``alpha/alpha.txt`` and ``beta/beta.txt`` fingerprint
        identically and collapse into one class. Without this, any tree that
        names a file after its directory — including everything pyarty itself
        writes from a ``{name}.txt`` pattern — would yield one class per
        directory.
        """
        file_part = sorted(self.token_for(f) for f in self.files)
        dir_part = sorted(child.shape() for child in self.subdirs)
        return json.dumps([file_part, dir_part], sort_keys=True)

    def is_self_named(self, file: Path) -> bool:
        """Whether ``file``'s stem is its containing directory's name."""
        return bool(self.path.name) and file.stem == self.path.name

    def token_for(self, file: Path) -> str:
        suffix = file.suffix.lower()
        if self.is_self_named(file):
            return f"{{name}}{suffix}"
        return f"{file.stem}{suffix}"


def scaffold_from_directory(
    directory: str | Path,
    *,
    root_class_name: str = "Root",
    max_depth: int = 8,
) -> str:
    """Return Python source declaring bundles that describe ``directory``.

    The result is a complete module: imports, class definitions in dependency
    order, and a short usage comment.
    """
    base = Path(directory).expanduser()
    if not base.exists() or not base.is_dir():
        raise ReadError(f"'{base}' does not exist or is not a directory.")

    tree = _scan(base, depth=0, max_depth=max_depth)
    builder = _Builder(root_class_name)
    builder.emit(tree, root_class_name)
    return builder.render(base)


def _scan(path: Path, *, depth: int, max_depth: int) -> _Node:
    if depth > max_depth:
        return _Node(path=path, files=[], subdirs=[])
    files: list[Path] = []
    subdirs: list[_Node] = []
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subdirs.append(_scan(entry, depth=depth + 1, max_depth=max_depth))
        elif entry.is_file():
            files.append(entry)
    return _Node(path=path, files=files, subdirs=subdirs)


class _Builder:
    def __init__(self, root_class_name: str) -> None:
        self.root_class_name = root_class_name
        #: Rendered class bodies, in emission order (children before parents).
        self.classes: list[tuple[str, list[str]]] = []
        self.used_names: set[str] = set()

    # ------------------------------------------------------------------
    def emit(self, node: _Node, class_name: str, *, named: bool = False) -> str:
        """Emit a class for ``node``, returning its final name.

        ``named`` marks a class whose directory name varies per instance; such
        a class gets a ``name`` field, and files named after the directory are
        written as ``{name}`` patterns so the value round-trips.
        """
        name = self._unique(class_name)
        declared: list[str] = []

        value_lines: list[str] = []
        if named:
            declared.append("name")
            value_lines.append("    name: str  # <- the directory name")

        file_lines: list[str] = []
        for file in node.files:
            annotation = _annotation_for(file)
            if named and node.is_self_named(file):
                pattern = f"{{name}}{file.suffix}"
                field = self._unique_field(_snake(_stem_hint(node, file)), declared)
            else:
                pattern = file.name
                field = self._unique_field(_snake(file.stem), declared)
            declared.append(field)
            file_lines.append(
                f'    {field}: File[{annotation}] = at("{pattern}")'
            )

        dir_lines: list[str] = []
        for members in _group_by_shape(node.subdirs):
            if len(members) > 1:
                # Sibling repetition: one class, one list field, {name} per dir.
                dir_lines.append(
                    self._emit_repeated(members, prefix=None, declared=declared)
                )
                continue

            member = members[0]
            collapsed = _repeated_container(member)
            if collapsed is not None:
                # `member` only holds repeated siblings, so fold it into the
                # pattern as "<dir>/{name}" instead of emitting a wrapper class.
                dir_lines.append(
                    self._emit_repeated(
                        collapsed, prefix=member.path.name, declared=declared
                    )
                )
                continue

            child_name = self.emit(member, _camel(member.path.name))
            field = self._unique_field(_snake(member.path.name), declared)
            declared.append(field)
            dir_lines.append(
                f'    {field}: Dir[{child_name}] = at("{member.path.name}")'
            )

        body = value_lines + file_lines + dir_lines or ["    pass"]
        self.classes.append((name, body))
        return name

    def _emit_repeated(
        self, members: list[_Node], *, prefix: str | None, declared: list[str]
    ) -> str:
        """Emit ``Dir[list[Child]]`` for a group of same-shaped directories."""
        stem = _common_stem(members)
        child_name = self.emit(members[0], _camel(stem), named=True)

        field_hint = prefix if prefix else _plural(stem)
        field = self._unique_field(_snake(field_hint), declared)
        declared.append(field)
        pattern = f"{prefix}/{{name}}" if prefix else "{name}"
        return f'    {field}: Dir[list[{child_name}]] = at("{pattern}")'

    # ------------------------------------------------------------------
    def _unique(self, candidate: str) -> str:
        base = candidate or "Node"
        name = base
        counter = 1
        while name in self.used_names:
            counter += 1
            name = f"{base}{counter}"
        self.used_names.add(name)
        return name

    @staticmethod
    def _unique_field(candidate: str, declared: list[str]) -> str:
        """Pick a field name not already used in this class."""
        base = candidate or "item"
        name = base
        counter = 1
        while name in declared:
            counter += 1
            name = f"{base}_{counter}"
        return name

    # ------------------------------------------------------------------
    def render(self, base: Path) -> str:
        header = [
            '"""Generated by pyarty.scaffold_from_directory().',
            "",
            f"Describes the layout of: {base.name}/",
            "Edit freely — this is ordinary Python.",
            '"""',
            "",
            "from pyarty import Dir, File, at, bundle",
            "",
            "",
            "",
        ]
        # Emission order already appends children before parents, which is
        # exactly the order Python needs: a class is defined before the class
        # that references it.
        chunks = [
            "\n".join(["@bundle", f"class {name}:", *body])
            for name, body in self.classes
        ]

        usage = [
            "",
            "",
            "# Round trip:",
            f"#   data = {self.root_class_name}.read({base.name!r})",
            '#   data.write("./copy")',
        ]
        return "\n".join(header) + "\n\n\n".join(chunks) + "\n".join(usage) + "\n"


# ----------------------------------------------------------------------
# Shape grouping
# ----------------------------------------------------------------------
def _group_by_shape(nodes: list[_Node]) -> list[list[_Node]]:
    """Partition sibling directories by structural fingerprint, order-stable."""
    groups: dict[str, list[_Node]] = {}
    for node in nodes:
        groups.setdefault(node.shape(), []).append(node)
    return list(groups.values())


def _repeated_container(node: _Node) -> list[_Node] | None:
    """Return ``node``'s children if it exists only to hold repeated siblings.

    ``reports/{alpha,beta}`` is such a container: it has no files of its own
    and its children all share one shape, so it collapses into the parent's
    pattern as ``reports/{name}`` rather than needing a class of its own.
    """
    if node.files or len(node.subdirs) < 2:
        return None
    groups = _group_by_shape(node.subdirs)
    if len(groups) != 1:
        return None
    return groups[0]


# ----------------------------------------------------------------------
# Annotation inference
# ----------------------------------------------------------------------
def _annotation_for(file: Path) -> str:
    suffix = file.suffix.lower()
    if suffix == ".json":
        return _json_annotation(file)
    if suffix == ".jsonl":
        return "list[dict]"
    if suffix in _KNOWN_TEXT:
        return "str"
    # Unknown/binary: copy it verbatim rather than guess an encoding.
    return "bytes"


def _json_annotation(file: Path) -> str:
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "dict"
    if isinstance(payload, dict):
        return "dict"
    if isinstance(payload, list):
        if payload and all(isinstance(row, dict) for row in payload):
            return "list[dict]"
        return "list"
    return "dict"


# ----------------------------------------------------------------------
# Naming helpers
# ----------------------------------------------------------------------
def _stem_hint(node: _Node, file: Path) -> str:
    """A field name for a file named after its own directory.

    The stem is the varying directory name, so it makes a poor field name;
    fall back to the extension ("body", "data") instead.
    """
    suffix = file.suffix.lstrip(".").lower()
    return {"txt": "body", "json": "data", "jsonl": "rows"}.get(suffix, suffix or "body")


def _common_stem(nodes: list[_Node]) -> str:
    """A class-name hint for a group of repeated sibling directories."""
    parent = nodes[0].path.parent.name
    if parent:
        return _singular(parent)
    return _singular(nodes[0].path.name)


def _singular(value: str) -> str:
    if value.endswith("ies") and len(value) > 3:
        return f"{value[:-3]}y"
    if value.endswith("ses") or value.endswith("xes"):
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _plural(value: str) -> str:
    if value.endswith("y") and len(value) > 1:
        return f"{value[:-1]}ies"
    if value.endswith(("s", "x", "z", "ch", "sh")):
        return f"{value}es"
    return f"{value}s"


def _camel(value: str) -> str:
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", value) if t]
    if not tokens:
        return "Node"
    joined = "".join(token[:1].upper() + token[1:] for token in tokens)
    return f"N{joined}" if joined[0].isdigit() else joined


def _snake(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
    if not text:
        text = "item"
    if text[0].isdigit():
        text = f"n_{text}"
    if keyword.iskeyword(text):
        text = f"{text}_"
    return text
