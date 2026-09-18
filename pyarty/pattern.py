"""Bidirectional path patterns.

A :class:`PathPattern` is the single source of truth for where a field lives on
disk. Unlike a ``str.format`` template, it works in *both* directions:

    >>> p = PathPattern.parse("reports/{name}.txt")
    >>> p.format({"name": "alpha"})
    'reports/alpha.txt'
    >>> p.match("reports/alpha.txt")
    {'name': 'alpha'}

That invertibility is what makes ``Bundle.read(Bundle.write(x)) == x`` possible.

Three pieces of syntax:

``{name}``
    A variable, captured on read and substituted on write.
``{name:charset}``
    A variable restricted to a character class. Without one a variable matches
    anything but ``/``, which silently mis-parses delimited names: matching
    ``sub-{sub}_task-{task}_bold.nii`` against
    ``sub-01_ses-pre_task-rest_bold.nii`` otherwise captures
    ``sub='01_ses-pre'``. ``{sub:alnum}`` makes that a clean non-match.
``[...]``
    An optional group, dropped on write when its variables are all ``None``.
    Needed wherever a naming scheme has optional components — BIDS entities,
    a wheel's build tag. A bracket pair containing no variable is a literal,
    so ``a[1].txt`` still means the file called ``a[1].txt``.

Patterns are always relative and always use ``/`` as the separator, on every
platform; they are resolved against a base directory at the last moment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterator, Mapping

from .errors import PatternError

__all__ = ["PathPattern", "CHARSETS"]


# A `{variable}` or `{variable:charset}` reference.
_TOKEN_RE = re.compile(r"\{([^{}]*)\}")

#: Named character classes for variables. ``any`` is non-greedy because it can
#: span delimiters and needs backtracking; the constrained classes are greedy
#: because they cannot cross their own delimiters, which makes matching both
#: faster and unambiguous.
CHARSETS: Mapping[str, str] = {
    "any": r"[^/]+?",
    "word": r"[0-9A-Za-z_]+",
    "alnum": r"[0-9A-Za-z]+",
    "alpha": r"[A-Za-z]+",
    "digits": r"[0-9]+",
    "hex": r"[0-9a-fA-F]+",
}

DEFAULT_CHARSET = "any"

#: Anchored forms used to validate a value on write.
_CHARSET_VALIDATORS = {
    name: re.compile(r"\A" + expr.rstrip("?") + r"\Z")
    for name, expr in CHARSETS.items()
}

#: Characters `pathlib.glob` treats specially; a literal must escape them, or a
#: name like "a[1].txt" writes but never reads back. Bracketing is glob's own
#: escape mechanism.
_GLOB_SPECIAL = {"*": "[*]", "?": "[?]", "[": "[[]"}


def _glob_escape(text: str) -> str:
    return "".join(_GLOB_SPECIAL.get(char, char) for char in text)


# ----------------------------------------------------------------------
# Pattern nodes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Literal:
    text: str


@dataclass(frozen=True)
class _Var:
    name: str
    charset: str = DEFAULT_CHARSET


@dataclass(frozen=True)
class _Optional:
    """A ``[...]`` group, present only when its variables have values."""

    parts: tuple["_Node", ...]


_Node = Any  # _Literal | _Var | _Optional


def _walk(parts: tuple[_Node, ...]) -> Iterator[_Node]:
    for part in parts:
        yield part
        if isinstance(part, _Optional):
            yield from _walk(part.parts)


def _variables_in(parts: tuple[_Node, ...]) -> tuple[str, ...]:
    return tuple(p.name for p in _walk(parts) if isinstance(p, _Var))


@dataclass(frozen=True)
class _Segment:
    """One ``/``-delimited component."""

    parts: tuple[_Node, ...]

    @property
    def variables(self) -> tuple[str, ...]:
        return _variables_in(self.parts)

    @property
    def is_dynamic(self) -> bool:
        return bool(self.variables)


@dataclass(frozen=True)
class PathPattern:
    """A relative path template that can both render and parse."""

    raw: str
    segments: tuple[_Segment, ...]
    variables: tuple[str, ...]
    optional_variables: frozenset[str]
    _regex: re.Pattern[str] = field(repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, raw: str, *, charset: str | None = None) -> "PathPattern":
        """Parse ``raw``; ``charset`` is the default for unqualified variables."""
        if not isinstance(raw, str):
            raise PatternError(
                f"Path pattern must be a string; got {type(raw).__name__}."
            )
        default_charset = charset or DEFAULT_CHARSET
        if default_charset not in CHARSETS:
            raise PatternError(
                f"Unknown charset {default_charset!r}. "
                f"Available: {', '.join(sorted(CHARSETS))}."
            )

        text = raw.strip()
        if not text:
            raise PatternError("Path pattern cannot be empty.")
        if text.startswith("/"):
            raise PatternError(
                f"Path pattern {raw!r} must be relative; remove the leading '/'."
            )
        if "\\" in text:
            raise PatternError(
                f"Path pattern {raw!r} must use '/' as the separator, even on Windows."
            )

        segments: list[_Segment] = []
        seen: list[str] = []
        for raw_segment in text.split("/"):
            if not raw_segment:
                raise PatternError(
                    f"Path pattern {raw!r} contains an empty path segment."
                )
            if raw_segment in (".", ".."):
                raise PatternError(
                    f"Path pattern {raw!r} may not contain '.' or '..' segments."
                )
            parts = _parse_parts(raw_segment, raw, default_charset)
            for name in _variables_in(parts):
                # A variable may repeat ("sub-{id}/anat/sub-{id}_T1w.nii");
                # later occurrences become backreferences.
                if name not in seen:
                    seen.append(name)
            segments.append(_Segment(parts=parts))

        optional = {
            name
            for segment in segments
            for part in segment.parts
            if isinstance(part, _Optional)
            for name in _variables_in(part.parts)
        }
        return cls(
            raw=text,
            segments=tuple(segments),
            variables=tuple(seen),
            optional_variables=frozenset(optional),
            _regex=_build_regex(segments),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def format(self, values: Mapping[str, Any]) -> str:
        """Render the pattern to a concrete relative path."""
        return "/".join(
            self._render_parts(segment.parts, values) for segment in self.segments
        )

    def _render_parts(
        self, parts: tuple[_Node, ...], values: Mapping[str, Any]
    ) -> str:
        out: list[str] = []
        for part in parts:
            if isinstance(part, _Literal):
                out.append(part.text)
            elif isinstance(part, _Var):
                out.append(self._render_var(part, values))
            else:  # _Optional
                names = _variables_in(part.parts)
                supplied = [
                    n for n in names if values.get(n) is not None
                ]
                if not supplied:
                    continue  # whole group omitted
                missing = [n for n in names if values.get(n) is None]
                if missing:
                    raise PatternError(
                        f"Pattern {self.raw!r}: optional group needs either all "
                        f"or none of {', '.join(names)}; "
                        f"{', '.join(missing)} is None while "
                        f"{', '.join(supplied)} is set."
                    )
                out.append(self._render_parts(part.parts, values))
        return "".join(out)

    def _render_var(self, var: _Var, values: Mapping[str, Any]) -> str:
        if var.name not in values:
            raise PatternError(
                f"Pattern {self.raw!r} needs a value for '{{{var.name}}}' "
                f"but none was supplied."
            )
        value = values[var.name]
        if value is None:
            raise PatternError(
                f"Pattern {self.raw!r} needs a value for '{{{var.name}}}' "
                f"but it is None."
            )
        text = str(value)
        if not text:
            raise PatternError(
                f"Pattern {self.raw!r} got an empty value for '{{{var.name}}}'."
            )
        if "/" in text or "\\" in text:
            raise PatternError(
                f"Pattern {self.raw!r} got {text!r} for '{{{var.name}}}', but a "
                "path separator cannot appear inside a single path component."
            )
        if not _CHARSET_VALIDATORS[var.charset].match(text):
            raise PatternError(
                f"Pattern {self.raw!r} got {text!r} for "
                f"'{{{var.name}:{var.charset}}}', which is not a valid "
                f"{var.charset} value ({CHARSETS[var.charset]})."
            )
        return text

    def match(self, relative_path: str | PurePosixPath) -> dict[str, str] | None:
        """Parse a relative path, returning captured variables or ``None``.

        Variables inside an omitted optional group are absent from the result,
        rather than present-and-``None``, so callers can tell the difference.
        """
        found = self._regex.match(str(PurePosixPath(relative_path)))
        if found is None:
            return None
        return {k: v for k, v in found.groupdict().items() if v is not None}

    def glob(self) -> str:
        """A glob that over-matches the pattern, for discovering candidates.

        Always paired with :meth:`match` to reject false positives. An optional
        group becomes ``*``, since a glob cannot express optionality.
        """
        rendered: list[str] = []
        for segment in self.segments:
            piece = "".join(_glob_parts(segment.parts))
            # Collapse runs of '*' that optional groups may have introduced.
            rendered.append(re.sub(r"\*{2,}", "*", piece))
        return "/".join(rendered)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def suffix(self) -> str:
        """Trailing extension of the final segment, e.g. ``'.json'``."""
        return PurePosixPath(self.glob()).suffix

    @property
    def required_variables(self) -> tuple[str, ...]:
        return tuple(v for v in self.variables if v not in self.optional_variables)

    @property
    def depth(self) -> int:
        return len(self.segments)

    @property
    def is_dynamic(self) -> bool:
        return bool(self.variables)

    @property
    def names_distinct_children(self) -> bool:
        """Whether the final segment varies per child.

        ``Dir[list[T]]`` requires this; without it every element renders to the
        same directory and all but the last are silently lost.
        """
        return self.segments[-1].is_dynamic

    def with_suffix(self, suffix: str) -> "PathPattern":
        """Return a copy whose final segment ends in ``suffix``."""
        if not suffix:
            return self
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if self.suffix == suffix:
            return self
        return PathPattern.parse(f"{self.raw}{suffix}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def _parse_parts(
    raw_segment: str, raw: str, default_charset: str
) -> tuple[_Node, ...]:
    """Split one path segment into literals, variables and optional groups."""
    parts: list[_Node] = []
    buffer: list[str] = []
    index = 0
    length = len(raw_segment)

    def flush() -> None:
        if buffer:
            parts.append(_Literal("".join(buffer)))
            buffer.clear()

    while index < length:
        char = raw_segment[index]

        if char == "[":
            close = _matching_bracket(raw_segment, index)
            inner = raw_segment[index + 1 : close] if close is not None else ""
            # Brackets only mean "optional" when they wrap a variable; that
            # keeps a literal name like "a[1].txt" working.
            if close is not None and _TOKEN_RE.search(inner):
                flush()
                parts.append(
                    _Optional(parts=_parse_parts(inner, raw, default_charset))
                )
                index = close + 1
                continue
            buffer.append(char)
            index += 1
            continue

        if char == "{":
            token = _TOKEN_RE.match(raw_segment, index)
            if token is None:
                raise PatternError(
                    f"Path pattern {raw!r} has an unbalanced '{{' or '}}'."
                )
            flush()
            parts.append(_parse_var(token.group(1), raw, default_charset))
            index = token.end()
            continue

        if char == "}":
            raise PatternError(
                f"Path pattern {raw!r} has an unbalanced '{{' or '}}'."
            )

        buffer.append(char)
        index += 1

    flush()
    return tuple(parts)


def _matching_bracket(text: str, start: int) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_var(body: str, raw: str, default_charset: str) -> _Var:
    name, _, charset = body.partition(":")
    name = name.strip()
    charset = charset.strip() or default_charset
    if not name:
        raise PatternError(
            f"Path pattern {raw!r} contains an empty '{{}}' placeholder."
        )
    if not name.isidentifier():
        raise PatternError(
            f"Path pattern {raw!r} variable '{{{name}}}' is not a valid Python "
            "identifier, so it cannot map onto a field."
        )
    if charset not in CHARSETS:
        raise PatternError(
            f"Path pattern {raw!r} variable '{{{name}}}' uses unknown charset "
            f"{charset!r}. Available: {', '.join(sorted(CHARSETS))}."
        )
    return _Var(name=name, charset=charset)


def _build_regex(segments: list[_Segment]) -> re.Pattern[str]:
    bound: set[str] = set()

    def render(parts: tuple[_Node, ...]) -> str:
        out: list[str] = []
        for part in parts:
            if isinstance(part, _Literal):
                out.append(re.escape(part.text))
            elif isinstance(part, _Var):
                if part.name in bound:
                    # A repeated variable must match the same text every time.
                    out.append(rf"(?P={part.name})")
                else:
                    out.append(rf"(?P<{part.name}>{CHARSETS[part.charset]})")
                    bound.add(part.name)
            else:  # _Optional
                out.append(f"(?:{render(part.parts)})?")
        return "".join(out)

    body = "/".join(render(segment.parts) for segment in segments)
    return re.compile(r"\A" + body + r"\Z")


def _glob_parts(parts: tuple[_Node, ...]) -> list[str]:
    out: list[str] = []
    for part in parts:
        if isinstance(part, _Literal):
            out.append(_glob_escape(part.text))
        elif isinstance(part, _Var):
            out.append("*")
        else:  # _Optional — a glob cannot express optionality
            out.append("*")
    return out
