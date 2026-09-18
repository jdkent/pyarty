"""Bidirectional path patterns.

A :class:`PathPattern` is the single source of truth for where a field lives on
disk. Unlike a ``str.format`` template, it works in *both* directions:

    >>> p = PathPattern("reports/{name}.txt")
    >>> p.format({"name": "alpha"})
    'reports/alpha.txt'
    >>> p.match("reports/alpha.txt")
    {'name': 'alpha'}

That invertibility is what makes ``Bundle.read(Bundle.write(x)) == x`` possible.
Patterns are always relative and always use ``/`` as the separator, on every
platform; they are resolved against a base directory at the last moment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from .errors import PatternError

__all__ = ["PathPattern"]


# A `{variable}` reference. Names must be valid Python identifiers so they can
# map onto dataclass fields.
_TOKEN_RE = re.compile(r"\{([^{}]*)\}")

#: Characters that `pathlib.glob` treats specially. A literal part of a pattern
#: must have them escaped, or a name like "a[1].txt" writes but never reads
#: back. Bracketing is glob's own escape for these.
_GLOB_SPECIAL = {"*": "[*]", "?": "[?]", "[": "[[]"}


def _glob_escape(text: str) -> str:
    return "".join(_GLOB_SPECIAL.get(char, char) for char in text)


@dataclass(frozen=True)
class _Segment:
    """One ``/``-delimited component, split into literal and variable parts."""

    #: ``(is_variable, text)`` pairs in source order.
    parts: tuple[tuple[bool, str], ...]

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(text for is_var, text in self.parts if is_var)

    @property
    def is_dynamic(self) -> bool:
        return any(is_var for is_var, _ in self.parts)


@dataclass(frozen=True)
class PathPattern:
    """A relative path template that can both render and parse.

    Construct via :meth:`parse`; the raw string is kept for error messages.
    """

    raw: str
    segments: tuple[_Segment, ...]
    variables: tuple[str, ...]
    _regex: re.Pattern[str] = field(repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, raw: str) -> "PathPattern":
        if not isinstance(raw, str):
            raise PatternError(
                f"Path pattern must be a string; got {type(raw).__name__}."
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

        raw_segments = text.split("/")
        segments: list[_Segment] = []
        seen: list[str] = []

        for raw_segment in raw_segments:
            if not raw_segment:
                raise PatternError(
                    f"Path pattern {raw!r} contains an empty path segment."
                )
            if raw_segment in (".", ".."):
                raise PatternError(
                    f"Path pattern {raw!r} may not contain '.' or '..' segments."
                )
            segment = cls._parse_segment(raw_segment, raw)
            for name in segment.variables:
                # A variable may repeat: "sub-{id}/anat/sub-{id}_T1w.nii" is a
                # normal layout. Later occurrences become backreferences, so
                # matching requires every occurrence to agree.
                if name not in seen:
                    seen.append(name)
            segments.append(segment)

        pattern = cls(
            raw=text,
            segments=tuple(segments),
            variables=tuple(seen),
            _regex=cls._build_regex(segments),
        )
        return pattern

    @staticmethod
    def _parse_segment(raw_segment: str, raw: str) -> _Segment:
        parts: list[tuple[bool, str]] = []
        cursor = 0
        for token in _TOKEN_RE.finditer(raw_segment):
            literal = raw_segment[cursor : token.start()]
            if literal:
                parts.append((False, literal))
            name = token.group(1).strip()
            if not name:
                raise PatternError(
                    f"Path pattern {raw!r} contains an empty '{{}}' placeholder."
                )
            if not name.isidentifier():
                raise PatternError(
                    f"Path pattern {raw!r} variable '{{{name}}}' is not a valid "
                    "Python identifier, so it cannot map onto a field."
                )
            parts.append((True, name))
            cursor = token.end()

        trailing = raw_segment[cursor:]
        if trailing:
            parts.append((False, trailing))

        # Any brace surviving tokenization is unbalanced.
        for is_var, text in parts:
            if not is_var and ("{" in text or "}" in text):
                raise PatternError(
                    f"Path pattern {raw!r} has an unbalanced '{{' or '}}'."
                )
        return _Segment(parts=tuple(parts))

    @staticmethod
    def _build_regex(segments: list[_Segment]) -> re.Pattern[str]:
        chunks: list[str] = []
        bound: set[str] = set()
        for segment in segments:
            for is_var, text in segment.parts:
                if not is_var:
                    chunks.append(re.escape(text))
                elif text in bound:
                    # A repeated variable must match the same text every time.
                    chunks.append(rf"(?P={text})")
                else:
                    # Non-greedy and separator-free: a variable never spans '/'.
                    # Backtracking resolves multi-variable segments such as
                    # "{subject}_{session}.json".
                    chunks.append(rf"(?P<{text}>[^/]+?)")
                    bound.add(text)
            chunks.append("/")
        chunks.pop()  # trailing separator
        return re.compile(r"\A" + "".join(chunks) + r"\Z")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def format(self, values: Mapping[str, Any]) -> str:
        """Render the pattern to a concrete relative path.

        Raises :class:`PatternError` naming the variable if a value is absent,
        empty, or contains a path separator.
        """
        rendered: list[str] = []
        for segment in self.segments:
            piece: list[str] = []
            for is_var, text in segment.parts:
                if not is_var:
                    piece.append(text)
                    continue
                if text not in values:
                    raise PatternError(
                        f"Pattern {self.raw!r} needs a value for '{{{text}}}' "
                        f"but none was supplied."
                    )
                value = values[text]
                if value is None:
                    raise PatternError(
                        f"Pattern {self.raw!r} needs a value for '{{{text}}}' "
                        f"but it is None."
                    )
                as_text = str(value)
                if not as_text:
                    raise PatternError(
                        f"Pattern {self.raw!r} got an empty value for '{{{text}}}'."
                    )
                if "/" in as_text or "\\" in as_text:
                    raise PatternError(
                        f"Pattern {self.raw!r} got {as_text!r} for '{{{text}}}', "
                        "but a path separator cannot appear inside a single "
                        "path component."
                    )
                piece.append(as_text)
            rendered.append("".join(piece))
        return "/".join(rendered)

    def match(self, relative_path: str | PurePosixPath) -> dict[str, str] | None:
        """Parse a relative path, returning captured variables or ``None``.

        A pattern with no variables still returns ``{}`` (a successful match)
        rather than ``None``, so callers can distinguish match from no-match.
        """
        text = str(PurePosixPath(relative_path))
        found = self._regex.match(text)
        if found is None:
            return None
        return dict(found.groupdict())

    def glob(self) -> str:
        """A glob that over-matches the pattern, for discovering candidates.

        Always paired with :meth:`match` to reject false positives.
        """
        rendered: list[str] = []
        for segment in self.segments:
            piece = [
                "*" if is_var else _glob_escape(text)
                for is_var, text in segment.parts
            ]
            rendered.append("".join(piece))
        return "/".join(rendered)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def suffix(self) -> str:
        """Trailing extension of the final segment, e.g. ``'.json'``.

        Empty when the pattern names a directory or an extension-less file.
        """
        return PurePosixPath(self.glob()).suffix

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
        """Return a copy whose final segment ends in ``suffix``.

        Used to apply a codec's default extension when the pattern omits one.
        """
        if not suffix:
            return self
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if self.suffix == suffix:
            return self
        return PathPattern.parse(f"{self.raw}{suffix}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw
