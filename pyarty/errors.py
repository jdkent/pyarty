"""Error hierarchy for pyarty.

Every error carries enough context (bundle, field, path) to fix the problem
without reading pyarty's source.
"""

from __future__ import annotations

__all__ = [
    "PyartyError",
    "PatternError",
    "SpecError",
    "LayoutError",
    "PayloadTypeError",
    "ReadError",
    "MissingFileError",
    "WriteError",
]


class PyartyError(Exception):
    """Base class for every pyarty error."""


class PatternError(PyartyError, ValueError):
    """A path pattern string is malformed."""


class SpecError(PyartyError, TypeError):
    """A bundle class is declared incorrectly.

    Raised at class-decoration time so mistakes surface on import rather than
    on the first ``write``.
    """


class LayoutError(SpecError):
    """A declared layout cannot round-trip.

    The canonical case is a ``Dir[list[T]]`` whose pattern has no variable in
    its final segment, so every child would collide into one directory.
    """


class PayloadTypeError(PyartyError, TypeError):
    """A field's runtime value does not match its declared annotation.

    Raised on write, before anything touches disk, so a bundle never produces
    a file whose contents contradict its own schema.
    """


class WriteError(PyartyError, RuntimeError):
    """A bundle could not be written to disk."""


class ReadError(PyartyError, RuntimeError):
    """A directory could not be read into a bundle."""


class MissingFileError(ReadError, FileNotFoundError):
    """A required field had no matching file on disk.

    Optional fields (``File[str] | None``) resolve to ``None`` instead.
    """
