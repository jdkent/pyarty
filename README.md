# pyarty

A two-way contract between directory trees and Python dataclasses.

Declare where each field lives with a relative path pattern. Then move between a
dataclass and a directory in **either** direction:

```python
from pyarty import Dir, File, at, bundle

@bundle
class Report:
    name: str                              # filled FROM the path
    body: File[str] = at("{name}.txt")
    metrics: File[dict] = at("metrics.json")

@bundle
class Corpus:
    reports: Dir[list[Report]] = at("reports/{name}")
    summary: File[dict] = at("summary.json")
```

```python
corpus = Corpus(
    reports=[
        Report(name="alpha", body="Hello alpha", metrics={"score": 10}),
        Report(name="beta", body="Hello beta", metrics={"score": 20}),
    ],
    summary={"count": 2},
)
corpus.write("./out")
```

```
out/
├── reports/
│   ├── alpha/
│   │   ├── alpha.txt
│   │   └── metrics.json
│   └── beta/
│       ├── beta.txt
│       └── metrics.json
└── summary.json
```

And back again:

```python
Corpus.read("./out") == corpus     # True
```

## The one guarantee

**`read` is the inverse of `write`.**

That is the whole point, and it is why patterns are *patterns* and not
`str.format` templates. `"{name}.txt"` is rendered on write **and parsed on
read**, so a value baked into a filename comes back as a field:

```python
Corpus.read("./out").reports[0].name    # "alpha" — recovered from the directory name
```

A template can only go one way. A pattern goes both, which is what lets your
own class be the thing on the other side of the round trip — not a generated
look-alike.

## The three field kinds

That is the entire model. Every field is exactly one of:

| Annotation | Means | Round-trips as |
|---|---|---|
| `File[T]` | one file, `T` picks the format | file contents |
| `Dir[T]` | one subdirectory | nested bundle |
| `Dir[list[T]]` | one subdirectory per element | nested bundles |
| anything else | a plain value | **captured from the path** |

A plain value is never written as its own file. It is carried by a
`{variable}` in some sibling's (or its parent's) pattern. That is how a name
encoded in a path survives the trip.

## Patterns

A pattern is a relative path, always `/`-separated, on every platform.

```python
at("metrics.json")              # a fixed name
at("{name}.txt")                # the stem comes from field `name`
at("reports/{name}")            # a subdirectory per child
at("sub-{subject}_ses-{session}.nii")   # several variables in one segment
at("data/{year}/{month}.csv")   # variables at any depth
```

Variables resolve against **the object being named**: for a `File`, its owner;
for a `Dir`, the child it names, falling back to the owner. There is no
"is this variable mine or my child's" flag to remember.

Omit the pattern and you get the field name plus the format's extension, which
covers the common case with no annotation at all:

```python
@bundle
class Defaults:
    notes: File[str]           # -> notes.txt
    config: File[dict]         # -> config.json
    rows: File[list[dict]]     # -> rows.jsonl
```

## Formats come from the annotation

The declared type picks the codec — never the runtime value, and never the file
extension. `File[dict]` is JSON because you *said* it was.

| Annotation | Extension | On disk |
|---|---|---|
| `str` | `.txt` | UTF-8 text |
| `bytes` | `.bin` | raw bytes |
| `int` / `float` | `.txt` | the number as text |
| `bool` | `.json` | `true` / `false` |
| `dict` | `.json` | indented JSON |
| `list[str]` | `.txt` | one item per line |
| `list[dict]` | `.jsonl` | one JSON object per line |
| `list` (other) | `.json` | indented JSON array |
| `Path` | (source's) | copied from the given path |

An extension written into the pattern always wins: `at("payload.cfg")` on a
`File[dict]` is still parsed as JSON.

## Strict write, lenient read

**Writing is strict and all-or-nothing.** The whole tree is validated in memory
first, so a bundle either produces a complete valid directory or raises without
leaving a partial one behind:

```python
Report(name="a", body={"oops": 1}, metrics={})
# PayloadTypeError: Field 'Report.body' is declared File[str],
# which expects str, but got dict: {'oops': 1}
```

**Reading is lenient**, because real trees have extra files in them. Unclaimed
files are ignored, and an `Optional` field with no match becomes `None`. Ask for
`strict=True` to assert a tree contains *exactly* what the schema describes:

```python
Corpus.read("./out")                 # ignores stray files
Corpus.read("./out", strict=True)    # ReadError if anything is unclaimed
```

Errors name the field, the declared type, and the path.

`write(overwrite=True)` replaces the files the bundle declares; it does not
delete anything else. Writing beside unrelated files is therefore safe, but a
file left behind by an earlier write of a now-`None` optional field will still
be there — and a later `read` will pick it up. Write to a fresh directory when
you want an exact snapshot.

## Layouts that cannot round-trip are rejected on import

A `Dir[list[...]]` whose pattern does not vary per child would write every
element to the same directory and silently lose all but one. That is a
declaration-time error, not a surprise on disk:

```python
@bundle
class Broken:
    reports: Dir[list[Report]] = at("reports")
# LayoutError: pattern 'reports', whose last segment has no {variable}.
# Every element would render to the same directory and all but one would
# be lost.
#   Try: at("reports/{name}") — where 'name' is a field on Report.
```

Same for two fields claiming one path, a pattern naming a field that does not
exist, and a variable whose value contains `/`.

## Inspecting a layout

```python
print(Corpus.layout())
```

```
Corpus/
  reports/{name}/*
    (name: str — from path)
    {name}.txt  [text]
    metrics.json  [json]
  summary.json  [json]
```

Use `plan_bundle(instance)` to see the exact files a write *would* produce,
without writing them.

## Starting from a directory you already have

`scaffold_from_directory` generates **source code** you can paste and edit.
Same-shaped sibling directories collapse into one class plus a `Dir[list[...]]`
field, rather than one class per directory:

```python
from pyarty import scaffold_from_directory

print(scaffold_from_directory("./out", root_class_name="Corpus"))
```

```python
@bundle
class Report:
    name: str  # <- the directory name
    body: File[str] = at("{name}.txt")
    metrics: File[dict] = at("metrics.json")


@bundle
class Corpus:
    summary: File[dict] = at("summary.json")
    reports: Dir[list[Report]] = at("reports/{name}")
```

That is the definition from the top of this README, recovered from the tree —
including the `{name}.txt` pattern, because a file named after its own
directory is detected as varying rather than fixed.

## Copying existing files

Annotate `File[Path]` to copy a file instead of serializing a payload. If the
pattern has no extension, the source's is kept so `read` can find it again.
A missing source is an error, not a warning:

```python
@bundle
class Release:
    model: File[Path] = at("model.bin")

Release(model="/tmp/model.bin").write("./release")
Release.read("./release").model      # Path to the copied file
```

## Optional fields

```python
from typing import Optional

@bundle
class Run:
    log: File[str] = at("log.txt")
    profile: Optional[File[dict]] = at("profile.json", default=None)
```

Absent on read → `None`. Absent on write → not written.

## Install

```bash
pip install pyarty
```

Python 3.11+, no dependencies.

## Errors

All inherit from `PyartyError`:

| Error | Raised when |
|---|---|
| `PatternError` | a pattern string, or a value going into one, is malformed |
| `SpecError` | a bundle class is declared incorrectly |
| `LayoutError` | a declared layout could not round-trip (a `SpecError`) |
| `PayloadTypeError` | a value contradicts its declared annotation, on write |
| `WriteError` | a tree could not be written |
| `ReadError` | a tree could not be read |
| `MissingFileError` | a required field had no matching file (a `ReadError`) |

## Development

```bash
pytest
```
