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
| `Files[dict[K, T]]` | **many** files sharing one pattern | mapping, keyed from the path |
| `Dir[T]` | one subdirectory | nested bundle |
| `Dir[list[T]]` | one subdirectory per element | nested bundles |
| `Group[list[T]]` | one record **across parallel trees** | records, keyed from the path |
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

### Constraining a variable

By default a variable matches anything except `/`, which silently mis-parses
delimited names — the wrong capture is indistinguishable from a real match:

```python
at("sub-{sub}_task-{task}_bold.nii")
# "sub-01_ses-pre_task-rest_bold.nii" -> sub='01_ses-pre'   silently wrong
```

Add a character class to make that a clean non-match:

```python
at("sub-{sub:alnum}_task-{task:alnum}_bold.nii")
# "sub-01_ses-pre_task-rest_bold.nii" -> no match
```

Available: `any` (default), `alnum`, `alpha`, `digits`, `word`, `hex`. A value
that violates the class is refused on write, too. Pass `charset=` to set the
default for every unqualified variable in one pattern.

### Optional parts

`[...]` marks a group that is dropped when its variables are `None`, for naming
schemes with optional components:

```python
@bundle
class Bold:
    sub: str
    task: str
    image: File[bytes] = at("sub-{sub}[_ses-{ses}]_task-{task}_bold.nii")
    ses: str | None = None

Bold(sub="01", task="rest", ses=None,  image=b"")  # sub-01_task-rest_bold.nii
Bold(sub="01", task="rest", ses="pre", image=b"")  # sub-01_ses-pre_task-rest_bold.nii
```

Both parse back, and an omitted variable reads as `None`. A group is all-or-
nothing: filling some of its variables and not others is an error, since the
result would not parse back. Brackets containing no variable are literal, so
`at("a[1].txt")` still means the file named `a[1].txt`.

Omit the pattern and you get the field name plus the format's extension, which
covers the common case with no annotation at all:

```python
@bundle
class Defaults:
    notes: File[str]           # -> notes.txt
    config: File[dict]         # -> config.json
    rows: File[list[dict]]     # -> rows.jsonl
```

## Many files, one pattern

`File[...]` matches exactly one file. When a pattern should match *many* —
shards, checksums, runs, per-image labels — use `Files[dict[key, payload]]`:

```python
@bundle
class Bag:
    declaration: File[str]           = at("bagit.txt")
    manifests: Files[dict[str, str]] = at("manifest-{algorithm}.txt")

Bag(declaration="BagIt-Version: 1.0",
    manifests={"md5": "...", "sha512": "..."}).write("./bag")
# bag/{bagit.txt, manifest-md5.txt, manifest-sha512.txt}
```

The mapping key supplies one pattern variable; **any other variable resolves
against the owning instance**, and is read back into its own field:

```python
@bundle
class Repo:
    total: str
    shards: Files[dict[str, bytes]] = at(
        "model-{shard:digits}-of-{total:digits}.safetensors", key="shard")

# model-00001-of-00002.safetensors, model-00002-of-00002.safetensors
# read back: shards={"00001": ..., "00002": ...}, total="00002"
```

`key=` names the keying variable; it defaults to the pattern's only variable
and is required when there is more than one. Declare the key as `dict[int, T]`
and it comes back an `int`. Matching no files gives an empty mapping rather
than an error, since a partly-filled directory is a normal state.

Only the mapping form exists. A `Files[list[T]]` is rejected at declaration
time, because a bare list of payloads cannot tell `write` what to name each
file — it could not round-trip.

## Records across parallel trees

Some layouts spread one logical record over *sibling* trees that agree on a
filename — a YOLO image and its label, a BIDS image and its sidecar. No single
directory holds a record, so nesting cannot express it. `Group[list[T]]` repeats
a bundle over the distinct values of a shared key, rooting every record in the
*same* directory as its owner:

```python
@bundle
class Sample:
    stem: str
    image: File[bytes] = at("images/train/{stem}.jpg")
    label: File[str]   = at("labels/train/{stem}.txt")

@bundle
class Dataset:
    names: File[dict] = at("data.json")
    samples: Group[list[Sample]] = at(key="stem")
```

```
out/
├── data.json
├── images/train/{img001.jpg, img002.jpg}
└── labels/train/{img001.txt, img002.txt}
```

```python
Dataset.read("./out").samples[0].image   # the .jpg
Dataset.read("./out").samples[0].label   # its paired .txt
```

`key=` names the field the records share; it must appear in at least one of the
record's patterns, or there would be nothing on disk to discover them from.
Records come back ordered by key.

Keys are the **union** across the record's patterns, not the intersection, so an
image with no label surfaces through that field's own rules — an error if it is
required, `None` if `Optional` — rather than disappearing. That is the whole
point: a silently dropped half-record is the bug this feature exists to prevent.

A `Group` creates no directory of its own. Its pattern, if given, is a fixed
prefix; it may not contain a variable, because nothing else would capture it on
read. For one group per directory, nest it instead:

```python
@bundle
class Split:
    split: str
    pairs: Group[list[Sample]] = at(key="stem")

@bundle
class Dataset:
    splits: Dir[list[Split]] = at("{split}")     # train/, val/
```

## Recursive layouts

A bundle may contain itself, for trees of unbounded depth:

```python
@bundle
class Catalog:
    catalog: File[dict] = at("catalog.json")
    children: Dir[list["Catalog"]] = at("{child_id}", default_factory=list)
    child_id: str = "root"    # the root has no parent to name it
```

Reading recurses until no subdirectory matches. Note that fields with defaults
must come last, as in any dataclass — so a collection field usually wants
`default_factory=list` (or `dict`) both to satisfy that ordering and to let
leaves omit it.

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
