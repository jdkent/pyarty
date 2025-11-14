# pyarty

`pyarty` is a declarative, schema-first toolkit for describing and materializing dataclasses. Define your structure once with the Python DSL, then write it to disk with a single call. No imperative path juggling, no guessing which files belong where.

## Installation

```bash
pip install pyarty  # once published
```

## Quick Start: Single Report

```python
from pyarty import File, bundle, twig

@bundle
class Report:
    name: str
    body: File[str] = twig(name="{name}")         # renders to "alpha/alpha.txt"
    metadata: File[dict]
```

```python
report = Report(name="alpha", body="Hello", metadata={"score": 10})
report.write("./out")
```

Filesystem:
```
out/
└── alpha/
    ├── alpha.txt
    └── metadata.json
```

## Level Up: Nested Bundles

```python
from pyarty import Dir

@bundle
class Experiment:
    reports: Dir[list[Report]] = twig(name="{name}", source="field")
    summary: File[dict]

experiment = Experiment(
    reports=[Report(name="a", body="A", metadata={}), Report(name="b", body="B", metadata={})],
    summary={"count": 2},
)
experiment.write("./experiments")
```

Filesystem:
```
experiments/
├── a/
│   ├── a.txt
│   └── metadata.json
├── b/
│   ├── b.txt
│   └── metadata.json
└── summary.json
```

## Advanced Naming

```python
def custom_dir_name(field_name, field, self, index):
    return f"{self.label}-{index}-{field.slug}"

@bundle
class Node:
    slug: str
    payload: File[str]

@bundle
class Tree:
    label: str
    nodes: Dir[list[Node]] = twig(name=custom_dir_name)

tree = Tree(label="demo", nodes=[Node(slug="n1", payload="X"), Node(slug="n2", payload="Y")])
```

Filesystem:
```
./tree/
├── tree-0-n1/
│   └── payload.txt
└── tree-1-n2/
    └── payload.txt
```

## File-Type Awareness

- `File[str]` → `txt`
- `File[dict]` → `json`
- `File[list[dict]]` → `jsonl`

Custom extensions are a `twig(extension="bin")` away.

## Why pyarty?

- **Schema-first**: clear bundle definitions, YAML/JSON friendly.
- **Type-driven**: `File[dict]` vs `File[list[str]]` matters.
- **Nameable**: templates, field references, callables, or defaults.
- **Batteries-included**: `.write(path)` takes care of directories, files, and serialization.

Define once, render anywhere. Life's a pyarty.
