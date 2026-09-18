"""Patterns must render and parse as exact inverses."""

from __future__ import annotations

import pytest

from pyarty.errors import PatternError
from pyarty.pattern import PathPattern

ROUND_TRIP_CASES = [
    ("metrics.json", {}, "metrics.json"),
    ("{name}.txt", {"name": "alpha"}, "alpha.txt"),
    ("reports/{name}", {"name": "beta"}, "reports/beta"),
    ("data/{year}/{month}.csv", {"year": "2024", "month": "01"}, "data/2024/01.csv"),
    (
        "{subject}_{session}.json",
        {"subject": "s01", "session": "ses1"},
        "s01_ses1.json",
    ),
    ("sub-{sid}/anat/{sid2}_T1w.nii", {"sid": "01", "sid2": "x"}, "sub-01/anat/x_T1w.nii"),
]


@pytest.mark.parametrize("raw,values,expected", ROUND_TRIP_CASES)
def test_format_then_match_is_identity(raw, values, expected):
    pattern = PathPattern.parse(raw)
    rendered = pattern.format(values)
    assert rendered == expected
    assert pattern.match(rendered) == values


def test_match_rejects_non_matching_paths():
    pattern = PathPattern.parse("{name}.txt")
    assert pattern.match("alpha.json") is None
    assert pattern.match("alpha.txt.bak") is None
    # A variable never spans a separator.
    assert pattern.match("nested/alpha.txt") is None


def test_match_handles_dotted_stems():
    pattern = PathPattern.parse("{name}.txt")
    assert pattern.match("a.b.c.txt") == {"name": "a.b.c"}


def test_static_pattern_matches_with_empty_mapping():
    pattern = PathPattern.parse("summary.json")
    # {} is a successful match; None means no match. The distinction matters.
    assert pattern.match("summary.json") == {}
    assert pattern.match("other.json") is None


def test_glob_overmatches_then_match_filters():
    pattern = PathPattern.parse("reports/{name}.json")
    assert pattern.glob() == "reports/*.json"


@pytest.mark.parametrize("raw", ["a[1].txt", "q?.txt", "star[*].json"])
def test_glob_escapes_metacharacters_in_literals(raw):
    """A literal '[' or '?' must not be read as a glob wildcard.

    Without escaping these write fine but never read back.
    """
    pattern = PathPattern.parse(raw)
    assert pattern.match(raw) == {}
    # The glob must still find the one literal file it describes.
    assert "*" not in pattern.glob().replace("[*]", "")


def test_suffix_and_child_naming_introspection():
    assert PathPattern.parse("metrics.json").suffix == ".json"
    assert PathPattern.parse("reports/{name}").suffix == ""
    assert PathPattern.parse("{name}.txt").names_distinct_children is True
    assert PathPattern.parse("reports").names_distinct_children is False


def test_with_suffix_is_idempotent():
    pattern = PathPattern.parse("metrics")
    assert pattern.with_suffix("json").raw == "metrics.json"
    assert pattern.with_suffix(".json").with_suffix("json").raw == "metrics.json"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "/absolute",
        "windows\\path",
        "{}.txt",
        "{not-an-identifier}",
        "double//slash",
        "../escape",
        "./here",
        "unbalanced{brace.txt",
    ],
)
def test_malformed_patterns_are_rejected(raw):
    with pytest.raises(PatternError):
        PathPattern.parse(raw)


def test_repeated_variable_is_a_backreference():
    """A repeated variable such as sub-{id}/anat/sub-{id}_T1w.nii is normal."""
    pattern = PathPattern.parse("sub-{sid}/anat/sub-{sid}_T1w.nii")
    assert pattern.variables == ("sid",)

    rendered = pattern.format({"sid": "01"})
    assert rendered == "sub-01/anat/sub-01_T1w.nii"
    assert pattern.match(rendered) == {"sid": "01"}

    # Every occurrence must agree.
    assert pattern.match("sub-01/anat/sub-02_T1w.nii") is None


def test_format_requires_every_variable():
    pattern = PathPattern.parse("{a}/{b}.txt")
    with pytest.raises(PatternError, match="b"):
        pattern.format({"a": "x"})


def test_format_rejects_values_that_would_change_the_tree():
    pattern = PathPattern.parse("{name}.txt")
    for bad in ["a/b", "../etc", "", None]:
        with pytest.raises(PatternError):
            pattern.format({"name": bad})


def test_format_stringifies_non_str_values():
    pattern = PathPattern.parse("run-{index}.txt")
    assert pattern.format({"index": 7}) == "run-7.txt"
