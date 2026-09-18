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


# ----------------------------------------------------------------------
# Charsets
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "charset,good,bad",
    [
        ("alnum", "rest01", "has_underscore"),
        ("digits", "0042", "01a"),
        ("alpha", "rest", "rest1"),
        ("word", "has_underscore", "has-hyphen"),
        ("hex", "deadBEEF", "xyz"),
    ],
)
def test_charset_constrains_both_directions(charset, good, bad):
    pattern = PathPattern.parse(f"v-{{value:{charset}}}.txt")

    rendered = pattern.format({"value": good})
    assert pattern.match(rendered) == {"value": good}

    # A value outside the class is refused on write...
    with pytest.raises(PatternError, match=charset):
        pattern.format({"value": bad})
    # ...and not captured on read.
    assert pattern.match(f"v-{bad}.txt") is None


def test_charset_prevents_the_silent_misparse():
    """Unconstrained variables span '_' and quietly capture the wrong text."""
    loose = PathPattern.parse("sub-{sub}_task-{task}_bold.nii")
    assert loose.match("sub-01_ses-pre_task-rest_bold.nii") == {
        "sub": "01_ses-pre",  # wrong, but indistinguishable from a real match
        "task": "rest",
    }

    strict = PathPattern.parse("sub-{sub:alnum}_task-{task:alnum}_bold.nii")
    assert strict.match("sub-01_ses-pre_task-rest_bold.nii") is None
    assert strict.match("sub-01_task-rest_bold.nii") == {"sub": "01", "task": "rest"}


def test_default_charset_applies_to_unqualified_variables():
    pattern = PathPattern.parse("{a}-{b:any}.txt", charset="alnum")
    assert pattern.match("x_y-z.txt") is None       # {a} inherited alnum
    assert pattern.match("x-y_z.txt") == {"a": "x", "b": "y_z"}


def test_unknown_charset_is_rejected():
    with pytest.raises(PatternError, match="charset"):
        PathPattern.parse("{a:nonsense}.txt")
    with pytest.raises(PatternError, match="charset"):
        PathPattern.parse("{a}.txt", charset="nonsense")


# ----------------------------------------------------------------------
# Optional groups
# ----------------------------------------------------------------------
def test_optional_group_is_dropped_when_its_variable_is_none():
    pattern = PathPattern.parse("sub-{sub}[_ses-{ses}]_bold.nii")

    assert pattern.format({"sub": "01", "ses": None}) == "sub-01_bold.nii"
    assert pattern.format({"sub": "01", "ses": "pre"}) == "sub-01_ses-pre_bold.nii"


def test_optional_group_round_trips_both_ways():
    pattern = PathPattern.parse("sub-{sub}[_ses-{ses}]_bold.nii")

    without = pattern.format({"sub": "01", "ses": None})
    # An omitted variable is absent, not present-and-None.
    assert pattern.match(without) == {"sub": "01"}

    with_ses = pattern.format({"sub": "01", "ses": "pre"})
    assert pattern.match(with_ses) == {"sub": "01", "ses": "pre"}


def test_several_optional_groups_are_independent():
    pattern = PathPattern.parse(
        "sub-{sub}[_ses-{ses}][_acq-{acq}][_run-{run}]_bold.nii"
    )
    values = {"sub": "01", "ses": None, "acq": "hi", "run": None}
    assert pattern.format(values) == "sub-01_acq-hi_bold.nii"
    assert pattern.match("sub-01_acq-hi_bold.nii") == {"sub": "01", "acq": "hi"}
    assert pattern.match("sub-01_ses-a_acq-hi_run-2_bold.nii") == {
        "sub": "01", "ses": "a", "acq": "hi", "run": "2",
    }


def test_optional_variables_are_reported_separately():
    pattern = PathPattern.parse("sub-{sub}[_ses-{ses}]_bold.nii")
    assert pattern.optional_variables == frozenset({"ses"})
    assert pattern.required_variables == ("sub",)
    assert set(pattern.variables) == {"sub", "ses"}


def test_partially_filled_optional_group_is_an_error():
    """All-or-nothing: a half-rendered group would not parse back."""
    pattern = PathPattern.parse("x[_a-{a}_b-{b}].txt")
    assert pattern.format({"a": None, "b": None}) == "x.txt"
    assert pattern.format({"a": "1", "b": "2"}) == "x_a-1_b-2.txt"
    with pytest.raises(PatternError, match="all or none"):
        pattern.format({"a": "1", "b": None})


def test_brackets_without_a_variable_stay_literal():
    """Back-compat: 'a[1].txt' is a filename, not an optional group."""
    pattern = PathPattern.parse("a[1].txt")
    assert pattern.variables == ()
    assert pattern.match("a[1].txt") == {}
    assert pattern.format({}) == "a[1].txt"


def test_optional_group_glob_overmatches():
    pattern = PathPattern.parse("sub-{sub}[_ses-{ses}]_bold.nii")
    glob = pattern.glob()
    # Must be loose enough to find the file whether or not ses is present.
    assert glob == "sub-*_bold.nii"
