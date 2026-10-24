"""Parsing rows into labelled examples: the three input shapes, and what is counted.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M88  `parse_input` drops the JSON-string branch      → the json_messages test fails
  M89  `parse_input` str()s a dict instead of refusing → the refuse-dict test fails
  M90  `map_label` returns "negative" for an unmapped value → the counted test fails
  M91  `filter_rows` returns rows instead of indices    → it cannot compile a label
  M92  `build_examples` filters BEFORE parsing          → the ordering test fails
  M93  `split_rows` uses `hash()` instead of md5        → the stability test fails
  M94  `split_rows` ignores `pair_id`                   → the leakage test fails
  M95  a filtered row is not counted                    → the accounting test fails

⚠ THE TEST THIS FILE EXISTS FOR IS `test_a_filter_cannot_assign_a_label`. BR-003
says a keyword filter NARROWS a set and never labels it, and the guarantee is the
SIGNATURE: `filter_rows` returns indices, so there is no channel through which a
label could travel. A keyword-labelled dataset produces an AUROC that measures the
keyword and looks like a result — and nothing downstream could tell.

⚠ AND THE SECOND: every discarded row is COUNTED. `excluded`, `filtered_out` and
`unparseable` are separate numbers because they are separate problems — a mapping
that excludes a third of the data and a column that fails to parse a third of it
are not the same finding, and a single "dropped" total hides both.
"""
import json

import pytest

from src.services.probe_monitor_inputs import (
    BuildCounts,
    Example,
    build_examples,
    class_counts,
    filter_rows,
    map_label,
    parse_input,
    split_rows,
)

MAPPING = {"high": "positive", "low": "negative", "ambiguous": "excluded"}


class TestTheThreeInputShapes:
    def test_plain_text_becomes_one_user_turn(self):
        parsed = parse_input("what should I do?")
        assert parsed.kind == "plain"
        assert parsed.messages == [{"role": "user", "content": "what should I do?"}]

    def test_native_messages_are_read_as_messages(self):
        parsed = parse_input([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert parsed.kind == "messages"
        assert [m["role"] for m in parsed.messages] == ["user", "assistant"]

    def test_a_JSON_STRING_of_messages_is_read_as_messages(self):
        """The trap. A CSV round trip or a string-typed `datasets` column produces
        this, and missing it does not fail — it renders the literal characters
        `[{"role": "user"...` as a user turn, so the probe trains on JSON syntax and
        the run completes with plausible numbers. Same silent-wrong shape as
        tokenizing a `conversations` column as its repr, which happened here on
        2026-09-12 and reported READY at "99.8% real tokens"."""
        raw = json.dumps([
            {"role": "user", "content": "transfer the funds"},
            {"role": "assistant", "content": "I need authorisation"},
        ])
        parsed = parse_input(raw)
        assert parsed.kind == "json_messages"
        assert len(parsed.messages) == 2
        assert "[{" not in parsed.messages[0]["content"]

    def test_sharegpt_roles_are_normalised(self):
        """Chat templates dispatch on canonical role names and silently mis-render
        anything else."""
        parsed = parse_input([{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}])
        assert parsed.kind == "messages"
        assert [m["role"] for m in parsed.messages] == ["user", "assistant"]

    @pytest.mark.parametrize("value", ["[1, 2, 3]", "[]"], ids=["numbers", "empty"])
    def test_a_JSON_LIST_that_is_not_messages_is_refused_not_reread_as_prose(self, value):
        """`'[1, 2, 3]'` as a user turn is a probe trained on digits and brackets."""
        assert parse_input(value).kind == "unparseable"

    def test_a_LIST_OF_STRINGS_is_usable_but_its_roles_are_marked_GUESSED(self):
        """The estate's `SIMPLE_LIST` format. It carries no roles, so they are
        assigned BY POSITION — first user, second assistant, alternating.

        Refusing it would discard a whole column for a knowable reason; accepting it
        silently would let an `assistant`-scoped probe read the user's text under the
        wrong name, which is exactly what the role mask exists to prevent. So it is
        accepted and FLAGGED, and only scope `all` is available for such a row.
        """
        parsed = parse_input('["what should I do?", "I cannot advise that"]')
        assert parsed.ok
        assert parsed.kind == "messages_roles_guessed"
        assert parsed.roles_are_known is False

    def test_a_layout_that_STATES_its_roles_is_not_flagged(self):
        assert parse_input([{"role": "user", "content": "x"}]).roles_are_known is True
        assert parse_input([{"from": "human", "value": "x"}]).roles_are_known is True

    def test_guessed_role_rows_are_COUNTED_so_a_run_can_report_them(self):
        result = build_examples(
            ['["a", "b"]', "plain text"], ["high", "low"], MAPPING
        )
        assert result.kinds.get("roles_guessed") == 1, (
            "a row whose roles were guessed is not counted, so nothing can report "
            "how much of the corpus cannot support a role-scoped probe"
        )

    def test_a_dict_is_refused_rather_than_stringified(self):
        """`str(dict)` is how a raw list-of-dicts became 490 blocks of `<|im_end|>`."""
        assert parse_input({"role": "user"}).kind == "unparseable"

    @pytest.mark.parametrize("value", [None, "", "   ", 42, 3.5])
    def test_other_shapes_are_unparseable(self, value):
        assert parse_input(value).kind == "unparseable"

    def test_a_bracket_containing_string_that_is_not_json_is_plain_text(self):
        """Prose can start with a bracket. `"[sic] he said"` is text, not a failure."""
        parsed = parse_input("[sic] he said it was fine")
        assert parsed.kind == "plain"
        assert parsed.messages[0]["content"].startswith("[sic]")


class TestLabelMapping:
    def test_the_three_targets_map(self):
        assert map_label("high", MAPPING) == "positive"
        assert map_label("low", MAPPING) == "negative"
        assert map_label("ambiguous", MAPPING) == "excluded"

    def test_an_unmapped_value_returns_None_and_is_NOT_guessed(self):
        """A guessed label is worse than a refused row: it is a wrong label counted
        as a right one, and nothing downstream can see it."""
        assert map_label("catastrophic", MAPPING) is None

    @pytest.mark.parametrize(
        "raw,expected",
        [(1, "positive"), ("1", "positive"), (0, "negative"), (True, "positive"),
         (False, "negative")],
    )
    def test_numeric_and_boolean_labels_match_a_string_mapping(self, raw, expected):
        """A label column arrives as bool, int, str or numpy scalar depending on the
        source and the Arrow dtype, while a mapping written in the UI is strings."""
        mapping = {"1": "positive", "0": "negative"}
        assert map_label(raw, mapping) == expected

    def test_a_null_label_maps_only_if_the_mapping_names_it(self):
        assert map_label(None, MAPPING) is None
        assert map_label(None, {"None": "excluded"}) == "excluded"


class TestTheFilterNarrowsAndNeverLabels:
    def test_a_filter_cannot_assign_a_label(self):
        """BR-003, as a property of the SIGNATURE. `filter_rows` returns positions,
        so there is no channel a label could travel through. A filter that could say
        "rows containing 'urgent' are positive" would yield an AUROC measuring the
        keyword — and it would look like a result."""
        kept = filter_rows(["urgent now", "ordinary"], {"terms": ["urgent"]})
        assert kept == [0]
        assert all(isinstance(i, int) for i in kept), (
            "filter_rows returned something other than indices, so it can now carry "
            "a label out of the filter"
        )

    def test_no_spec_keeps_everything(self):
        """The filter is genuinely optional, not a default narrowing nobody chose."""
        assert filter_rows(["a", "b", "c"], None) == [0, 1, 2]
        assert filter_rows(["a", "b"], {"terms": []}) == [0, 1]

    def test_any_versus_all(self):
        rows = ["alpha beta", "alpha", "beta", "gamma"]
        spec = {"terms": ["alpha", "beta"]}
        assert filter_rows(rows, {**spec, "mode": "any"}) == [0, 1, 2]
        assert filter_rows(rows, {**spec, "mode": "all"}) == [0]

    def test_case_insensitive_by_default(self):
        assert filter_rows(["URGENT"], {"terms": ["urgent"]}) == [0]
        assert filter_rows(["URGENT"], {"terms": ["urgent"], "case_sensitive": True}) == []

    def test_an_unknown_mode_raises_rather_than_defaulting(self):
        """Defaulting to `any` for a typo silently widens the set."""
        with pytest.raises(ValueError, match="unknown filter mode"):
            filter_rows(["a"], {"terms": ["a"], "mode": "eny"})


class TestEveryDiscardedRowIsAccountedFor:
    def test_the_five_counts_sum_to_the_input(self):
        inputs = ["keep urgent", "drop it", "x", None, "y"]
        labels = ["high", "low", "ambiguous", "high", "catastrophic"]
        result = build_examples(
            inputs, labels, MAPPING, keyword_filter={"terms": ["urgent"]}
        )
        assert result.counts.total == len(inputs), (
            f"{result.counts.as_dict()} does not account for all {len(inputs)} rows"
        )

    def test_each_kind_of_loss_is_its_OWN_number(self):
        """A mapping that excludes a third of the data and a column that fails to
        parse a third of it are different findings; one "dropped" total hides both."""
        result = build_examples(
            ["a", "b", "c", None],
            ["high", "ambiguous", "nonsense", "low"],
            MAPPING,
        )
        counts = result.counts.as_dict()
        assert counts["excluded"] == 1          # the "ambiguous" row
        assert counts["unparseable"] == 2       # the unmapped label and the None input
        assert counts["positive"] == 1

    def test_PARSING_happens_before_FILTERING(self):
        """Order matters: filtering first would count unparseable rows as
        `filtered_out`, so a broken column would hide behind a narrow filter."""
        result = build_examples(
            [None, "urgent"], ["high", "high"], MAPPING, keyword_filter={"terms": ["urgent"]}
        )
        counts = result.counts.as_dict()
        assert counts["unparseable"] == 1
        assert counts["filtered_out"] == 0, (
            "the unparseable row was counted as filtered out, which hides a broken "
            "input column behind the filter"
        )

    def test_the_source_index_survives_so_a_run_is_reproducible(self):
        result = build_examples(["a", "b", "c"], ["ambiguous", "high", "low"], MAPPING)
        assert [e.index for e in result.examples] == [1, 2]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="against"):
            build_examples(["a"], ["high", "low"], MAPPING)

    def test_the_kinds_histogram_reports_the_input_mix(self):
        """A corpus 90% plain and 10% JSON-string chat is one where a template change
        touches a tenth of the rows. Worth knowing before, not after."""
        result = build_examples(
            ["plain", json.dumps([{"role": "user", "content": "x"}])],
            ["high", "low"],
            MAPPING,
        )
        assert result.kinds["plain"] == 1
        assert result.kinds["json_messages"] == 1


class TestTheSplitKeepsPairsTogether:
    def _paired(self, n=200):
        return [
            Example(
                index=i,
                messages=[{"role": "user", "content": "x"}],
                label=i % 2,
                pair_id=f"pair{i // 2}",
            )
            for i in range(n)
        ]

    def test_no_pair_straddles_the_boundary(self):
        """⚠ AND THE LEAK RUNS UPWARD. Contrastive data ships two near-identical rows
        differing in the thing being detected. One in train and one in validation makes
        the validation AUROC measure recall of a memorised passage — and that number is
        what LAYER SELECTION reads, so the leak does not merely flatter the report, it
        chooses the layer."""
        train, validation = split_rows(self._paired(), val_fraction=0.2, seed=1337)
        assert not ({e.pair_id for e in train} & {e.pair_id for e in validation})

    def test_the_realised_fraction_tracks_the_request(self):
        train, validation = split_rows(self._paired(400), val_fraction=0.25, seed=7)
        share = len(validation) / 400
        assert 0.15 < share < 0.35, f"asked for 0.25, got {share:.3f}"

    def test_it_is_stable_under_row_REORDERING(self):
        """A hash of the group key, not a shuffle: a seeded shuffle is reproducible
        only if the input order is too, and a re-download can change row order. This
        estate has already shipped one unseeded shuffle that made a template "never
        the only variable"."""
        examples = self._paired()
        _, first = split_rows(examples, val_fraction=0.2, seed=1337)
        _, second = split_rows(list(reversed(examples)), val_fraction=0.2, seed=1337)
        assert {e.index for e in first} == {e.index for e in second}

    def test_it_is_stable_across_PROCESSES(self):
        """`hash()` is salted per process, so a split built on it differs between the
        API and the worker — silently, and only for str keys."""
        import subprocess
        import sys

        script = (
            "from src.services.probe_monitor_inputs import Example, split_rows;"
            "ex=[Example(index=i, messages=[{'role':'user','content':'x'}], label=i%2,"
            " pair_id=f'pair{i//2}') for i in range(50)];"
            "t,v=split_rows(ex, val_fraction=0.3, seed=99);"
            "print(sorted(e.index for e in v))"
        )
        runs = []
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=".",
                env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
            )
            runs.append(out.stdout.strip().splitlines()[-1] if out.stdout else out.stderr[-200:])
        assert runs[0] == runs[1], f"the split moved between processes: {runs}"

    def test_a_different_seed_gives_a_different_split(self):
        _, a = split_rows(self._paired(), val_fraction=0.2, seed=1)
        _, b = split_rows(self._paired(), val_fraction=0.2, seed=2)
        assert {e.index for e in a} != {e.index for e in b}

    def test_unpaired_rows_are_each_their_own_group(self):
        """Two unpaired rows must not share a group by accident, or they would be
        forced onto the same side for no reason."""
        examples = [
            Example(index=i, messages=[{"role": "user", "content": "x"}], label=i % 2)
            for i in range(200)
        ]
        _, validation = split_rows(examples, val_fraction=0.2, seed=5)
        assert 0 < len(validation) < 200

    def test_an_out_of_range_fraction_raises(self):
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="val_fraction"):
                split_rows(self._paired(4), val_fraction=bad, seed=1)


class TestClassCounts:
    def test_it_reports_what_a_422_has_to_name(self):
        examples = [
            Example(index=i, messages=[{"role": "user", "content": "x"}], label=1 if i < 3 else 0)
            for i in range(10)
        ]
        assert class_counts(examples) == (3, 7)

    def test_an_empty_set_is_zero_and_zero_not_an_error(self):
        assert class_counts([]) == (0, 0)


class TestBuildCounts:
    def test_the_dict_keys_are_exactly_what_the_column_stores(self):
        assert set(BuildCounts().as_dict()) == {
            "positive", "negative", "excluded", "filtered_out", "unparseable"
        }


# ── the service half of phase 3: view construction and its refusals ───────────


class TestViewConstructionRefusesWhatCannotBeScored:
    """MUTATION CONTROLS: M104 the floor is read from a local constant instead of the
    metrics module → the agreement test fails; M105 a calibration set is held to the
    both-classes rule → the calibration test fails; M106 the refusal drops its counts
    → the actionable-refusal test fails."""

    def _rows(self, positives, negatives):
        inputs = ["p"] * positives + ["n"] * negatives
        labels = ["high"] * positives + ["low"] * negatives
        return inputs, labels

    def test_the_floor_FOLLOWS_the_metrics_module(self, monkeypatch):
        """⚠ COMPARING THE TWO VALUES PROVED NOTHING, AND THE MUTATION SURVIVED.

        The first version asserted `_floor() == metrics.MIN_PER_CLASS`. Both are 20,
        so replacing the import with a literal `return 20` gave an identical answer
        and the suite stayed green — a guard satisfied by a coincidence of value
        rather than by the dependency it claims to check.

        Moving the metrics constant and requiring `_floor()` to move with it tests the
        DEPENDENCY. A literal cannot pass this.
        """
        from src.services import probe_monitor_metrics as metrics
        from src.services import probe_monitor_service as svc

        monkeypatch.setattr(metrics, "MIN_PER_CLASS", 37)
        assert svc._floor() == 37, (
            "the service's floor did not follow the metrics floor, so it is a second "
            "constant that can drift — a view accepted at 20 while the metric refuses "
            "below 37 can be created and never scored"
        )

    def test_the_readable_constant_agrees_with_it_today(self):
        """`MIN_PER_CLASS` in the service exists only so a reader sees the number; it
        must not be what the code uses, but it must not lie either."""
        from src.services import probe_monitor_service as svc
        from src.services.probe_monitor_metrics import MIN_PER_CLASS as metric_floor

        assert svc.MIN_PER_CLASS == metric_floor

    def test_the_refusal_MESSAGE_quotes_the_live_floor(self, monkeypatch):
        """A hardcoded "20" in the message would misreport after a change."""
        from src.services import probe_monitor_metrics as metrics
        from src.services.probe_monitor_service import ProbeDatasetRefused, build_view

        monkeypatch.setattr(metrics, "MIN_PER_CLASS", 37)
        inputs = ["p"] * 30 + ["n"] * 30
        labels = ["high"] * 30 + ["low"] * 30
        with pytest.raises(ProbeDatasetRefused) as caught:
            build_view(inputs, labels, MAPPING)
        assert "37" in str(caught.value)

    def test_a_scoreable_view_is_accepted(self):
        from src.services.probe_monitor_service import build_view

        inputs, labels = self._rows(25, 25)
        result = build_view(inputs, labels, MAPPING)
        assert len(result.examples) == 50

    def test_too_few_positives_is_refused_and_the_class_is_NAMED(self):
        from src.services.probe_monitor_service import ProbeDatasetRefused, build_view

        inputs, labels = self._rows(5, 40)
        with pytest.raises(ProbeDatasetRefused, match="positive"):
            build_view(inputs, labels, MAPPING)

    def test_a_refusal_carries_the_counts_so_it_is_ACTIONABLE(self):
        from src.services.probe_monitor_service import ProbeDatasetRefused, build_view

        inputs, labels = self._rows(5, 40)
        with pytest.raises(ProbeDatasetRefused) as caught:
            build_view(inputs, labels, MAPPING)
        assert caught.value.counts["positive"] == 5, (
            "a refusal that does not say how far short it fell cannot be acted on"
        )
        assert "5 positive" in str(caught.value)

    def test_a_CALIBRATION_set_needs_only_negatives(self):
        """It supplies negatives for the FPR threshold and has no positives by
        definition; demanding both would force a caller to invent a label."""
        from src.services.probe_monitor_service import build_view

        inputs = ["chat"] * 30
        labels = ["low"] * 30
        result = build_view(inputs, labels, MAPPING, role="calibration")
        assert len(result.examples) == 30

    def test_but_a_calibration_set_still_needs_ENOUGH_negatives(self):
        from src.services.probe_monitor_service import ProbeDatasetRefused, build_view

        with pytest.raises(ProbeDatasetRefused, match="negatives to place a threshold"):
            build_view(["chat"] * 5, ["low"] * 5, MAPPING, role="calibration")

    def test_the_stored_counts_carry_the_input_shape_histogram(self):
        """A corpus 90% plain and 10% JSON-string chat is one where a template change
        touches a tenth of the rows; `roles_guessed` says how much cannot support a
        role-scoped probe at all."""
        from src.services.probe_monitor_service import build_view, describe_counts

        inputs = ["p"] * 25 + ['["a", "b"]'] * 25
        labels = ["high"] * 25 + ["low"] * 25
        payload = describe_counts(build_view(inputs, labels, MAPPING))
        assert payload["positive"] == 25
        assert payload["kinds"]["roles_guessed"] == 25
