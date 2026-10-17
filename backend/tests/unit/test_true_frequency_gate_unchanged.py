"""The dead-neuron gate must keep reading the POST-filter frequency.

WHY THIS EXISTS
---------------
This is a REVERSE control. Every other test in this arc asserts that something
changed; this one asserts that something did not.

Adding `activation_frequency_true` makes it trivially tempting to point the
dead-neuron gate at it — the true frequency is, after all, the honest number.
Doing so would change WHICH FEATURES EXIST in every future extraction, in the
same commit that adds a column, and nothing downstream would report the
difference. A feature that does not clear `min_activation_frequency` is not
merely unlabelled: `extraction_service` skips it entirely, so it gets no
`Feature` row and its stored examples are discarded.

The gate is instrumented instead. `filter_suppressed_neurons` counts neurons cut
as dead that fired often enough on the TRUE frequency — i.e. cut BY THE FILTER
rather than by being dead. That number is the first measurement of what
`filter_fragments` and its siblings actually cost, and it is the evidence any
future decision to move the gate would need.

MUTATION CONTROLS:
  C83 point the gate at true_activation_frequencies
       -> test_the_gate_reads_the_post_filter_frequency
  C84 drop the filter_suppressed_neurons counter
       -> test_the_gate_counts_what_the_filter_suppressed
  C85 drop it from the reported statistics
       -> test_the_suppression_count_is_reported
"""

import ast
import inspect
import textwrap

import pytest

from src.services.extraction_service import ExtractionService


@pytest.fixture(scope="module")
def gate_source() -> str:
    return textwrap.dedent(
        inspect.getsource(ExtractionService.extract_features_for_sae)
    )


def _gate_nodes(source: str):
    """Every `if <freq> < min_activation_frequency:` in the function."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not any(isinstance(op, ast.Lt) for op in test.ops):
            continue
        if "min_activation_frequency" not in ast.dump(test):
            continue
        found.append(node)
    return found


class TestTheGateIsUnchanged:
    def test_the_gate_reads_the_post_filter_frequency(self, gate_source):
        """C83. RESOLVE THE NAME, do not just read the comparison.

        The first version of this test inspected the comparison operands and
        SURVIVED the mutation, because the gate compares a local
        (`neuron_activation_freq < min_activation_frequency`) and the mutation
        changes the ASSIGNMENT one line above. Reading the comparison proved
        only that a name was compared.

        So: find the name the gate compares, then find what that name was bound
        to, and check the source array there. A guard that stops at the
        comparison is exactly the kind that fails open.
        """
        gates = _gate_nodes(gate_source)
        assert gates, (
            "no dead-neuron gate found; the pattern changed and this test is "
            "inert"
        )

        tree = ast.parse(gate_source)

        def _source_arrays(name: str) -> str:
            """Every subscripted array this name is assigned from."""
            dumps = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                targets = {
                    t.id for t in node.targets if isinstance(t, ast.Name)
                }
                if name in targets:
                    dumps.append(ast.dump(node.value))
            return "".join(dumps)

        for gate in gates:
            operands = [gate.test.left, *gate.test.comparators]
            resolved = ""
            for operand in operands:
                resolved += ast.dump(operand)
                if isinstance(operand, ast.Name):
                    resolved += _source_arrays(operand.id)

            # SELF-CHECK: the resolution must actually find the array, or the
            # assertion below is vacuous.
            assert "activation_frequencies" in resolved, (
                "could not resolve the gate's operand back to a frequency "
                "array; the pattern changed and this test is inert"
            )
            assert "true_activation_frequencies" not in resolved, (
                "the dead-neuron gate now reads the TRUE frequency. That "
                "changes which features exist in every future extraction, "
                "silently — a feature cut here loses its row AND its stored "
                "examples. If this is intended it needs its own increment and "
                "its own measurement, not a side effect of adding a column."
            )

    def test_the_empty_heap_case_counts_as_suppression(self, gate_source):
        """The TOTAL suppression case, which the gate matcher cannot see.

        `_gate_nodes` selects `if <x> < min_activation_frequency:`. The
        empty-heap branch — `if not heap_items:` — is a different shape, so
        deleting the suppression count from it left the suite green. That branch
        is the most suppressed a neuron can be: every one of its peaks landed on
        a filtered token, so it emitted nothing at all.

        Omitting it biases `filter_suppressed_neurons` toward "the filter costs
        little", which is the conclusion the instrument exists to test.
        """
        tree = ast.parse(gate_source)

        empty_heap_branches = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and "heap_items" in ast.dump(node.test)
        ]
        # SELF-CHECK: the branch must exist, or this asserts nothing.
        assert empty_heap_branches, (
            "no `if not heap_items:` branch found; the pattern moved and this "
            "test is inert"
        )

        body = "".join(
            ast.dump(ast.Module(body=b.body, type_ignores=[]))
            for b in empty_heap_branches
        )
        assert "filter_suppressed_neurons" in body, (
            "a neuron whose every peak was filtered away — total suppression — "
            "is not counted as suppressed"
        )
        assert "true_activation_frequencies" in body, (
            "the empty-heap branch does not distinguish a genuinely dead neuron "
            "from one the filter emptied"
        )

    def test_the_gate_counts_what_the_filter_suppressed(self, gate_source):
        """C84. The instrumentation is the whole justification for not moving
        the gate. Without it, the divergence is invisible and the decision to
        leave the gate alone can never be revisited on evidence."""
        gates = _gate_nodes(gate_source)
        bodies = "".join(
            ast.dump(ast.Module(body=g.body, type_ignores=[])) for g in gates
        )

        assert "true_activation_frequencies" in bodies, (
            "the gate does not compare against the true frequency at all, so "
            "nothing measures how many neurons the junk filter suppressed"
        )
        assert "filter_suppressed_neurons" in bodies, (
            "the gate notices the divergence and does not count it"
        )

    def test_the_suppression_count_is_reported(self, gate_source):
        """C85. A number computed and not surfaced is not a measurement.

        This is the same failure as `token_positions` — built on every example,
        stored nowhere, and therefore unavailable to the one decision it exists
        to inform.
        """
        assert '"filter_suppressed_neurons"' in gate_source, (
            "filter_suppressed_neurons is counted but never reported in the "
            "extraction statistics"
        )


class TestTheTwoFrequenciesAreBothPersisted:
    def test_each_column_is_fed_its_own_array(self, gate_source):
        """RESOLVE THE ARGUMENT, do not match the keyword.

        The first version asserted `"activation_frequency_true=float(" in
        source`. That says a keyword is present and nothing about what feeds
        it: writing `activation_frequency_true=float(activation_frequencies[i])`
        — the DISHONEST post-filter number into the column whose entire purpose
        is honesty — passed. Same for `activation_count_true`, whose stated
        justification is "so the denominator is auditable rather than assumed".

        This reads the keyword's VALUE expression off the AST instead.
        """
        tree = ast.parse(gate_source)

        feature_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Feature"
        ]
        # SELF-CHECK: no Feature(...) construction means the scan has stopped
        # matching and every assertion below is vacuous.
        assert feature_calls, (
            "no Feature(...) construction found; the pattern changed and this "
            "test is inert"
        )

        expected = {
            "activation_frequency": "activation_frequencies",
            "activation_frequency_true": "true_activation_frequencies",
            "activation_count_true": "feature_fired_counts",
        }
        seen = {}
        for call in feature_calls:
            for kw in call.keywords:
                if kw.arg in expected:
                    seen[kw.arg] = ast.dump(kw.value)

        missing = sorted(set(expected) - set(seen))
        assert not missing, f"these columns are never written: {missing}"

        for column, array in expected.items():
            assert array in seen[column], (
                f"{column} is not fed from {array}. It is fed from "
                f"{seen[column]!r} — so the column carries a different "
                f"quantity than its name and its migration promise."
            )

        # And the two frequency columns must not be fed the SAME array, which
        # the per-column check above cannot catch on its own because
        # "activation_frequencies" is a substring of
        # "true_activation_frequencies".
        assert "true_activation_frequencies" not in seen["activation_frequency"], (
            "the post-filter column is being fed the true frequency, which "
            "silently changes every steering auto-baseline in the estate"
        )

    def test_the_model_keeps_them_apart(self):
        """Two columns, two meanings — and the old one stays NOT NULL.

        Making `activation_frequency` nullable would let a future extraction
        write NULL there and silently break the steering auto-baseline, which
        falls back to strength 10 on a missing frequency.
        """
        from src.models.feature import Feature

        columns = Feature.__table__.columns
        assert columns["activation_frequency"].nullable is False
        assert columns["activation_frequency_true"].nullable is True, (
            "the true frequency must be nullable: every row written before it "
            "existed genuinely did not measure it, and a default would claim "
            "otherwise"
        )
