"""The contrast-flag backfill must compute the SAME hash the writer does.

WHY THIS EXISTS
---------------
`2a81f3b18250` turns `include_negative_examples` off on every template. That
column is inside `prompt_fingerprint`, so every template's fingerprint moves and
every succeeded feature reads as STALE — on L46, 53,088 features at ~8 s each,
roughly 118 GPU-hours offered as one click, to regenerate labels from prompts
byte-identical to the ones that produced them (the block never rendered before,
and does not render now).

`3e2dc9b2fc44` carries those verdicts forward by rewriting
`features.label_prompt_fingerprint` from the old hash to the new one.

THE WHOLE THING RESTS ON ONE PROPERTY: the migration's own hash must equal what
`labeling_fingerprint.prompt_fingerprint` produces. A migration duplicates the
hashing rather than importing it — deliberately, so its behaviour cannot change
under it when the live denylist is edited — and that duplication is exactly what
can silently drift. If it drifts, the UPDATE matches zero rows, prints a
reassuring "carried 0", and the estate is stale anyway.

MUTATION CONTROLS:
  C111 change the migration's json.dumps (drop sort_keys)
        -> test_the_migration_hash_matches_the_writer
  C112 add a field to IDENTITY_FIELDS in the migration only
        -> same
"""

import importlib.util
import pathlib

import pytest

from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.services.labeling_fingerprint import prompt_fingerprint

_MIGRATION = (
    pathlib.Path(__file__).resolve().parents[2]
    / "alembic" / "versions"
    / "3e2dc9b2fc44_carry_verdicts_across_the_contrast_flag_.py"
)


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("_carry_mig", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _template(**overrides):
    values = dict(
        id="lpt_fp", name="n", description="d",
        system_message="sys", user_prompt_template="body {examples_block}",
        temperature=0.2, max_tokens=300, top_p=0.9,
        template_type="mistudio_context", max_examples=10,
        include_prefix=True, include_suffix=True, prime_token_marker="<<>>",
        include_logit_effects=False,
        top_promoted_tokens_count=None, top_suppressed_tokens_count=None,
        include_negative_examples=True, num_negative_examples=None,
        example_sampling="top_k", activation_display="absolute",
        is_detection_template=False, include_nlp_analysis=False,
        is_default=True, is_system=False, created_by=None,
    )
    values.update(overrides)
    return LabelingPromptTemplate(**values)


def _row_mapping(template):
    """What the migration reads out of information_schema + a SELECT."""
    return {
        c.name: getattr(template, c.name)
        for c in LabelingPromptTemplate.__table__.columns
    }


class TestTheMigrationAgreesWithTheWriter:
    def test_the_migration_hash_matches_the_writer(self, migration):
        """C111/C112. Both flag values, against the real implementation."""
        columns = [c.name for c in LabelingPromptTemplate.__table__.columns]

        for flag in (True, False):
            template = _template(include_negative_examples=flag)
            expected = prompt_fingerprint(template)
            actual = migration._fingerprint(
                _row_mapping(template), columns,
                include_negative_examples=flag,
            )
            assert actual == expected, (
                f"the migration computes a different hash than the writer for "
                f"include_negative_examples={flag}. Its UPDATE would match zero "
                f"rows, print 'carried 0', and leave the estate reading as "
                f"stale — with nothing reporting a problem."
            )

    def test_the_two_flag_values_differ(self, migration):
        """The premise. If they matched, the migration would be pointless —
        and the test above would pass vacuously."""
        columns = [c.name for c in LabelingPromptTemplate.__table__.columns]
        row = _row_mapping(_template(include_negative_examples=True))

        before = migration._fingerprint(row, columns, include_negative_examples=True)
        after = migration._fingerprint(row, columns, include_negative_examples=False)
        assert before != after

    def test_the_identity_denylist_is_still_in_sync(self, migration):
        """The duplication is deliberate; the drift is not.

        A migration must describe the schema as it was when it ran, so importing
        the live denylist would make this revision's behaviour change under it.
        That is right — and it means a divergence has to be caught here rather
        than by the code agreeing with itself.
        """
        from src.services.labeling_fingerprint import IDENTITY_FIELDS

        assert migration._IDENTITY_FIELDS == IDENTITY_FIELDS, (
            "the migration's identity denylist has drifted from the live one; "
            "its hashes no longer match the writer's and the carry silently "
            "matches nothing"
        )

    def test_a_template_changed_for_another_reason_is_left_stale(self, migration):
        """Only the flag may be carried.

        A template whose prompt body also changed is GENUINELY stale, and
        carrying it would assert that a verdict from a different prompt is
        current.
        """
        columns = [c.name for c in LabelingPromptTemplate.__table__.columns]
        edited = _row_mapping(_template(system_message="A DIFFERENT JUDGE"))
        original = _row_mapping(_template())

        edited_before = migration._fingerprint(
            edited, columns, include_negative_examples=True)
        original_before = migration._fingerprint(
            original, columns, include_negative_examples=True)

        assert edited_before != original_before, (
            "an edited prompt body produces the same hash, so the carry cannot "
            "distinguish a flag change from a real one"
        )


class TestTheHoldOffMigrationActuallyHoldsOff:
    """M10. The migration that turns the contrast block off must turn it off.

    `UPDATE ... WHERE include_negative_examples = true` reverted to
    `WHERE false` left 4545 tests green — the migration had no test at all.

    Its effect is not cosmetic: without it, wiring the block switches on an
    estate-wide ~50% prompt-volume increase on the ACTIVE default template, with
    no measurement, through the one channel where the recorded 38%->3.5%
    refusal collapse transfers literally.
    """

    def test_the_migration_targets_rows_that_are_on(self):
        import ast
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "alembic" / "versions"
            / "2a81f3b18250_hold_the_contrast_block_off_until_it_is_.py"
        )
        assert path.exists(), "the hold-off migration moved; this test is inert"

        spec = importlib.util.spec_from_file_location("_holdoff", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source = pathlib.Path(path).read_text()
        tree = ast.parse(source)

        upgrades = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upgrade"
        ]
        assert upgrades, "no upgrade() found"
        body = ast.dump(ast.Module(body=upgrades[0].body, type_ignores=[]))

        assert "include_negative_examples = false" in body, (
            "the migration does not turn the flag off"
        )
        assert "include_negative_examples = true" in body, (
            "the migration's WHERE clause no longer selects rows that are ON, "
            "so it updates nothing and the contrast block ships enabled "
            "estate-wide, unmeasured"
        )
        assert "WHERE false" not in body

    def test_the_carry_migration_follows_the_hold_off(self):
        """Order matters: carrying before the flag moves would carry nothing."""
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "alembic" / "versions"
            / "3e2dc9b2fc44_carry_verdicts_across_the_contrast_flag_.py"
        )
        spec = importlib.util.spec_from_file_location("_carry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.down_revision == "2a81f3b18250", (
            "the fingerprint carry no longer runs immediately after the flag "
            "change; run in the wrong order it rewrites nothing and the estate "
            "reads as stale"
        )
