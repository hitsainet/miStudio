"""No hand-written field list may drift behind the template's columns.

WHY THIS EXISTS
---------------
`labeling_prompt_templates` has its prompt-affecting fields copied into FIVE
other places. Every one was hand-written, and by 2026-09-10 three had already
drifted:

  * `clone_template` dropped include_negative_examples, num_negative_examples
    and include_nlp_analysis — so "clone the baseline and change one field",
    which is how an A/B arm is built, silently moved four variables.
  * the bulk `template_config` omitted both negative-example keys, so the bulk
    path could not see them however a template was configured — while the trial
    path carried them. A trial therefore did not predict the bulk run it was
    trialling.
  * the no-template fallback retyped the values and disagreed with every seeded
    template on `prime_token_marker` ('>>>' vs '<<>>'), so a run that fell back
    marked prime tokens differently from every other run, invisibly.

`labeling_fingerprint` already solved this for itself — it DERIVES its field set
from `__table__.columns` minus an identity denylist, and its docstring names
hand-listing as the defect. The other five sites did not get the lesson.

These tests use the derived sets as the oracle, so they cannot themselves drift.
A new column is covered the day it is added, without editing this file.

MUTATION CONTROLS:
  C64 re-hand-list the fields in freeze_template
       -> test_freeze_template_covers_every_fingerprinted_field
  C65 drop a key from TEMPLATE_CONFIG_KEYS
       -> test_the_renderer_reads_only_keys_the_config_carries
  C66 retype a value in the no-template fallback
       -> test_the_fallback_matches_the_column_defaults
"""

import inspect

import pytest

from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.services.labeling_fingerprint import prompt_fingerprint_fields
from src.services.labeling_trial_service import (
    TEMPLATE_CONFIG_KEYS,
    freeze_template,
)


def _template(**overrides) -> LabelingPromptTemplate:
    """A template whose every column differs from its default.

    A field that agrees with its default by construction cannot reveal a drop —
    the fixture trap this repo keeps rediscovering.
    """
    values = dict(
        id="lpt_probe", name="probe", description="d",
        system_message="sys", user_prompt_template="body {examples_block}",
        temperature=0.11, max_tokens=321, top_p=0.77,
        template_type="mistudio_context", max_examples=17,
        include_prefix=False, include_suffix=False, prime_token_marker="[[]]",
        include_logit_effects=True,
        top_promoted_tokens_count=7, top_suppressed_tokens_count=8,
        include_negative_examples=False, num_negative_examples=3,
        is_detection_template=False, include_nlp_analysis=True,
        example_sampling="stratified", activation_display="percent_of_max",
        is_default=False, is_system=False, created_by=None,
    )
    values.update(overrides)
    return LabelingPromptTemplate(**values)


class TestFreezeTemplateIsComplete:
    def test_freeze_template_covers_every_fingerprinted_field(self):
        """C64. A frozen run must record every variable it actually used.

        The fingerprint's field set is the authority on "what changes the
        prompt". Anything in it and absent from the freeze is a variable that
        moved without being recorded — and a trial's whole job is to say what
        it ran.
        """
        template = _template()
        frozen = freeze_template(template)
        fingerprinted = prompt_fingerprint_fields(template)

        missing = sorted(set(fingerprinted) - set(frozen))
        assert not missing, (
            "freeze_template does not record these prompt-affecting fields, so "
            f"a trial using them could not say so afterwards: {missing}"
        )

        wrong = {
            k: (v, frozen[k])
            for k, v in fingerprinted.items()
            if frozen.get(k) != v
        }
        assert not wrong, f"frozen values disagree with the template: {wrong}"

    def test_the_freeze_still_carries_its_own_extras(self):
        """Negative control: deriving must not drop what only the freeze has.

        `template_id`, `template_name` and `body_sha256` are deliberately NOT
        fingerprinted (identity, and a digest of two fields already covered),
        so a derivation that returned only the fingerprint set would pass the
        test above while losing them.
        """
        frozen = freeze_template(_template())
        assert frozen["template_id"] == "lpt_probe"
        assert frozen["template_name"] == "probe"
        assert len(frozen["body_sha256"]) == 16


class TestTemplateConfigKeysAreOneList:
    def test_the_renderer_reads_only_keys_the_config_carries(self):
        """C65. Every field the formatter reads must be in the shared list.

        Scans EVERY consumer, not just the formatter, and both access forms.

        The first version of this test scanned `labeling_context_formatter`
        alone and looked only for `.get(...)`. Dropping `include_nlp_analysis`
        from the shared list left it GREEN — that key is read in
        `openai_labeling_service`, and `is_detection_template` / `max_examples`
        are read with subscript syntax. A source scan that looks in one file for
        one syntax fails open, which is the failure mode this repo has recorded
        twice; the self-check below is what stops it failing open silently.
        """
        import re

        # CONSUMERS ONLY. `labeling_service` BUILDS a template_config, so
        # scanning it matches assignments — writes, not reads — and an earlier
        # version of this test consequently "found" a key that no consumer ever
        # reads and demanded it be carried.
        from src.services import (
            labeling_context_formatter,
            openai_labeling_service,
        )

        read_keys: set = set()
        for module in (
            labeling_context_formatter,
            openai_labeling_service,
        ):
            source = inspect.getsource(module)
            read_keys |= set(re.findall(
                r"template_config\.get\(\s*['\"]([a-z_]+)['\"]", source))
            read_keys |= set(re.findall(
                r"template_config\[\s*['\"]([a-z_]+)['\"]", source))

        # SELF-CHECK: the scan must find the keys we KNOW are read, or it has
        # stopped matching the code and is asserting nothing.
        for known in ("template_type", "include_nlp_analysis",
                      "prime_token_marker", "include_prefix"):
            assert known in read_keys, (
                f"the scan no longer finds {known!r}, which is read in these "
                f"modules — the pattern has drifted and this test is inert"
            )

        # Not every read key is a column — the formatter also reads derived
        # values passed alongside. Only require the ones that ARE columns.
        columns = set(LabelingPromptTemplate.__table__.columns.keys())
        missing = sorted((read_keys & columns) - set(TEMPLATE_CONFIG_KEYS))
        assert not missing, (
            "the formatter reads these template columns but no template_config "
            f"carries them, so they silently take their defaults: {missing}"
        )

    def test_every_config_key_is_a_real_column(self):
        """The reverse direction: a typo in the list must not pass silently.

        A key that is not a column reads as None from every construction site,
        which looks exactly like a field that is legitimately unset.
        """
        columns = set(LabelingPromptTemplate.__table__.columns.keys())
        bogus = sorted(set(TEMPLATE_CONFIG_KEYS) - columns)
        assert not bogus, f"TEMPLATE_CONFIG_KEYS names non-columns: {bogus}"

    def test_the_new_sampling_fields_are_carried(self):
        """The fields this arc adds must reach the renderer.

        Named explicitly rather than left to the derived checks: these two are
        the variable under test in the sampling experiment, and a config that
        silently dropped them would make every arm identical while reporting
        four distinct fingerprints.
        """
        assert "example_sampling" in TEMPLATE_CONFIG_KEYS
        assert "activation_display" in TEMPLATE_CONFIG_KEYS


class TestTheFallbackMatchesTheSchema:
    def test_the_fallback_is_built_from_the_column_defaults(self):
        """C66. Tested where the defect LIVED, not where the helper lives.

        The first version asserted `_column_default(key) == getattr(column.default,
        "arg", None)` — which is that helper's implementation retyped as its own
        oracle. It could not fail, and it never touched the fallback DICT in
        `label_features_for_extraction`, which is where the original defect was:
        ten hand-typed values, one of which ('>>>') disagreed with every seeded
        template's prime-token marker, so a run that fell back marked prime
        tokens differently from every other run and nothing reported it.

        Re-adding `template_config['prime_token_marker'] = '>>>'` after the
        comprehension reproduces that defect verbatim, and the old test stayed
        green. This one reads the fallback's construction off the AST and
        requires every value to come from `_column_default`.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_service import LabelingService

        source = textwrap.dedent(
            inspect.getsource(LabelingService.label_features_for_extraction)
        )
        tree = ast.parse(source)

        # The fallback is the dict comprehension over TEMPLATE_CONFIG_KEYS.
        comprehensions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.DictComp)
            and "TEMPLATE_CONFIG_KEYS" in ast.dump(node)
        ]
        # SELF-CHECK: no comprehension means the fallback was hand-written
        # again, which IS the defect.
        assert comprehensions, (
            "the no-template fallback is no longer derived from "
            "TEMPLATE_CONFIG_KEYS; if its values are retyped they will drift "
            "from the schema, as '>>>' did from '<<>>'"
        )

        derived = [
            c for c in comprehensions if "_column_default" in ast.dump(c.value)
        ]
        assert derived, (
            "the fallback no longer reads its values from the column defaults"
        )

        # AND NO LITERAL OVERRIDE MAY FOLLOW IT, in any of the three shapes
        # that reproduce the original drift. Checking only subscript assignment
        # left `.update({...})` and a `{**comprehension, 'key': literal}` merge
        # as open routes back to the defect.
        overrides = set()

        # (a) template_config['key'] = ...
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "template_config"
                    and isinstance(getattr(t, "slice", None), ast.Constant)
                ):
                    overrides.add(t.slice.value)

        # (b) template_config.update({'key': ...})
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "template_config"
            ):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    overrides.update(
                        k.value for k in arg.keys if isinstance(k, ast.Constant)
                    )

        # (c) {**{comprehension}, 'key': literal}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            if not any(k is None for k in node.keys):  # no ** unpacking
                continue
            overrides.update(
                k.value for k in node.keys if isinstance(k, ast.Constant)
            )
        # `max_examples` is the ONE documented job-level override.
        assert overrides <= {"max_examples"}, (
            f"template_config keys are being overwritten with literals after "
            f"construction: {sorted(overrides - {'max_examples'})}. That is how "
            f"the fallback's prime-token marker came to disagree with every "
            f"seeded template."
        )

    def test_the_marker_default_is_the_seeded_one(self):
        """Pins the specific disagreement that was found.

        A regression test written against the mechanism alone would pass if the
        default itself drifted; this names the value.
        """
        from src.services.labeling_service import _column_default

        assert _column_default("prime_token_marker") == "<<>>"
