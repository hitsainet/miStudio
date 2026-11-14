"""A scoring template must be refused, loudly, before any feature is judged.

WHY THIS EXISTS
---------------
`_format_examples_block` dispatched `template_type == 'eleutherai_detection'`
to `format_eleutherai_detection(examples=..., template_config=...)` while that
function's signature is `(feature_explanation, test_examples)`
(labeling_context_formatter.py:361-364). Every call was a guaranteed TypeError.

It was invisible because `generate_label_from_examples` wraps the render in a
broad `except Exception` and returns an error label. So selecting the seeded
`lpt_eleutherai_detection` template for a bulk run produced a job in which
EVERY feature failed, each with its own per-feature error, for a single
job-level mistake — the exact confusion between job-level and feature-level
faults this codebase has already paid for once.

A detection template scores an explanation that already exists. It cannot
produce one, and `LabelingTrialService.start_trial` has always refused them
(labeling_trial_service.py:168-172). The bulk path never learned to.

MUTATION CONTROLS:
  C58 restore the format_eleutherai_detection call in the dispatch
       -> test_the_dispatch_refuses_instead_of_miscalling
  C59 drop the is_detection_template guard from label_features_for_extraction
       -> test_the_bulk_path_refuses_a_detection_template_at_job_start
"""

import inspect

import pytest

from src.services.labeling_context_formatter import LabelingContextFormatter
from src.services.openai_labeling_service import OpenAILabelingService


class TestTheDispatchRefusesInsteadOfMiscalling:
    def test_the_dispatch_refuses_instead_of_miscalling(self):
        """C58. Calling the real dispatch, not scraping its source.

        A source guard would pass against a branch that still mis-calls, which
        is how the original defect survived. This invokes the branch.
        """
        svc = OpenAILabelingService.__new__(OpenAILabelingService)

        with pytest.raises(NotImplementedError) as excinfo:
            svc._format_examples_block(
                examples=[{
                    "prefix_tokens": ["a"], "prime_token": "b",
                    "suffix_tokens": ["c"], "max_activation": 1.0,
                }],
                template_config={"template_type": "eleutherai_detection"},
                feature_id="feat_1",
            )

        # Not merely "it raised" — a TypeError would also raise. The message
        # must explain the category error, because that is the whole point of
        # refusing at all.
        assert "scoring template" in str(excinfo.value)

    def test_a_labeling_template_still_renders(self):
        """Negative control: the guard must not refuse everything.

        A branch that raises for every template_type passes the test above and
        ships a labeling service that cannot label.
        """
        svc = OpenAILabelingService.__new__(OpenAILabelingService)

        rendered = svc._format_examples_block(
            examples=[{
                "prefix_tokens": ["the", "server"], "prime_token": "running",
                "suffix_tokens": ["fast"], "max_activation": 6.8,
            }],
            template_config={
                "template_type": "mistudio_context",
                "prime_token_marker": "<<>>",
                "include_prefix": True,
                "include_suffix": True,
            },
            feature_id="feat_1",
        )
        assert "running" in rendered

    def test_the_formatter_signature_is_what_the_dispatch_assumed_it_was_not(self):
        """Pins WHY the branch cannot simply be re-wired.

        If someone later gives `format_eleutherai_detection` an
        `(examples, template_config)` signature, the refusal above becomes
        arguable and this test says so out loud rather than letting the two
        drift back into disagreement.
        """
        params = list(inspect.signature(
            LabelingContextFormatter.format_eleutherai_detection
        ).parameters)
        assert params == ["feature_explanation", "test_examples"], (
            "the detection formatter's signature changed; revisit whether the "
            "dispatch should still refuse"
        )


class TestTheBulkPathRefusesAtJobStart:
    def test_the_bulk_path_refuses_a_detection_template_at_job_start(self):
        """C59. Job-level fault, raised once — never once per feature.

        Asserted against the real source of the guard's condition rather than
        by running a whole labeling job, because the surrounding function needs
        a live DB, an SAE and a judge. The condition and its placement are what
        matter: BEFORE the per-feature loop.
        """
        import ast
        import textwrap

        from src.services.labeling_service import LabelingService

        src = textwrap.dedent(
            inspect.getsource(LabelingService.label_features_for_extraction)
        )
        tree = ast.parse(src)

        guards = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "is_detection_template" in ast.dump(node.test)
        ]
        assert guards, (
            "the bulk path does not refuse a detection template, so every "
            "feature in the run fails individually for one job-level mistake"
        )

        for guard in guards:
            raises = [
                n for n in ast.walk(ast.Module(body=guard.body, type_ignores=[]))
                if isinstance(n, ast.Raise)
            ]
            assert raises, "the guard notices the problem and continues anyway"
