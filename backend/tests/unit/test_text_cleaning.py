"""The text cleaner must not destroy what a model reads.

MEASURED 2026-09-12. With `enable_cleaning` on (the default at the time), stored
OpenWebText and OpenHermes tokenizations contained NONE of the 306
newline-bearing token types (0 of 614,400 sampled tokens each). The real
cleaner turned

    '<|startoftext|><|im_start|>user\\nHi<|im_end|>...'  ->  'user Hi assistant Hello'
    'if a < b and c > d:'                               ->  'if a d:'
    'Map<String, List<Integer>> m = new HashMap<>();'   ->  'Map > m = new HashMap<<<'

and a real github-code-clean file went from 149 newlines to 0.

Three patterns did it: `<[^>]+>` treated special tokens and comparisons as HTML,
`\\s+ -> ' '` flattened every line and indent, and a "4+ mixed punctuation" rule
rewrote ordinary code syntax. Cleaning is now OFF by default; these tests pin
that, pin the rewritten patterns, and pin that chat-rendered text is never
cleaned at all. The cleaner had no tests before this file.
"""

import ast
import inspect

import pytest

from src.services.tokenization_service import TokenizationService
from src.utils.text_cleaning import get_standard_cleaner


@pytest.fixture
def clean():
    return get_standard_cleaner().clean


class TestLineStructureSurvives:

    def test_paragraph_breaks_survive(self, clean):
        assert clean("First paragraph.\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."

    def test_python_indentation_and_comparisons_survive(self, clean):
        src = "def f(xs):\n    if a < b and c > d:\n        return [x for x in xs]"
        assert clean(src) == src

    def test_blank_line_runs_are_capped_not_removed(self, clean):
        assert clean("One point.\n\n\n\n\nTwo point.") == "One point.\n\nTwo point."

    def test_interior_space_runs_collapse(self, clean):
        assert clean("word     word and   more words") == "word word and more words"

    def test_trailing_spaces_are_dropped_but_the_newline_kept(self, clean):
        assert clean("a line of text   \nnext line here") == "a line of text\nnext line here"

    def test_windows_line_endings_are_unified_not_flattened(self, clean):
        assert clean("first line\r\nsecond line") == "first line\nsecond line"


class TestModelSyntaxSurvives:

    def test_chat_special_tokens_survive(self, clean):
        s = "<|startoftext|><|im_start|>user\nHi there<|im_end|>"
        assert clean(s) == s

    def test_empty_generic_and_call_survive(self, clean):
        """The old mixed-punctuation rule turned this into `HashMap<<<`."""
        s = "m = new HashMap<>();"
        assert clean(s) == s


class TestCleaningStillDoesItsJob:
    """NEGATIVE CONTROLS. A cleaner that did nothing would pass every test above."""

    def test_real_markup_is_still_removed(self, clean):
        assert clean("<p>Hello world</p> and more text") == "Hello world and more text"

    def test_same_character_punctuation_runs_are_capped(self, clean):
        assert clean("Really!!!!!!! That is amazing") == "Really!!! That is amazing"


class TestChatRenderedTextIsNeverCleaned:

    @pytest.mark.parametrize(
        "requested,rendered_chat,expected",
        [(True, False, True), (True, True, False), (False, True, False), (False, False, False)],
    )
    def test_truth_table(self, requested, rendered_chat, expected):
        assert TokenizationService.resolve_enable_cleaning(requested, rendered_chat) is expected

    @staticmethod
    def _task_node():
        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "tokenize_dataset_task":
                return node
        raise AssertionError("tokenize_dataset_task not found")

    def test_worker_rebinds_enable_cleaning_through_the_resolver_before_use(self):
        """Assert the ASSIGNMENT, not merely a call: a resolver whose result is
        computed and discarded is the defect this arc has hit repeatedly."""
        node = self._task_node()
        assign_line = None
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "enable_cleaning" for t in sub.targets)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Attribute)
                and sub.value.func.attr == "resolve_enable_cleaning"
            ):
                assign_line = sub.lineno
        assert assign_line, "enable_cleaning must be reassigned from resolve_enable_cleaning(...)"

        use_line = None
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "tokenize_dataset"
            ):
                kw = {k.arg: k.value for k in sub.keywords}
                assert isinstance(kw.get("enable_cleaning"), ast.Name)
                assert kw["enable_cleaning"].id == "enable_cleaning"
                use_line = sub.lineno
        assert use_line and assign_line < use_line, "the resolved value must be set BEFORE tokenize_dataset uses it"


class TestCleaningIsOffByDefault:

    def test_the_request_default_is_off(self):
        from src.schemas.dataset import DatasetTokenizeRequest

        assert DatasetTokenizeRequest(model_id="m_x").enable_cleaning is False

    def test_every_enable_cleaning_parameter_defaults_to_false(self):
        """Service and worker: a caller that forgets the argument gets the text
        the model reads, not a rewritten copy."""
        from src.services import tokenization_service
        from src.workers import dataset_tasks

        found = []
        for module in (tokenization_service, dataset_tasks):
            tree = ast.parse(inspect.getsource(module))
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                args = fn.args.args + fn.args.kwonlyargs
                defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
                defaults += list(fn.args.kw_defaults)
                for arg, default in zip(args, defaults):
                    if arg.arg == "enable_cleaning":
                        found.append((module.__name__, fn.name, default))
        assert found, "expected enable_cleaning parameters"
        wrong = [(m, f) for m, f, d in found if not (isinstance(d, ast.Constant) and d.value is False)]
        assert not wrong, f"these default enable_cleaning to something other than False: {wrong}"
