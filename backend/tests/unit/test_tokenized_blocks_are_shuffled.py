"""A tokenization's blocks are permuted, so an extraction prefix samples the whole corpus.

WHY THIS FILE EXISTS. Activation extraction reads a tokenized dataset with
`dataset.select(range(max_samples))` (`services/activation_service.py`) — a straight
PREFIX from block 0 — and nothing upstream shuffled: the raw dataset is read in
download order, `dataset.map` preserves order, and `iter_packed_blocks` emits
blocks sequentially. Verified absent repo-wide in the dataset path on 2026-09-17:
`.shuffle(`, `.select(`, `.sort(`, `train_test_split`.

So every extraction in this estate used the FIRST N blocks of each corpus, in
original order — Bloomberg 9,000 of 138,400 blocks (6.5%), codeparrot 13,500 of
55,926.

MEASURED 2026-09-17, after this shipped, by decoding the corpora themselves.
Bloomberg and codeparrot — the two this file originally accused — are NOT
ordered: Bloomberg scores Spearman -0.06 between row order and article date
(4,025 of 446,762 rows sampled), and codeparrot's first quarter spans 30.3% of
its 33,090 repositories. The real case is OpenHermes-2.5, which is written one
source at a time: 14 of its 15 labeled sources occupy a single contiguous row
range, so the first 10% of rows holds only airoboros2.2 and CamelAI — 11.3% of
the corpus — and never reaches glaive-code-assist (18.2%) or the unlabeled
49.6% remainder. The defect is real; the prediction of where it lived was not.

THE BIAS IS UNDETECTABLE DOWNSTREAM. A dictionary trained on one slice looks
healthy on every metric this project collects — train_47db6506 reached FVU 0.17,
CE delta 0.975 nats and zero dead latents — because the held-out evaluation draws
its rows from just AFTER the prefix (`select_unseen_rows`): adjacent data, same
slice, same bias. No metric can see it, which is why the fix has to be structural
and why it is pinned here.

WHY A FULL PERMUTATION AND NOT A SHUFFLE BUFFER. The requirement is that a prefix
sample the whole corpus. A buffered shuffle mixes locally and leaves the far end
of a large corpus unreachable to a short prefix; only a full permutation makes
block i a uniformly random block from anywhere. `test_the_first_tenth_spans_every_decile`
is the executable form of that requirement — a buffer fails it.

WHY THE DECISION IS EXTRACTED. `resolve_shuffle_seed` and `shuffle_blocks` are
small pure functions on `TokenizationService` rather than inline in a 1,200-line
Celery task. Inline, the only ways to test them are to drive the whole task or to
scrape source, and a source scrape fails open — this repo has five recorded cases
of a guard satisfied by the wrong occurrence, including two that left the full
suite green. So: unit-test the decision, assert the CALL by AST.

MUTATION CONTROLS (2026-09-17; each applied alone, source restored by bytes and
the sha256 verified):
  S1 `shuffle_blocks` returns the dataset unchanged
        -> test_the_first_tenth_spans_every_decile, test_every_block_survives_the_permutation
  S2 `.flatten_indices()` dropped from `shuffle_blocks`
        -> test_the_new_order_is_written_down_physically
  S3 `resolve_shuffle_seed` uses `shuffle_seed or <derived>` instead of `is not None`
        -> test_seed_zero_is_a_real_seed_not_a_sentinel
  S4 the derived seed drops `max_length` from its key
        -> test_the_seed_is_keyed_on_the_whole_identity
  S5 the worker's `drop_remainder` hardcoded False
        -> test_a_shuffled_pack_leaves_no_ragged_block
  S6 the worker calls `.shuffle()` inline instead of `shuffle_blocks`
        -> test_the_worker_calls_the_extracted_decisions
"""

import ast
import inspect

import pytest
from datasets import Dataset

from src.services.tokenization_service import TokenizationService
from src.workers import dataset_tasks


BLOCKS = 1000
PREFIX = 100  # what an extraction with a small max_samples would take


def _ordered_corpus(n: int = BLOCKS) -> Dataset:
    """Blocks that carry their own original position, so order is recoverable."""
    return Dataset.from_dict({
        "input_ids": [[i] for i in range(n)],
        "attention_mask": [[1] for _ in range(n)],
    })


def _positions(dataset: Dataset) -> list:
    return [row[0] for row in dataset["input_ids"]]


# ── the requirement: a prefix must sample the WHOLE corpus ───────────────────

class TestAPrefixSamplesTheWholeCorpus:
    """The property the extraction prefix actually depends on."""

    def test_the_first_tenth_spans_every_decile(self):
        """THE REQUIREMENT. A shuffle buffer would fail this; a permutation passes.

        Extraction takes the first N blocks. Those N must come from everywhere,
        not from the opening slice — that is the whole defect.
        """
        shuffled = TokenizationService.shuffle_blocks(_ordered_corpus(), seed=1234)
        prefix = _positions(shuffled)[:PREFIX]
        deciles = {p // (BLOCKS // 10) for p in prefix}
        assert deciles == set(range(10)), (
            f"the first {PREFIX} blocks touch only deciles {sorted(deciles)} of the "
            "corpus; an extraction would see that slice and no other"
        )

    def test_the_prefix_is_not_the_opening_slice(self):
        """NEGATIVE CONTROL for the above: unshuffled, the prefix IS deciles 0-0."""
        prefix = _positions(_ordered_corpus())[:PREFIX]
        deciles = {p // (BLOCKS // 10) for p in prefix}
        assert deciles == {0}, "the unshuffled corpus should be exactly the defect"

    def test_every_block_survives_the_permutation(self):
        """A permutation, not a sample: nothing added, nothing lost, nothing doubled."""
        shuffled = TokenizationService.shuffle_blocks(_ordered_corpus(), seed=7)
        assert sorted(_positions(shuffled)) == list(range(BLOCKS))

    def test_the_new_order_is_written_down_physically(self):
        """`shuffle()` alone leaves an indices mapping over data still in corpus order.

        Every later reader would then indirect through it, and the saved directory
        would carry an `indices.arrow` the existing tokenizations do not have.
        """
        shuffled = TokenizationService.shuffle_blocks(_ordered_corpus(), seed=7)
        assert shuffled._indices is None, (
            "the permutation is a view; the data is still physically in corpus order"
        )


# ── the seed: reproducible, and 0 is a real value ────────────────────────────

class TestTheSeedIsHonestAboutItself:

    def test_the_same_seed_reproduces_the_same_order(self):
        a = _positions(TokenizationService.shuffle_blocks(_ordered_corpus(), seed=99))
        b = _positions(TokenizationService.shuffle_blocks(_ordered_corpus(), seed=99))
        assert a == b

    def test_a_different_seed_gives_a_different_order(self):
        a = _positions(TokenizationService.shuffle_blocks(_ordered_corpus(), seed=1))
        b = _positions(TokenizationService.shuffle_blocks(_ordered_corpus(), seed=2))
        assert a != b

    def test_no_seed_means_no_shuffle_and_the_same_object(self):
        corpus = _ordered_corpus()
        assert TokenizationService.shuffle_blocks(corpus, seed=None) is corpus

    def test_a_derived_seed_is_deterministic(self):
        kw = dict(dataset_id="ds-1", model_id="m_f0271325", max_length=2048)
        assert (TokenizationService.resolve_shuffle_seed(True, None, **kw)
                == TokenizationService.resolve_shuffle_seed(True, None, **kw))

    def test_seed_zero_is_a_real_seed_not_a_sentinel(self):
        """MUTATION S3. `shuffle_seed or derived` would silently discard 0."""
        got = TokenizationService.resolve_shuffle_seed(
            True, 0, dataset_id="ds-1", model_id="m", max_length=2048
        )
        assert got == 0, "an explicit seed of 0 was replaced by a derived one"

    @pytest.mark.parametrize(
        "changed",
        [
            dict(dataset_id="ds-2"),
            dict(model_id="m_other"),
            dict(max_length=1024),
        ],
    )
    def test_the_seed_is_keyed_on_the_whole_identity(self, changed):
        """MUTATION S4. The triple is what identifies the row AND its directory."""
        base = dict(dataset_id="ds-1", model_id="m_f0271325", max_length=2048)
        other = {**base, **changed}
        assert (TokenizationService.resolve_shuffle_seed(True, None, **base)
                != TokenizationService.resolve_shuffle_seed(True, None, **other))

    def test_not_shuffling_records_no_seed(self):
        """None is the ONLY way to say "unshuffled" — no sentinel is available."""
        assert TokenizationService.resolve_shuffle_seed(
            False, 77, dataset_id="ds-1", model_id="m", max_length=2048
        ) is None


# ── the ragged block must not reach the packed-corpus probe ──────────────────

class TestAShuffledPackLeavesNoRaggedBlock:
    """`_dataset_looks_packed` probes the FIRST 32 rows and requires every
    attention_mask to be all-ones. Unshuffled, the one padded block is guaranteed
    last. Shuffled it lands anywhere — P = 32/N, negligible on OpenWebText
    (0.006%) but ~32% on a small corpus — and if it lands in the probe the
    "max_samples selects BLOCKS, not documents" warning silently stops firing.
    """

    def test_dropping_the_remainder_leaves_every_mask_full(self):
        blocks = list(TokenizationService.iter_packed_blocks(
            [list(range(17))] * 20, max_length=8, eos_token_id=None,
            drop_remainder=True,
        ))
        assert blocks, "the fixture produced no blocks"
        assert all(set(b["attention_mask"]) == {1} for b in blocks), (
            "a padded block survived; shuffled, it can land in the 32-row probe"
        )

    def test_keeping_the_remainder_leaves_exactly_one_ragged_block(self):
        """NEGATIVE CONTROL: the ragged block is real, and only the last one."""
        blocks = list(TokenizationService.iter_packed_blocks(
            [list(range(17))] * 20, max_length=8, eos_token_id=None,
            drop_remainder=False,
        ))
        ragged = [i for i, b in enumerate(blocks) if set(b["attention_mask"]) != {1}]
        assert ragged == [len(blocks) - 1]


# ── wiring: the worker must call these, and by AST ───────────────────────────

class TestTheWorkerCallsTheExtractedDecisions:
    """A capability is not shipped until a test fails when its wiring is removed.

    These read the AST for a CALL, not the text for a NAME: this file and the
    worker both explain the mechanism in prose, and a substring search matches the
    comments — a guard that passes for the wrong reason.
    """

    @staticmethod
    def _service_calls():
        tree = ast.parse(inspect.getsource(dataset_tasks))
        task = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "tokenize_dataset_task")
        return [
            n.func.attr for n in ast.walk(task)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and getattr(n.func.value, "id", None) == "TokenizationService"
        ]

    @pytest.mark.parametrize("name", ["resolve_shuffle_seed", "shuffle_blocks"])
    def test_the_worker_calls_the_extracted_decisions(self, name):
        """MUTATION S6."""
        assert name in self._service_calls(), (
            f"tokenize_dataset_task never calls {name}; the extraction is dead code "
            "and the rule it holds is not the rule that runs"
        )

    def test_the_worker_keeps_no_second_copy_of_the_rule(self):
        """Two copies drift. That divergence is what this whole change is about."""
        src = inspect.getsource(dataset_tasks)
        assert "zlib.crc32" not in src, "the worker derives its own seed again"
        assert ".shuffle(seed=" not in src, "the worker shuffles inline again"

    def test_drop_remainder_is_tied_to_the_seed_not_hardcoded(self):
        """MUTATION S5. Hardcoding False lets the padded block reach the probe."""
        tree = ast.parse(inspect.getsource(dataset_tasks))
        packed = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "iter_packed_blocks"
        ]
        assert packed, "the packer is no longer called"
        kw = {k.arg: k.value for c in packed for k in c.keywords}
        assert "drop_remainder" in kw, "the packer keeps its ragged block when shuffling"
        assert not isinstance(kw["drop_remainder"], ast.Constant), (
            "drop_remainder is a constant; it must follow whether a seed is in play"
        )


class TestTheEndpointForwardsTheRequest:
    """Accepted by the schema is not the same as reaching the worker. Three fields
    in this arc were accepted and never forwarded, so the row recorded defaults."""

    def test_the_tokenize_endpoint_forwards_both_fields(self):
        from src.api.v1.endpoints import datasets as datasets_endpoint

        tree = ast.parse(inspect.getsource(datasets_endpoint))
        delays = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "delay"
            and getattr(n.func.value, "id", "") == "tokenize_dataset_task"
        ]
        assert delays, "the tokenize endpoint no longer dispatches the task"
        forwarded = {k.arg for c in delays for k in c.keywords}
        for field in ("shuffle", "shuffle_seed"):
            assert field in forwarded, (
                f"{field} is accepted by the schema and never forwarded; the blocks "
                "are written in corpus order while the row says otherwise"
            )
