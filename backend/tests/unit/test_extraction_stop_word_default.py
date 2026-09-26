"""Stop-word filtering is ON by default for extraction.

THE EVIDENCE. The 16k L12 pair differed in exactly this one field
(`extr_20260919_135431` off, `extr_20260920_070000` on), same SAE, same five
corpora, same top-k, same sample count:

    single-token evidence degeneracy   23.02%  ->  17.04%
    mean distinct prime tokens          8.19   ->  10.47
    'Ġthe' as a stored prime token     12,739  ->  0
    features that lost evidence                    0
    runtime                            4h56m   ->  3h49m

Nothing was starved: every one of the 16,381 features still filled all 25 slots,
from non-stop-word peaks. So the function-word degeneracy was an
evidence-SELECTION artefact, not a property of the dictionary.

WHY THIS TEST EXISTS. A default is a decision, and an untested default is a
decision nothing defends. Flipping it back must go red rather than silently
change what every future extraction stores.

MUTATION CONTROLS:
  * set any of the three `filter_stop_words` defaults back to False
      -> the corresponding test below fails
  * change a SIBLING filter's default (e.g. filter_fragments -> False)
      -> `test_the_other_filters_are_unchanged` fails, proving this change did
         not quietly move anything else
"""

from src.schemas.extraction import (
    BatchExtractionRequest,
    ExtractionConfigRequest,
)


def test_single_extraction_filters_stop_words_by_default():
    assert ExtractionConfigRequest().filter_stop_words is True


def test_batch_extraction_filters_stop_words_by_default():
    """A batch run must not quietly differ from a single run."""
    assert BatchExtractionRequest(sae_ids=['sae_1'], dataset_id='ds_1').filter_stop_words is True


def test_an_explicit_false_is_still_honoured():
    """The default is a default, not a policy — the old behaviour stays reachable."""
    assert ExtractionConfigRequest(filter_stop_words=False).filter_stop_words is False


def test_the_other_filters_are_unchanged():
    """Negative control: this change moved ONE field, not the filter block."""
    cfg = ExtractionConfigRequest()

    assert cfg.filter_special is True
    assert cfg.filter_single_char is True
    assert cfg.filter_punctuation is True
    assert cfg.filter_numbers is True
    assert cfg.filter_fragments is True
