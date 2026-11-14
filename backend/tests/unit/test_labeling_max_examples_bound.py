"""What `max_examples` a labeling request may carry.

The bound used to be a hardcoded `le=50` that bore no relation to what any
extraction had retained. It was wrong in both directions:

  * against the current top-k of 25 it advertised 50, and the extra 25 were
    silently unavailable -- `TOP_K_EXAMPLES_SQL` selects `rank <= :max_examples`
    and simply returns fewer rows, with no error;
  * against the older top-k-100 extractions it REFUSED a request for more than
    50, locking out half of what was on disk.

The ceiling is now the extraction parameter's own ceiling (`top_k_examples` is
10-1000), and how many examples actually exist is bounded by the run, not by
this field. The UI narrows the control to the run's real retention; this field
only has to stop nonsense.

MUTATION CONTROLS:
  * restore `le=50`            -> `test_permits_more_than_the_old_fifty` fails
  * drop `ge=10`               -> `test_refuses_below_the_floor` fails
  * drop `le=1000`             -> `test_refuses_above_the_extraction_ceiling` fails
  * give the field a default   -> `test_absent_means_use_the_template_default` fails
"""

import pytest
from pydantic import ValidationError

from src.schemas.labeling import LabelingConfigRequest


def _request(**overrides):
    payload = {
        "extraction_job_id": "extr_1",
        "labeling_method": "openai",
        **overrides,
    }
    return LabelingConfigRequest(**payload)


def test_permits_more_than_the_old_fifty():
    """The top-k-100 extractions must be fully readable."""
    assert _request(max_examples=100).max_examples == 100


def test_refuses_below_the_floor():
    with pytest.raises(ValidationError):
        _request(max_examples=9)


def test_refuses_above_the_extraction_ceiling():
    """1000 is `top_k_examples`' own maximum; nothing can have stored more."""
    with pytest.raises(ValidationError):
        _request(max_examples=1001)


def test_absent_means_use_the_template_default():
    """None is not 25 -- it defers to the template, which the worker relies on."""
    assert _request().max_examples is None
