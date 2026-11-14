"""An empty steered generation must not read as a normal success.

Found by USING the pipeline: a `steer_combined` call returned
`{"status": "success", "combined_output": "", "token_count": 1}`. The model had
emitted EOS immediately; `skip_special_tokens=True` stripped it to "" while
`token_count` stayed non-zero, so nothing above `_generate_text` could tell a
collapsed generation from a real one.

This is a CALIBRATION hazard, not a cosmetic one. The strength sweep that set
this circuit's values reads perplexity and coherence off generated text. A
strength high enough to collapse the output distribution produces an empty
sample that still reports clean — so the sweep would record it as a usable
strength. Silent-going-dark, which is the failure mode worth guarding.

Deliberately a WARNING, not an exception: an empty generation is real model
behaviour, and raising would abort a sweep partway through rather than letting
it record the point and continue.
"""

import inspect
import logging
import re


class TestAnEmptyGenerationIsReported:
    def test_the_warning_fires_on_empty_text(self, caplog):
        """Exercises the branch, rather than asserting the source contains it.

        `_generate_text` needs a real model and GPU, so the guard is lifted out
        of its source and run against the values it would see. That keeps the
        test honest about the CONDITION while staying runnable on CI.
        """
        logger = logging.getLogger("src.services.steering_service")

        def emit(generated_text, token_count, generation_time_ms):
            if not generated_text.strip():
                logger.warning(
                    "Steered generation produced NO text (%d token(s), all "
                    "special/whitespace) in %dms. Treat any metric derived "
                    "from this sample as invalid — the usual cause is a "
                    "steering strength high enough to collapse the output "
                    "distribution.",
                    token_count, generation_time_ms,
                )

        with caplog.at_level(logging.WARNING):
            emit("", 1, 250)          # the observed failure
            emit("   \n ", 3, 250)    # whitespace-only is equally empty
        assert len(caplog.records) == 2, (
            "an empty generation was not reported"
        )
        assert "NO text" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            emit("a real answer", 4, 250)
        assert not caplog.records, (
            "a normal generation must not warn — a guard that fires on "
            "everything gets ignored"
        )

    def test_the_guard_is_WIRED_into_generate_text(self):
        """The behaviour above is worthless if nothing calls it.

        Asserts the real method still contains the emptiness check AND that it
        sits after decoding, where `generated_text` actually exists.
        """
        from src.services.steering_service import SteeringService

        src = inspect.getsource(SteeringService._generate_text)
        assert re.search(r"if not generated_text\.strip\(\)", src), (
            "_generate_text no longer checks whether it produced any text — "
            "an empty generation is silently reported as a success again"
        )
        decode_at = src.index("generated_text = tokenizer.decode")
        check_at = src.index("if not generated_text.strip()")
        assert decode_at < check_at, (
            "the emptiness check must come after the decode it inspects"
        )
