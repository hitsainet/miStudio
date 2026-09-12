"""Is the judge itself the problem, or is this feature the problem?

A labeling failure has two very different causes, and the whole adjudication
design depends on telling them apart:

  FEATURE-LEVEL  the judge ran and could not label THIS feature — an
                 unparseable answer, no activating examples, a prompt that does
                 not fit. One feature, one attempt, one recorded reason.

  JOB-LEVEL      the judge was never in a position to answer ANYTHING — the
                 model does not exist, the endpoint refused the credentials,
                 the host is unreachable. Nothing was learned about any feature.

Recording a job-level failure as N feature-level failures is not a cosmetic
error. It writes N identical reasons, and — worse — it spends N retry attempts
on features that were never attempted in any meaningful sense.

OBSERVED IN PRODUCTION. Two resume clicks on an extraction whose March-era judge
(`granite-3.3-8b-instruct`) had since been removed from the server took 39
features from one attempt to three, exhausting their retry budget in fourteen
seconds of 404s. A third click then reported "Resume 0 of 39" and refused to do
anything, because the features had used up retries they never received.

At 14,000 features one wrong endpoint would exhaust an entire extraction in
about a minute, permanently, with nothing on the row to say the judge was
simply absent.
"""

from typing import Any, Optional

#: Exception class names that mean THE JUDGE, not the feature.
#:
#: Matched by name rather than by importing the classes: the OpenAI SDK, httpx
#: and the local path raise from different hierarchies, and this module must not
#: acquire an import dependency on all three to answer a question about a string.
JOB_LEVEL_ERROR_TYPES = frozenset(
    {
        # The model named in the job configuration is not on the server. This is
        # the one that cost 54 features their retries: a judge removed or
        # renamed between the original run and the resume.
        "NotFoundError",
        # Credentials the server will not accept. Every feature would fail
        # identically for as long as this holds.
        "AuthenticationError",
        "PermissionDeniedError",
        # The endpoint is unreachable or not answering. Distinct from a
        # per-request timeout, which IS feature-level: a slow feature is a fact
        # about that feature.
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        # The server is up but refusing work wholesale.
        "InternalServerError",
    }
)

#: Substrings that identify a job-level failure when only the message survives.
#: Deliberately narrow — a phrase that could describe one feature does not
#: belong here, because a false positive aborts a job that should continue.
JOB_LEVEL_MESSAGE_MARKERS = (
    "does not exist or has not been downloaded",
    "model not found",
    "invalid api key",
    "incorrect api key",
)


class JudgeUnavailable(RuntimeError):
    """The judge cannot answer for any feature, so the JOB has failed.

    Raised instead of writing per-feature failures. Nothing about the features
    is recorded, because nothing about them was learned.
    """


def is_job_level_failure(reason: Optional[str]) -> bool:
    """Does this reason describe the judge rather than the feature?

    Takes the recorded reason string — the same one that reaches `label_error`
    — so the classification happens in one place regardless of which service
    produced it.
    """
    if not reason:
        return False
    head = reason.split(":", 1)[0].strip()
    if head in JOB_LEVEL_ERROR_TYPES:
        return True
    lowered = reason.lower()
    return any(marker in lowered for marker in JOB_LEVEL_MESSAGE_MARKERS)


def judge_identity(config: Any) -> Optional[str]:
    """The model name a job will ask for, whatever method it uses."""
    if config is None:
        return None
    get = config.get if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))
    return (
        get("openai_compatible_model")
        or get("openai_model")
        or get("local_model")
        or None
    )
