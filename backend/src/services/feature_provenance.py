"""Where a feature came from.

MIS-E2E-135 / -100. `Feature.training_id` is NULL for essentially every feature
in this product, BY DESIGN, and three separate consumers read it directly.

Features are extracted against an entry in the SAE registry, so the row carries
`external_sae_id` and the training — when there was one — lives one hop away on
`ExternalSAE.training_id`. A downloaded community SAE has no training at all,
and downloading one from HuggingFace is a first-class documented workflow.

The consequences were all silent:

  * the Logit Lens tab 500'd (the training lookup returned None and the endpoint
    rendered it as "Feature not found", blaming the feature it had just loaded);
  * Correlations scoped its comparison set on `Feature.training_id == NULL`,
    which in SQL matches nothing;
  * `browse_sae_features` fell through to a placeholder branch, so every feature
    of an external SAE rendered with no label, no statistics and **no
    `activation_frequency`** — which is what the frequency-derived auto-baseline
    computes from, so every such feature silently took the default strength of
    10 instead of its measured one.

`Feature.source_id` already existed and nothing used it. These resolvers are the
missing half: `source_id` answers "which dictionary", and these answer "which
training and which model", which is what the consumers actually wanted.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def resolve_training_id(db: AsyncSession, feature: Any) -> Optional[str]:
    """The training behind `feature`, following the SAE hop when needed.

    Returns None when there genuinely is no training — a downloaded SAE — which
    callers must treat as "no checkpoint to load", not as an error. That
    distinction is the whole finding: `None` here has always been a legitimate
    answer, and every consumer read it as a failure.
    """
    direct = getattr(feature, "training_id", None)
    if direct:
        return direct

    external_sae_id = getattr(feature, "external_sae_id", None)
    if not external_sae_id:
        return None

    from ..models.external_sae import ExternalSAE

    result = await db.execute(
        select(ExternalSAE.training_id).where(ExternalSAE.id == external_sae_id)
    )
    return result.scalar_one_or_none()


def resolve_training_id_sync(db, feature: Any) -> Optional[str]:
    """`resolve_training_id` for a synchronous session (Celery workers)."""
    direct = getattr(feature, "training_id", None)
    if direct:
        return direct

    external_sae_id = getattr(feature, "external_sae_id", None)
    if not external_sae_id:
        return None

    from ..models.external_sae import ExternalSAE

    return db.execute(
        select(ExternalSAE.training_id).where(ExternalSAE.id == external_sae_id)
    ).scalar_one_or_none()


def feature_scope_clause(feature: Any):
    """A SQLAlchemy clause selecting the features that share `feature`'s source.

    Use this instead of `Feature.training_id == feature.training_id`, which
    compiles to `IS NULL` for a registry-sourced feature and therefore matches
    every OTHER training-less feature in the database — or nothing at all,
    depending on the dialect. Either way it is not "the features beside this
    one".
    """
    from ..models.feature import Feature

    if getattr(feature, "external_sae_id", None):
        return Feature.external_sae_id == feature.external_sae_id
    if getattr(feature, "training_id", None):
        return Feature.training_id == feature.training_id
    # Neither set: scope to the feature itself rather than to "everything with
    # a NULL source", which is what the bare comparison did.
    return Feature.id == feature.id
