"""
Compatibility patches for transformers library.

This module provides workarounds for compatibility issues with newer models
that require features not yet in stable transformers releases.
"""

import logging
from typing import TypedDict, Optional

logger = logging.getLogger(__name__)


class LossKwargs(TypedDict, total=False):
    """
    Compatibility class for models requiring LossKwargs.

    This is a temporary workaround for Phi-4 and other models that use custom
    modeling code expecting LossKwargs which isn't yet in stable transformers.

    LossKwargs is used to pass loss-related arguments to model forward passes.
    """
    reduction: Optional[str]  # "mean", "sum", "none"
    label_smoothing: Optional[float]


def patch_transformers_compatibility():
    """
    Apply compatibility patches to transformers library.

    This adds missing classes/functions that some models expect.
    Should be called once at application startup.
    """
    import transformers.utils

    # Add LossKwargs if it doesn't exist
    if not hasattr(transformers.utils, 'LossKwargs'):
        transformers.utils.LossKwargs = LossKwargs
        logger.info("Applied LossKwargs compatibility patch to transformers.utils")

    logger.info("Transformers compatibility patches applied successfully")
