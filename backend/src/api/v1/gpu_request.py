"""Turn a job request's ``gpu`` field into the value its row stores.

Every endpoint that starts GPU work calls :func:`resolve_gpu_request` before it
creates the job row, so an unknown card is a 400 naming the cards that exist
rather than a job that fails later in a worker. So is ``"all"`` asked of a job
that runs on one card: it would otherwise take its place on the GPU queue, wait
behind whatever holds it, and be refused only when a worker placed it.
"""

from fastapi import HTTPException, status

from ...services.gpu_placement import GpuPlacementError, GpuRequest, is_all, resolve_request

#: The refusal a job that cannot run split gives ``"all"`` — the placement's own words.
SPLIT_REFUSED = "This job cannot run split across GPUs. Choose Auto or one GPU."


def resolve_gpu_request(requested: GpuRequest, *, can_split: bool) -> str:
    """``"auto"``, ``"all"``, or the UUID of the card the request names.

    Args:
        requested: The request's ``gpu`` field.
        can_split: False for a job whose worker places it on ONE card (it does not
            pass ``allow_shard``): a J-lens fit, training on cached activations,
            and the logit lens behind the Neuronpedia export, dashboard data and
            push. Such a job refuses ``"all"`` here, at submit.

    Raises:
        HTTPException: 400 when the request names a card this node does not have,
            or asks a job that cannot split for ``"all"``.
    """
    if not can_split and is_all(requested):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=SPLIT_REFUSED)
    try:
        return resolve_request(requested)
    except GpuPlacementError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
