#!/usr/bin/env python3
"""Record the hook and layer of SAE rows that recorded neither (review R3-B, R3B-11).

A NULL ``external_saes.hook_type`` reads as residual at every refusal, and a NULL
``layer`` was turned into **layer 0 without a word** by seven consumers. Downloads and
local imports have recorded both since R3B-6; rows written before it have not, and the
consumers refuse a NULL layer only once those rows are repaired. Run this first.

    # in the backend pod, which mounts DATA_DIR:
    kubectl -n mistudio exec deploy/mistudio-backend -- \
        python scripts/backfill_sae_hooks.py            # dry run: prints the report
    kubectl -n mistudio exec deploy/mistudio-backend -- \
        python scripts/backfill_sae_hooks.py --apply    # writes

Dry run is the DEFAULT. Only NULL fields are written; a recorded hook or layer is never
overwritten, and a recorded layer that disagrees with the files is reported, not changed.
``source='trained'`` rows are excluded: their hook came from the export directory's
suffix, and a pre-A5 training export's cfg.json names ``resid_post`` for every hook, so
re-resolving one would write a wrong hook over a right one.

Idempotent: a second run finds nothing to change. The resolution and the write rules live
in ``src/db/sae_hook_backfill.py``, so they are unit-tested rather than discovered here.

Exit status: 0 on success (dry run included), 1 when ``--apply`` is refused because a job
is still reading one of the rows it would change.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Run from anywhere: the backend root holds the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.database import AsyncSessionLocal  # noqa: E402
from src.db.sae_hook_backfill import (  # noqa: E402
    BackfillBlocked,
    apply_backfill,
    format_report,
    plan_backfill,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_sae_hooks")


async def run(apply: bool) -> int:
    """Plan, report, and (with ``apply``) write. Returns the process exit status."""
    async with AsyncSessionLocal() as db:
        plans = await db.run_sync(plan_backfill)

        if not apply:
            print(format_report(plans, applied=False))
            return 0

        try:
            changed = await db.run_sync(lambda session: apply_backfill(session, plans))
        except BackfillBlocked as blocked:
            # Printed, not raised as a traceback: the operator needs the job list.
            print(format_report(plans, applied=False))
            print(f"\nREFUSED: {blocked}")
            return 1

        # Re-planned after the write, so the report shows what the rows now hold rather
        # than what they held when the plan was built.
        print(format_report(await db.run_sync(plan_backfill), applied=True))
        print(f"\nWrote {changed} row(s).")
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the changes (without it, the script only reports them)",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run(args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
