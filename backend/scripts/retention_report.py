#!/usr/bin/env python3
"""READ-ONLY checkpoint-retention report (OSD-36a/36b, suite decision D10).

Prints what the retention policy WOULD prune — row-driven plans per training plus the
orphaned step directories no row references — and deletes nothing, whatever
`checkpoint_prune_dry_run` says. This is the report an operator reads BEFORE turning
`dry_run` off.

    python scripts/retention_report.py            # human summary
    python scripts/retention_report.py --json     # the full report

In the cluster (the backend pod has the database and /data):

    microk8s kubectl -n mistudio exec deploy/mistudio-backend -c backend -- \\
        python scripts/retention_report.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    from src.core.config import settings
    from src.core.database import get_sync_db
    from src.services.checkpoint_retention import retention_report

    with get_sync_db() as db:
        report = retention_report(db, settings.data_dir)
        db.rollback()  # read-only: nothing to keep, and nothing is written

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    p = report["policy"]
    print(f"policy: enabled={p['enabled']} dry_run={p['dry_run']} keep_last={p['keep_last']} "
          f"keep_best={p['keep_best']} min_age_hours={p['min_age_hours']}")
    print(f"data_dir: {settings.data_dir}   checkpoint rows in table: {report['checkpoint_rows_total']}")
    for t in report["trainings"]:
        print(f"  {t['training_id']:<18} {t['rows']:>4} rows  ~{t['estimated_bytes'] / 1e9:7.1f} GB  "
              f"prune steps {t['prunable_steps']}  keep {t['kept_steps']}")
    print(f"ROW-DRIVEN TOTAL: {report['rows_to_prune']} rows, ~{report['estimated_bytes'] / 1e9:.1f} GB")
    print(f"ORPHANED STEP DIRS (OSD-36b): {len(report['orphaned_dirs'])}, "
          f"~{report['orphaned_bytes'] / 1e9:.1f} GB")
    for o in report["orphaned_dirs"]:
        print(f"  {o['path']}  ({o['reason']}, ~{o['size_bytes'] / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
