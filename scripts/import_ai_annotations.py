from __future__ import annotations

import argparse
from pathlib import Path

from diff_detect.ai_annotations import (
    AI_USER_ID,
    import_ai_annotations,
    iter_ai_annotation_outcomes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import AI bounding-box annotations as ExplainOutcome rows."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Optional single download directory. Defaults to Butterfly and FlyButter."
        ),
    )
    parser.add_argument(
        "--user-id",
        default=AI_USER_ID,
        help="Base user id prefix for imported AI annotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse annotations and print a summary without writing to the database.",
    )
    args = parser.parse_args()

    if args.dry_run:
        outcomes = list(
            iter_ai_annotation_outcomes(args.download_dir, user_id=args.user_id)
        )
        object_count = sum(
            0 if outcome.annotation is None else len(outcome.annotation.raw.objects)
            for outcome in outcomes
        )
        print(
            f"Parsed {len(outcomes)} AI outcomes with {object_count} boxes "
            f"for user prefix {args.user_id!r}."
        )
        return

    from diff_detect._storage._storage_sqlite import SqliteStorage

    storage = SqliteStorage()
    outcomes = import_ai_annotations(
        storage, args.download_dir, user_id=args.user_id
    )
    print(f"Imported {len(outcomes)} AI outcomes for user prefix {args.user_id!r}.")


if __name__ == "__main__":
    main()
