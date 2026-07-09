from __future__ import annotations

import argparse
from typing import Any, Iterable

from diff_detect.challenges import get_available_explain_challenges
from diff_detect.models import (
    ExplainOutcome,
    ExplainTask,
    User,
    UserKind,
    UserRole,
)

DUMMY_USERS = (
    ("DummyUser1A", 0, "1A"),
    ("DummyUser2A", 1, "2A"),
    ("DummyUser3A", 2, "3A"),
    ("DummyUser1B", 0, "1B"),
    ("DummyUser2B", 1, "2B"),
    ("DummyUser3B", 2, "3B"),
)


def iter_dummy_users() -> Iterable[User]:
    for user_id, _, _ in DUMMY_USERS:
        yield User(
            id=user_id,
            name=user_id,
            lab="Dummy",
            kind=UserKind.HUMAN,
            role=UserRole.PARTICIPANT,
            hashed_password=None,
        )


def iter_dummy_explain_outcomes() -> Iterable[ExplainOutcome]:
    _, explain_challenges = get_available_explain_challenges(UserRole.PARTICIPANT)
    for challenge in explain_challenges.values():
        for task in challenge.tasks:
            if not isinstance(task, ExplainTask):
                continue
            for user_id, selection_idx, explanation in DUMMY_USERS:
                selected_image = task.image_ids[selection_idx]
                reference_images = task.references_for(selected_image)
                yield ExplainOutcome(
                    dataset_id=task.dataset_id,
                    annotated_image=selected_image,
                    reference_image1=reference_images[0],
                    reference_image2=reference_images[1],
                    user=user_id,
                    explanation=explanation,
                    annotation=None,
                )


def import_dummy_explanations(storage: Any) -> list[ExplainOutcome]:
    for user in iter_dummy_users():
        if storage.fetch_user(user.id) is None:
            storage.add_user(user)

    outcomes = list(iter_dummy_explain_outcomes())
    for outcome in outcomes:
        storage.upsert_explain_outcome(outcome)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import dummy human explanations for bootstrapping rate tasks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dummy explanations and print a summary without writing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        users = list(iter_dummy_users())
        outcomes = list(iter_dummy_explain_outcomes())
        print(f"Parsed {len(outcomes)} dummy outcomes for {len(users)} users.")
        return

    from diff_detect._storage._storage_sqlite import SqliteStorage

    storage = SqliteStorage()
    outcomes = import_dummy_explanations(storage)
    print(f"Imported {len(outcomes)} dummy outcomes.")


if __name__ == "__main__":
    main()
