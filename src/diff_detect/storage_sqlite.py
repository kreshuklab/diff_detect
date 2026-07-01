import random
from typing import assert_never

import streamlit as st
from sqlmodel import Session, select

from ._state import state
from .challenges import get_available_explain_challenges
from .models import (
    ChallengeData,
    ExplainOutcome,
    ExplainTask,
    RateChallenge,
    RateChallengeId,
    RateOutcome,
    RateTask,
    TaskKey,
    User,
    UserId,
    UserKind,
    UserRole,
    get_sqlite_engine,
)


class SqliteStorage:
    """A storage backend for diff-detect that uses Sqlite."""

    def __init__(self):
        self.engine = get_sqlite_engine()

    def fetch_user(self, user_id: UserId) -> User | None:
        with Session(self.engine) as session:
            statement = select(User).where(User.id == user_id)
            return session.exec(statement).first()

    def add_user(self, user: User) -> None:
        with Session(self.engine) as session:
            session.add(user.model_copy())
            session.commit()

    def fetch_explain_outcomes(self, user_id: UserId):
        """Fetch all explanations submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(ExplainOutcome).where(ExplainOutcome.user == user_id)
            explanations = session.exec(statement)

            print(explanations)
            return list(explanations)

    def fetch_rate_outcomes(self, user_id: UserId):
        """Fetch all ratings submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(RateOutcome).where(RateOutcome.own == user_id)
            ratings = session.exec(statement)
            return list(ratings)

    @st.cache_data(scope="session")
    def fetch_random_reference_explain_outcome(
        self, own_explain_outcome: ExplainOutcome, user_kind: UserKind
    ):
        """Fetch all explanations of the same task by other users of a specific kind (human or AI)."""
        with Session(self.engine) as session:
            other_users_statement = select(User.id).where(
                User.kind == user_kind, User.id != own_explain_outcome.user
            )
            other_users = list(session.exec(other_users_statement))

            statement = (
                select(ExplainOutcome)
                .where(
                    ExplainOutcome.annotated_image
                    == own_explain_outcome.annotated_image,
                    ExplainOutcome.reference_image1
                    == own_explain_outcome.reference_image1,
                    ExplainOutcome.reference_image2
                    == own_explain_outcome.reference_image2,
                    ExplainOutcome.user in other_users,
                )
                .limit(10)
            )
            peer_ratings = list(session.exec(statement))
            if not peer_ratings:
                return None
            else:
                return random.choice(list(peer_ratings))

    # @st.cache_data(ttl=30)
    def fetch_challenges(self, user: User) -> ChallengeData:
        if state.task:
            return state.task.challenge_data

        datasets, explain_challenges = get_available_explain_challenges(
            user_role=user.role
        )

        # replace explain tasks with outcomes if they exist
        own_explain_outcomes = {
            out.task_key: out for out in self.fetch_explain_outcomes(user.id)
        }
        for challenge in explain_challenges.values():
            for task_idx in range(len(challenge.tasks)):
                task = challenge.tasks[task_idx]
                assert isinstance(task, ExplainTask)
                if task.task_key in own_explain_outcomes:
                    outcome = own_explain_outcomes[task.task_key]
                    challenge.tasks[task_idx] = outcome

        rate_outcomes = {out.task_key: out for out in self.fetch_rate_outcomes(user.id)}

        rate_challenges: dict[RateChallengeId, RateChallenge] = {}
        reference_explain_outcomes: dict[tuple[TaskKey, UserId], ExplainOutcome] = {}
        for explain_id, explain_challenge in explain_challenges.items():
            if not explain_challenge.finished:
                continue

            if explain_id == "explain_dummy":
                rate_id = "rate_dummy"
            elif explain_id == "explain_butterfly_easy":
                rate_id = "rate_butterfly_easy"
            elif explain_id == "explain_butterfly_difficult":
                rate_id = "rate_butterfly_difficult"
            else:
                assert_never(explain_id)

            tasks: list[RateTask | RateOutcome] = []
            for explain_outcome in explain_challenge.tasks:
                assert isinstance(explain_outcome, ExplainOutcome)
                if explain_outcome.task_key in rate_outcomes:
                    tasks.append(rate_outcomes[explain_outcome.task_key])
                    continue

                peer_explain_outcome = self.fetch_random_reference_explain_outcome(
                    explain_outcome, UserKind.HUMAN
                )
                if peer_explain_outcome is None:
                    # If no peer outcome is found, we cannot create a rate task for this explain outcome.
                    if user.role == UserRole.MAINTAINER:
                        # If the user is a maintainer, we can use the same outcome as both the own and peer outcome for testing purposes.
                        peer_explain_outcome = explain_outcome
                    else:
                        continue

                ai_explain_outcome = self.fetch_random_reference_explain_outcome(
                    explain_outcome, UserKind.AI
                )
                if ai_explain_outcome is None:
                    # If no AI outcome is found, we cannot create a rate task for this explain outcome.
                    if user.role == UserRole.MAINTAINER:
                        # If the user is a maintainer, we can use the same outcome as the AI outcome for testing purposes.
                        ai_explain_outcome = explain_outcome
                    else:
                        continue

                reference_explain_outcomes[
                    (peer_explain_outcome.task_key, peer_explain_outcome.user)
                ] = peer_explain_outcome

                reference_explain_outcomes[
                    (ai_explain_outcome.task_key, ai_explain_outcome.user)
                ] = ai_explain_outcome

                tasks.append(
                    RateTask(
                        annotated_image=explain_outcome.annotated_image,
                        reference_image1=explain_outcome.reference_image1,
                        reference_image2=explain_outcome.reference_image2,
                        own=user.id,
                        peer=peer_explain_outcome.user,
                        ai=ai_explain_outcome.user,
                    ),
                )

            rate_challenges[rate_id] = RateChallenge(id=rate_id, tasks=tasks)

        return ChallengeData(
            datasets=datasets,
            explain_challenges=explain_challenges,
            reference_explain_outcomes=reference_explain_outcomes,
            rate_challenges=rate_challenges,
        )
