import random
from typing import assert_never

from sqlalchemy import Engine, text
from sqlmodel import Session, select

from ..challenges import get_available_explain_challenges
from ..models import (
    ChallengeData,
    ChallengeId,
    ExplainChallenge,
    ExplainOutcome,
    ExplainTask,
    RateChallenge,
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

    def __init__(self, *, engine: Engine | None = None):
        self.engine = engine if engine is not None else get_sqlite_engine()
        self._allow_partial_rate_outcomes()

    def _allow_partial_rate_outcomes(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return

        with self.engine.begin() as connection:
            columns = list(connection.execute(text("PRAGMA table_info(rateoutcome)")))
            if not columns:
                return
            if not any(
                column.name in {"most_convincing", "most_likely_ai"} and column.notnull
                for column in columns
            ):
                return

            connection.execute(text("ALTER TABLE rateoutcome RENAME TO rateoutcome_old"))
            RateOutcome.__table__.create(connection)
            column_names = [column.name for column in columns]
            copied_columns = ", ".join(column_names)
            connection.execute(
                text(
                    f"INSERT INTO rateoutcome ({copied_columns}) "
                    f"SELECT {copied_columns} FROM rateoutcome_old"
                )
            )
            connection.execute(text("DROP TABLE rateoutcome_old"))

    def fetch_user(self, user_id: UserId) -> User | None:
        with Session(self.engine) as session:
            statement = select(User).where(User.id == user_id)
            return session.exec(statement).first()

    def fetch_lab_options(self) -> list[str]:
        with Session(self.engine) as session:
            statement = select(User.lab).distinct()
            labs = session.exec(statement)
            return [lab for lab in labs if lab is not None]

    def fetch_users(self) -> list[User]:
        with Session(self.engine) as session:
            return list(session.exec(select(User)))

    def add_user(self, user: User) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(user)
            session.commit()

    def upsert_explain_outcome(self, outcome: ExplainOutcome) -> None:
        """Create or replace a user's explanation for one task."""
        with Session(self.engine) as session:
            existing_outcomes = session.exec(
                select(ExplainOutcome).where(ExplainOutcome.user == outcome.user)
            )
            for existing_outcome in existing_outcomes:
                if (
                    existing_outcome.candidate_key == outcome.candidate_key
                    and existing_outcome.task_key != outcome.task_key
                ):
                    session.delete(existing_outcome)

            session.merge(outcome)
            session.commit()

    def delete_explain_outcome(self, outcome: ExplainOutcome) -> None:
        """Delete a user's explanation for one task."""
        with Session(self.engine) as session:
            existing_outcomes = session.exec(
                select(ExplainOutcome).where(ExplainOutcome.user == outcome.user)
            )
            for existing_outcome in existing_outcomes:
                if existing_outcome.candidate_key == outcome.candidate_key:
                    session.delete(existing_outcome)
            session.commit()

    def fetch_explain_outcomes(self, user_id: UserId):
        """Fetch all explanations submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(ExplainOutcome).where(ExplainOutcome.user == user_id)
            explanations = session.exec(statement)

            return list(explanations)

    def fetch_all_explain_outcomes(self) -> list[ExplainOutcome]:
        with Session(self.engine) as session:
            return list(session.exec(select(ExplainOutcome)))

    def fetch_rate_outcomes(self, user_id: UserId):
        """Fetch all ratings submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(RateOutcome).where(RateOutcome.own == user_id)
            ratings = session.exec(statement)
            return list(ratings)

    def fetch_all_rate_outcomes(self) -> list[RateOutcome]:
        with Session(self.engine) as session:
            return list(session.exec(select(RateOutcome)))

    def upsert_rate_outcome(self, outcome: RateOutcome) -> None:
        """Create or replace a user's rating for one task."""
        with Session(self.engine) as session:
            session.merge(outcome)
            session.commit()

    def fetch_reference_explain_outcome(
        self, candidate_key: TaskKey, user_id: UserId
    ) -> ExplainOutcome | None:
        with Session(self.engine) as session:
            outcomes = session.exec(
                select(ExplainOutcome).where(ExplainOutcome.user == user_id)
            )
            for outcome in outcomes:
                if outcome.candidate_key == candidate_key:
                    return outcome

        return None

    def fetch_random_reference_explain_outcome(
        self, own_explain_outcome: ExplainOutcome, user_kind: UserKind
    ):
        """Fetch all explanations of the same task by other users of a specific kind (human or AI)."""
        with Session(self.engine) as session:
            other_users_statement = select(User.id).where(
                User.kind == user_kind, User.id != own_explain_outcome.user
            )
            other_users = list(session.exec(other_users_statement))
            if not other_users:
                return None

            statement = (
                select(ExplainOutcome)
                .where(
                    ExplainOutcome.user.in_(other_users),
                )
            )
            peer_ratings = [
                outcome
                for outcome in session.exec(statement)
                if outcome.candidate_key == own_explain_outcome.candidate_key
            ]
            if not peer_ratings:
                return None
            else:
                return random.choice(list(peer_ratings))

    def fetch_challenges(self, user: User) -> ChallengeData:
        datasets, explain_challenges = get_available_explain_challenges(
            user_role=user.role
        )

        # replace explain tasks with outcomes if they exist
        own_explain_outcomes = {
            out.candidate_key: out for out in self.fetch_explain_outcomes(user.id)
        }
        for challenge in explain_challenges.values():
            for task_idx in range(len(challenge.tasks)):
                task = challenge.tasks[task_idx]
                assert isinstance(task, ExplainTask)
                if task.candidate_key in own_explain_outcomes:
                    outcome = own_explain_outcomes[task.candidate_key]
                    challenge.tasks[task_idx] = outcome

        rate_outcomes = {
            out.candidate_key: out for out in self.fetch_rate_outcomes(user.id)
        }

        challenges: dict[ChallengeId, ExplainChallenge | RateChallenge] = {
            k: v for k, v in explain_challenges.items()
        }
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
                reference_explain_outcomes[
                    (explain_outcome.candidate_key, explain_outcome.user)
                ] = explain_outcome

                rate_outcome = rate_outcomes.get(explain_outcome.candidate_key)
                if rate_outcome is not None:
                    for reference_user in (rate_outcome.peer, rate_outcome.ai):
                        reference_outcome = self.fetch_reference_explain_outcome(
                            explain_outcome.candidate_key, reference_user
                        )
                        if reference_outcome is not None:
                            reference_explain_outcomes[
                                (
                                    reference_outcome.candidate_key,
                                    reference_outcome.user,
                                )
                            ] = reference_outcome

                    tasks.append(rate_outcome)
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
                    (peer_explain_outcome.candidate_key, peer_explain_outcome.user)
                ] = peer_explain_outcome

                reference_explain_outcomes[
                    (ai_explain_outcome.candidate_key, ai_explain_outcome.user)
                ] = ai_explain_outcome

                tasks.append(
                    RateTask(
                        dataset_id=explain_outcome.dataset_id,
                        annotated_image=explain_outcome.annotated_image,
                        reference_image1=explain_outcome.reference_image1,
                        reference_image2=explain_outcome.reference_image2,
                        own=user.id,
                        peer=peer_explain_outcome.user,
                        ai=ai_explain_outcome.user,
                    ),
                )

            challenges[rate_id] = RateChallenge(id=rate_id, tasks=tasks)

        return ChallengeData(
            datasets=datasets,
            reference_explain_outcomes=reference_explain_outcomes,
            challenges=challenges,
        )
