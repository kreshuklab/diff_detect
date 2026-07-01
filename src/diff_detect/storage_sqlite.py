from sqlmodel import Session, create_engine, select

from .models import ExplainedDifference, ExplanationRating, User, UserId, UserRole


class SqliteStorage:
    """A storage backend for diff-detect that uses Sqlite."""

    def __init__(self):
        self.engine = create_engine("sqlite:///database.db")

    def fetch_user_role(self, user_id: UserId) -> UserRole:
        with Session(self.engine) as session:
            statement = select(User).where(User.id == user_id)
            user = session.exec(statement).first()
            if user is None:
                raise ValueError(f"User {user_id} not found.")

            return user.role

    def fetch_explanations(self, user_id: UserId):
        """Fetch all explanations submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(ExplainedDifference).where(ExplainedDifference.user == user_id)
            explanations = session.exec(statement)

            print(explanations)
            return list(explanations)

    def fetch_ratings(self, user_id: UserId):
        """Fetch all ratings submitted by a given user."""
        with Session(self.engine) as session:
            statement = select(ExplanationRating).where(ExplanationRating.self == user_id)
            ratings = session.exec(statement)
            return list(ratings)

    def load_challenge_progress(
        user: UserId
    ) -> :
        progress: list[ChallengeProgress] = []
        dataset_ids = sorted(
            {challenge.dataset_id for challenge in challenges.values()}
        )
        for challenge_id in dataset_ids:
            submissions = fetch_user_submissions(supabase, username, dataset_id)
            ratings = fetch_user_ratings(supabase, username, dataset_id)
            dataset_challenges = sorted(
                (
                    challenge
                    for challenge in challenges.values()
                    if challenge.dataset_id == dataset_id
                ),
                key=lambda challenge: challenge.challenge_id,
            )
            for challenge in dataset_challenges:
                challenge_id = challenge.challenge_id
                if challenge.load_error:
                    progress.append(
                        ChallengeProgress(
                            dataset_id=dataset_id,
                            challenge_id=challenge_id,
                            task_count=0,
                            submitted_count=0,
                            rated_count=0,
                            load_error=challenge.load_error,
                        )
                    )
                    continue

                task_ids = {task.task_id for task in challenge.rounds}
                progress.append(
                    ChallengeProgress(
                        dataset_id=dataset_id,
                        challenge_id=challenge_id,
                        task_count=challenge.task_count,
                        submitted_count=len(completed_task_ids(submissions, task_ids)),
                        rated_count=len(completed_task_ids(ratings, task_ids)),
                    )
                )
        return progress


# def is_list_of_dicts(data: Any) -> TypeGuard[list[dict[str, Any]]]:
#     return isinstance(data, list) and all(isinstance(item, dict) for item in data)


# def fetch_selection_choices(
#     supabase: SupabaseConnection, user: UserId, dataset: DatasetId
# ) -> list[SelectionChoice]:
#     """Fetch all selection choices for a given user and dataset."""
#     response = (
#         supabase.table("selection_choices")
#         .select("*")
#         .eq("username", user)
#         .eq("dataset_id", dataset)
#         .execute()
#     )
#     return [SelectionChoice.model_validate(item) for item in response.data]
