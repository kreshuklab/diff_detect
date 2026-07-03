from st_supabase_connection import SupabaseConnection

from ..models import UserId


class SupabaseStorage:
    """A storage backend for diff-detect that uses Supabase."""

    def __init__(self, supabase: SupabaseConnection):
        self.supabase = supabase

    @property
    def table(self):
        return self.supabase.table

    def fetch_user_role(self, user: UserId):
        """Fetch the role of a user from the Supabase database."""
        response = self.table("users").select("role").eq("username", user).execute()
        if not response.data:
            raise ValueError(f"User {user} not found.")
        row = response.data[0]
        if not isinstance(row, dict) or "role" not in row:
            raise ValueError(f"Invalid response for user {user}: {type(row)}")
        return row["role"]

    def fetch_selection_choices(self, user: UserId):
        """Fetch all selection choices for a given user."""
        response = self.table("selection_choices").select("*").eq("user", user).execute()
        return [SelectionChoice.model_validate(item) for item in response.data]

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
