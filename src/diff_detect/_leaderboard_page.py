from collections import Counter, defaultdict

import streamlit as st

from diff_detect.challenges import get_available_explain_challenges
from diff_detect.common import CHALLENGE_NAMES
from diff_detect.models import (
    Dataset,
    DatasetId,
    ExplainChallenge,
    ExplainOutcome,
    ExplainTask,
    RateOutcome,
    User,
    UserKind,
    UserRole,
)

from ._page_utils import PageKey
from ._state import state
from ._storage import storage

Scores = dict[str, list[int]]
LabScores = dict[str, dict[str, int]]


def _add_score(scores: Scores, key: str, correct: bool) -> None:
    scores[key][0] += int(correct)
    scores[key][1] += 1


def _add_lab_score(scores: LabScores, lab: str, user: str, correct: bool) -> None:
    scores[lab][user] = scores[lab].get(user, 0) + int(correct)


def _user_label(user_id: str, users: dict[str, User]) -> str:
    user = users.get(user_id)
    if user is None:
        return user_id
    if user.lab:
        return f"{user.name} ({user.lab})"
    else:
        return user.name


def _lab_label(user_id: str, users: dict[str, User]) -> str:
    user = users.get(user_id)
    if user is None:
        return "Unknown"
    return user.lab or "No lab"


def _is_participant(user_id: str, users: dict[str, User]) -> bool:
    if user_id.lower().startswith("dummyuser"):
        return False
    else:
        user = users.get(user_id)
        return user is None or user.kind != UserKind.AI


def _ranked_rows(scores: Scores, label: str) -> list[dict[str, object]]:
    rows = [
        {
            label: key,
            "Score": correct,
        }
        for key, (correct, total) in scores.items()
        if total
    ]
    return sorted(rows, key=lambda row: (-row["Score"], row[label]))


def _ranked_lab_rows(scores: LabScores) -> list[dict[str, object]]:
    rows = [
        {
            "Lab": lab,
            "Score": sum(user_scores.values()) / len(user_scores),
        }
        for lab, user_scores in scores.items()
        if user_scores
    ]
    return sorted(rows, key=lambda row: (-row["Score"], row["Lab"]))


def _correct_odd_image(
    task: ExplainTask, datasets: dict[DatasetId, Dataset] | None
) -> str:
    if datasets is None or task.dataset_id not in datasets:
        return task.annotated_image

    task_images = [
        datasets[task.dataset_id].images.get(image_id) for image_id in task.image_ids
    ]
    if any(image is None for image in task_images):
        return task.annotated_image

    taxa = [
        (image.image_info.get("species"), image.image_info.get("subspecies"))
        for image in task_images
        if image is not None
    ]
    counts = Counter(taxa)
    odd_images = [
        image.image_id
        for image, taxon in zip(task_images, taxa)
        if image is not None and counts[taxon] == 1
    ]
    return odd_images[0] if len(odd_images) == 1 else task.annotated_image


def _score_explain(
    challenge: ExplainChallenge,
    outcomes: list[ExplainOutcome],
    users: dict[str, User],
    datasets: dict[DatasetId, Dataset] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers = {
        task.candidate_key: _correct_odd_image(task, datasets)
        for task in challenge.tasks
    }
    user_scores: Scores = defaultdict(lambda: [0, 0])
    lab_scores: LabScores = defaultdict(dict)
    for outcome in outcomes:
        answer = answers.get(outcome.candidate_key)
        if answer is None or not _is_participant(outcome.user, users):
            continue
        correct = outcome.annotated_image == answer
        user_label = _user_label(outcome.user, users)
        _add_score(user_scores, user_label, correct)
        _add_lab_score(lab_scores, _lab_label(outcome.user, users), user_label, correct)
    return _ranked_rows(user_scores, "User"), _ranked_lab_rows(lab_scores)


def _score_rate(
    challenge: ExplainChallenge,
    outcomes: list[RateOutcome],
    users: dict[str, User],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers = {task.candidate_key for task in challenge.tasks}
    user_scores: Scores = defaultdict(lambda: [0, 0])
    lab_scores: LabScores = defaultdict(dict)
    for outcome in outcomes:
        if outcome.candidate_key not in answers or not _is_participant(
            outcome.own, users
        ):
            continue
        if outcome.most_likely_ai is None:
            continue
        correct = outcome.most_likely_ai == outcome.ai
        user_label = _user_label(outcome.own, users)
        _add_score(user_scores, user_label, correct)
        _add_lab_score(lab_scores, _lab_label(outcome.own, users), user_label, correct)
    return _ranked_rows(user_scores, "User"), _ranked_lab_rows(lab_scores)


def _render_board(rows: list[dict[str, object]]) -> None:
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No submissions yet.")


def render_leaderboard_page() -> PageKey | None:
    st.set_page_config(initial_sidebar_state="expanded", layout="wide")

    user = state.user
    if user is None:
        return "login"

    # remove some padding from the top and sides of the page to make more room for the canvas
    st.markdown(
        """
    <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 1rem;
                padding-left: 4rem;
                padding-right: 4rem;
            }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("Leaderboard")

    datasets, challenges = get_available_explain_challenges(UserRole.PARTICIPANT)
    users = {user.id: user for user in storage.fetch_users()}
    explain_outcomes = storage.fetch_all_explain_outcomes()
    rate_outcomes = storage.fetch_all_rate_outcomes()

    for challenge_id, challenge in challenges.items():
        st.subheader(CHALLENGE_NAMES[challenge_id])
        explain_user_rows, explain_lab_rows = _score_explain(
            challenge, explain_outcomes, users, datasets
        )
        rate_user_rows, rate_lab_rows = _score_rate(challenge, rate_outcomes, users)

        explain_users, explain_labs, rate_users, rate_labs = st.columns(4)
        with explain_users:
            st.markdown("**Single specimen users**")
            _render_board(explain_user_rows)
        with explain_labs:
            st.markdown("**Single specimen labs ⌀**")
            _render_board(explain_lab_rows)
        with rate_users:
            st.markdown("**AI detection users**")
            _render_board(rate_user_rows)
        with rate_labs:
            st.markdown("**AI detection labs ⌀**")
            _render_board(rate_lab_rows)
