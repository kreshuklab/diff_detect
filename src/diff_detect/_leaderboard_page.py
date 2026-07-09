from collections import defaultdict

import streamlit as st

from diff_detect.challenges import get_available_explain_challenges
from diff_detect.common import CHALLENGE_NAMES
from diff_detect.models import (
    ExplainChallenge,
    ExplainOutcome,
    RateOutcome,
    User,
    UserKind,
    UserRole,
)

from ._page_utils import PageKey
from ._state import state
from ._storage import storage

Scores = dict[str, list[int]]


def _add_score(scores: Scores, key: str, correct: bool) -> None:
    scores[key][0] += int(correct)
    scores[key][1] += 1


def _user_label(user_id: str, users: dict[str, User]) -> str:
    user = users.get(user_id)
    if user is None:
        return user_id
    if user.name == user.id:
        return user.id
    return f"{user.name} ({user.id})"


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
            "Correct": correct,  # for sorting
            # "Total": total,
            # "Accuracy": correct / total,
            "Score": correct,
        }
        for key, (correct, total) in scores.items()
        if total
    ]
    # rows = sorted(rows, key=lambda row: (-row["Accuracy"], -row["Correct"], row[label]))
    rows = sorted(rows, key=lambda row: (-row["Correct"], row[label]))
    # drop "Correct" column
    rows = [{k: v for k, v in row.items() if k != "Correct"} for row in rows]
    # for row in rows:
    #     row["Accuracy"] = f"{row['Accuracy']:.0%}"
    return rows


def _score_explain(
    challenge: ExplainChallenge,
    outcomes: list[ExplainOutcome],
    users: dict[str, User],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers = {task.candidate_key: task.annotated_image for task in challenge.tasks}
    user_scores: Scores = defaultdict(lambda: [0, 0])
    lab_scores: Scores = defaultdict(lambda: [0, 0])
    for outcome in outcomes:
        answer = answers.get(outcome.candidate_key)
        if answer is None or not _is_participant(outcome.user, users):
            continue
        correct = outcome.annotated_image == answer
        _add_score(user_scores, _user_label(outcome.user, users), correct)
        _add_score(lab_scores, _lab_label(outcome.user, users), correct)
    return _ranked_rows(user_scores, "User"), _ranked_rows(lab_scores, "Lab")


def _score_rate(
    challenge: ExplainChallenge,
    outcomes: list[RateOutcome],
    users: dict[str, User],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers = {task.candidate_key for task in challenge.tasks}
    user_scores: Scores = defaultdict(lambda: [0, 0])
    lab_scores: Scores = defaultdict(lambda: [0, 0])
    for outcome in outcomes:
        if outcome.candidate_key not in answers or not _is_participant(
            outcome.own, users
        ):
            continue
        if outcome.most_likely_ai is None:
            continue
        correct = outcome.most_likely_ai == outcome.ai
        _add_score(user_scores, _user_label(outcome.own, users), correct)
        _add_score(lab_scores, _lab_label(outcome.own, users), correct)
    return _ranked_rows(user_scores, "User"), _ranked_rows(lab_scores, "Lab")


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

    _, challenges = get_available_explain_challenges(UserRole.PARTICIPANT)
    users = {user.id: user for user in storage.fetch_users()}
    explain_outcomes = storage.fetch_all_explain_outcomes()
    rate_outcomes = storage.fetch_all_rate_outcomes()

    for challenge_id, challenge in challenges.items():
        st.subheader(CHALLENGE_NAMES[challenge_id])
        explain_user_rows, explain_lab_rows = _score_explain(
            challenge, explain_outcomes, users
        )
        rate_user_rows, rate_lab_rows = _score_rate(challenge, rate_outcomes, users)

        explain_users, explain_labs, rate_users, rate_labs = st.columns(4)
        with explain_users:
            st.markdown("**Single specimen users**")
            _render_board(explain_user_rows)
        with explain_labs:
            st.markdown("**Single specimen labs**")
            _render_board(explain_lab_rows)
        with rate_users:
            st.markdown("**AI detection users**")
            _render_board(rate_user_rows)
        with rate_labs:
            st.markdown("**AI detection labs**")
            _render_board(rate_lab_rows)
