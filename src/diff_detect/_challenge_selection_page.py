from typing import assert_never

import streamlit as st

from diff_detect.common import CHALLENGE_NAMES

from ._page_utils import PageKey
from ._state import state
from ._storage import storage
from .models import ActiveExplainChallenge, ActiveRateChallenge


def render_challenge_selection_page() -> PageKey | None:
    st.set_page_config(layout="centered")
    user = state.user
    if user is None:
        return "login"

    st.header("Challenges")
    challenge_data = storage.fetch_challenges(user)
    if not challenge_data.explain_challenges:
        st.error("No challenges found.")
        st.stop()

    explain_col, rate_col = st.columns(2)
    with explain_col:
        st.subheader("🔍 Find the Imposter")
    with rate_col:
        st.subheader("🕵 Detective Showdown")

    for (
        explain_challenge_id,
        explain_challenge,
    ) in challenge_data.explain_challenges.items():
        explain_col, rate_col = st.columns(2)
        with explain_col:
            if st.button(
                CHALLENGE_NAMES[explain_challenge_id], key=explain_challenge_id
            ):
                # challenge = challenge_data.explain_challenges[explain_challenge_id]
                state.active_challenge = ActiveExplainChallenge(
                    challenge_data=challenge_data, challenge_id=explain_challenge_id
                )
                state.task_idx = explain_challenge.first_undone or 0
                return "task"
            st.progress(
                explain_challenge.progress,
                text=f"{explain_challenge.done_count}/{explain_challenge.task_count}",
            )

        if explain_challenge_id == "explain_dummy":
            rate_challenge_id = "rate_dummy"
        elif explain_challenge_id == "explain_butterfly_easy":
            rate_challenge_id = "rate_butterfly_easy"
        elif explain_challenge_id == "explain_butterfly_difficult":
            rate_challenge_id = "rate_butterfly_difficult"
        elif explain_challenge_id == "explain_flybutter_easy":
            rate_challenge_id = "rate_flybutter_easy"
        else:
            assert_never(explain_challenge_id)

        rate_challenge = challenge_data.challenges.get(rate_challenge_id)
        with rate_col:
            if (
                st.button(
                    CHALLENGE_NAMES[rate_challenge_id],
                    key=rate_challenge_id,
                    disabled=rate_challenge is None,
                )
                and rate_challenge is not None
            ):
                state.active_challenge = ActiveRateChallenge(
                    challenge_data=challenge_data, challenge_id=rate_challenge_id
                )
                state.task_idx = rate_challenge.first_undone or 0
                return "task"
            st.progress(
                0 if rate_challenge is None else rate_challenge.progress,
                text=f"{0 if rate_challenge is None else rate_challenge.done_count}/{explain_challenge.task_count if rate_challenge is None else rate_challenge.task_count}",
            )
